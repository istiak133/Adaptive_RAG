"""Full ingestion pipeline orchestrator.

One command populates PostgreSQL (sections, chunks, topics, section_topics,
chunk_topics) and ChromaDB (chunks + embeddings + metadata) from the source
PDF. Idempotent — safe to re-run; existing rows are dropped first.

Run with:
    .venv/bin/python -m src.ingestion.indexer
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from src.config import settings
from src.ingestion.chunker import chunk_sub_sections
from src.ingestion.embedder import get_embedder
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.topic_tagger import (
    derive_section_topics,
    extract_topics_from_glossary,
    tag_chunks,
)
from src.kb.chroma_repo import ChromaRepo
from src.kb.models import (
    Chunk as ChunkModel,
    ChunkTopic,
    Section,
    SectionTopic,
    Topic as TopicModel,
)


def run_indexer() -> Dict[str, int]:
    print("=" * 70)
    print("INGESTION PIPELINE")
    print("=" * 70)

    t_total = time.time()

    # ── Pipeline stages 1-4: parse → chunk → tag → embed ─────────────
    print("\n[1/5] Parsing PDF...")
    t0 = time.time()
    sub_sections = parse_pdf(settings.paths.pdf_corpus)
    print(f"  {len(sub_sections)} sub-sections  ({time.time() - t0:.1f}s)")

    print("\n[2/5] Chunking...")
    t0 = time.time()
    chunks = chunk_sub_sections(sub_sections)
    print(f"  {len(chunks)} chunks  ({time.time() - t0:.1f}s)")

    print("\n[3/5] Topic tagging...")
    t0 = time.time()
    topics = extract_topics_from_glossary(chunks)
    chunk_tags = tag_chunks(chunks, topics)
    section_links = derive_section_topics(chunks, chunk_tags)
    print(f"  {len(topics)} topics, {len(section_links)} section-topic links  "
          f"({time.time() - t0:.1f}s)")

    print("\n[4/5] Embedding chunks...")
    t0 = time.time()
    embedder = get_embedder()
    texts = [c.content for c in chunks]
    embeddings = embedder.embed(texts, show_progress=False)
    print(f"  {embeddings.shape}  ({time.time() - t0:.1f}s)")

    # ── Stage 5: write to PostgreSQL + ChromaDB ──────────────────────
    print("\n[5/5] Writing to databases...")
    t0 = time.time()

    # Build section metadata from sub-section spans
    section_meta: OrderedDict = OrderedDict()
    for ss in sub_sections:
        if ss.section_id not in section_meta:
            section_meta[ss.section_id] = {
                "id": ss.section_id,
                "title": ss.section_title,
                "page_start": ss.page_start,
                "page_end": ss.page_end,
                "chunk_count": 0,
            }
        else:
            section_meta[ss.section_id]["page_end"] = ss.page_end

    for c in chunks:
        if c.section_id in section_meta:
            section_meta[c.section_id]["chunk_count"] += 1

    engine = create_engine(settings.secrets.database_url, pool_pre_ping=True)
    chunk_topics_inserted = 0

    with Session(engine) as session:
        # Clear existing rows in dependency-safe order (idempotent re-run)
        print("  Clearing existing rows...")
        session.execute(delete(ChunkTopic))
        session.execute(delete(SectionTopic))
        session.execute(delete(ChunkModel))
        session.execute(delete(TopicModel))
        session.execute(delete(Section))
        session.commit()

        # Sections
        print("  Inserting sections...")
        for sec in section_meta.values():
            session.add(Section(
                id=sec["id"],
                title=sec["title"],
                page_start=sec["page_start"],
                page_end=sec["page_end"],
                chunk_count=sec["chunk_count"],
            ))
        session.commit()

        # Topics — need IDs after commit for FK references
        print("  Inserting topics...")
        topic_objs = []
        for t in topics:
            obj = TopicModel(
                name=t.name,
                slug=t.slug,
                description=t.description,
                category=t.category,
                importance=t.importance,
            )
            session.add(obj)
            topic_objs.append(obj)
        session.commit()
        slug_to_topic_id = {t.slug: obj.id for t, obj in zip(topics, topic_objs)}

        # Section-topic links
        print("  Inserting section_topics...")
        for link in section_links:
            topic_id = slug_to_topic_id.get(link.topic_slug)
            if topic_id is None:
                continue
            session.add(SectionTopic(
                section_id=link.section_id,
                topic_id=topic_id,
                depth=link.depth,
                relevance_score=link.relevance_score,
            ))
        session.commit()

        # Chunks
        print("  Inserting chunks...")
        chunk_objs = []
        for idx, c in enumerate(chunks):
            obj = ChunkModel(
                section_id=c.section_id,
                sub_section_id=c.sub_section_id,
                sub_section_title=c.sub_section_title,
                content=c.content,
                token_count=c.token_count,
                chunk_order=idx,
                page_start=c.page_start,
                page_end=c.page_end,
                has_table=c.has_table,
                has_bullets=c.has_bullets,
                chunk_kind=c.chunk_kind,
                cross_refs=c.cross_refs,
            )
            session.add(obj)
            chunk_objs.append(obj)
        session.commit()
        sub_id_to_chunk_id = {c.sub_section_id: obj.id
                              for c, obj in zip(chunks, chunk_objs)}

        # Chunk-topic links
        print("  Inserting chunk_topics...")
        for tag in chunk_tags:
            chunk_sub_id = chunks[tag.chunk_index].sub_section_id
            chunk_id = sub_id_to_chunk_id.get(chunk_sub_id)
            if chunk_id is None:
                continue
            for slug in tag.topic_slugs:
                topic_id = slug_to_topic_id.get(slug)
                if topic_id is None:
                    continue
                session.add(ChunkTopic(chunk_id=chunk_id, topic_id=topic_id))
                chunk_topics_inserted += 1
        session.commit()
        print(f"    {chunk_topics_inserted} chunk-topic links")

    # ChromaDB
    print("  Writing to ChromaDB...")
    repo = ChromaRepo()
    repo.reset()
    n_chroma = repo.add_chunks(chunks, embeddings)
    print(f"    {n_chroma} chunks in ChromaDB")

    db_elapsed = time.time() - t0
    total_elapsed = time.time() - t_total

    print(f"\n  DB write time:   {db_elapsed:.1f}s")
    print(f"  TOTAL elapsed:   {total_elapsed:.1f}s")
    print()
    print("=" * 70)
    print("✓ INGESTION COMPLETE")
    print("=" * 70)

    return {
        "sub_sections": len(sub_sections),
        "chunks": len(chunks),
        "topics": len(topics),
        "section_topics": len(section_links),
        "chunk_topics": chunk_topics_inserted,
        "embeddings": int(embeddings.shape[0]),
        "chromadb_count": n_chroma,
        "elapsed_seconds": round(total_elapsed, 1),
    }


def verify_db_state() -> None:
    """Quick post-ingestion verification."""
    from sqlalchemy import func, select

    engine = create_engine(settings.secrets.database_url)
    print("\n=== POST-INGESTION VERIFICATION ===\n")
    print("PostgreSQL row counts:")
    with Session(engine) as session:
        for model, label in [
            (Section, "sections"),
            (TopicModel, "topics"),
            (SectionTopic, "section_topics"),
            (ChunkModel, "chunks"),
            (ChunkTopic, "chunk_topics"),
        ]:
            count = session.scalar(select(func.count()).select_from(model))
            print(f"  {label:<20} {count}")

    repo = ChromaRepo()
    print(f"\nChromaDB:")
    print(f"  collection count    {repo.count()}")

    # Spot-check: §2.13 round-trip
    print("\nSpot-check: §2.13 (Echo Lock) round-trip from PostgreSQL → ChromaDB")
    with Session(engine) as session:
        row = session.execute(
            select(ChunkModel).where(ChunkModel.sub_section_id == "2.13")
        ).scalar_one_or_none()
        if row:
            print(f"  Postgres:  id={row.id}, tokens={row.token_count}, "
                  f"title='{row.sub_section_title}'")
        else:
            print("  Postgres:  ✗ not found")

    chroma_record = repo.get_by_id("2.13")
    if chroma_record:
        print(f"  ChromaDB:  id='{chroma_record['id']}', "
              f"section={chroma_record['metadata']['section_id']}")
    else:
        print("  ChromaDB:  ✗ not found")


if __name__ == "__main__":
    stats = run_indexer()
    verify_db_state()
    print("\nFinal stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
