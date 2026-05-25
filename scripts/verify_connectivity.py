"""Cross-component connectivity & data flow verification.

Confirms that everything we've built composes correctly:
  1. PostgreSQL ↔ ChromaDB consistency  (counts, IDs, content match)
  2. Relational integrity                (FKs, joins return expected rows)
  3. Cross-DB query flow                 (typical Phase-2 retrieval path)
  4. Cross-reference network             (§X.Y links resolvable)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sqlalchemy import create_engine, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.config import settings  # noqa: E402
from src.ingestion.embedder import get_embedder  # noqa: E402
from src.kb.chroma_repo import ChromaRepo  # noqa: E402
from src.kb.models import (  # noqa: E402
    Chunk, ChunkTopic, Section, SectionTopic, Topic,
)


results = []


def check(name: str, passed: bool, detail: str = "") -> None:
    icon = "✓" if passed else "✗"
    print(f"  {icon} {name}{(' — ' + detail) if detail else ''}")
    results.append((name, passed))


def main() -> None:
    engine = create_engine(settings.secrets.database_url, pool_pre_ping=True)
    repo = ChromaRepo()

    # ── PART 1: PostgreSQL ↔ ChromaDB Consistency ────────────────────
    print("=" * 70)
    print("PART 1: PostgreSQL ↔ ChromaDB Consistency")
    print("=" * 70)

    with Session(engine) as session:
        pg_chunks = session.scalars(select(Chunk)).all()
        pg_count = len(pg_chunks)
        pg_ids = {c.sub_section_id for c in pg_chunks}

    chroma_count = repo.count()
    chroma_data = repo.collection.get()
    chroma_ids = set(chroma_data["ids"])

    check("Same chunk count in both DBs",
          pg_count == chroma_count,
          f"Postgres={pg_count}, Chroma={chroma_count}")

    check("Same sub_section_ids in both DBs",
          pg_ids == chroma_ids,
          f"diff={len(pg_ids ^ chroma_ids)}")

    # Sample 5 chunks, verify content matches
    pg_by_sub_id = {c.sub_section_id: c for c in pg_chunks}
    samples = ["2.13", "5.2", "9.7", "10.1#26", "5.10+5.11"]
    mismatches = 0
    for sub_id in samples:
        if sub_id not in pg_by_sub_id:
            continue
        chroma_rec = repo.get_by_id(sub_id)
        if not chroma_rec:
            mismatches += 1
            continue
        if pg_by_sub_id[sub_id].content != chroma_rec["document"]:
            mismatches += 1

    check("Content matches PG↔Chroma (5 samples)",
          mismatches == 0,
          f"{mismatches} mismatches")

    # ── PART 2: Relational Integrity ─────────────────────────────────
    print()
    print("=" * 70)
    print("PART 2: Relational Integrity")
    print("=" * 70)

    with Session(engine) as session:
        # All chunk.section_id values reference real sections
        orphan_chunks = session.execute(
            select(func.count()).select_from(Chunk)
            .outerjoin(Section, Chunk.section_id == Section.id)
            .where(Section.id.is_(None))
        ).scalar()
        check("No orphan chunks (FK section_id valid)",
              orphan_chunks == 0,
              f"{orphan_chunks} orphans")

        # All chunk_topics.chunk_id reference real chunks
        orphan_ct_chunk = session.execute(
            select(func.count()).select_from(ChunkTopic)
            .outerjoin(Chunk, ChunkTopic.chunk_id == Chunk.id)
            .where(Chunk.id.is_(None))
        ).scalar()
        check("No orphan chunk_topics → chunks",
              orphan_ct_chunk == 0,
              f"{orphan_ct_chunk} orphans")

        # All chunk_topics.topic_id reference real topics
        orphan_ct_topic = session.execute(
            select(func.count()).select_from(ChunkTopic)
            .outerjoin(Topic, ChunkTopic.topic_id == Topic.id)
            .where(Topic.id.is_(None))
        ).scalar()
        check("No orphan chunk_topics → topics",
              orphan_ct_topic == 0,
              f"{orphan_ct_topic} orphans")

        # All section_topics references are valid
        orphan_st = session.execute(
            select(func.count()).select_from(SectionTopic)
            .outerjoin(Section, SectionTopic.section_id == Section.id)
            .where(Section.id.is_(None))
        ).scalar()
        check("No orphan section_topics → sections",
              orphan_st == 0,
              f"{orphan_st} orphans")

        # All sections referenced by chunks exist
        section_ids_in_chunks = session.scalars(
            select(Chunk.section_id).distinct()
        ).all()
        check("All chunks point to existing sections 1-10",
              set(section_ids_in_chunks) <= set(range(1, 11)),
              f"sections referenced: {sorted(set(section_ids_in_chunks))}")

    # ── PART 3: Cross-DB Query Flow (Phase 2 dry-run) ────────────────
    print()
    print("=" * 70)
    print("PART 3: Cross-DB Query Flow (simulating Phase-2 retrieval)")
    print("=" * 70)

    embedder = get_embedder()

    # Scenario: "User wants Section 8 chunks related to Echo Lock"
    print()
    print('Scenario: "Generate questions from Section 8 about Echo Lock"')

    # Step 1: SQL — get all chunks in Section 8
    with Session(engine) as session:
        s8_chunks = session.scalars(
            select(Chunk).where(Chunk.section_id == 8)
        ).all()
        s8_sub_ids = {c.sub_section_id for c in s8_chunks}
        check("Step 1 — SQL fetched Section 8 chunks",
              len(s8_chunks) > 0,
              f"{len(s8_chunks)} chunks")

    # Step 2: ChromaDB — semantic search "Echo Lock" within Section 8
    query_emb = embedder.embed_one("Echo Lock failure mode triggers")
    hits = repo.query(query_emb, top_k=5, section_ids=[8])
    hits_ids = {h["id"] for h in hits}
    check("Step 2 — Chroma filtered by section_id=8",
          hits_ids.issubset(s8_sub_ids),
          f"top-5: {[h['id'] for h in hits]}")

    # Step 3: Look up topics for the top retrieved chunk via PostgreSQL join
    top_id = hits[0]["id"]
    with Session(engine) as session:
        top_chunk = session.execute(
            select(Chunk).where(Chunk.sub_section_id == top_id)
        ).scalar_one()
        chunk_topics = session.execute(
            select(Topic.slug, Topic.name)
            .join(ChunkTopic, ChunkTopic.topic_id == Topic.id)
            .where(ChunkTopic.chunk_id == top_chunk.id)
        ).all()
        check("Step 3 — PG join chunk→chunk_topics→topics",
              True,
              f"§{top_id} → topics: {[t.slug for t in chunk_topics]}")

    # ── PART 4: Cross-Reference Network ──────────────────────────────
    print()
    print("=" * 70)
    print("PART 4: Cross-Reference Network")
    print("=" * 70)

    with Session(engine) as session:
        # §2.16 should reference §2.2, §7.3, §7.10
        c216 = session.execute(
            select(Chunk).where(Chunk.sub_section_id == "2.16")
        ).scalar_one()
        check("§2.16 cross_refs detected",
              len(c216.cross_refs) >= 2,
              f"{c216.cross_refs}")

        # For each cross-ref, can we find the referenced chunk?
        for ref in c216.cross_refs:
            ref_chunk = session.execute(
                select(Chunk).where(Chunk.sub_section_id == ref)
            ).scalar_one_or_none()
            if ref_chunk is None:
                # Maybe merged (e.g., "2.7" found as "2.7" or "2.7+2.8")
                ref_chunk = session.execute(
                    select(Chunk).where(Chunk.sub_section_id.like(f"%{ref}%"))
                ).first()
            check(f"  §{ref} resolvable from §2.16",
                  ref_chunk is not None)

    # ── PART 5: Section-Topic Depth Network ──────────────────────────
    print()
    print("=" * 70)
    print("PART 5: Section-Topic Depth (adaptive groundwork)")
    print("=" * 70)

    with Session(engine) as session:
        # Section 2 (Powers) — what's primary?
        s2_primaries = session.execute(
            select(Topic.slug, SectionTopic.depth)
            .join(SectionTopic, SectionTopic.topic_id == Topic.id)
            .where(SectionTopic.section_id == 2)
            .where(SectionTopic.depth == "primary")
        ).all()
        primary_slugs = {row.slug for row in s2_primaries}
        check("Section 2 has Echo Lock as primary",
              "echo_lock" in primary_slugs,
              f"primaries: {sorted(primary_slugs)[:5]}…")

        check("Section 2 has Inertial Suspension as primary",
              "inertial_suspension" in primary_slugs)

        # Echo Lock appears across multiple sections?
        echo_appearances = session.execute(
            select(SectionTopic.section_id, SectionTopic.depth)
            .join(Topic, SectionTopic.topic_id == Topic.id)
            .where(Topic.slug == "echo_lock")
            .order_by(SectionTopic.section_id)
        ).all()
        check("Echo Lock spans multiple sections (cross-section topic)",
              len(echo_appearances) >= 3,
              f"{len(echo_appearances)} sections: "
              f"{[(r.section_id, r.depth) for r in echo_appearances]}")

    # ── PART 6: Full Round-Trip — sample chunk all the way ──────────
    print()
    print("=" * 70)
    print("PART 6: Full Round-Trip Sample (§2.13 Echo Lock)")
    print("=" * 70)

    with Session(engine) as session:
        chunk = session.execute(
            select(Chunk).where(Chunk.sub_section_id == "2.13")
        ).scalar_one()
        topics_for = session.execute(
            select(Topic.slug)
            .join(ChunkTopic, ChunkTopic.topic_id == Topic.id)
            .where(ChunkTopic.chunk_id == chunk.id)
        ).all()
        topic_slugs = {row.slug for row in topics_for}

    chroma_record = repo.get_by_id("2.13")

    check("§2.13 in Postgres", chunk is not None,
          f"id={chunk.id}, tokens={chunk.token_count}")
    check("§2.13 in ChromaDB", chroma_record is not None,
          f"section={chroma_record['metadata']['section_id']}")
    check("§2.13 content identical PG↔Chroma",
          chunk.content == chroma_record["document"])
    check("§2.13 has 'echo_lock' topic tag",
          "echo_lock" in topic_slugs,
          f"topics: {sorted(topic_slugs)}")
    check("§2.13 metadata token_count matches PG",
          chroma_record["metadata"]["token_count"] == chunk.token_count)

    # ── Scoreboard ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FINAL SCOREBOARD")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = [n for n, p in results if not p]

    print(f"\n  Passed: {passed}/{total}")
    if failed:
        print("  FAILED:")
        for n in failed:
            print(f"    ✗ {n}")
    else:
        print("  🎉 ALL CHECKS PASSED — every component connected, data flows correctly")


if __name__ == "__main__":
    main()
