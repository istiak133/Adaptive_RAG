"""End-to-end verification of everything built so far.

Run with:  .venv/bin/python -m scripts.verify_pipeline
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2  # noqa: E402
import numpy as np  # noqa: E402

# Track results
results = []


def check(name: str, passed: bool, detail: str = "") -> None:
    icon = "✓" if passed else "✗"
    print(f"  {icon} {name}{(' — ' + detail) if detail else ''}")
    results.append((name, passed))


def main() -> None:
    # PART 1: Foundation
    print("=" * 70)
    print("PART 1: FOUNDATION CHECKS")
    print("=" * 70)
    print()

    print("Python & environment:")
    import platform
    check("Python 3.9+", sys.version_info >= (3, 9), f"got {platform.python_version()}")

    from src.config import settings
    check(".env: GROQ_API_KEY set", bool(settings.secrets.groq_api_key))
    check(".env: GOOGLE_API_KEY set", bool(settings.secrets.google_api_key))
    check(".env: DATABASE_URL set", bool(settings.secrets.database_url))

    print()
    print("Config loader:")
    check("config.yaml loaded", True, f"app: {settings.app.name}")
    check("17 config sections present", len(settings.model_fields) >= 17)

    tm = settings.token_management
    check(
        "Token budget arithmetic sound",
        (tm.system_prompt_budget + tm.instructions_budget
         + tm.history_budget + tm.content_budget + tm.output_reserve)
        == tm.context_limit,
    )

    print()
    print("Supabase connectivity:")
    try:
        conn = psycopg2.connect(settings.secrets.database_url, connect_timeout=10)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        pg_version = cur.fetchone()[0].split(",")[0]
        check("Supabase reachable", True, pg_version)

        cur.execute("""
            SELECT count(*) FROM information_schema.tables
            WHERE table_schema='public' AND table_type='BASE TABLE';
        """)
        table_count = cur.fetchone()[0]
        check("12 tables in DB (11 + alembic_version)",
              table_count == 12, f"got {table_count}")

        cur.execute("SELECT version_num FROM alembic_version;")
        av = cur.fetchone()
        check("Alembic version present", av is not None,
              av[0] if av else "none")

        cur.execute("SELECT count(*) FROM pg_constraint WHERE contype='f';")
        fk_count = cur.fetchone()[0]
        check("Foreign key constraints ≥ 30", fk_count >= 30,
              f"{fk_count} FKs")

        cur.close()
        conn.close()
    except Exception as e:
        check("Supabase reachable", False, str(e)[:80])

    print()
    print("FastAPI / module imports:")
    try:
        from src.main import app
        check("src.main imports", True, f"FastAPI title: {app.title}")
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        check("/api/v1/health route registered",
              "/api/v1/health" in routes)
    except Exception as e:
        check("src.main imports", False, str(e)[:80])

    try:
        from src.kb.models import Base
        check("SQLAlchemy models import",
              True, f"{len(Base.metadata.tables)} tables")
    except Exception as e:
        check("SQLAlchemy models import", False, str(e)[:80])

    # PART 2: Pipeline Components
    print()
    print("=" * 70)
    print("PART 2: PIPELINE COMPONENTS")
    print("=" * 70)
    print()
    print("Parser:")

    from src.ingestion.pdf_parser import parse_pdf
    t0 = time.time()
    sub_sections = parse_pdf(settings.paths.pdf_corpus)
    parse_time = time.time() - t0
    check("Parser runs without error", True, f"{parse_time:.1f}s")
    check("133 sub-sections parsed",
          len(sub_sections) == 133, f"got {len(sub_sections)}")

    section_ids = set(ss.section_id for ss in sub_sections)
    check("All 10 sections present", section_ids == set(range(1, 11)))

    multi_level = [ss for ss in sub_sections
                   if ss.sub_section_id.count(".") == 2]
    check("§9.9.5/6/7 captured (multi-level)",
          len(multi_level) == 3,
          ", ".join(m.sub_section_id for m in multi_level))

    empty = [ss for ss in sub_sections if not ss.content.strip()]
    check("No empty sub-sections", len(empty) == 0, f"{len(empty)} empty")

    print()
    print("Chunker:")
    from src.ingestion.chunker import chunk_sub_sections

    t0 = time.time()
    chunks = chunk_sub_sections(sub_sections)
    chunk_time = time.time() - t0
    check("Chunker runs without error", True, f"{chunk_time:.1f}s")
    check("Got 195 chunks", len(chunks) == 195, f"got {len(chunks)}")

    max_tokens = max(c.token_count for c in chunks)
    check("All chunks ≤ 500 tokens",
          max_tokens <= 500, f"max={max_tokens}")

    kinds = Counter(c.chunk_kind for c in chunks)
    check("Narrative chunks present", kinds["narrative"] >= 100,
          f"{kinds['narrative']} narrative")
    check("Glossary entries present", kinds["glossary_entry"] >= 60,
          f"{kinds['glossary_entry']} glossary")

    narrative_with_title = [
        c for c in chunks
        if c.chunk_kind == "narrative"
        and c.content.startswith(f"§{c.sub_section_id.split('+')[0]}")
    ]
    check("Titles prepended in narrative chunks",
          len(narrative_with_title) >= 100,
          f"{len(narrative_with_title)}/{kinds['narrative']}")

    cross_refs_count = sum(1 for c in chunks if c.cross_refs)
    check("Cross-refs extracted", cross_refs_count >= 40,
          f"{cross_refs_count} chunks have cross-refs")

    chunk_section_ids = set(c.section_id for c in chunks)
    check("All chunk section_ids valid",
          chunk_section_ids.issubset(section_ids))

    print()
    print("Topic tagger:")
    from src.ingestion.topic_tagger import (
        CRITICAL_TOPIC_NAMES,
        derive_section_topics,
        extract_topics_from_glossary,
        tag_chunks,
    )

    topics = extract_topics_from_glossary(chunks)
    check("Topics extracted from glossary",
          len(topics) >= 65, f"{len(topics)} topics")

    critical_found = sum(1 for t in topics if t.importance == "critical")
    check(
        f"All {len(CRITICAL_TOPIC_NAMES)} critical topics matched",
        critical_found == len(CRITICAL_TOPIC_NAMES),
        f"{critical_found}/{len(CRITICAL_TOPIC_NAMES)}",
    )

    chunk_tags = tag_chunks(chunks, topics)
    tagged_count = sum(1 for t in chunk_tags if t.topic_slugs)
    coverage = tagged_count / len(chunks) * 100
    check("Chunk-topic coverage ≥ 90%",
          coverage >= 90, f"{coverage:.1f}%")

    section_links = derive_section_topics(chunks, chunk_tags)
    check("Section-topic links derived",
          len(section_links) >= 200, f"{len(section_links)} links")

    depth_counts = Counter(link.depth for link in section_links)
    check("All three depth levels used",
          all(d in depth_counts for d in
              ("primary", "secondary", "mention")))

    print()
    print("Embedder:")
    from src.ingestion.embedder import get_embedder

    t0 = time.time()
    embedder = get_embedder()
    embedder._ensure_loaded()
    load_time = time.time() - t0
    check("Embedder loads", True, f"{load_time:.1f}s")

    texts = [c.content for c in chunks]
    t0 = time.time()
    embeddings = embedder.embed(texts, show_progress=False)
    embed_time = time.time() - t0
    check("All chunks embedded",
          embeddings.shape[0] == len(chunks), f"{embed_time:.1f}s")
    check("Embedding dim = 384", embeddings.shape[1] == 384)

    norms = np.linalg.norm(embeddings, axis=1)
    check("All vectors normalised (L2 ≈ 1)",
          np.allclose(norms, 1.0, atol=1e-3))

    # PART 3: Integration — Retrieval tests
    print()
    print("=" * 70)
    print("PART 3: REAL QUERY RETRIEVAL TESTS")
    print("=" * 70)
    print()
    print("For each query, EXPECTED chunk should appear in top-3 results.")
    print()

    test_queries = [
        ("What triggers Echo Lock?", "2.13"),
        ("Mass ceiling for Inertial Suspension", "2.3"),
        ("Effective range of suspension", "2.5"),
        ("Andina-7 hardshell uniform specs", "4.2"),
        ("Tungsten throwing baton mass and length", "4.4"),
        ("Three-Two-One Rule three activations", "5.2"),
        ("Twelve combat directives", "5.10+5.11"),
        ("Salta engagement catastrophic outcome", "9.7"),
        ("Tropopausa pressure manipulator powers", "7.3"),
        ("Quebrantadero bone density manipulation", "7.5"),
        ("Cuartel Valparaíso primary base", "8.1"),
        ("PAMC directorate structure", "6.1"),
    ]

    passed_count = 0
    for query, expected_sub_id in test_queries:
        query_emb = embedder.embed_one(query)
        sims = embeddings @ query_emb
        top3_indices = np.argsort(-sims)[:3]
        top3_ids = [chunks[i].sub_section_id for i in top3_indices]

        if expected_sub_id in top3_ids:
            passed_count += 1
            rank = top3_ids.index(expected_sub_id) + 1
            score = sims[top3_indices[top3_ids.index(expected_sub_id)]]
            print(f"  ✓ '{query[:50]:<52}' → §{expected_sub_id} "
                  f"(rank #{rank}, score {score:.3f})")
        else:
            print(f"  ✗ '{query[:50]:<52}' → expected §{expected_sub_id}, "
                  f"got top-3: {top3_ids}")

    print()
    print(f"  Retrieval accuracy: {passed_count}/{len(test_queries)} "
          f"({passed_count*100/len(test_queries):.0f}%)")
    check("Retrieval ≥ 80% accuracy",
          passed_count >= 0.8 * len(test_queries),
          f"{passed_count}/{len(test_queries)}")

    # Final scoreboard
    print()
    print("=" * 70)
    print("FINAL SCOREBOARD")
    print("=" * 70)
    print()

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed_items = [n for n, p in results if not p]

    print(f"  Total checks: {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {total - passed}")
    print()

    if failed_items:
        print("  FAILED:")
        for n in failed_items:
            print(f"    ✗ {n}")
    else:
        print("  🎉 ALL CHECKS PASSED")

    print()
    print(f"  Score: {passed/total*100:.1f}%")


if __name__ == "__main__":
    main()
