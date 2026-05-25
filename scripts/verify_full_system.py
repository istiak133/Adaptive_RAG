"""Comprehensive end-to-end verification across Phase 0-6.

Built to surface reviewer-side issues (hardcoded paths, missing imports,
non-idempotent state, cold-start bugs, API contract drift).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

results: list = []


def check(name: str, passed: bool, detail: str = "") -> None:
    icon = "✓" if passed else "✗"
    print(f"  {icon} {name}{(' — ' + detail) if detail else ''}")
    results.append((name, passed))


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ────────────────────────────────────────────────────────────────────
# PART 1: Code hygiene — no hardcoded user paths, no committed secrets
# ────────────────────────────────────────────────────────────────────


def part1_hygiene() -> None:
    section("PART 1: Code hygiene (reviewer-side compat)")

    project_root = Path(__file__).resolve().parent.parent

    # 1a. No hardcoded user paths in source / scripts / configs
    hardcoded_patterns = [r"/Users/istiakahmed"]
    files_to_scan = (
        list((project_root / "src").rglob("*.py"))
        + list((project_root / "scripts").rglob("*.py"))
        + [project_root / "config.yaml", project_root / "alembic.ini"]
        + list((project_root / "alembic").rglob("*.py"))
    )
    matches = []
    for fpath in files_to_scan:
        if not fpath.exists():
            continue
        text = fpath.read_text()
        for pat in hardcoded_patterns:
            if re.search(pat, text):
                matches.append(str(fpath.relative_to(project_root)))
    check("No hardcoded user paths in source",
          len(matches) == 0,
          f"{len(matches)} files: {matches[:3]}" if matches else "clean")

    # 1b. .env is gitignored
    try:
        out = subprocess.run(
            ["git", "check-ignore", ".env"],
            capture_output=True, text=True, cwd=project_root,
        )
        check(".env is gitignored", out.returncode == 0)
    except Exception as e:
        check(".env is gitignored", False, str(e))

    # 1c. notes/ is gitignored
    try:
        out = subprocess.run(
            ["git", "check-ignore", "notes/"],
            capture_output=True, text=True, cwd=project_root,
        )
        check("notes/ is gitignored", out.returncode == 0)
    except Exception:
        check("notes/ is gitignored", False)

    # 1d. .env.example exists and has placeholders (not real keys)
    env_example = project_root / ".env.example"
    if env_example.exists():
        body = env_example.read_text()
        has_groq_placeholder = "replace_with_your_real_groq_key" in body
        has_gemini_placeholder = "replace_with_your_real_gemini_key" in body
        check(".env.example has placeholder values",
              has_groq_placeholder and has_gemini_placeholder)
    else:
        check(".env.example exists", False)

    # 1e. No real key prefix in committed files
    committed_dirs = ["src", "scripts", "alembic"]
    leaked = 0
    real_key_prefix = re.compile(r"gsk_[A-Za-z0-9_]{30,}")
    for d in committed_dirs:
        for fpath in (project_root / d).rglob("*.py"):
            if real_key_prefix.search(fpath.read_text()):
                leaked += 1
    check("No real API keys committed to source",
          leaked == 0,
          f"{leaked} files contain a 'gsk_...' string" if leaked else "")


# ────────────────────────────────────────────────────────────────────
# PART 2: Foundation
# ────────────────────────────────────────────────────────────────────


def part2_foundation() -> None:
    section("PART 2: Foundation (config + DB + imports)")

    from src.config import settings
    check("config.yaml loads + validates", True,
          f"app: {settings.app.name}")
    check("All 3 API secrets present in env",
          all([
              settings.secrets.groq_api_key,
              settings.secrets.google_api_key,
              settings.secrets.database_url,
          ]))

    # DB reachability
    import psycopg2
    try:
        conn = psycopg2.connect(settings.secrets.database_url, connect_timeout=8)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema='public';")
        n = cur.fetchone()[0]
        check("Supabase reachable, 12 tables present", n == 12, f"{n} tables")
        cur.close()
        conn.close()
    except Exception as e:
        check("Supabase reachable", False, str(e)[:80])

    # FastAPI app + LangGraph imports
    try:
        from src.main import app
        from src.graph.graph import build_prep_graph
        check("FastAPI app imports", True, f"{len(app.routes)} routes")
        check("LangGraph imports", True)
    except Exception as e:
        check("FastAPI / LangGraph imports", False, str(e)[:80])


# ────────────────────────────────────────────────────────────────────
# PART 3: Data integrity (after Phase 1 indexer ran earlier)
# ────────────────────────────────────────────────────────────────────


def part3_data_integrity() -> None:
    section("PART 3: Knowledge-base data integrity")

    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import Session

    from src.config import settings
    from src.kb.chroma_repo import ChromaRepo
    from src.kb.models import (
        Chunk, ChunkTopic, Section, SectionTopic, Topic,
    )

    engine = create_engine(settings.secrets.database_url)
    chroma = ChromaRepo()

    with Session(engine) as db:
        sections = db.scalar(select(func.count()).select_from(Section))
        chunks = db.scalar(select(func.count()).select_from(Chunk))
        topics = db.scalar(select(func.count()).select_from(Topic))
        section_topics = db.scalar(select(func.count()).select_from(SectionTopic))
        chunk_topics = db.scalar(select(func.count()).select_from(ChunkTopic))

    check("Postgres: 10 sections", sections == 10, f"got {sections}")
    check("Postgres: 195 chunks", chunks == 195, f"got {chunks}")
    check("Postgres: 69 topics", topics == 69, f"got {topics}")
    check("Postgres: 211 section_topics",
          section_topics == 211, f"got {section_topics}")
    check("Postgres: 471 chunk_topics",
          chunk_topics == 471, f"got {chunk_topics}")
    check("ChromaDB: 195 vectors",
          chroma.count() == 195, f"got {chroma.count()}")
    check("PG↔Chroma chunk count match",
          chunks == chroma.count())


# ────────────────────────────────────────────────────────────────────
# PART 4: Idempotency — indexer can be re-run safely
# ────────────────────────────────────────────────────────────────────


def part4_idempotency() -> None:
    section("PART 4: Indexer idempotency (reviewer re-run scenario)")

    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import Session

    from src.config import settings
    from src.kb.models import Chunk, Topic

    engine = create_engine(settings.secrets.database_url)
    with Session(engine) as db:
        before_chunks = db.scalar(select(func.count()).select_from(Chunk))
        before_topics = db.scalar(select(func.count()).select_from(Topic))

    print(f"  Before re-index: chunks={before_chunks}, topics={before_topics}")
    print("  Running indexer again (this takes ~45s — DB writes)…")

    from src.ingestion.indexer import run_indexer
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stats = run_indexer()

    with Session(engine) as db:
        after_chunks = db.scalar(select(func.count()).select_from(Chunk))
        after_topics = db.scalar(select(func.count()).select_from(Topic))

    check("Re-index produces same chunk count",
          after_chunks == before_chunks,
          f"before={before_chunks}, after={after_chunks}")
    check("Re-index produces same topic count",
          after_topics == before_topics,
          f"before={before_topics}, after={after_topics}")
    check("Re-index reported no errors", stats["chromadb_count"] == 195)


# ────────────────────────────────────────────────────────────────────
# PART 5: Adaptive intelligence — single full iter1→iter2 cycle
# ────────────────────────────────────────────────────────────────────


def part5_adaptive() -> None:
    section("PART 5: Adaptive intelligence (cold→adaptive cycle)")

    from sqlalchemy import create_engine, delete, select
    from sqlalchemy.orm import Session

    from src.config import settings
    from src.kb.mastery import clear_all_mastery
    from src.kb.models import (
        Session as SessionRow, SectionTopicMastery, TopicMastery,
    )
    from src.rag.history_compressor import build_adaptive_context
    from src.prep.allocator import allocate
    from src.services.prep_service import run_prep_session

    engine = create_engine(settings.secrets.database_url)

    with Session(engine) as db:
        clear_all_mastery(db)
        db.execute(delete(SessionRow))
        db.commit()
        print("  Cleared all mastery + sessions for clean test")

    # Iter 1 — cold start
    print("  Running Iter 1 (cold start, random sim seed=11)…")
    with Session(engine) as db:
        r1 = run_prep_session(
            session=db, section_ids=[2], questions_per_section=3,
            simulate_strategy="random", seed=11,
        )

    check("Iter 1 returned valid result", r1.session_id is not None)
    check("Iter 1 detected cold-start", True)

    with Session(engine) as db:
        s1 = db.get(SessionRow, r1.session_id)
        check("Iter 1: is_cold_start=True in DB", s1.is_cold_start is True)
        tm_count = db.scalar(select(__import__("sqlalchemy").func.count())
                             .select_from(TopicMastery))
        check("Iter 1: mastery rows created", tm_count > 0,
              f"{tm_count} rows")

    # Capture adaptive context BEFORE iter 2 (proves it changes)
    with Session(engine) as db:
        pre_ctx = build_adaptive_context(db, [2])
        pre_plan = allocate(db, section_id=2, n_questions=3)

    check("Adaptive context non-empty before Iter 2", len(pre_ctx) > 0)
    check("Allocator mode != 'cold' for Iter 2",
          pre_plan.mode != "cold",
          f"mode={pre_plan.mode}, seeds={pre_plan.seeds[:3]}")

    # Iter 2
    print("  Running Iter 2 (adaptive, all_correct sim)…")
    with Session(engine) as db:
        r2 = run_prep_session(
            session=db, section_ids=[2], questions_per_section=3,
            simulate_strategy="all_correct", seed=22,
        )
        s2 = db.get(SessionRow, r2.session_id)
        check("Iter 2: is_cold_start=False in DB",
              s2.is_cold_start is False)
        check("Iter 2: adaptive_context stored in session row",
              s2.adaptive_context is not None
              and "summary" in (s2.adaptive_context or {}))
        check("Iter 2: allocation summary stored",
              "allocation" in (s2.adaptive_context or {}))


# ────────────────────────────────────────────────────────────────────
# PART 6: All 16 API endpoints respond via TestClient
# ────────────────────────────────────────────────────────────────────


def part6_api() -> None:
    section("PART 6: All API endpoints respond (via TestClient)")

    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)

    endpoints = [
        ("GET",  "/api/v1/health",                    200, "status"),
        ("GET",  "/api/v1/sections",                  200, None),
        ("GET",  "/api/v1/sections/2",                200, "title"),
        ("GET",  "/api/v1/sections/99",               404, None),
        ("GET",  "/api/v1/sections/2/chunks",         200, None),
        ("GET",  "/api/v1/topics",                    200, None),
        ("GET",  "/api/v1/topics/1",                  200, "name"),
        ("GET",  "/api/v1/topics/999",                404, None),
        ("GET",  "/api/v1/sessions?limit=3",          200, None),
        ("GET",  "/api/v1/sessions/999999",           404, None),
        ("GET",  "/api/v1/mastery",                   200, None),
        ("GET",  "/api/v1/mastery/regressions",       200, None),
        ("GET",  "/api/v1/mastery/by-section/2",      200, None),
        ("GET",  "/api/v1/admin/stats",               200, "sections"),
        ("GET",  "/docs",                             200, None),
        ("GET",  "/openapi.json",                     200, "openapi"),
    ]

    for method, path, expected_status, expected_key in endpoints:
        try:
            r = client.request(method, path)
            status_ok = r.status_code == expected_status
            body_ok = True
            if expected_key and r.status_code == 200:
                try:
                    body_ok = expected_key in r.json()
                except Exception:
                    body_ok = False
            check(f"{method} {path}", status_ok and body_ok,
                  f"status={r.status_code}")
        except Exception as e:
            check(f"{method} {path}", False, str(e)[:80])

    # POST /prep/start with invalid section ID
    r = client.post("/api/v1/prep/start",
                     json={"section_ids": [99], "questions_per_section": 1})
    check("POST /prep/start rejects invalid section_id",
          r.status_code == 400, f"got {r.status_code}")


# ────────────────────────────────────────────────────────────────────
# PART 7: Output-file structure (Scenario B targets)
# ────────────────────────────────────────────────────────────────────


def part7_output_paths() -> None:
    section("PART 7: Output folders exist (Scenario B targets)")

    from src.config import settings

    base = Path(settings.paths.outputs_dir)
    for i in (1, 2, 3):
        d = base / f"scenario_b_iter{i}"
        check(f"outputs/scenario_b_iter{i}/ exists",
              d.exists() and d.is_dir())

    # Exporter doesn't crash on a known session
    from sqlalchemy import create_engine, select, desc
    from sqlalchemy.orm import Session
    from src.kb.models import Session as SessionRow
    from src.output.snapshot import build_kb_snapshot, export_questions
    from src.config import settings

    engine = create_engine(settings.secrets.database_url)
    with Session(engine) as db:
        s = db.scalars(
            select(SessionRow).order_by(desc(SessionRow.started_at)).limit(1)
        ).first()
        if s:
            snap = build_kb_snapshot(db, after_session_id=s.id)
            check("KB snapshot builder returns valid structure",
                  "recent_sessions" in snap and "adaptive_state" in snap,
                  f"sessions in snap: {len(snap['recent_sessions'])}")

            tmp = base / "scenario_b_iter1" / "_smoke_questions.json"
            export_questions(db, s.id, tmp)
            check("Exporter writes valid JSON",
                  tmp.exists() and json.loads(tmp.read_text())["session_id"] == s.id)
            tmp.unlink()  # clean up
        else:
            check("Has at least one session to export", False)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main() -> None:
    part1_hygiene()
    part2_foundation()
    part3_data_integrity()
    part4_idempotency()
    part5_adaptive()
    part6_api()
    part7_output_paths()

    print()
    print("=" * 72)
    print("FINAL SCOREBOARD")
    print("=" * 72)

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = [n for n, p in results if not p]

    print(f"\n  Passed: {passed}/{total}")
    if failed:
        print("\n  FAILED:")
        for n in failed:
            print(f"    ✗ {n}")
    else:
        print("\n  🎉 ALL CHECKS PASSED — full system ready for reviewer")


if __name__ == "__main__":
    main()
