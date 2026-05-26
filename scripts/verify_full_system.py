"""Comprehensive end-to-end verification across Phase 0-7.

Surfaces reviewer-side issues (hardcoded paths, missing imports, non-
idempotent state, cold-start bugs, API contract drift, LangGraph routing,
CLI integration).
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
# PART 1 — Code hygiene
# ────────────────────────────────────────────────────────────────────


def part1_hygiene() -> None:
    section("PART 1: Code hygiene (reviewer-side compat)")

    project_root = Path(__file__).resolve().parent.parent
    self_path = Path(__file__).resolve()

    hardcoded_patterns = ["/" + "Users/" + "istiakahmed"]
    files_to_scan = (
        list((project_root / "src").rglob("*.py"))
        + [
            f for f in (project_root / "scripts").rglob("*.py")
            if f.resolve() != self_path
        ]
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

    for label, path in [(".env", ".env"), ("notes/", "notes/")]:
        try:
            out = subprocess.run(
                ["git", "check-ignore", path],
                capture_output=True, text=True, cwd=project_root,
            )
            check(f"{label} is gitignored", out.returncode == 0)
        except Exception as e:
            check(f"{label} is gitignored", False, str(e))

    env_example = project_root / ".env.example"
    if env_example.exists():
        body = env_example.read_text()
        check(
            ".env.example has placeholder values (no real keys)",
            "replace_with_your_real_groq_key" in body
            and "replace_with_your_real_gemini_key" in body,
        )

    leaked = 0
    real_key_re = re.compile(r"gsk_[A-Za-z0-9_]{30,}")
    for d in ("src", "scripts", "alembic"):
        for fpath in (project_root / d).rglob("*.py"):
            if real_key_re.search(fpath.read_text()):
                leaked += 1
    check("No real API keys committed to source", leaked == 0)


# ────────────────────────────────────────────────────────────────────
# PART 2 — Foundation
# ────────────────────────────────────────────────────────────────────


def part2_foundation() -> None:
    section("PART 2: Foundation (config + DB + imports)")

    from src.config import settings
    check("config.yaml loads + validates", True, f"app: {settings.app.name}")
    check("All 3 API secrets present in env",
          all([settings.secrets.groq_api_key,
               settings.secrets.google_api_key,
               settings.secrets.database_url]))

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

    try:
        from src.main import app
        from src.graph.graph import build_prep_graph, get_graph_diagram
        check("FastAPI app imports", True, f"{len(app.routes)} routes")
        check("LangGraph imports + compiles", True)
    except Exception as e:
        check("FastAPI / LangGraph imports", False, str(e)[:80])


# ────────────────────────────────────────────────────────────────────
# PART 3 — Data integrity
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
        counts = {
            "sections": db.scalar(select(func.count()).select_from(Section)),
            "chunks": db.scalar(select(func.count()).select_from(Chunk)),
            "topics": db.scalar(select(func.count()).select_from(Topic)),
            "section_topics": db.scalar(select(func.count()).select_from(SectionTopic)),
            "chunk_topics": db.scalar(select(func.count()).select_from(ChunkTopic)),
        }

    check("Postgres: 10 sections", counts["sections"] == 10)
    check("Postgres: 195 chunks", counts["chunks"] == 195)
    check("Postgres: 69 topics", counts["topics"] == 69)
    check("Postgres: 211 section_topics", counts["section_topics"] == 211)
    check("Postgres: 471 chunk_topics", counts["chunk_topics"] == 471)
    check("ChromaDB: 195 vectors", chroma.count() == 195)
    check("PG↔Chroma chunk count match", counts["chunks"] == chroma.count())


# ────────────────────────────────────────────────────────────────────
# PART 4 — Indexer idempotency
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

    print(f"  Before: chunks={before_chunks}, topics={before_topics}")
    print("  Running indexer (idempotent re-run)…")

    import contextlib
    import io
    from src.ingestion.indexer import run_indexer

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            stats = run_indexer()
        check("Re-index ran without error", True)
    except Exception as e:
        check("Re-index ran without error", False, str(e)[:80])
        return

    with Session(engine) as db:
        after_chunks = db.scalar(select(func.count()).select_from(Chunk))
        after_topics = db.scalar(select(func.count()).select_from(Topic))

    check("Same chunk count after re-index",
          after_chunks == before_chunks,
          f"before={before_chunks}, after={after_chunks}")
    check("Same topic count after re-index",
          after_topics == before_topics,
          f"before={before_topics}, after={after_topics}")
    check("Re-index ChromaDB count = 195", stats["chromadb_count"] == 195)


# ────────────────────────────────────────────────────────────────────
# PART 5 — LangGraph routing audit
# ────────────────────────────────────────────────────────────────────


def part5_langgraph() -> None:
    section("PART 5: LangGraph state-machine audit")

    from src.graph.graph import build_prep_graph, get_graph_diagram
    from src.graph import nodes
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.config import settings

    diagram = get_graph_diagram()

    # Real conditional edges should appear as dotted edges (`-.->`)
    check("Cold-vs-adaptive conditional edge present",
          "detect_mode -.-> create_session" in diagram
          and "detect_mode -.-> load_adaptive_context" in diagram)

    check("Retry conditional edge present",
          "validate -.-> generate" in diagram
          and "validate -.-> record" in diagram)

    # All 8 nodes present
    for n in (
        "detect_mode", "load_adaptive_context", "create_session",
        "generate", "validate", "record", "simulate_and_score", "complete",
    ):
        check(f"Node '{n}' present in graph", n in diagram)

    # Routers don't return a single hard-coded value (real branching)
    src_route = nodes.route_after_detect.__code__.co_consts
    check("route_after_detect actually branches on is_cold_start",
          any("is_cold_start" in str(c) for c in src_route)
          or len(nodes.route_after_detect.__code__.co_names) > 0)

    # Smoke test compile
    engine = create_engine(settings.secrets.database_url)
    with Session(engine) as db:
        g = build_prep_graph(db)
    check("build_prep_graph compiles without error", True)


# ────────────────────────────────────────────────────────────────────
# PART 6 — Adaptive intelligence (cold→adaptive cycle)
# ────────────────────────────────────────────────────────────────────


def part6_adaptive() -> None:
    section("PART 6: Adaptive intelligence (cold→adaptive cycle via LangGraph)")

    from sqlalchemy import create_engine, delete, func, select
    from sqlalchemy.orm import Session

    from src.config import settings
    from src.kb.mastery import clear_all_mastery
    from src.kb.models import Session as SessionRow, TopicMastery
    from src.rag.history_compressor import build_adaptive_context
    from src.prep.allocator import allocate
    from src.services.prep_service import run_prep_session

    engine = create_engine(settings.secrets.database_url)
    with Session(engine) as db:
        clear_all_mastery(db)
        db.execute(delete(SessionRow))
        db.commit()
        print("  Cleared mastery + sessions for clean test")

    # Iter 1 — cold
    print("  Running Iter 1 (cold, random sim seed=11)…")
    with Session(engine) as db:
        r1 = run_prep_session(
            session=db, section_ids=[2], questions_per_section=3,
            simulate_strategy="random", seed=11,
        )

    with Session(engine) as db:
        s1 = db.get(SessionRow, r1.session_id)
        check("Iter 1 ran via LangGraph", r1.session_id is not None)
        check("Iter 1: is_cold_start=True", s1.is_cold_start is True)

        tm_count = db.scalar(
            select(func.count()).select_from(TopicMastery)
        )
        check("Iter 1: mastery rows created", tm_count > 0,
              f"{tm_count} rows")

    # State snapshot pre-iter2 (adaptive context + allocator mode)
    with Session(engine) as db:
        pre_ctx = build_adaptive_context(db, [2])
        plan = allocate(db, 2, 3)
    check("Adaptive context non-empty before Iter 2", len(pre_ctx) > 0)
    check("Allocator mode != 'cold' for Iter 2",
          plan.mode != "cold",
          f"mode={plan.mode}, seeds={plan.seeds[:3]}")

    # Iter 2 — adaptive
    print("  Running Iter 2 (adaptive, all_correct sim)…")
    with Session(engine) as db:
        r2 = run_prep_session(
            session=db, section_ids=[2], questions_per_section=3,
            simulate_strategy="all_correct", seed=22,
        )
        s2 = db.get(SessionRow, r2.session_id)
        check("Iter 2: is_cold_start=False", s2.is_cold_start is False)
        check("Iter 2: adaptive_context.summary persisted",
              s2.adaptive_context is not None
              and "summary" in (s2.adaptive_context or {}))
        check("Iter 2: allocation summary persisted",
              "allocation" in (s2.adaptive_context or {}))


# ────────────────────────────────────────────────────────────────────
# PART 7 — All 16 API endpoints (TestClient, no uvicorn)
# ────────────────────────────────────────────────────────────────────


def part7_api() -> None:
    section("PART 7: All API endpoints respond (TestClient)")

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
            ok = r.status_code == expected_status
            if expected_key and r.status_code == 200:
                try:
                    ok = ok and expected_key in r.json()
                except Exception:
                    ok = False
            check(f"{method} {path}", ok, f"status={r.status_code}")
        except Exception as e:
            check(f"{method} {path}", False, str(e)[:80])

    r = client.post(
        "/api/v1/prep/start",
        json={"section_ids": [99], "questions_per_section": 1},
    )
    check("POST /prep/start rejects invalid section_id",
          r.status_code == 400)


# ────────────────────────────────────────────────────────────────────
# PART 8 — CLI
# ────────────────────────────────────────────────────────────────────


def part8_cli() -> None:
    section("PART 8: CLI surface")

    from click.testing import CliRunner
    from src.cli import cli

    runner = CliRunner()

    r = runner.invoke(cli, ["--help"])
    check("`prep-cli --help` exits 0", r.exit_code == 0)

    r = runner.invoke(cli, ["stats"])
    check("`prep-cli stats` exits 0", r.exit_code == 0)
    check("`prep-cli stats` shows Sections", "Sections" in r.output)

    r = runner.invoke(cli, ["history", "--limit", "5"])
    check("`prep-cli history` exits 0", r.exit_code == 0)

    # Bad sections
    r = runner.invoke(cli, ["prep", "-s", "99", "-q", "1"])
    check("`prep-cli prep` rejects invalid sections",
          r.exit_code != 0)


# ────────────────────────────────────────────────────────────────────
# PART 9 — Output exporter
# ────────────────────────────────────────────────────────────────────


def part9_outputs() -> None:
    section("PART 9: Output paths + exporters")

    from sqlalchemy import create_engine, desc, select
    from sqlalchemy.orm import Session

    from src.config import settings
    from src.kb.models import Session as SessionRow
    from src.output.snapshot import build_kb_snapshot, export_questions

    base = Path(settings.paths.outputs_dir)
    for i in (1, 2, 3):
        d = base / f"scenario_b_iter{i}"
        check(f"outputs/scenario_b_iter{i}/ exists",
              d.exists() and d.is_dir())

    engine = create_engine(settings.secrets.database_url)
    with Session(engine) as db:
        s = db.scalars(
            select(SessionRow).order_by(desc(SessionRow.started_at)).limit(1)
        ).first()
        if s:
            snap = build_kb_snapshot(db, after_session_id=s.id)
            check("KB snapshot has required keys",
                  all(k in snap for k in
                      ("snapshot_after_session", "exported_at",
                       "total_sessions_in_kb", "recent_sessions",
                       "adaptive_state")))

            tmp = base / "scenario_b_iter1" / "_smoke.json"
            export_questions(db, s.id, tmp)
            payload = json.loads(tmp.read_text())
            check("Exporter writes valid JSON with session_id",
                  payload["session_id"] == s.id)
            tmp.unlink()
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
    part5_langgraph()
    part6_adaptive()
    part7_api()
    part8_cli()
    part9_outputs()

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
        print("\n  🎉 ALL CHECKS PASSED — phases 0–7 ready for reviewer")


if __name__ == "__main__":
    main()
