"""Command-line interface for the Adaptive Document Preparation System.

Run any command:
    python -m src.cli --help
    python -m src.cli ingest
    python -m src.cli prep --sections 5 8 --questions 5
    python -m src.cli scenario-b
    python -m src.cli history --limit 10
    python -m src.cli snapshot 5
    python -m src.cli stats
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import click
from sqlalchemy import create_engine, desc, func, select, text
from sqlalchemy.orm import Session

from src.config import settings
from src.kb.chroma_repo import ChromaRepo
from src.kb.mastery import clear_all_mastery
from src.kb.models import (
    Answer, Chunk, ChunkTopic, Question,
    Section, SectionTopic, Session as SessionRow, Topic,
)
from src.output.snapshot import build_kb_snapshot, export_questions, export_snapshot
from src.services.prep_service import run_prep_session


VALID_SECTIONS = list(range(1, 11))


# ── helpers ──────────────────────────────────────────────────────────


def _engine():
    return create_engine(settings.secrets.database_url, pool_pre_ping=True)


def _row_count(db: Session, model) -> int:
    return db.scalar(select(func.count()).select_from(model)) or 0


def _print_header(title: str) -> None:
    click.echo()
    click.secho("=" * 70, fg="cyan")
    click.secho(title, fg="cyan", bold=True)
    click.secho("=" * 70, fg="cyan")


def _print_score(score_pct: float, correct: int, total: int) -> None:
    colour = "green" if score_pct >= 70 else ("yellow" if score_pct >= 40 else "red")
    click.secho(
        f"Score: {correct}/{total} ({score_pct:.1f}%)",
        fg=colour, bold=True,
    )


def _validate_sections(ctx, param, value):
    """Click callback to ensure section IDs are in 1..10."""
    if not value:
        return value
    bad = [s for s in value if s not in VALID_SECTIONS]
    if bad:
        raise click.BadParameter(
            f"Invalid section IDs: {bad}. Valid range is 1–10."
        )
    return list(value)


# ── CLI group ────────────────────────────────────────────────────────


@click.group()
@click.version_option(settings.app.version, prog_name="prep-cli")
def cli():
    """Adaptive Document Preparation System — CLI."""


# ── ingest ───────────────────────────────────────────────────────────


@cli.command("ingest")
def ingest_cmd():
    """Run the full ingestion pipeline (PDF → KB)."""
    _print_header("Ingestion Pipeline")
    from src.ingestion.indexer import run_indexer

    stats = run_indexer()
    click.echo()
    click.secho("✓ Ingestion complete", fg="green", bold=True)
    for k, v in stats.items():
        click.echo(f"  {k:<20} {v}")


# ── prep ─────────────────────────────────────────────────────────────


@cli.command("prep")
@click.option(
    "--sections", "-s",
    type=int, multiple=True, required=True,
    callback=_validate_sections,
    help="Section IDs to study (1–10). Repeat for multiple: -s 5 -s 8",
)
@click.option(
    "--questions", "-q", "questions_per_section",
    type=click.IntRange(1, 20), default=5, show_default=True,
    help="MCQs per section.",
)
@click.option(
    "--difficulty", "-d",
    type=click.Choice(["easy", "medium", "hard"]),
    default=None,
    help="Difficulty override (auto-picked from history otherwise).",
)
@click.option(
    "--simulate",
    type=click.Choice(["weighted", "random", "all_correct"]),
    default="weighted", show_default=True,
    help="Answer simulation strategy.",
)
@click.option(
    "--seed",
    type=int, default=None,
    help="Random seed for reproducibility.",
)
@click.option(
    "--save-outputs/--no-save-outputs",
    default=False,
    help="Write questions.json + snapshot.json to outputs/manual_run/.",
)
def prep_cmd(
    sections: List[int],
    questions_per_section: int,
    difficulty: Optional[str],
    simulate: str,
    seed: Optional[int],
    save_outputs: bool,
):
    """Run one adaptive prep session for the given sections."""
    _print_header(f"Prep session — sections {sections}")
    click.echo(f"  questions/section: {questions_per_section}")
    click.echo(f"  difficulty:        {difficulty or 'auto'}")
    click.echo(f"  simulate:          {simulate}")
    click.echo()

    with Session(_engine()) as db:
        with click.progressbar(
            length=1, label="Running prep session",
            show_eta=False, show_percent=False,
        ):
            result = run_prep_session(
                session=db,
                section_ids=sections,
                questions_per_section=questions_per_section,
                difficulty=difficulty,
                simulate_strategy=simulate,
                seed=seed,
            )

    r = result.score_report
    click.echo()
    _print_score(r.score_pct, r.correct, r.total)
    click.echo(f"Session ID: {result.session_id}")
    click.echo(f"Cold start: {result.session_id is not None}")
    click.echo(f"Tokens used: {result.token_usage.get('total_input', 0)} input "
               f"+ {result.token_usage.get('output_reserve', 0)} reserve")
    click.echo(f"Elapsed: {result.elapsed_seconds}s")

    # Per-question summary
    click.echo()
    click.secho("Per-question results:", bold=True)
    for i, q in enumerate(r.per_question, 1):
        mark = click.style("✓", fg="green") if q.is_correct else click.style("✗", fg="red")
        click.echo(
            f"  {mark} Q{i}  user={q.user_answer} correct={q.correct_answer} "
            f"— {q.question_text[:70]}"
        )
        if not q.is_correct:
            click.echo(click.style(f"      → {q.explanation[:90]}", fg="yellow"))

    if save_outputs:
        out_dir = Path(settings.paths.outputs_dir) / "manual_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        with Session(_engine()) as db:
            export_questions(db, result.session_id,
                             out_dir / f"questions_session_{result.session_id}.json")
            export_snapshot(db, out_dir / f"snapshot_session_{result.session_id}.json",
                            after_session_id=result.session_id)
        click.secho(f"\n✓ Outputs saved to {out_dir}", fg="green")


# ── scenario-b ───────────────────────────────────────────────────────


SCENARIO_B_PLAN = [
    {"iter": 1, "sections": [5, 8]},
    {"iter": 2, "sections": [6, 8, 9]},
    {"iter": 3, "sections": [8]},
]


@cli.command("scenario-b")
@click.option(
    "--questions", "-q", "questions_per_section",
    type=click.IntRange(1, 10), default=5, show_default=True,
)
@click.option(
    "--simulate",
    type=click.Choice(["weighted", "random", "all_correct"]),
    default="weighted", show_default=True,
)
@click.option(
    "--seed", type=int, default=42, show_default=True,
)
@click.option(
    "--reset-state/--keep-state",
    default=True,
    help="Clear mastery state before running (recommended for clean Scenario B).",
)
def scenario_b_cmd(
    questions_per_section: int,
    simulate: str,
    seed: int,
    reset_state: bool,
):
    """Run the 3-iteration Scenario B and write all required JSON outputs."""
    _print_header("SCENARIO B — 3 consecutive iterations")
    click.echo(f"  Iter 1: sections {SCENARIO_B_PLAN[0]['sections']}")
    click.echo(f"  Iter 2: sections {SCENARIO_B_PLAN[1]['sections']}")
    click.echo(f"  Iter 3: sections {SCENARIO_B_PLAN[2]['sections']}")
    click.echo(f"  questions/section: {questions_per_section}")
    click.echo(f"  reset_state:       {reset_state}")
    click.echo()

    t0 = time.time()
    outputs_root = Path(settings.paths.outputs_dir)

    if reset_state:
        with Session(_engine()) as db:
            clear_all_mastery(db)
        click.secho("  ↺ Mastery state cleared", fg="yellow")

    results = []
    for plan in SCENARIO_B_PLAN:
        n = plan["iter"]
        secs = plan["sections"]
        click.echo()
        click.secho(f"─── Iter {n} (sections {secs}) ───", fg="cyan", bold=True)

        with Session(_engine()) as db:
            result = run_prep_session(
                session=db,
                section_ids=secs,
                questions_per_section=questions_per_section,
                simulate_strategy=simulate,
                seed=seed,
            )
            out_dir = outputs_root / f"scenario_b_iter{n}"
            export_questions(db, result.session_id,
                             out_dir / f"questions_iter{n}.json")
            export_snapshot(db, out_dir / f"kb_snapshot_iter{n}.json",
                            after_session_id=result.session_id)

        r = result.score_report
        _print_score(r.score_pct, r.correct, r.total)
        click.echo(f"  session_id: {result.session_id}")
        click.echo(f"  elapsed:    {result.elapsed_seconds}s")
        click.secho(f"  ✓ Wrote {out_dir}/questions_iter{n}.json", fg="green")
        click.secho(f"  ✓ Wrote {out_dir}/kb_snapshot_iter{n}.json", fg="green")
        results.append((n, secs, result))

    elapsed = round(time.time() - t0, 1)
    click.echo()
    click.secho("=" * 70, fg="green")
    click.secho(f"✓ SCENARIO B COMPLETE — {elapsed}s total", fg="green", bold=True)
    click.secho("=" * 70, fg="green")
    click.echo()
    click.echo("Outputs:")
    for n, secs, _ in results:
        click.echo(f"  outputs/scenario_b_iter{n}/questions_iter{n}.json")
        click.echo(f"  outputs/scenario_b_iter{n}/kb_snapshot_iter{n}.json")


# ── history ──────────────────────────────────────────────────────────


@cli.command("history")
@click.option("--limit", "-n", type=int, default=10, show_default=True)
def history_cmd(limit: int):
    """Show recent session history."""
    _print_header(f"Session history (latest {limit})")

    with Session(_engine()) as db:
        rows = db.scalars(
            select(SessionRow)
            .order_by(desc(SessionRow.started_at))
            .limit(limit)
        ).all()

    if not rows:
        click.echo("  (no sessions yet)")
        return

    click.echo(f"{'ID':>5}  {'Sections':<18}  {'Score':>6}  {'Cold':>5}  "
               f"{'Difficulty':<8}  {'Started'}")
    click.echo("-" * 90)
    for r in rows:
        sections_str = ",".join(str(s) for s in (r.sections_studied or []))
        cold = "yes" if r.is_cold_start else "no"
        score = f"{r.score_pct:.0f}%"
        started = r.started_at.strftime("%Y-%m-%d %H:%M") if r.started_at else ""
        click.echo(
            f"{r.id:>5}  [{sections_str:<16}]  {score:>6}  {cold:>5}  "
            f"{r.difficulty_level:<8}  {started}"
        )


# ── snapshot ─────────────────────────────────────────────────────────


@cli.command("snapshot")
@click.argument("session_id", type=int, required=False)
@click.option(
    "--save",
    type=click.Path(),
    help="Write snapshot to this path (otherwise stdout).",
)
def snapshot_cmd(session_id: Optional[int], save: Optional[str]):
    """Export the KB snapshot (top-5 recent sessions + adaptive state)."""
    with Session(_engine()) as db:
        if session_id is None:
            latest = db.scalars(
                select(SessionRow)
                .order_by(desc(SessionRow.started_at))
                .limit(1)
            ).first()
            if latest is None:
                click.secho("No sessions yet — run a prep first.", fg="yellow")
                return
            session_id = latest.id

        snap = build_kb_snapshot(db, after_session_id=session_id)

    pretty = json.dumps(snap, indent=2, ensure_ascii=False)
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(pretty)
        click.secho(f"✓ Snapshot saved to {out}", fg="green")
    else:
        click.echo(pretty)


# ── stats ────────────────────────────────────────────────────────────


@cli.command("stats")
def stats_cmd():
    """Show database row counts + adaptive state summary."""
    _print_header("System statistics")

    chroma = ChromaRepo()
    with Session(_engine()) as db:
        try:
            av = db.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            ).scalar()
        except Exception:
            av = None

        rows = [
            ("Sections", _row_count(db, Section)),
            ("Chunks", _row_count(db, Chunk)),
            ("Topics", _row_count(db, Topic)),
            ("Section-Topic links", _row_count(db, SectionTopic)),
            ("Chunk-Topic links", _row_count(db, ChunkTopic)),
            ("Sessions", _row_count(db, SessionRow)),
            ("Questions", _row_count(db, Question)),
            ("Answers", _row_count(db, Answer)),
            ("ChromaDB vectors", chroma.count()),
            ("Alembic version", av or "—"),
        ]

    for label, value in rows:
        click.echo(f"  {label:<24} {value}")


if __name__ == "__main__":
    cli()
