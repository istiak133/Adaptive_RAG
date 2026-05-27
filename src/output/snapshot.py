"""KB snapshot exporter — assessment-required human-readable JSON.

Per spec: "top-5 most recent session records at the moment an iteration
completes". Includes adaptive_state so the reviewer can verify history
shapes the next iteration's prompts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session, joinedload

from src.config import settings
from src.kb.models import (
    Answer, Question, SectionTopicMastery,
    Session as SessionRow, Topic, TopicMastery,
)


def build_kb_snapshot(
    session: Session, after_session_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the top-N recent session records + adaptive state.

    When `after_session_id` is supplied, the snapshot is filtered to sessions
    that existed at or before that ID. This makes per-iteration snapshots
    point-in-time correct — re-running the export later won't bleed in
    sessions that came after the iteration completed.
    """
    limit = settings.snapshot.recent_sessions_count

    stmt = (
        select(SessionRow)
        .order_by(desc(SessionRow.started_at))
        .options(joinedload(SessionRow.questions).joinedload(Question.answer))
    )
    if after_session_id is not None:
        stmt = stmt.where(SessionRow.id <= after_session_id)
    stmt = stmt.limit(limit)

    recent = session.execute(stmt).unique().scalars().all()

    session_records = []
    for s in recent:
        questions_block = []
        if settings.snapshot.include_questions:
            for q in s.questions:
                # Shape matches the `QuestionView` Pydantic model used by the
                # /sessions/{id}/snapshot REST endpoint. Includes the four MCQ
                # choices so a reviewer reading the snapshot can see exactly
                # what options were presented for each question — without this,
                # only the correct-answer letter is visible.
                questions_block.append({
                    "id": q.id,
                    "section_id": q.section_id,
                    "question_text": q.question_text,
                    "choice_a": q.choice_a,
                    "choice_b": q.choice_b,
                    "choice_c": q.choice_c,
                    "choice_d": q.choice_d,
                    "correct_answer": q.correct_answer,
                    "user_answer": q.answer.user_answer if q.answer else None,
                    "is_correct": q.answer.is_correct if q.answer else None,
                    "explanation": q.explanation,
                    "source_quote": q.source_quote,
                })

        session_records.append({
            "session_id": s.id,
            "sections_studied": s.sections_studied,
            "is_cold_start": s.is_cold_start,
            "difficulty_level": s.difficulty_level,
            "score_pct": s.score_pct,
            "correct_count": s.correct_count,
            "wrong_count": s.wrong_count,
            "total_questions": s.total_questions,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            "adaptive_context": s.adaptive_context if settings.snapshot.include_adaptive_state else None,
            "token_usage": s.token_usage if settings.snapshot.include_token_usage else None,
            "questions": questions_block,
        })

    # Adaptive state — top weak + mastered topics globally
    adaptive_state: Dict[str, Any] = {}
    if settings.snapshot.include_adaptive_state:
        weak = session.execute(
            select(Topic.name, Topic.slug, TopicMastery)
            .join(TopicMastery, TopicMastery.topic_id == Topic.id)
            .where(TopicMastery.weight > 1.0)
            .order_by(desc(TopicMastery.weight))
            .limit(15)
        ).all()
        mastered = session.execute(
            select(Topic.name, Topic.slug, TopicMastery)
            .join(TopicMastery, TopicMastery.topic_id == Topic.id)
            .where(TopicMastery.is_mastered.is_(True))
            .order_by(TopicMastery.weight.asc())
            .limit(15)
        ).all()
        adaptive_state = {
            "top_weak_topics": [
                {
                    "name": r[0], "slug": r[1],
                    "weight": r[2].weight,
                    "times_wrong": r[2].times_wrong,
                    "times_correct": r[2].times_correct,
                    "streak": r[2].current_streak,
                }
                for r in weak
            ],
            "mastered_topics": [
                {
                    "name": r[0], "slug": r[1],
                    "weight": r[2].weight,
                    "times_correct": r[2].times_correct,
                }
                for r in mastered
            ],
        }

    total_sessions = session.scalar(
        select(func.count()).select_from(SessionRow)
    ) or 0

    return {
        "snapshot_after_session": after_session_id or (
            recent[0].id if recent else 0
        ),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_sessions_in_kb": total_sessions,
        "recent_sessions": session_records,
        "adaptive_state": adaptive_state,
    }


def export_snapshot(
    session: Session,
    output_path: Path | str,
    after_session_id: Optional[int] = None,
) -> Path:
    """Write the snapshot JSON to disk."""
    snap = build_kb_snapshot(session, after_session_id)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if settings.snapshot.pretty_print else None
    out.write_text(json.dumps(snap, indent=indent, ensure_ascii=False))
    return out


def export_questions(
    session: Session,
    session_id: int,
    output_path: Path | str,
) -> Path:
    """Write the questions+answers for one session to disk."""
    s = session.execute(
        select(SessionRow)
        .where(SessionRow.id == session_id)
        .options(joinedload(SessionRow.questions).joinedload(Question.answer))
    ).unique().scalar_one_or_none()
    if s is None:
        raise ValueError(f"Session {session_id} not found")

    payload = {
        "session_id": s.id,
        "sections_studied": s.sections_studied,
        "is_cold_start": s.is_cold_start,
        "score_pct": s.score_pct,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "questions": [
            {
                "id": q.id,
                "section_id": q.section_id,
                "source_chunk_ids": q.source_chunk_ids,
                "difficulty": q.difficulty,
                "question_text": q.question_text,
                "choices": {
                    "A": q.choice_a, "B": q.choice_b,
                    "C": q.choice_c, "D": q.choice_d,
                },
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
                "source_quote": q.source_quote,
                "user_answer": q.answer.user_answer if q.answer else None,
                "is_correct": q.answer.is_correct if q.answer else None,
            }
            for q in s.questions
        ],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if settings.snapshot.pretty_print else None
    out.write_text(json.dumps(payload, indent=indent, ensure_ascii=False))
    return out
