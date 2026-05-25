"""Difficulty controller based on recent score for a section.

≥ 85% avg → hard,  50–85% → medium,  < 50% → easy
"""

from __future__ import annotations

from typing import List

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.config import settings
from src.kb.models import Question, Session as SessionRow


def pick_difficulty(
    session: Session,
    section_id: int,
    lookback_sessions: int = 5,
) -> str:
    """Return easy/medium/hard for this section based on recent score."""
    default = settings.mcq.default_difficulty

    rows = session.execute(
        select(SessionRow.score_pct)
        .join(Question, Question.session_id == SessionRow.id)
        .where(Question.section_id == section_id)
        .where(SessionRow.completed_at.is_not(None))
        .group_by(SessionRow.id, SessionRow.score_pct, SessionRow.completed_at)
        .order_by(desc(SessionRow.completed_at))
        .limit(lookback_sessions)
    ).all()

    if not rows:
        return default

    avg = sum(r.score_pct for r in rows) / len(rows)

    if not settings.adaptive.difficulty_escalation:
        return default

    if avg >= 85:
        return "hard"
    if avg < 50:
        return "easy"
    return "medium"
