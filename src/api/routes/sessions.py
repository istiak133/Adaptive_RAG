"""Session history + KB snapshot endpoints."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session, joinedload

from src.api.dependencies import get_db
from src.api.schemas import (
    KBSnapshot, QuestionView, SessionDetail, SessionSummary,
)
from src.kb.models import Question, Session as SessionRow
from src.output.snapshot import build_kb_snapshot


router = APIRouter(prefix="/sessions", tags=["sessions"])


def _row_to_summary(s: SessionRow) -> SessionSummary:
    return SessionSummary(
        session_id=s.id,
        sections_studied=s.sections_studied or [],
        score_pct=s.score_pct,
        is_cold_start=s.is_cold_start,
        total_questions=s.total_questions,
        correct_count=s.correct_count,
        wrong_count=s.wrong_count,
        started_at=s.started_at.isoformat() if s.started_at else None,
        completed_at=s.completed_at.isoformat() if s.completed_at else None,
    )


@router.get("", response_model=List[SessionSummary])
def list_sessions(
    db: Session = Depends(get_db),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    rows = db.scalars(
        select(SessionRow)
        .order_by(desc(SessionRow.started_at))
        .limit(limit)
        .offset(offset)
    ).all()
    return [_row_to_summary(r) for r in rows]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: int, db: Session = Depends(get_db)):
    s = db.execute(
        select(SessionRow)
        .where(SessionRow.id == session_id)
        .options(joinedload(SessionRow.questions).joinedload(Question.answer))
    ).unique().scalar_one_or_none()
    if s is None:
        raise HTTPException(404, f"Session {session_id} not found")

    summary = _row_to_summary(s)
    return SessionDetail(
        **summary.model_dump(),
        difficulty_level=s.difficulty_level,
        adaptive_context=s.adaptive_context,
        token_usage=s.token_usage,
        questions=[
            QuestionView(
                id=q.id, section_id=q.section_id,
                question_text=q.question_text,
                choice_a=q.choice_a, choice_b=q.choice_b,
                choice_c=q.choice_c, choice_d=q.choice_d,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                source_quote=q.source_quote,
                user_answer=q.answer.user_answer if q.answer else None,
                is_correct=q.answer.is_correct if q.answer else None,
            )
            for q in s.questions
        ],
    )


@router.get("/{session_id}/snapshot", response_model=KBSnapshot)
def get_snapshot(session_id: int, db: Session = Depends(get_db)):
    """Top-5 recent sessions + adaptive state (assessment requirement)."""
    s = db.get(SessionRow, session_id)
    if s is None:
        raise HTTPException(404, f"Session {session_id} not found")
    snap = build_kb_snapshot(db, after_session_id=session_id)
    return KBSnapshot(**snap)
