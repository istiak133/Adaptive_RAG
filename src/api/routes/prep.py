"""Prep flow endpoint — run one adaptive session end-to-end."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import PrepSessionResponse, PrepStartRequest, QuestionView
from src.kb.models import Question, SectionTopicMastery
from src.kb.models import Session as SessionRow
from src.services.prep_service import run_prep_session


router = APIRouter(prefix="/prep", tags=["prep"])


@router.post("/start", response_model=PrepSessionResponse)
def start_prep(req: PrepStartRequest, db: Session = Depends(get_db)):
    """Run a full adaptive prep session (LangGraph state machine).

    Generates MCQs, simulates answers, updates mastery, returns score.
    """
    # Validate section IDs
    valid_ids = set(range(1, 11))
    bad = [s for s in req.section_ids if s not in valid_ids]
    if bad:
        raise HTTPException(
            400, f"Invalid section_ids {bad}; valid range is 1..10",
        )

    try:
        result = run_prep_session(
            session=db,
            section_ids=req.section_ids,
            questions_per_section=req.questions_per_section,
            difficulty=req.difficulty,
            simulate_strategy=req.simulate_strategy,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(500, f"Prep session failed: {e}")

    # Compose questions with answers
    qs = db.scalars(
        Question.__table__.select().where(
            Question.session_id == result.session_id,
        )
    ).all()
    # actually use ORM-style
    from sqlalchemy import select
    rows = db.execute(
        select(Question).where(Question.session_id == result.session_id)
    ).scalars().all()

    sess_row = db.get(SessionRow, result.session_id)
    regressions = []
    if sess_row and sess_row.adaptive_context:
        regressions = sess_row.adaptive_context.get("regressions", []) or []

    return PrepSessionResponse(
        session_id=result.session_id,
        is_cold_start=sess_row.is_cold_start if sess_row else False,
        difficulty=sess_row.difficulty_level if sess_row else "medium",
        score_pct=result.score_report.score_pct,
        correct=result.score_report.correct,
        wrong=result.score_report.wrong,
        total_questions=result.score_report.total,
        questions=[
            QuestionView(
                id=q.id,
                section_id=q.section_id,
                question_text=q.question_text,
                choice_a=q.choice_a, choice_b=q.choice_b,
                choice_c=q.choice_c, choice_d=q.choice_d,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                source_quote=q.source_quote,
                user_answer=q.answer.user_answer if q.answer else None,
                is_correct=q.answer.is_correct if q.answer else None,
            )
            for q in rows
        ],
        chunks_used={str(k): v for k, v in result.chunks_used_per_section.items()},
        token_usage=result.token_usage,
        generation_rejects=result.generation_rejects,
        elapsed_seconds=result.elapsed_seconds,
        regressions=regressions,
    )
