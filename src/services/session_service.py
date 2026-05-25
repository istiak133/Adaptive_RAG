"""Session lifecycle and question/answer persistence.

Responsibilities:
  - Start / complete sessions
  - Persist generated MCQs (with question_topics linking)
  - Record user answers and compute is_correct
  - Fetch session history for snapshots
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from src.config import settings
from src.kb.models import (
    Answer, Chunk, Question, QuestionTopic,
    Session as SessionRow, Topic,
)
from src.rag.validator import MCQ


_SLUG_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)


def _slugify(name: str) -> str:
    s = name.lower()
    s = _SLUG_RE.sub("", s)
    s = re.sub(r"[\s-]+", "_", s)
    return s.strip("_")


@dataclass
class GeneratedQuestion:
    """An MCQ tied to the section it came from + source chunk sub_ids."""
    section_id: int
    mcq: MCQ
    source_sub_ids: List[str]


def start_session(
    session: Session,
    section_ids: List[int],
    is_cold_start: bool,
    difficulty: str = "medium",
    adaptive_context: Optional[dict] = None,
) -> int:
    row = SessionRow(
        sections_studied=section_ids,
        difficulty_level=difficulty,
        is_cold_start=is_cold_start,
        adaptive_context=adaptive_context,
    )
    session.add(row)
    session.commit()
    return row.id


def _resolve_topic_id(session: Session, topic_name: str) -> Optional[int]:
    """Match LLM-produced topic strings to our DB topics.

    Order:
      1. Exact name match
      2. Slug match
      3. Substring overlap (longest match wins)
    """
    name = topic_name.strip()
    if not name:
        return None

    tid = session.scalar(select(Topic.id).where(Topic.name == name))
    if tid:
        return tid

    slug = _slugify(name)
    tid = session.scalar(select(Topic.id).where(Topic.slug == slug))
    if tid:
        return tid

    # Fuzzy fallback: longest substring match in either direction.
    name_l = name.lower()
    candidates = session.execute(
        select(Topic.id, Topic.name).order_by(func.length(Topic.name).desc())
    ).all()
    best_id, best_overlap = None, 0
    for cand_id, cand_name in candidates:
        cand_l = cand_name.lower()
        if cand_l in name_l or name_l in cand_l:
            overlap = min(len(cand_l), len(name_l))
            if overlap > best_overlap and overlap >= 4:
                best_id, best_overlap = cand_id, overlap
    return best_id


def _resolve_chunk_ids(session: Session, sub_ids: List[str]) -> List[int]:
    if not sub_ids:
        return []
    rows = session.execute(
        select(Chunk.id, Chunk.sub_section_id).where(
            Chunk.sub_section_id.in_(sub_ids)
        )
    ).all()
    sub_to_id = {r.sub_section_id: r.id for r in rows}
    # Preserve original order
    return [sub_to_id[s] for s in sub_ids if s in sub_to_id]


def record_questions(
    session: Session,
    session_id: int,
    generated: List[GeneratedQuestion],
    difficulty: str = "medium",
) -> List[int]:
    """Insert questions + question_topics. Returns new question IDs."""
    question_ids: List[int] = []

    for gq in generated:
        chunk_ids = _resolve_chunk_ids(session, gq.source_sub_ids)
        q = Question(
            session_id=session_id,
            section_id=gq.section_id,
            source_chunk_ids=chunk_ids,
            difficulty=difficulty,
            question_text=gq.mcq.question_text,
            choice_a=gq.mcq.choice_a,
            choice_b=gq.mcq.choice_b,
            choice_c=gq.mcq.choice_c,
            choice_d=gq.mcq.choice_d,
            correct_answer=gq.mcq.correct_answer,
            explanation=gq.mcq.explanation,
            source_quote=gq.mcq.source_quote,
        )
        session.add(q)
        session.flush()

        # Link primary topic (MCQ.topic)
        topic_id = _resolve_topic_id(session, gq.mcq.topic)
        if topic_id is not None:
            session.add(QuestionTopic(
                question_id=q.id, topic_id=topic_id, is_primary=True,
            ))

        question_ids.append(q.id)

    # Update sessions.total_questions
    sess = session.get(SessionRow, session_id)
    if sess:
        sess.total_questions = (sess.total_questions or 0) + len(question_ids)
    session.commit()
    return question_ids


def record_answer(
    session: Session,
    question_id: int,
    user_answer: str,
) -> bool:
    """Insert one answer; returns is_correct."""
    user_answer = user_answer.strip().upper()
    if user_answer not in ("A", "B", "C", "D"):
        raise ValueError(f"user_answer must be A/B/C/D, got {user_answer!r}")

    q = session.get(Question, question_id)
    if q is None:
        raise ValueError(f"Question {question_id} not found")

    is_correct = (user_answer == q.correct_answer)
    session.add(Answer(
        question_id=question_id,
        user_answer=user_answer,
        is_correct=is_correct,
    ))
    session.commit()
    return is_correct


def complete_session(
    session: Session,
    session_id: int,
    token_usage: Optional[dict] = None,
) -> Dict[str, int]:
    """Compute final score and mark session complete."""
    sess = session.get(SessionRow, session_id)
    if sess is None:
        raise ValueError(f"Session {session_id} not found")

    correct = session.scalar(
        select(func.count())
        .select_from(Answer)
        .join(Question, Answer.question_id == Question.id)
        .where(Question.session_id == session_id)
        .where(Answer.is_correct.is_(True))
    ) or 0
    total = session.scalar(
        select(func.count())
        .select_from(Answer)
        .join(Question, Answer.question_id == Question.id)
        .where(Question.session_id == session_id)
    ) or 0

    sess.correct_count = correct
    sess.wrong_count = total - correct
    sess.score_pct = (correct / total * 100.0) if total else 0.0
    sess.completed_at = datetime.now(timezone.utc)
    if token_usage:
        sess.token_usage = token_usage
    session.commit()

    return {
        "correct": correct,
        "wrong": total - correct,
        "total": total,
        "score_pct": sess.score_pct,
    }


def get_session_detail(session: Session, session_id: int) -> Optional[dict]:
    """Full session record including questions and answers."""
    sess = session.execute(
        select(SessionRow)
        .where(SessionRow.id == session_id)
        .options(joinedload(SessionRow.questions).joinedload(Question.answer))
    ).unique().scalar_one_or_none()

    if sess is None:
        return None

    return {
        "session_id": sess.id,
        "sections_studied": sess.sections_studied,
        "score_pct": sess.score_pct,
        "correct": sess.correct_count,
        "wrong": sess.wrong_count,
        "total_questions": sess.total_questions,
        "is_cold_start": sess.is_cold_start,
        "started_at": sess.started_at.isoformat() if sess.started_at else None,
        "completed_at": sess.completed_at.isoformat() if sess.completed_at else None,
        "token_usage": sess.token_usage,
        "questions": [
            {
                "id": q.id,
                "section_id": q.section_id,
                "question_text": q.question_text,
                "correct_answer": q.correct_answer,
                "user_answer": q.answer.user_answer if q.answer else None,
                "is_correct": q.answer.is_correct if q.answer else None,
            }
            for q in sess.questions
        ],
    }


def recent_sessions(session: Session, limit: int = 5) -> List[dict]:
    rows = session.scalars(
        select(SessionRow)
        .order_by(SessionRow.started_at.desc())
        .limit(limit)
    ).all()
    return [
        {
            "session_id": r.id,
            "sections_studied": r.sections_studied,
            "score_pct": r.score_pct,
            "is_cold_start": r.is_cold_start,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in rows
    ]
