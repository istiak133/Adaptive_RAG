"""Session scoring & per-question result formatting.

Computes the per-question right/wrong list with the same `explanation`
the LLM produced — so wrong answers always show clarification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.kb.models import Answer, Question


@dataclass
class QuestionResult:
    question_id: int
    question_text: str
    user_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    explanation: str
    source_quote: Optional[str]


@dataclass
class ScoreReport:
    session_id: int
    total: int
    correct: int
    wrong: int
    score_pct: float
    per_question: List[QuestionResult]

    def wrong_results(self) -> List[QuestionResult]:
        return [r for r in self.per_question if not r.is_correct]


def score_session(session: Session, session_id: int) -> ScoreReport:
    rows = session.execute(
        select(Question, Answer)
        .join(Answer, Answer.question_id == Question.id, isouter=True)
        .where(Question.session_id == session_id)
        .order_by(Question.id)
    ).all()

    per_question: List[QuestionResult] = []
    correct_count = 0

    for q, a in rows:
        is_correct = bool(a and a.is_correct)
        if is_correct:
            correct_count += 1
        per_question.append(QuestionResult(
            question_id=q.id,
            question_text=q.question_text,
            user_answer=a.user_answer if a else None,
            correct_answer=q.correct_answer,
            is_correct=is_correct,
            explanation=q.explanation,
            source_quote=q.source_quote,
        ))

    total = len(per_question)
    return ScoreReport(
        session_id=session_id,
        total=total,
        correct=correct_count,
        wrong=total - correct_count,
        score_pct=(correct_count / total * 100.0) if total else 0.0,
        per_question=per_question,
    )
