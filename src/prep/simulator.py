"""Realistic answer simulation for Scenario B and dry-runs.

Strategies:
  - all_correct  : always pick the correct answer
  - random       : uniform random A/B/C/D
  - weighted     : skew toward correct, but make some chunks "weak" so the
                   sim demonstrates adaptive behaviour later. Topic weights
                   from `topic_mastery` (when populated) bias accuracy.
"""

from __future__ import annotations

import random
from typing import List, Literal, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.kb.models import Question, QuestionTopic, Topic, TopicMastery
from src.rag.validator import MCQ


Strategy = Literal["all_correct", "random", "weighted"]


def _correct_prob_for_question(
    session: Session,
    question_id: int,
) -> float:
    """Look up the dominant topic's mastery weight to bias accuracy.

    No mastery row yet → use moderate default. Higher weight = weak topic
    = lower prob of correct.
    """
    sim = settings.simulation
    topic_ids = session.scalars(
        select(QuestionTopic.topic_id)
        .where(QuestionTopic.question_id == question_id)
    ).all()

    if not topic_ids:
        return sim.scenario_b_correct_ratio

    masteries = session.scalars(
        select(TopicMastery).where(TopicMastery.topic_id.in_(topic_ids))
    ).all()

    if not masteries:
        return sim.scenario_b_correct_ratio

    avg_weight = sum(m.weight for m in masteries) / len(masteries)
    if avg_weight >= 2.0:
        return sim.weighted_correct_ratio_weak
    if avg_weight >= 1.0:
        return sim.weighted_correct_ratio_moderate
    return sim.weighted_correct_ratio_strong


def _wrong_choice(correct: str, rng: random.Random) -> str:
    options = [c for c in ("A", "B", "C", "D") if c != correct]
    return rng.choice(options)


def simulate_answer_for(
    correct_answer: str,
    correct_prob: float,
    rng: random.Random,
) -> str:
    if rng.random() < correct_prob:
        return correct_answer
    return _wrong_choice(correct_answer, rng)


def simulate_answers(
    session: Session,
    question_ids: List[int],
    strategy: Strategy = "weighted",
    seed: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """Return list of (question_id, simulated_user_answer)."""
    sim = settings.simulation
    rng = random.Random(seed)

    if not question_ids:
        return []

    rows = session.execute(
        select(Question.id, Question.correct_answer)
        .where(Question.id.in_(question_ids))
    ).all()
    correct_by_qid = {r.id: r.correct_answer for r in rows}

    results: List[Tuple[int, str]] = []
    for qid in question_ids:
        correct = correct_by_qid.get(qid)
        if correct is None:
            continue

        if strategy == "all_correct":
            ans = correct
        elif strategy == "random":
            ans = rng.choice(["A", "B", "C", "D"])
        else:  # weighted
            prob = _correct_prob_for_question(session, qid)
            ans = simulate_answer_for(correct, prob, rng)

        results.append((qid, ans))

    return results


if __name__ == "__main__":
    print("Strategy testing (no DB needed):")
    rng = random.Random(42)
    for strategy in ("all_correct", "random", "weighted"):
        if strategy == "weighted":
            n_correct = sum(
                1 for _ in range(100)
                if simulate_answer_for("B", 0.65, rng) == "B"
            )
        elif strategy == "all_correct":
            n_correct = 100
        else:
            n_correct = sum(
                1 for _ in range(100)
                if rng.choice(["A", "B", "C", "D"]) == "B"
            )
        print(f"  {strategy:<12} 100 questions → {n_correct} correct")
