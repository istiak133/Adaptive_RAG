"""Mastery weight calculation and update logic.

Closes the adaptive loop: every answer cascades through `question_topics`
and updates both global and section-specific mastery rows.

Weight formula:
    base = 1.0 + (times_wrong × 0.5 × impact)
    if streak ≥ mastery_threshold:   weight = weight_min  (mastered)
    elif streak ≥ 2:                  weight × weight_decay_on_correct
    elif streak < 0:                  weight × (1 + |streak| × 0.3)
    clamp to [weight_min, weight_max]

Impact: 1.0 for primary topic on a question, 0.5 for secondary.

Regression: if a topic was `is_mastered=True` and the user gets it wrong,
flip the flag and bump weight to `regression_weight_floor` (default 2.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.kb.models import (
    Question, QuestionTopic, SectionTopicMastery, TopicMastery,
)


# ── Weight math ──────────────────────────────────────────────────────


def calculate_weight(
    times_wrong: int,
    times_correct: int,
    streak: int,
    impact: float = 1.0,
) -> float:
    a = settings.adaptive

    if streak >= a.mastery_threshold:
        return a.weight_min  # mastered

    base = 1.0 + (times_wrong * a.weight_boost_on_wrong * impact)

    if streak >= 2:
        base *= a.weight_decay_on_correct
    elif streak < 0:
        base *= 1.0 + abs(streak) * 0.3

    return max(a.weight_min, min(base, a.weight_max))


# ── Update entry point ──────────────────────────────────────────────


@dataclass
class MasteryDelta:
    topic_id: int
    topic_name: str
    impact: float
    new_weight: float
    new_streak: int
    is_mastered: bool
    regression_detected: bool


def update_mastery_after_answer(
    session: Session,
    question_id: int,
    is_correct: bool,
) -> List[MasteryDelta]:
    """Cascade-update mastery for every topic linked to the question."""
    q = session.get(Question, question_id)
    if q is None:
        raise ValueError(f"Question {question_id} not found")

    section_id = q.section_id

    # Topics linked to this question
    link_rows = session.execute(
        select(QuestionTopic.topic_id, QuestionTopic.is_primary)
        .where(QuestionTopic.question_id == question_id)
    ).all()
    if not link_rows:
        return []

    now = datetime.now(timezone.utc)
    deltas: List[MasteryDelta] = []
    a = settings.adaptive

    for topic_id, is_primary in link_rows:
        impact = 1.0 if is_primary else 0.5

        # ── Global mastery ──
        mastery = session.execute(
            select(TopicMastery).where(TopicMastery.topic_id == topic_id)
        ).scalar_one_or_none()
        if mastery is None:
            mastery = TopicMastery(topic_id=topic_id)
            session.add(mastery)
            session.flush()

        was_mastered = mastery.is_mastered

        mastery.times_asked += 1
        if is_correct:
            mastery.times_correct += 1
            mastery.current_streak = max(0, mastery.current_streak) + 1
        else:
            mastery.times_wrong += 1
            mastery.current_streak = min(0, mastery.current_streak) - 1
            mastery.last_wrong_at = now
        mastery.last_asked_at = now

        mastery.weight = calculate_weight(
            times_wrong=mastery.times_wrong,
            times_correct=mastery.times_correct,
            streak=mastery.current_streak,
            impact=impact,
        )

        regression = False
        if mastery.current_streak >= a.mastery_threshold:
            mastery.is_mastered = True
        elif was_mastered and not is_correct:
            mastery.is_mastered = False
            mastery.weight = max(mastery.weight, a.regression_weight_floor)
            regression = True

        # ── Section-specific mastery ──
        stm = session.execute(
            select(SectionTopicMastery).where(
                SectionTopicMastery.section_id == section_id,
                SectionTopicMastery.topic_id == topic_id,
            )
        ).scalar_one_or_none()
        if stm is None:
            stm = SectionTopicMastery(section_id=section_id, topic_id=topic_id)
            session.add(stm)
            session.flush()

        stm.times_asked += 1
        if is_correct:
            stm.times_correct += 1
        else:
            stm.times_wrong += 1
        stm.weight = calculate_weight(
            times_wrong=stm.times_wrong,
            times_correct=stm.times_correct,
            streak=mastery.current_streak,  # global streak drives section too
            impact=impact,
        )

        # Topic name for logging
        from src.kb.models import Topic
        topic_name = session.scalar(
            select(Topic.name).where(Topic.id == topic_id)
        ) or f"<id={topic_id}>"

        deltas.append(MasteryDelta(
            topic_id=topic_id,
            topic_name=topic_name,
            impact=impact,
            new_weight=mastery.weight,
            new_streak=mastery.current_streak,
            is_mastered=mastery.is_mastered,
            regression_detected=regression,
        ))

    session.commit()
    return deltas


# ── Read helpers ────────────────────────────────────────────────────


def get_topic_weights_for_sections(
    session: Session, section_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    """Returns per-section-per-topic state:
    {section_id: {topic_id: {"global_weight": x, "section_weight": y,
                              "is_mastered": bool}}}.
    """
    from src.kb.models import SectionTopic
    rows = session.execute(
        select(
            SectionTopic.section_id,
            SectionTopic.topic_id,
            SectionTopic.depth,
            SectionTopic.relevance_score,
        ).where(SectionTopic.section_id.in_(section_ids))
    ).all()

    topic_ids = list({r.topic_id for r in rows})
    if not topic_ids:
        return {}

    global_rows = session.execute(
        select(TopicMastery.topic_id, TopicMastery.weight,
               TopicMastery.is_mastered, TopicMastery.times_wrong)
        .where(TopicMastery.topic_id.in_(topic_ids))
    ).all()
    global_map = {r.topic_id: r for r in global_rows}

    section_rows = session.execute(
        select(SectionTopicMastery.section_id,
               SectionTopicMastery.topic_id,
               SectionTopicMastery.weight)
        .where(SectionTopicMastery.section_id.in_(section_ids))
        .where(SectionTopicMastery.topic_id.in_(topic_ids))
    ).all()
    section_map = {(r.section_id, r.topic_id): r.weight for r in section_rows}

    result: Dict[int, Dict[int, Dict[str, float]]] = {}
    for r in rows:
        g = global_map.get(r.topic_id)
        gweight = g.weight if g else 1.0
        is_mastered = g.is_mastered if g else False
        times_wrong = g.times_wrong if g else 0
        result.setdefault(r.section_id, {})[r.topic_id] = {
            "global_weight": gweight,
            "section_weight": section_map.get((r.section_id, r.topic_id), 1.0),
            "depth": r.depth,
            "depth_score": r.relevance_score,
            "is_mastered": is_mastered,
            "times_wrong": times_wrong,
        }
    return result


def clear_all_mastery(session: Session) -> int:
    """Test helper — wipes mastery state."""
    from sqlalchemy import delete
    n1 = session.execute(delete(SectionTopicMastery)).rowcount
    n2 = session.execute(delete(TopicMastery)).rowcount
    session.commit()
    return n1 + n2
