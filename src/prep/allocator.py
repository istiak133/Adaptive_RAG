"""Weighted question allocation for adaptive prep.

Given a section's topics + mastery weights, produce N "query seeds" that
bias the retriever toward the right chunks. Each seed becomes a separate
LLM call (one MCQ per seed) for per-topic targeting.

Edge cases:
  - All mastered  → escalate difficulty, pick `general key concepts` seeds
  - All weak      → focus on top-3 weakest, drop difficulty
  - No mastery    → cold start, evenly distributed seeds
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from sqlalchemy.orm import Session

from src.config import settings
from src.kb.mastery import get_topic_weights_for_sections


_DEPTH_MULT = {"primary": 1.0, "secondary": 0.6, "mention": 0.3}


@dataclass
class TopicAllocation:
    topic_id: int
    topic_name: str
    effective_weight: float
    questions: int


@dataclass
class AllocationPlan:
    section_id: int
    seeds: List[str]                  # Length == n_questions
    allocations: List[TopicAllocation]
    mode: str                          # cold | adaptive | all_mastered | all_weak


def _get_topic_names(session: Session, topic_ids: List[int]) -> dict:
    from src.kb.models import Topic
    from sqlalchemy import select
    rows = session.execute(
        select(Topic.id, Topic.name).where(Topic.id.in_(topic_ids))
    ).all()
    return {r.id: r.name for r in rows}


def allocate(
    session: Session,
    section_id: int,
    n_questions: int,
) -> AllocationPlan:
    state = get_topic_weights_for_sections(session, [section_id]).get(section_id, {})

    if not state:
        return AllocationPlan(
            section_id=section_id,
            seeds=[f"key concepts and definitions in section {section_id}"] * n_questions,
            allocations=[],
            mode="cold",
        )

    name_map = _get_topic_names(session, list(state.keys()))

    # Effective weight = global × section × depth_multiplier
    # Mastered → tiny weight to mostly skip
    weighted: List[TopicAllocation] = []
    for topic_id, st in state.items():
        depth_mult = _DEPTH_MULT.get(st["depth"], 0.5)
        if st["is_mastered"]:
            eff = settings.adaptive.weight_min
        else:
            eff = st["global_weight"] * st["section_weight"] * depth_mult
        weighted.append(TopicAllocation(
            topic_id=topic_id,
            topic_name=name_map.get(topic_id, f"<id={topic_id}>"),
            effective_weight=eff,
            questions=0,
        ))

    weighted.sort(key=lambda x: x.effective_weight, reverse=True)

    # Detect mode
    all_mastered = all(state[t.topic_id]["is_mastered"] for t in weighted)
    weak_threshold = 2.0
    high_weight_topics = [t for t in weighted if t.effective_weight >= weak_threshold]
    all_weak = (
        len(high_weight_topics) >= len(weighted) * 0.6
        and len(weighted) >= 3
    )

    mode = "adaptive"
    if all_mastered:
        mode = "all_mastered"
    elif all_weak:
        mode = "all_weak"
        # Trim to focus on worst-N as per config
        weighted = weighted[: settings.adaptive.all_weak_focus_topics]

    # Proportional allocation
    total_weight = sum(t.effective_weight for t in weighted)
    if total_weight <= 0:
        for t in weighted:
            t.effective_weight = 1.0
        total_weight = sum(t.effective_weight for t in weighted)

    remaining = n_questions
    for t in weighted:
        share = (t.effective_weight / total_weight) * n_questions
        t.questions = max(0, min(remaining, round(share)))
        remaining -= t.questions

    # Ensure each high-weight topic (eff > 2.0) gets at least 1
    if remaining > 0 or any(
        t.questions == 0 and t.effective_weight > weak_threshold for t in weighted
    ):
        # Force minimum 1 for weak topics
        for t in weighted:
            if t.effective_weight > weak_threshold and t.questions == 0 and remaining > 0:
                t.questions += 1
                remaining -= 1

    # Distribute any remaining to highest-weight
    i = 0
    while remaining > 0 and weighted:
        weighted[i % len(weighted)].questions += 1
        remaining -= 1
        i += 1

    # Build seed strings
    seeds: List[str] = []
    for t in weighted:
        for _ in range(t.questions):
            seeds.append(t.topic_name)

    # Pad if short (defensive)
    while len(seeds) < n_questions:
        seeds.append(f"key concepts in section {section_id}")
    seeds = seeds[:n_questions]

    return AllocationPlan(
        section_id=section_id,
        seeds=seeds,
        allocations=weighted,
        mode=mode,
    )
