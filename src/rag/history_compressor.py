"""Build the adaptive_context string injected into the LLM prompt.

Reads mastery state + recent session questions, formats a compact summary
under the history_budget (1200 tokens default). Three compression levels
based on how much state exists.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.config import settings
from src.kb.models import (
    Question, SectionTopicMastery, Session as SessionRow,
    Topic, TopicMastery,
)
from src.rag.token_budget import count_tokens


def _format_topic_line(
    name: str,
    weight: float,
    wrong: int,
    correct: int,
) -> str:
    return f"  • {name} — weight {weight:.2f}, wrong {wrong}× / correct {correct}×"


def _weak_topics_for_sections(
    session: Session,
    section_ids: List[int],
    limit: int = 10,
) -> List[dict]:
    """Top weak topics in these sections (section-specific + global)."""
    rows = session.execute(
        select(
            Topic.name,
            Topic.slug,
            SectionTopicMastery.section_id,
            SectionTopicMastery.weight.label("section_weight"),
            SectionTopicMastery.times_wrong,
            SectionTopicMastery.times_correct,
            TopicMastery.weight.label("global_weight"),
            TopicMastery.is_mastered,
        )
        .join(SectionTopicMastery, SectionTopicMastery.topic_id == Topic.id)
        .outerjoin(TopicMastery, TopicMastery.topic_id == Topic.id)
        .where(SectionTopicMastery.section_id.in_(section_ids))
        .where(SectionTopicMastery.weight > 1.0)
        .order_by(desc(SectionTopicMastery.weight))
        .limit(limit)
    ).all()

    return [
        {
            "name": r.name,
            "slug": r.slug,
            "section_id": r.section_id,
            "section_weight": r.section_weight,
            "global_weight": r.global_weight or 1.0,
            "wrong": r.times_wrong,
            "correct": r.times_correct,
            "is_mastered": bool(r.is_mastered),
        }
        for r in rows
    ]


def _mastered_topics_for_sections(
    session: Session,
    section_ids: List[int],
    limit: int = 5,
) -> List[dict]:
    """Topics user has mastered in these sections."""
    rows = session.execute(
        select(
            Topic.name,
            TopicMastery.weight,
            TopicMastery.times_correct,
        )
        .join(TopicMastery, TopicMastery.topic_id == Topic.id)
        .join(
            SectionTopicMastery,
            (SectionTopicMastery.topic_id == Topic.id)
            & (SectionTopicMastery.section_id.in_(section_ids)),
        )
        .where(TopicMastery.is_mastered.is_(True))
        .order_by(TopicMastery.weight.asc())
        .limit(limit)
    ).unique().all()

    return [
        {"name": r.name, "weight": r.weight, "correct": r.times_correct}
        for r in rows
    ]


def _recent_question_themes(
    session: Session,
    section_ids: List[int],
    limit: int = 10,
) -> List[str]:
    """Last N question texts asked in these sections (so LLM avoids them)."""
    rows = session.execute(
        select(Question.question_text, SessionRow.id.label("sid"))
        .join(SessionRow, Question.session_id == SessionRow.id)
        .where(Question.section_id.in_(section_ids))
        .order_by(desc(SessionRow.started_at), desc(Question.id))
        .limit(limit)
    ).all()
    return [r.question_text[:120] for r in rows]


def build_adaptive_context(
    session: Session,
    section_ids: List[int],
    iteration_hint: Optional[int] = None,
) -> str:
    """Render the adaptive history block for these sections.

    Returns "" if there is no prior state (cold start).
    """
    h = settings.history
    if not h.enable_compression:
        return ""

    weak = _weak_topics_for_sections(session, section_ids, limit=10)
    mastered = _mastered_topics_for_sections(session, section_ids, limit=5)
    recent_qs = (
        _recent_question_themes(session, section_ids, limit=8)
        if h.include_question_themes_to_avoid
        else []
    )

    if not weak and not mastered and not recent_qs:
        return ""

    parts: List[str] = []

    if weak:
        parts.append("WEAK TOPICS — generate questions emphasising these:")
        for w in weak:
            parts.append(
                f"  • {w['name']} (§{w['section_id']}) "
                f"— section weight {w['section_weight']:.2f}, "
                f"global {w['global_weight']:.2f}, "
                f"wrong {w['wrong']}×"
            )
        parts.append("")

    if mastered:
        parts.append("MASTERED — avoid heavy focus, vary angle if asked at all:")
        for m in mastered:
            parts.append(
                f"  • {m['name']} — weight {m['weight']:.2f}, "
                f"correct {m['correct']}× consecutive"
            )
        parts.append("")

    if recent_qs:
        parts.append("RECENT QUESTION THEMES (do NOT repeat these verbatim):")
        for q in recent_qs:
            parts.append(f"  - {q}")
        parts.append("")

    full = "\n".join(parts).strip()

    # Compress if over budget
    if count_tokens(full) > h.compression_threshold_tokens:
        # Drop the verbatim question list first (it's the bulkiest)
        if recent_qs:
            recent_qs = recent_qs[:3]
            full = _re_render(weak, mastered, recent_qs)
        if count_tokens(full) > h.compression_threshold_tokens:
            weak = weak[:5]
            mastered = mastered[:3]
            full = _re_render(weak, mastered, recent_qs)
        if count_tokens(full) > h.compression_threshold_tokens:
            weak = weak[:3]
            full = _re_render(weak, mastered, recent_qs)

    return full


def _re_render(weak, mastered, recent_qs) -> str:
    parts: List[str] = []
    if weak:
        parts.append("WEAK TOPICS:")
        for w in weak:
            parts.append(
                f"  • {w['name']} (§{w['section_id']}) "
                f"weight {w['section_weight']:.2f}, wrong {w['wrong']}×"
            )
        parts.append("")
    if mastered:
        parts.append("MASTERED (skip):")
        for m in mastered:
            parts.append(f"  • {m['name']}")
        parts.append("")
    if recent_qs:
        parts.append("AVOID repeating recent themes:")
        for q in recent_qs:
            parts.append(f"  - {q[:80]}")
    return "\n".join(parts).strip()
