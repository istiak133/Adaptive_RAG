"""Adaptive state inspection endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import MasteryView, SectionMasteryView
from src.kb.models import SectionTopicMastery, Topic, TopicMastery


router = APIRouter(prefix="/mastery", tags=["mastery"])


@router.get("", response_model=List[MasteryView])
def list_mastery(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Topic.id, Topic.name, Topic.slug, TopicMastery)
        .join(TopicMastery, TopicMastery.topic_id == Topic.id)
        .order_by(TopicMastery.weight.desc())
    ).all()
    return [
        MasteryView(
            topic_id=r[0],
            topic_name=r[1],
            topic_slug=r[2],
            times_asked=r[3].times_asked,
            times_correct=r[3].times_correct,
            times_wrong=r[3].times_wrong,
            current_streak=r[3].current_streak,
            weight=r[3].weight,
            is_mastered=r[3].is_mastered,
            last_asked_at=r[3].last_asked_at.isoformat() if r[3].last_asked_at else None,
            last_wrong_at=r[3].last_wrong_at.isoformat() if r[3].last_wrong_at else None,
        )
        for r in rows
    ]


@router.get("/regressions", response_model=List[MasteryView])
def list_regressions(db: Session = Depends(get_db)):
    """Topics that were mastered but recently got wrong."""
    rows = db.execute(
        select(Topic.id, Topic.name, Topic.slug, TopicMastery)
        .join(TopicMastery, TopicMastery.topic_id == Topic.id)
        .where(TopicMastery.is_mastered.is_(False))
        .where(TopicMastery.times_correct >= 3)
        .order_by(TopicMastery.weight.desc())
    ).all()
    return [
        MasteryView(
            topic_id=r[0], topic_name=r[1], topic_slug=r[2],
            times_asked=r[3].times_asked,
            times_correct=r[3].times_correct,
            times_wrong=r[3].times_wrong,
            current_streak=r[3].current_streak,
            weight=r[3].weight,
            is_mastered=r[3].is_mastered,
            last_asked_at=r[3].last_asked_at.isoformat() if r[3].last_asked_at else None,
            last_wrong_at=r[3].last_wrong_at.isoformat() if r[3].last_wrong_at else None,
        )
        for r in rows
    ]


@router.get("/by-section/{section_id}", response_model=List[SectionMasteryView])
def mastery_by_section(section_id: int, db: Session = Depends(get_db)):
    rows = db.execute(
        select(SectionTopicMastery, Topic.name)
        .join(Topic, SectionTopicMastery.topic_id == Topic.id)
        .where(SectionTopicMastery.section_id == section_id)
        .order_by(SectionTopicMastery.weight.desc())
    ).all()
    return [
        SectionMasteryView(
            section_id=r[0].section_id,
            topic_id=r[0].topic_id,
            topic_name=r[1],
            times_asked=r[0].times_asked,
            times_correct=r[0].times_correct,
            times_wrong=r[0].times_wrong,
            weight=r[0].weight,
        )
        for r in rows
    ]
