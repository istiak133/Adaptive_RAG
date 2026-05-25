"""GET sections + topics + chunks (read-only metadata)."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import ChunkView, SectionView, TopicView
from src.kb.models import Chunk, Section, Topic


router = APIRouter(tags=["catalogue"])


@router.get("/sections", response_model=List[SectionView])
def list_sections(db: Session = Depends(get_db)):
    rows = db.scalars(select(Section).order_by(Section.id)).all()
    return [
        SectionView(
            id=r.id, title=r.title,
            page_start=r.page_start, page_end=r.page_end,
            chunk_count=r.chunk_count,
        )
        for r in rows
    ]


@router.get("/sections/{section_id}", response_model=SectionView)
def get_section(section_id: int, db: Session = Depends(get_db)):
    r = db.get(Section, section_id)
    if r is None:
        raise HTTPException(404, f"Section {section_id} not found")
    return SectionView(
        id=r.id, title=r.title,
        page_start=r.page_start, page_end=r.page_end,
        chunk_count=r.chunk_count,
    )


@router.get("/sections/{section_id}/chunks", response_model=List[ChunkView])
def list_chunks_for_section(section_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Chunk)
        .where(Chunk.section_id == section_id)
        .order_by(Chunk.chunk_order)
    ).all()
    if not rows:
        raise HTTPException(404, f"No chunks for section {section_id}")
    return [
        ChunkView(
            id=c.id, section_id=c.section_id,
            sub_section_id=c.sub_section_id,
            sub_section_title=c.sub_section_title,
            token_count=c.token_count, chunk_kind=c.chunk_kind,
            page_start=c.page_start, page_end=c.page_end,
            cross_refs=c.cross_refs or [],
        )
        for c in rows
    ]


@router.get("/topics", response_model=List[TopicView])
def list_topics(db: Session = Depends(get_db)):
    rows = db.scalars(select(Topic).order_by(Topic.name)).all()
    return [
        TopicView(
            id=t.id, name=t.name, slug=t.slug,
            category=t.category, importance=t.importance,
            description=t.description,
        )
        for t in rows
    ]


@router.get("/topics/{topic_id}", response_model=TopicView)
def get_topic(topic_id: int, db: Session = Depends(get_db)):
    t = db.get(Topic, topic_id)
    if t is None:
        raise HTTPException(404, f"Topic {topic_id} not found")
    return TopicView(
        id=t.id, name=t.name, slug=t.slug,
        category=t.category, importance=t.importance,
        description=t.description,
    )
