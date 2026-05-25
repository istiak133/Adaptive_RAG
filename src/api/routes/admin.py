"""Admin / inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from src.api.dependencies import get_chroma, get_db
from src.api.schemas import AdminStats, ReindexResponse
from src.kb.chroma_repo import ChromaRepo
from src.kb.models import (
    Answer, Chunk, ChunkTopic, Question,
    Section, SectionTopic, Session as SessionRow, Topic,
)


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats", response_model=AdminStats)
def stats(
    db: Session = Depends(get_db),
    chroma: ChromaRepo = Depends(get_chroma),
):
    def count(model):
        return db.scalar(select(func.count()).select_from(model)) or 0

    try:
        av = db.execute(
            text("SELECT version_num FROM alembic_version LIMIT 1")
        ).scalar()
    except Exception:
        av = None

    return AdminStats(
        sections=count(Section),
        chunks=count(Chunk),
        topics=count(Topic),
        section_topics=count(SectionTopic),
        chunk_topics=count(ChunkTopic),
        questions=count(Question),
        sessions=count(SessionRow),
        answers=count(Answer),
        chromadb_count=chroma.count(),
        alembic_version=av,
    )


@router.post("/reindex", response_model=ReindexResponse)
def reindex():
    """Re-run the full ingestion pipeline (PDF → KB). Idempotent."""
    try:
        from src.ingestion.indexer import run_indexer
        stats = run_indexer()
        return ReindexResponse(status="ok", stats=stats)
    except Exception as e:
        return ReindexResponse(status="failed", error=str(e))
