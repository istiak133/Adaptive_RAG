"""FastAPI dependencies — DB session, ChromaDB repo, settings."""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.kb.chroma_repo import ChromaRepo


_engine = create_engine(settings.secrets.database_url, pool_pre_ping=True)
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


_chroma_repo: ChromaRepo | None = None


def get_chroma() -> ChromaRepo:
    global _chroma_repo
    if _chroma_repo is None:
        _chroma_repo = ChromaRepo()
    return _chroma_repo
