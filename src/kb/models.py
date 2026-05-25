"""SQLAlchemy 2.0 models for the Adaptive Document Preparation System.

11 tables across 4 conceptual layers:
  1. PDF Structure   — sections, chunks
  2. Topic System    — topics, section_topics, chunk_topics, question_topics
  3. Sessions        — sessions, questions, answers
  4. Mastery         — topic_mastery, section_topic_mastery
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Single declarative base for all models — Alembic uses Base.metadata."""


# ── Layer 1: PDF Structure ───────────────────────────────────────────


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    page_start: Mapped[Optional[int]] = mapped_column(Integer)
    page_end: Mapped[Optional[int]] = mapped_column(Integer)
    raw_text_length: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="section", cascade="all, delete-orphan"
    )
    section_topics: Mapped[List["SectionTopic"]] = relationship(
        back_populates="section", cascade="all, delete-orphan"
    )
    section_topic_masteries: Mapped[List["SectionTopicMastery"]] = relationship(
        back_populates="section", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("sections.id", ondelete="CASCADE"), nullable=False
    )
    sub_section_id: Mapped[str] = mapped_column(String(64), nullable=False)
    sub_section_title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    has_table: Mapped[bool] = mapped_column(default=False, nullable=False)
    has_bullets: Mapped[bool] = mapped_column(default=False, nullable=False)
    chunk_kind: Mapped[str] = mapped_column(String(32), default="narrative", nullable=False)
    cross_refs: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    section: Mapped[Section] = relationship(back_populates="chunks")
    chunk_topics: Mapped[List["ChunkTopic"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_chunks_section_id", "section_id"),
        Index("ix_chunks_sub_section_id", "sub_section_id"),
        CheckConstraint(
            "chunk_kind IN ('narrative', 'glossary_entry', 'split')",
            name="ck_chunks_chunk_kind",
        ),
    )


# ── Layer 2: Topic System (the brain) ────────────────────────────────


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    importance: Mapped[str] = mapped_column(String(16), default="normal", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    section_topics: Mapped[List["SectionTopic"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )
    chunk_topics: Mapped[List["ChunkTopic"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )
    question_topics: Mapped[List["QuestionTopic"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )
    topic_mastery: Mapped[Optional["TopicMastery"]] = relationship(
        back_populates="topic", uselist=False, cascade="all, delete-orphan"
    )
    section_topic_masteries: Mapped[List["SectionTopicMastery"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_topics_slug", "slug"),
        Index("ix_topics_category", "category"),
        CheckConstraint(
            "importance IN ('critical', 'normal', 'minor')",
            name="ck_topics_importance",
        ),
    )


class SectionTopic(Base):
    """Which topics appear in which section, and at what depth."""

    __tablename__ = "section_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("sections.id", ondelete="CASCADE"), nullable=False
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False
    )
    depth: Mapped[str] = mapped_column(String(16), nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    section: Mapped[Section] = relationship(back_populates="section_topics")
    topic: Mapped[Topic] = relationship(back_populates="section_topics")

    __table_args__ = (
        UniqueConstraint("section_id", "topic_id", name="uq_section_topics"),
        CheckConstraint(
            "depth IN ('primary', 'secondary', 'mention')",
            name="ck_section_topics_depth",
        ),
        Index("ix_section_topics_topic_id", "topic_id"),
    )


class ChunkTopic(Base):
    """Which chunks cover which topics (many-to-many)."""

    __tablename__ = "chunk_topics"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True
    )

    chunk: Mapped[Chunk] = relationship(back_populates="chunk_topics")
    topic: Mapped[Topic] = relationship(back_populates="chunk_topics")

    __table_args__ = (Index("ix_chunk_topics_topic_id", "topic_id"),)


# ── Layer 3: Sessions, Questions, Answers ────────────────────────────


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sections_studied: Mapped[list] = mapped_column(JSONB, nullable=False)
    total_questions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    wrong_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    score_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    difficulty_level: Mapped[str] = mapped_column(
        String(16), default="medium", nullable=False
    )
    is_cold_start: Mapped[bool] = mapped_column(nullable=False)
    adaptive_context: Mapped[Optional[dict]] = mapped_column(JSONB)
    token_usage: Mapped[Optional[dict]] = mapped_column(JSONB)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    questions: Mapped[List["Question"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_sessions_started_at", "started_at"),
        Index("ix_sessions_completed_at", "completed_at"),
        CheckConstraint(
            "difficulty_level IN ('easy', 'medium', 'hard')",
            name="ck_sessions_difficulty",
        ),
    )


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    section_id: Mapped[int] = mapped_column(
        ForeignKey("sections.id", ondelete="RESTRICT"), nullable=False
    )
    source_chunk_ids: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(16), default="medium", nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    choice_a: Mapped[str] = mapped_column(Text, nullable=False)
    choice_b: Mapped[str] = mapped_column(Text, nullable=False)
    choice_c: Mapped[str] = mapped_column(Text, nullable=False)
    choice_d: Mapped[str] = mapped_column(Text, nullable=False)
    correct_answer: Mapped[str] = mapped_column(String(1), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    source_quote: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship(back_populates="questions")
    answer: Mapped[Optional["Answer"]] = relationship(
        back_populates="question", uselist=False, cascade="all, delete-orphan"
    )
    question_topics: Mapped[List["QuestionTopic"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_questions_session_id", "session_id"),
        Index("ix_questions_section_id", "section_id"),
        CheckConstraint(
            "correct_answer IN ('A', 'B', 'C', 'D')",
            name="ck_questions_correct_answer",
        ),
        CheckConstraint(
            "difficulty IN ('easy', 'medium', 'hard')",
            name="ck_questions_difficulty",
        ),
    )


class QuestionTopic(Base):
    """One question can test multiple topics — primary vs secondary impact."""

    __tablename__ = "question_topics"

    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True
    )
    is_primary: Mapped[bool] = mapped_column(default=True, nullable=False)

    question: Mapped[Question] = relationship(back_populates="question_topics")
    topic: Mapped[Topic] = relationship(back_populates="question_topics")

    __table_args__ = (Index("ix_question_topics_topic_id", "topic_id"),)


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    user_answer: Mapped[str] = mapped_column(String(1), nullable=False)
    is_correct: Mapped[bool] = mapped_column(nullable=False)
    answered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    question: Mapped[Question] = relationship(back_populates="answer")

    __table_args__ = (
        Index("ix_answers_is_correct", "is_correct"),
        CheckConstraint(
            "user_answer IN ('A', 'B', 'C', 'D')",
            name="ck_answers_user_answer",
        ),
    )


# ── Layer 4: Mastery Tracking (adaptive state) ───────────────────────


class TopicMastery(Base):
    """Global mastery state per topic (across all sections)."""

    __tablename__ = "topic_mastery"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    times_asked: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    times_correct: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    times_wrong: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_mastered: Mapped[bool] = mapped_column(default=False, nullable=False)
    last_asked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_wrong_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    topic: Mapped[Topic] = relationship(back_populates="topic_mastery")

    __table_args__ = (
        Index("ix_topic_mastery_weight", "weight"),
        Index("ix_topic_mastery_is_mastered", "is_mastered"),
    )


class SectionTopicMastery(Base):
    """Section-specific mastery — same topic can be mastered in §2 but weak
    in §9 (e.g., user knows Echo Lock definition but not its application)."""

    __tablename__ = "section_topic_mastery"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("sections.id", ondelete="CASCADE"), nullable=False
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False
    )
    times_asked: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    times_correct: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    times_wrong: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    section: Mapped[Section] = relationship(back_populates="section_topic_masteries")
    topic: Mapped[Topic] = relationship(back_populates="section_topic_masteries")

    __table_args__ = (
        UniqueConstraint("section_id", "topic_id", name="uq_section_topic_mastery"),
        Index("ix_section_topic_mastery_weight", "weight"),
    )
