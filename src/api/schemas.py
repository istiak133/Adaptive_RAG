"""Pydantic request/response schemas for the FastAPI layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Choice = Literal["A", "B", "C", "D"]
Strategy = Literal["weighted", "random", "all_correct"]


# ── Sections / Topics / Chunks ───────────────────────────────────────


class SectionView(BaseModel):
    id: int
    title: str
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_count: int


class TopicView(BaseModel):
    id: int
    name: str
    slug: str
    category: str
    importance: str
    description: Optional[str] = None


class ChunkView(BaseModel):
    id: int
    section_id: int
    sub_section_id: str
    sub_section_title: str
    token_count: int
    chunk_kind: str
    page_start: int
    page_end: int
    cross_refs: List[str] = []


# ── Prep flow ────────────────────────────────────────────────────────


class PrepStartRequest(BaseModel):
    section_ids: List[int] = Field(..., min_length=1)
    questions_per_section: int = Field(default=5, ge=1, le=20)
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    simulate_strategy: Strategy = "weighted"
    seed: Optional[int] = None


class QuestionView(BaseModel):
    id: int
    section_id: int
    question_text: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    correct_answer: Choice
    explanation: str
    source_quote: Optional[str] = None
    user_answer: Optional[Choice] = None
    is_correct: Optional[bool] = None


class PrepSessionResponse(BaseModel):
    session_id: int
    is_cold_start: bool
    difficulty: str
    score_pct: float
    correct: int
    wrong: int
    total_questions: int
    questions: List[QuestionView]
    chunks_used: Dict[str, List[str]]
    token_usage: Dict[str, int]
    generation_rejects: int
    elapsed_seconds: float
    regressions: List[Dict[str, Any]] = []


# ── Sessions ─────────────────────────────────────────────────────────


class SessionSummary(BaseModel):
    session_id: int
    sections_studied: List[int]
    score_pct: float
    is_cold_start: bool
    total_questions: int
    correct_count: int
    wrong_count: int
    started_at: Optional[str]
    completed_at: Optional[str]


class SessionDetail(SessionSummary):
    difficulty_level: str
    adaptive_context: Optional[Dict[str, Any]] = None
    token_usage: Optional[Dict[str, Any]] = None
    questions: List[QuestionView] = []


class KBSnapshot(BaseModel):
    snapshot_after_session: int
    exported_at: str
    total_sessions_in_kb: int
    recent_sessions: List[SessionDetail]
    adaptive_state: Dict[str, Any]


# ── Mastery ──────────────────────────────────────────────────────────


class MasteryView(BaseModel):
    topic_id: int
    topic_name: str
    topic_slug: str
    times_asked: int
    times_correct: int
    times_wrong: int
    current_streak: int
    weight: float
    is_mastered: bool
    last_asked_at: Optional[str] = None
    last_wrong_at: Optional[str] = None


class SectionMasteryView(BaseModel):
    section_id: int
    topic_id: int
    topic_name: str
    times_asked: int
    times_correct: int
    times_wrong: int
    weight: float


# ── Scenarios ────────────────────────────────────────────────────────


class ScenarioBRunRequest(BaseModel):
    questions_per_section: int = Field(default=5, ge=1, le=10)
    simulate_strategy: Strategy = "weighted"
    seed: Optional[int] = 42
    reset_state: bool = True


class ScenarioBIterationResult(BaseModel):
    iteration: int
    section_ids: List[int]
    session_id: int
    score_pct: float
    output_dir: str


class ScenarioBRunResponse(BaseModel):
    iterations: List[ScenarioBIterationResult]
    total_elapsed_seconds: float
    output_root: str


# ── Admin ────────────────────────────────────────────────────────────


class AdminStats(BaseModel):
    sections: int
    chunks: int
    topics: int
    section_topics: int
    chunk_topics: int
    questions: int
    sessions: int
    answers: int
    chromadb_count: int
    alembic_version: Optional[str] = None


class ReindexResponse(BaseModel):
    status: Literal["ok", "failed"]
    stats: Dict[str, Any] = {}
    error: Optional[str] = None
