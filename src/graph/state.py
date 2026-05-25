"""LangGraph state for the prep flow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from src.prep.scorer import ScoreReport
from src.services.session_service import GeneratedQuestion


class PrepState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────
    section_ids: List[int]
    questions_per_section: int
    difficulty: Optional[str]
    simulate_strategy: str
    seed: Optional[int]

    # ── Runtime / branching ──────────────────────────────────────────
    is_cold_start: bool
    adaptive_context: str
    allocation_summary: Dict[int, Dict[str, Any]]

    # ── Generation outputs ───────────────────────────────────────────
    session_id: int
    all_generated: List[GeneratedQuestion]
    chunks_used: Dict[int, List[str]]
    token_usage: Dict[str, int]
    rejected_count: int
    retry_count: int

    # ── Final ────────────────────────────────────────────────────────
    question_ids: List[int]
    score_report: ScoreReport
    mastery_changes: List[Dict[str, Any]]
    elapsed: float
