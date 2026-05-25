"""Node functions for the prep-flow state machine.

Each node takes the current PrepState and returns a partial state update.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.graph.state import PrepState
from src.kb.mastery import update_mastery_after_answer
from src.kb.models import Session as SessionRow
from src.prep import scorer, simulator
from src.prep.allocator import allocate
from src.prep.difficulty import pick_difficulty
from src.rag.history_compressor import build_adaptive_context
from src.services import session_service
from src.services.session_service import GeneratedQuestion


# ── Helpers ──────────────────────────────────────────────────────────


def _has_prior_history(session: Session, section_ids: List[int]) -> bool:
    rows = session.execute(
        select(SessionRow.sections_studied)
        .where(SessionRow.completed_at.is_not(None))
    ).all()
    target = set(section_ids)
    for (studied,) in rows:
        if studied and set(studied) & target:
            return True
    return False


def _zero_token_usage() -> Dict[str, int]:
    return {
        "system_prompt": 0, "instructions": 0,
        "history": 0, "content": 0,
        "output_reserve": 0, "total_input": 0, "total": 0,
    }


# ── Nodes ────────────────────────────────────────────────────────────


def detect_mode_node(state: PrepState, *, session: Session) -> Dict[str, Any]:
    """Decide cold-start vs adaptive, build adaptive context, pick difficulty."""
    section_ids = state["section_ids"]

    is_cold_start = not _has_prior_history(session, section_ids)

    adaptive_context = ""
    if not is_cold_start:
        adaptive_context = build_adaptive_context(session, section_ids)

    difficulty = state.get("difficulty")
    if difficulty is None:
        difficulty = (
            pick_difficulty(session, section_ids[0])
            if not is_cold_start
            else settings.mcq.default_difficulty
        )

    return {
        "is_cold_start": is_cold_start,
        "adaptive_context": adaptive_context,
        "difficulty": difficulty,
    }


def create_session_node(state: PrepState, *, session: Session) -> Dict[str, Any]:
    """Insert the sessions row and record the run's metadata."""
    sid = session_service.start_session(
        session,
        section_ids=state["section_ids"],
        is_cold_start=state["is_cold_start"],
        difficulty=state["difficulty"],
        adaptive_context=(
            {"summary": state["adaptive_context"]}
            if state["adaptive_context"]
            else None
        ),
    )
    return {
        "session_id": sid,
        "all_generated": [],
        "chunks_used": {},
        "token_usage": _zero_token_usage(),
        "rejected_count": 0,
        "allocation_summary": {},
    }


def generate_node(state: PrepState, *, session: Session) -> Dict[str, Any]:
    """Per-section allocation → per-seed LLM call with retry (max 3)."""
    from src.services.prep_service import generate_mcqs  # avoid circular

    section_ids = state["section_ids"]
    n_per_section = state["questions_per_section"]
    adaptive_context = state["adaptive_context"]
    difficulty = state["difficulty"]

    all_generated: List[GeneratedQuestion] = []
    chunks_used: Dict[int, List[str]] = {}
    token_usage = dict(state["token_usage"])
    rejected_count = state.get("rejected_count", 0)
    allocation_summary: Dict[int, Dict[str, Any]] = {}

    for sec_id in section_ids:
        plan = allocate(session, sec_id, n_per_section)
        allocation_summary[sec_id] = {
            "mode": plan.mode,
            "seeds": plan.seeds,
            "weights": [
                {
                    "topic": a.topic_name,
                    "weight": round(a.effective_weight, 2),
                    "questions": a.questions,
                }
                for a in plan.allocations[:5]
            ],
        }
        chunks_used[sec_id] = []

        for seed_topic in plan.seeds:
            # Retry loop — up to settings.llm.max_retries
            mcq_added = False
            for attempt in range(settings.llm.max_retries):
                gen = generate_mcqs(
                    session=session,
                    section_ids=[sec_id],
                    n_questions=1,
                    difficulty=difficulty,
                    query_seed=seed_topic,
                    adaptive_context=adaptive_context,
                    topic_hint=seed_topic if plan.mode != "cold" else "",
                )
                chunks_used[sec_id].extend(gen.chunks_used)
                for k in token_usage:
                    token_usage[k] += gen.token_usage.get(k, 0)

                if gen.mcqs:
                    for mcq in gen.mcqs:
                        all_generated.append(GeneratedQuestion(
                            section_id=sec_id,
                            mcq=mcq,
                            source_sub_ids=gen.chunks_used,
                        ))
                    mcq_added = True
                    break

                rejected_count += len(gen.rejected)

            if not mcq_added:
                rejected_count += 1  # exhausted retries

    return {
        "all_generated": all_generated,
        "chunks_used": chunks_used,
        "token_usage": token_usage,
        "rejected_count": rejected_count,
        "allocation_summary": allocation_summary,
    }


def record_node(state: PrepState, *, session: Session) -> Dict[str, Any]:
    """Persist all generated questions and link topics."""
    qids = session_service.record_questions(
        session,
        state["session_id"],
        state["all_generated"],
        difficulty=state["difficulty"],
    )
    return {"question_ids": qids}


def simulate_and_score_node(state: PrepState, *, session: Session) -> Dict[str, Any]:
    """Simulate user answers, record them, update mastery, score the session."""
    sim_pairs = simulator.simulate_answers(
        session,
        state["question_ids"],
        strategy=state["simulate_strategy"],
        seed=state.get("seed"),
    )

    mastery_changes: List[Dict[str, Any]] = []
    for qid, ans in sim_pairs:
        is_correct = session_service.record_answer(session, qid, ans)
        deltas = update_mastery_after_answer(session, qid, is_correct)
        for d in deltas:
            if d.regression_detected:
                mastery_changes.append({
                    "type": "regression",
                    "topic": d.topic_name,
                    "weight": round(d.new_weight, 2),
                })

    report = scorer.score_session(session, state["session_id"])
    return {"score_report": report, "mastery_changes": mastery_changes}


def complete_node(state: PrepState, *, session: Session) -> Dict[str, Any]:
    """Finalize the session row + persist allocation/regression metadata."""
    session_service.complete_session(
        session,
        state["session_id"],
        token_usage=state["token_usage"],
    )

    sess = session.get(SessionRow, state["session_id"])
    if sess:
        existing = sess.adaptive_context or {}
        sess.adaptive_context = {
            **existing,
            "allocation": state.get("allocation_summary", {}),
            "regressions": state.get("mastery_changes", []),
        }
        session.commit()

    return {}


# ── Conditional routers ──────────────────────────────────────────────


def route_after_detect(state: PrepState) -> str:
    """Single edge for now — both paths converge into create_session.
    Kept as an explicit router so future variants can branch (e.g., skip
    record/simulate during dry-runs)."""
    return "create_session"
