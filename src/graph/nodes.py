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
    """Decide cold-start vs adaptive; pick difficulty.

    Does NOT build the adaptive context — that's a separate node so the
    cold-start branch can skip it entirely.
    """
    section_ids = state["section_ids"]
    is_cold_start = not _has_prior_history(session, section_ids)

    difficulty = state.get("difficulty")
    if difficulty is None:
        difficulty = (
            pick_difficulty(session, section_ids[0])
            if not is_cold_start
            else settings.mcq.default_difficulty
        )

    return {
        "is_cold_start": is_cold_start,
        "difficulty": difficulty,
        "adaptive_context": "",   # filled in by load_adaptive_context (adaptive branch only)
    }


def load_adaptive_context_node(
    state: PrepState, *, session: Session,
) -> Dict[str, Any]:
    """Build WEAK/MASTERED context from history.

    Skipped on cold-start runs via the route_after_detect router.
    """
    ac = build_adaptive_context(session, state["section_ids"])
    return {"adaptive_context": ac}


def create_session_node(
    state: PrepState, *, session: Session,
) -> Dict[str, Any]:
    """Insert the sessions row + initialise accumulators."""
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
        "retry_count": 0,
    }


def generate_node(state: PrepState, *, session: Session) -> Dict[str, Any]:
    """Per-section allocation → per-seed LLM call with retry (max 3)."""
    from src.services.prep_service import generate_mcqs  # avoid circular

    section_ids = state["section_ids"]
    n_per_section = state["questions_per_section"]
    adaptive_context = state["adaptive_context"]
    difficulty = state["difficulty"]

    # On retry runs we keep what we already generated and only fill the gap.
    all_generated: List[GeneratedQuestion] = list(state.get("all_generated") or [])
    chunks_used: Dict[int, List[str]] = dict(state.get("chunks_used") or {})
    token_usage = dict(state.get("token_usage") or _zero_token_usage())
    rejected_count = state.get("rejected_count", 0)
    allocation_summary: Dict[int, Dict[str, Any]] = (
        dict(state.get("allocation_summary") or {})
    )
    retry_count = state.get("retry_count", 0)

    expected_total = n_per_section * len(section_ids)
    already_have = len(all_generated)

    for sec_id in section_ids:
        # How many do we already have from this section?
        have_in_sec = sum(1 for g in all_generated if g.section_id == sec_id)
        need_in_sec = n_per_section - have_in_sec
        if need_in_sec <= 0:
            continue

        plan = allocate(session, sec_id, need_in_sec)
        allocation_summary.setdefault(sec_id, {
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
        })
        chunks_used.setdefault(sec_id, [])

        for seed_topic in plan.seeds:
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
                rejected_count += 1

    return {
        "all_generated": all_generated,
        "chunks_used": chunks_used,
        "token_usage": token_usage,
        "rejected_count": rejected_count,
        "allocation_summary": allocation_summary,
        "retry_count": retry_count + 1,
    }


def validate_generation_node(
    state: PrepState, *, session: Session,
) -> Dict[str, Any]:
    """Decide whether generation produced enough MCQs to proceed."""
    expected_total = (
        state["questions_per_section"] * len(state["section_ids"])
    )
    actual = len(state.get("all_generated") or [])
    return {"generation_complete": actual >= expected_total}


def record_node(state: PrepState, *, session: Session) -> Dict[str, Any]:
    """Persist all generated questions and link topics."""
    qids = session_service.record_questions(
        session,
        state["session_id"],
        state["all_generated"],
        difficulty=state["difficulty"],
    )
    return {"question_ids": qids}


def simulate_and_score_node(
    state: PrepState, *, session: Session,
) -> Dict[str, Any]:
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


# ── Conditional routers (REAL branching) ─────────────────────────────


def route_after_detect(state: PrepState) -> str:
    """Cold-start skips adaptive-context loading."""
    if state["is_cold_start"]:
        return "create_session"
    return "load_adaptive_context"


def route_after_validation(state: PrepState) -> str:
    """Proceed if we have enough MCQs; otherwise loop back to generate
    until max_retries is hit."""
    if state.get("generation_complete", False):
        return "record"
    if state.get("retry_count", 0) >= settings.llm.max_retries:
        return "record"   # give up after exhausting retries
    return "generate"     # loop back
