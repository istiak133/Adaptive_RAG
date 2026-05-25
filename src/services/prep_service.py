"""Prep service — orchestrates retrieve → prompt → LLM → validate, and (in
Phase 3) the full session lifecycle: start → generate → record → simulate
→ score → complete.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.kb.mastery import update_mastery_after_answer
from src.kb.models import Session as SessionRow
from src.llm.providers import get_llm_with_fallback
from src.prep import scorer, simulator
from src.prep.allocator import AllocationPlan, allocate
from src.prep.difficulty import pick_difficulty
from src.rag.history_compressor import build_adaptive_context
from src.rag.prompt_builder import (
    BuiltPrompt, build_mcq_prompt, to_langchain_messages,
)
from src.rag.retriever import retrieve, RetrievalResult
from src.rag.token_budget import build_accounting, assert_fits
from src.rag.validator import MCQ, ValidationReport, validate_response
from src.services import session_service
from src.services.session_service import GeneratedQuestion


@dataclass
class MCQGenerationResult:
    mcqs: List[MCQ] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)
    chunks_used: List[str] = field(default_factory=list)  # sub_section_ids
    token_usage: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    raw_llm_output: str = ""


def generate_mcqs(
    session: Session,
    section_ids: List[int],
    n_questions: int,
    difficulty: str = "medium",
    query_seed: Optional[str] = None,
    adaptive_context: str = "",
    topic_hint: str = "",
) -> MCQGenerationResult:
    """Generate MCQs from given sections.

    `query_seed`         — biases the semantic search toward a topic
    `adaptive_context`   — WEAK/MASTERED summary injected into the prompt
    `topic_hint`         — single-topic focus for this batch (used by allocator)
    """
    t0 = time.time()

    # 1. Retrieve relevant chunks
    query = query_seed or (
        f"key concepts, definitions, and critical facts in "
        f"section {', '.join(str(s) for s in section_ids)}"
    )
    retrieval = retrieve(
        session=session,
        section_ids=section_ids,
        query=query,
        top_k=settings.retrieval.vector_top_k,
        exclude_kinds=None,
    )

    # 2. Build prompt
    prompt = build_mcq_prompt(
        chunks=retrieval.chunks,
        n_questions=n_questions,
        difficulty=difficulty,
        adaptive_context=adaptive_context,
        topic_hint=topic_hint,
    )

    # 3. Check token budget
    accounting = build_accounting(
        system_prompt=prompt.system,
        instructions=prompt.instructions,
        history=prompt.adaptive_context,
        content=prompt.content,
    )
    assert_fits(accounting)

    # 4. LLM call (with fallback chain)
    llm = get_llm_with_fallback()
    messages = to_langchain_messages(prompt)
    response = llm.invoke(messages)
    raw_output = response.content if hasattr(response, "content") else str(response)

    # 5. Validate
    report = validate_response(
        raw_text=raw_output,
        context=prompt.content,
        expected_count=n_questions,
    )

    return MCQGenerationResult(
        mcqs=report.accepted,
        rejected=report.rejected,
        chunks_used=[c.sub_section_id for c in retrieval.chunks],
        token_usage=accounting.to_dict(),
        elapsed_seconds=round(time.time() - t0, 2),
        raw_llm_output=raw_output,
    )


# ── Full session lifecycle (Phase 3) ────────────────────────────────


@dataclass
class PrepSessionResult:
    session_id: int
    score_report: scorer.ScoreReport
    chunks_used_per_section: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    generation_rejects: int = 0


def _has_prior_history_for_sections(
    session: Session, section_ids: List[int],
) -> bool:
    """Return True if any prior completed session covered any of these sections."""
    rows = session.execute(
        select(SessionRow.sections_studied)
        .where(SessionRow.completed_at.is_not(None))
    ).all()
    target = set(section_ids)
    for (studied,) in rows:
        if studied and set(studied) & target:
            return True
    return False


def run_prep_session(
    session: Session,
    section_ids: List[int],
    questions_per_section: int = 5,
    difficulty: Optional[str] = None,
    simulate_strategy: simulator.Strategy = "weighted",
    seed: Optional[int] = None,
) -> PrepSessionResult:
    """End-to-end adaptive prep session (executed as a LangGraph state machine).

    Auto-detects cold-start vs adaptive based on prior history. On adaptive
    runs, builds WEAK/MASTERED context and biases the allocator. After scoring,
    updates mastery state to close the loop.
    """
    from src.graph.graph import build_prep_graph

    t0 = time.time()

    initial_state = {
        "section_ids": section_ids,
        "questions_per_section": questions_per_section,
        "difficulty": difficulty,
        "simulate_strategy": simulate_strategy,
        "seed": seed,
    }

    graph = build_prep_graph(session)
    final_state = graph.invoke(initial_state)

    return PrepSessionResult(
        session_id=final_state["session_id"],
        score_report=final_state["score_report"],
        chunks_used_per_section=final_state.get("chunks_used", {}),
        token_usage=final_state.get("token_usage", {}),
        elapsed_seconds=round(time.time() - t0, 2),
        generation_rejects=final_state.get("rejected_count", 0),
    )


def _run_prep_session_legacy(
    session: Session,
    section_ids: List[int],
    questions_per_section: int = 5,
    difficulty: Optional[str] = None,
    simulate_strategy: simulator.Strategy = "weighted",
    seed: Optional[int] = None,
) -> PrepSessionResult:
    """OBSOLETE — kept for reference. The LangGraph version above is used."""
    t0 = time.time()

    # 1. Detect cold-start vs adaptive
    is_cold_start = not _has_prior_history_for_sections(session, section_ids)

    # 2. Pre-compute adaptive context (used in prompts) — empty if cold
    adaptive_context = (
        ""
        if is_cold_start
        else build_adaptive_context(session, section_ids)
    )

    # 3. Difficulty picker (per-section average); fall back to caller override
    if difficulty is None and not is_cold_start:
        # Pick from first section as proxy for the run
        difficulty = pick_difficulty(session, section_ids[0])
    elif difficulty is None:
        difficulty = settings.mcq.default_difficulty

    # 4. Start session
    session_id = session_service.start_session(
        session,
        section_ids=section_ids,
        is_cold_start=is_cold_start,
        difficulty=difficulty,
        adaptive_context={"summary": adaptive_context} if adaptive_context else None,
    )

    chunks_used: dict = {}
    aggregate_tokens = {
        "system_prompt": 0, "instructions": 0,
        "history": 0, "content": 0,
        "output_reserve": 0, "total_input": 0, "total": 0,
    }
    total_rejected = 0
    allocation_summary: Dict[int, dict] = {}

    # 5. Per section: weighted allocation → one MCQ per seed
    all_generated: List[GeneratedQuestion] = []
    for sec_id in section_ids:
        plan = allocate(session, sec_id, questions_per_section)
        allocation_summary[sec_id] = {
            "mode": plan.mode,
            "seeds": plan.seeds,
            "weights": [
                {"topic": a.topic_name, "weight": round(a.effective_weight, 2),
                 "questions": a.questions}
                for a in plan.allocations[:5]
            ],
        }
        chunks_used[sec_id] = []
        for seed_topic in plan.seeds:
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
            total_rejected += len(gen.rejected)
            for k in aggregate_tokens:
                aggregate_tokens[k] += gen.token_usage.get(k, 0)
            for mcq in gen.mcqs:
                all_generated.append(GeneratedQuestion(
                    section_id=sec_id,
                    mcq=mcq,
                    source_sub_ids=gen.chunks_used,
                ))

    # 6. Persist questions
    question_ids = session_service.record_questions(
        session, session_id, all_generated, difficulty=difficulty,
    )

    # 7. Simulate user answers
    sim_pairs = simulator.simulate_answers(
        session, question_ids, strategy=simulate_strategy, seed=seed,
    )

    # 8. Record answers + update mastery (the closing-the-loop step)
    mastery_changes: List[dict] = []
    for qid, ans in sim_pairs:
        is_correct = session_service.record_answer(session, qid, ans)
        deltas = update_mastery_after_answer(session, qid, is_correct)
        for d in deltas:
            if d.regression_detected:
                mastery_changes.append({
                    "type": "regression",
                    "topic": d.topic_name,
                    "weight": d.new_weight,
                })

    # 9. Score
    report = scorer.score_session(session, session_id)

    # 10. Complete session
    session_service.complete_session(
        session, session_id, token_usage=aggregate_tokens,
    )

    # Persist allocation summary into session metadata for traceability
    sess = session.get(SessionRow, session_id)
    if sess and sess.adaptive_context is None:
        sess.adaptive_context = {"allocation": allocation_summary}
    elif sess and isinstance(sess.adaptive_context, dict):
        sess.adaptive_context = {
            **sess.adaptive_context,
            "allocation": allocation_summary,
            "regressions": mastery_changes,
        }
    session.commit()

    return PrepSessionResult(
        session_id=session_id,
        score_report=report,
        chunks_used_per_section=chunks_used,
        token_usage=aggregate_tokens,
        elapsed_seconds=round(time.time() - t0, 2),
        generation_rejects=total_rejected,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from sqlalchemy import create_engine

    engine = create_engine(settings.secrets.database_url, pool_pre_ping=True)

    print("=" * 70)
    print("PHASE 2 MILESTONE — Real MCQ Generation")
    print("=" * 70)
    print()
    print("Generating 5 MCQs from Section 2 (Powers)...")
    print()

    with Session(engine) as session:
        result = generate_mcqs(
            session=session,
            section_ids=[2],
            n_questions=5,
            difficulty="medium",
        )

    print(f"✓ Generated in {result.elapsed_seconds}s")
    print(f"  Chunks used:   {len(result.chunks_used)} → {result.chunks_used[:6]}")
    print(f"  Token usage:   {result.token_usage}")
    print(f"  Accepted MCQs: {len(result.mcqs)} / 5")
    print(f"  Rejected:      {len(result.rejected)}")
    if result.rejected:
        for r in result.rejected:
            print(f"    - {r['reason'][:80]}")
    print()

    for i, m in enumerate(result.mcqs, 1):
        print(f"───── Q{i} [{m.topic}] ─────")
        print(f"Q: {m.question_text}")
        print(f"  A) {m.choice_a}")
        print(f"  B) {m.choice_b}")
        print(f"  C) {m.choice_c}")
        print(f"  D) {m.choice_d}")
        print(f"Correct: {m.correct_answer}")
        print(f"Explanation: {m.explanation}")
        print(f"Source: \"{m.source_quote[:100]}…\"" if len(m.source_quote) > 100
              else f"Source: \"{m.source_quote}\"")
        print()
