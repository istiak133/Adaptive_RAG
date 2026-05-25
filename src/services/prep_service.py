"""Prep service — orchestrates retrieve → prompt → LLM → validate.

This is the Phase-2 minimum: cold-start MCQ generation. Phase 4 adds
adaptive history-aware prompting on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy.orm import Session

from src.config import settings
from src.llm.providers import get_llm_with_fallback
from src.rag.prompt_builder import (
    BuiltPrompt, build_mcq_prompt, to_langchain_messages,
)
from src.rag.retriever import retrieve, RetrievalResult
from src.rag.token_budget import build_accounting, assert_fits
from src.rag.validator import MCQ, ValidationReport, validate_response


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
) -> MCQGenerationResult:
    """Cold-start MCQ generation for the given sections.

    `query_seed` biases the semantic search toward a topic. If absent we
    use a generic "key concepts" prompt.
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
        adaptive_context="",
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
