"""Token budget management for the RAG pipeline.

Strict allocation per prompt component so we never exceed the LLM's
context window:

    system_prompt + instructions + history + content + output_reserve
    ≤ context_limit

Provides:
  - count_tokens(text)            — tiktoken-based counting
  - select_chunks_within_budget() — greedy fit of relevant chunks
  - TokenAccounting               — per-call audit trail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import tiktoken

from src.config import settings
from src.kb.models import Chunk as ChunkModel


_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class TokenAccounting:
    """Per-call audit trail showing how the budget was spent."""
    system_prompt: int = 0
    instructions: int = 0
    history: int = 0
    content: int = 0
    output_reserve: int = 0

    @property
    def total_input(self) -> int:
        return (self.system_prompt + self.instructions
                + self.history + self.content)

    @property
    def total(self) -> int:
        return self.total_input + self.output_reserve

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "instructions": self.instructions,
            "history": self.history,
            "content": self.content,
            "output_reserve": self.output_reserve,
            "total_input": self.total_input,
            "total": self.total,
        }

    def fits(self, context_limit: Optional[int] = None) -> bool:
        limit = context_limit or settings.token_management.context_limit
        return self.total <= limit


@dataclass
class ChunkSelection:
    """Result of selecting chunks to fit a budget."""
    chosen: List[ChunkModel] = field(default_factory=list)
    skipped: List[Tuple[ChunkModel, str]] = field(default_factory=list)
    tokens_used: int = 0
    tokens_remaining: int = 0


def select_chunks_within_budget(
    chunks: Sequence[ChunkModel],
    content_budget: Optional[int] = None,
) -> ChunkSelection:
    """Greedy fill: pick chunks in order until the next one would overflow.

    Caller is responsible for ordering chunks by relevance before passing in.
    """
    budget = content_budget or settings.token_management.content_budget
    selection = ChunkSelection(tokens_remaining=budget)

    for chunk in chunks:
        token_count = chunk.token_count
        if selection.tokens_used + token_count <= budget:
            selection.chosen.append(chunk)
            selection.tokens_used += token_count
            selection.tokens_remaining = budget - selection.tokens_used
        else:
            selection.skipped.append(
                (chunk, f"would exceed budget ({selection.tokens_used + token_count} > {budget})")
            )

    return selection


def build_accounting(
    *,
    system_prompt: str,
    instructions: str,
    history: str = "",
    content: str = "",
) -> TokenAccounting:
    """Compute an accounting from concrete prompt strings."""
    return TokenAccounting(
        system_prompt=count_tokens(system_prompt),
        instructions=count_tokens(instructions),
        history=count_tokens(history),
        content=count_tokens(content),
        output_reserve=settings.token_management.output_reserve,
    )


def assert_fits(accounting: TokenAccounting) -> None:
    """Raise if the proposed prompt won't fit the LLM's context window."""
    limit = settings.token_management.context_limit
    if not accounting.fits(limit):
        raise ValueError(
            f"Prompt total {accounting.total} exceeds context limit {limit}. "
            f"Breakdown: {accounting.to_dict()}"
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    engine = create_engine(settings.secrets.database_url)

    print("=== count_tokens sanity ===")
    print(f"  'Hello world'                 → {count_tokens('Hello world')} tokens")
    print(f"  'Echo Lock failure mode' x100 → {count_tokens('Echo Lock failure mode ' * 100)} tokens")
    print()

    print("=== Token budget allocation (from config) ===")
    tm = settings.token_management
    print(f"  context_limit:     {tm.context_limit}")
    print(f"  output_reserve:    {tm.output_reserve}")
    print(f"  system_prompt:     {tm.system_prompt_budget}")
    print(f"  instructions:      {tm.instructions_budget}")
    print(f"  history:           {tm.history_budget}")
    print(f"  content:           {tm.content_budget}")
    print()

    print("=== Greedy chunk selection (Section 2, content budget) ===")
    with Session(engine) as session:
        s2_chunks = session.scalars(
            select(ChunkModel)
            .where(ChunkModel.section_id == 2)
            .order_by(ChunkModel.chunk_order)
        ).all()
        print(f"  Section 2 has {len(s2_chunks)} chunks")
        total_s2_tokens = sum(c.token_count for c in s2_chunks)
        print(f"  Total tokens in S2: {total_s2_tokens}")

        sel = select_chunks_within_budget(s2_chunks)
        print(f"  Selected within budget {tm.content_budget}: {len(sel.chosen)} chunks, "
              f"{sel.tokens_used} tokens used, {sel.tokens_remaining} remaining")
        if sel.skipped:
            print(f"  Skipped: {len(sel.skipped)} chunks")

    print()
    print("=== build_accounting + assert_fits ===")
    acc = build_accounting(
        system_prompt="You are an MCQ generator." + " word" * 80,
        instructions="Return 5 questions in JSON format." + " word" * 50,
        history="Past weak topics: Echo Lock (4x wrong)" + " word" * 100,
        content="Some PDF content here." + " word" * 500,
    )
    print(f"  Accounting: {acc.to_dict()}")
    print(f"  Fits 8000-token window? {acc.fits()}")
    assert_fits(acc)
    print("  ✓ assert_fits passed")
