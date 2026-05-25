"""Prompt assembly for MCQ generation.

Builds the four-part prompt:
  system  +  adaptive context  +  content chunks  +  output instructions

Each part lives in its own token budget bucket. For Phase 2 the adaptive
context is empty (cold-start); Phase 4 will populate it from mastery state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.kb.models import Chunk as ChunkModel


SYSTEM_PROMPT = """\
You are an expert exam-question writer working from a single source PDF
(SLATEFALL_DOSSIER). Your job is to generate multiple-choice questions
that are precise, factually grounded, and structurally valid.

RULES (non-negotiable):
1. EVERY question must be answerable from the supplied CONTEXT only.
   Do NOT invent facts not present in the context.
2. Each question has EXACTLY 4 choices (A, B, C, D), all distinct.
3. EXACTLY ONE choice is correct.
4. Provide a concise explanation (1–3 sentences) referencing the context.
5. Include a `source_quote`: a verbatim 10-40 word substring from the
   context that justifies the correct answer.
6. Distractors must be plausible but wrong — close enough to require
   real understanding, not random.

Return ONLY valid JSON. No prose before or after.
"""


OUTPUT_INSTRUCTIONS = """\
Generate exactly {n_questions} MCQs at {difficulty} difficulty.

Output schema:
{{
  "questions": [
    {{
      "question_text": "...",
      "choice_a": "...",
      "choice_b": "...",
      "choice_c": "...",
      "choice_d": "...",
      "correct_answer": "A" | "B" | "C" | "D",
      "explanation": "...",
      "source_quote": "...",
      "topic": "<short topic name, e.g., 'Echo Lock', 'Targeting Procedure'>"
    }}
  ]
}}

Return only the JSON object. No commentary."""


@dataclass
class BuiltPrompt:
    system: str
    user: str
    adaptive_context: str = ""
    content: str = ""
    instructions: str = ""


def render_content_block(chunks: Sequence[ChunkModel]) -> str:
    """Render chunks as a single context block."""
    parts = []
    for c in chunks:
        parts.append(c.content.strip())
    return "\n\n---\n\n".join(parts)


def build_mcq_prompt(
    chunks: Sequence[ChunkModel],
    n_questions: int,
    difficulty: str = "medium",
    adaptive_context: str = "",
    topic_hint: str = "",
) -> BuiltPrompt:
    """Assemble the four-part MCQ-generation prompt.

    `adaptive_context` is the WEAK/MASTERED summary built from history.
    `topic_hint` biases this batch toward a specific topic (single-MCQ calls).
    """
    content_block = render_content_block(chunks)
    instructions = OUTPUT_INSTRUCTIONS.format(
        n_questions=n_questions,
        difficulty=difficulty,
    )

    sections = []
    if adaptive_context.strip():
        sections.append(
            "USER HISTORY SIGNAL (steer question selection toward weak topics, "
            "vary or avoid mastered ones):\n" + adaptive_context.strip()
        )
    if topic_hint.strip():
        sections.append(
            f"TOPIC FOCUS for this batch: {topic_hint.strip()}\n"
            "Each question must primarily test understanding of the topic above."
        )
    sections.append("CONTEXT (verbatim excerpts from the PDF):\n" + content_block)
    sections.append(instructions)

    user_msg = "\n\n".join(sections)

    return BuiltPrompt(
        system=SYSTEM_PROMPT,
        user=user_msg,
        adaptive_context=adaptive_context,
        content=content_block,
        instructions=instructions,
    )


def to_langchain_messages(prompt: BuiltPrompt) -> list:
    """Convert BuiltPrompt → LangChain message list for .invoke()."""
    return [
        SystemMessage(content=prompt.system),
        HumanMessage(content=prompt.user),
    ]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.rag.retriever import retrieve
    from src.rag.token_budget import count_tokens

    engine = create_engine(settings.secrets.database_url, pool_pre_ping=True)

    print("=== Build prompt for Section 5 ===")
    with Session(engine) as session:
        r = retrieve(session, [5], "combat doctrine and tactics", top_k=5)
        prompt = build_mcq_prompt(
            chunks=r.chunks,
            n_questions=5,
            difficulty="medium",
        )

    print(f"System prompt:    {count_tokens(prompt.system):>4} tokens")
    print(f"Content block:    {count_tokens(prompt.content):>4} tokens")
    print(f"Instructions:     {count_tokens(prompt.instructions):>4} tokens")
    print(f"User message:     {count_tokens(prompt.user):>4} tokens")
    total = count_tokens(prompt.system) + count_tokens(prompt.user)
    print(f"Total input:      {total:>4} tokens (limit: {settings.token_management.context_limit})")
    print()
    print(f"Chunks included: {len(r.chunks)}")
    for c in r.chunks:
        print(f"  §{c.sub_section_id:<12} {c.sub_section_title[:55]}")
    print()
    print("=== Prompt preview (first 600 chars of user message) ===")
    print(prompt.user[:600])
    print("...")
