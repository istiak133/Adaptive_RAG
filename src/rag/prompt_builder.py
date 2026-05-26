"""Prompt assembly for MCQ generation using LangChain ChatPromptTemplate.

Builds a multi-part chat prompt:
  • system message       — generation rules (no hallucination, structural)
  • human message        — adaptive context + topic hint + content + format instructions

The format instructions come from LangChain's parser so the LLM's contract
matches our Pydantic schema exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
5. Include a `source_quote`: a verbatim 10–40 word substring from the
   context that justifies the correct answer.
6. Distractors must be plausible but wrong — close enough to require
   real understanding, not random.

Return ONLY valid JSON conforming to the schema. No prose around it.
"""


HUMAN_TEMPLATE = (
    "{adaptive_block}"
    "{topic_block}"
    "CONTEXT (verbatim excerpts from the PDF):\n{content}\n\n"
    "TASK: Generate exactly {n_questions} MCQs at {difficulty} difficulty.\n\n"
    "{format_instructions}\n"
)


@dataclass
class BuiltPrompt:
    """Output of build_mcq_prompt — keep token counts/content for accounting."""
    messages: list[BaseMessage]
    system: str
    user: str
    content: str
    instructions: str
    adaptive_context: str


def render_content_block(chunks: Sequence[ChunkModel]) -> str:
    return "\n\n---\n\n".join(c.content.strip() for c in chunks)


def build_mcq_prompt(
    chunks: Sequence[ChunkModel],
    n_questions: int,
    difficulty: str = "medium",
    adaptive_context: str = "",
    topic_hint: str = "",
    parser: Optional[JsonOutputParser] = None,
) -> BuiltPrompt:
    """Assemble the chat prompt via LangChain's ChatPromptTemplate.

    The format instructions are pulled from the parser when supplied so the
    LLM's output schema is in sync with our Pydantic model.
    """
    content_block = render_content_block(chunks)

    adaptive_block = ""
    if adaptive_context.strip():
        adaptive_block = (
            "USER HISTORY SIGNAL (steer questions toward weak topics, "
            "vary or avoid mastered ones):\n"
            + adaptive_context.strip() + "\n\n"
        )

    topic_block = ""
    if topic_hint.strip():
        topic_block = (
            f"TOPIC FOCUS for this batch: {topic_hint.strip()}\n"
            "Each question must primarily test understanding of the topic above.\n\n"
        )

    format_instructions = (
        parser.get_format_instructions()
        if parser is not None
        else 'Return a JSON object: {"questions": [{...}]}'
    )

    template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])

    messages = template.format_messages(
        adaptive_block=adaptive_block,
        topic_block=topic_block,
        content=content_block,
        n_questions=n_questions,
        difficulty=difficulty,
        format_instructions=format_instructions,
    )

    user_text = messages[1].content if len(messages) > 1 else ""

    return BuiltPrompt(
        messages=messages,
        system=SYSTEM_PROMPT,
        user=user_text,
        content=content_block,
        instructions=format_instructions,
        adaptive_context=adaptive_context,
    )


def to_langchain_messages(prompt: BuiltPrompt) -> list[BaseMessage]:
    """Already a list of LangChain messages — passthrough for compatibility."""
    return prompt.messages
