"""LLM response parsing + validation using LangChain primitives.

Pipeline:
  1. LangChain JsonOutputParser handles markdown stripping + JSON extraction
     + initial Pydantic-schema validation.
  2. We add a hallucination check (source_quote must appear fuzzily in the
     retrieved context) — this is our defensive layer beyond LangChain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Literal, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, field_validator

from src.config import settings


Choice = Literal["A", "B", "C", "D"]


# ── Pydantic schema (used by LangChain parser too) ────────────────────


class MCQ(BaseModel):
    question_text: str = Field(min_length=10)
    choice_a: str = Field(min_length=1)
    choice_b: str = Field(min_length=1)
    choice_c: str = Field(min_length=1)
    choice_d: str = Field(min_length=1)
    correct_answer: Choice
    explanation: str = Field(min_length=10)
    source_quote: str = Field(min_length=5)
    topic: str = Field(default="")

    @field_validator("question_text", "explanation", "source_quote")
    @classmethod
    def _stripped(cls, v: str) -> str:
        return v.strip()

    def choices_unique(self) -> bool:
        choices = [self.choice_a, self.choice_b, self.choice_c, self.choice_d]
        return len({c.strip().lower() for c in choices}) == 4


class MCQBatch(BaseModel):
    questions: List[MCQ]


# Singleton — used by prompt builder + parser
_parser: Optional[JsonOutputParser] = None


def get_parser() -> JsonOutputParser:
    global _parser
    if _parser is None:
        _parser = JsonOutputParser(pydantic_object=MCQBatch)
    return _parser


# ── Result types ─────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    accepted: List[MCQ] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)
    raw_parsed_count: int = 0

    def add_rejection(self, item: dict, reason: str) -> None:
        self.rejected.append({"item": item, "reason": reason})

    @property
    def is_complete(self) -> bool:
        return len(self.rejected) == 0


# ── Hallucination check ──────────────────────────────────────────────


def quote_appears_in_context(
    quote: str,
    context: str,
    threshold: Optional[float] = None,
) -> bool:
    """Fuzzy substring match for the LLM's source_quote."""
    threshold = threshold or settings.mcq.hallucination_fuzzy_threshold
    quote_norm = re.sub(r"\s+", " ", quote.strip().lower())
    context_norm = re.sub(r"\s+", " ", context.lower())
    if quote_norm in context_norm:
        return True
    matcher = SequenceMatcher(None, quote_norm, context_norm, autojunk=False)
    longest = matcher.find_longest_match(0, len(quote_norm),
                                          0, len(context_norm))
    coverage = longest.size / max(len(quote_norm), 1)
    return coverage >= threshold


def _normalise_for_dedup(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation/§ refs.

    Two questions that differ only in §-citation or trailing period should
    compare identical for duplicate-detection purposes.
    """
    t = re.sub(r"\s+", " ", text.lower().strip())
    t = re.sub(r"[§]\s*\d+(\.\d+)*", "", t)  # drop §X.Y references
    t = re.sub(r"[^\w\s]", "", t)            # drop punctuation
    return t.strip()


def is_near_duplicate(
    candidate: str,
    previously_asked: List[str],
    threshold: Optional[float] = None,
) -> Optional[float]:
    """Return the similarity ratio if `candidate` is close enough to any
    string in `previously_asked` to count as a duplicate, else None.

    Uses SequenceMatcher's `ratio()` — symmetric, 0–1.
    """
    if not previously_asked:
        return None
    threshold = (
        threshold
        if threshold is not None
        else settings.mcq.near_duplicate_threshold
    )
    cand = _normalise_for_dedup(candidate)
    if not cand:
        return None
    for prev in previously_asked:
        prev_norm = _normalise_for_dedup(prev)
        if not prev_norm:
            continue
        ratio = SequenceMatcher(None, cand, prev_norm, autojunk=False).ratio()
        if ratio >= threshold:
            return ratio
    return None


# ── Public validation entry point ────────────────────────────────────


def validate_response(
    raw_text: str,
    context: str,
    expected_count: int,
    require_source_quote: Optional[bool] = None,
    recent_question_texts: Optional[List[str]] = None,
) -> ValidationReport:
    """Parse and validate the LLM's raw response.

    LangChain's JsonOutputParser does the heavy lifting (markdown fences,
    JSON extraction, Pydantic schema validation). On top of that we apply:
      • a hallucination guard (source_quote must appear in retrieved context)
      • a near-duplicate guard (question_text vs recent questions in the KB)

    The duplicate guard also tracks question_texts accepted *in this batch*
    so the LLM can't sneak two near-identical MCQs past in a single call.
    """
    require_source_quote = (
        require_source_quote
        if require_source_quote is not None
        else settings.mcq.require_source_quote
    )
    dedup_enabled = settings.mcq.reject_near_duplicates
    # Local copy so we can append in-batch accepted texts as we go.
    seen_texts: List[str] = list(recent_question_texts or [])

    report = ValidationReport()
    parser = get_parser()

    try:
        parsed_dict = parser.invoke(raw_text)
    except (OutputParserException, ValueError) as e:
        report.add_rejection({}, f"langchain_parse_error: {e}")
        return report
    except Exception as e:
        report.add_rejection({}, f"unexpected_parser_error: {e}")
        return report

    raw_questions = parsed_dict.get("questions", []) if isinstance(parsed_dict, dict) else []
    report.raw_parsed_count = len(raw_questions)

    for item in raw_questions:
        # Pydantic re-validation (parser may give dicts in some versions)
        try:
            mcq = item if isinstance(item, MCQ) else MCQ(**item)
        except Exception as e:
            report.add_rejection(item, f"schema_error: {e}")
            continue

        if not mcq.choices_unique():
            report.add_rejection(mcq.model_dump(), "duplicate_choices")
            continue

        if require_source_quote and not quote_appears_in_context(
            mcq.source_quote, context,
        ):
            report.add_rejection(
                mcq.model_dump(),
                f"hallucination: source_quote not found "
                f"(quote: {mcq.source_quote[:80]!r})",
            )
            continue

        if dedup_enabled:
            dup_ratio = is_near_duplicate(mcq.question_text, seen_texts)
            if dup_ratio is not None:
                report.add_rejection(
                    mcq.model_dump(),
                    f"near_duplicate: ratio={dup_ratio:.2f} "
                    f"(question: {mcq.question_text[:80]!r})",
                )
                continue

        report.accepted.append(mcq)
        seen_texts.append(mcq.question_text)

    return report


if __name__ == "__main__":
    print("=== Test 1: Valid response ===")
    sample = """```json
{
  "questions": [
    {
      "question_text": "What triggers Echo Lock failure mode?",
      "choice_a": "Acoustic shock ≥ 145 dB during targeting",
      "choice_b": "Visible-spectrum illumination",
      "choice_c": "Aymara recitation",
      "choice_d": "Mountain altitude over 5,000 m",
      "correct_answer": "A",
      "explanation": "Per §2.13, Echo Lock is triggered by acoustic shock ≥ 145 dB.",
      "source_quote": "Acoustic shock during the targeting window (≥ 145 dB)",
      "topic": "Echo Lock"
    }
  ]
}
```"""
    context = (
        "Echo Lock is a neurological seizure-equivalent. "
        "Triggering conditions include: "
        "Acoustic shock during the targeting window (≥ 145 dB)."
    )
    report = validate_response(sample, context, expected_count=1)
    print(f"  Accepted: {len(report.accepted)}, Rejected: {len(report.rejected)}")

    print("\n=== Test 2: Hallucinated quote ===")
    bad = sample.replace(
        'Acoustic shock during the targeting window (≥ 145 dB)',
        'Calexin-7 antidote prevents Echo Lock entirely',
    )
    r2 = validate_response(bad, context, expected_count=1)
    print(f"  Rejected: {len(r2.rejected)}")
    for r in r2.rejected:
        print(f"    {r['reason'][:90]}")

    print("\n=== Test 3: Bad JSON ===")
    r3 = validate_response("not json at all", context, expected_count=1)
    print(f"  Rejected: {len(r3.rejected)}")
    for r in r3.rejected:
        print(f"    {r['reason'][:90]}")
