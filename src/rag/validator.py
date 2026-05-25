"""LLM response validation and hallucination check.

Pipeline:
  1. Strip markdown fences (LLMs sometimes wrap JSON in ```json … ```)
  2. Parse JSON
  3. Validate structure via Pydantic (4 choices, valid answer, etc.)
  4. Hallucination check: source_quote must appear (fuzzy) in the context
  5. Filter and report results
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.config import settings


# ── Pydantic models ──────────────────────────────────────────────────


Choice = Literal["A", "B", "C", "D"]


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
        normalised = [c.strip().lower() for c in choices]
        return len(set(normalised)) == 4


class MCQBatch(BaseModel):
    questions: List[MCQ]


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


# ── JSON extraction ──────────────────────────────────────────────────


_MARKDOWN_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?|\n?```\s*$",
                                 flags=re.IGNORECASE)


def _strip_markdown(text: str) -> str:
    return _MARKDOWN_FENCE_RE.sub("", text).strip()


def _extract_json(text: str) -> str:
    """Find the outermost JSON object in a string (best-effort)."""
    cleaned = _strip_markdown(text)
    # Find the first '{' and the matching last '}'
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return cleaned
    return cleaned[first : last + 1]


def parse_llm_json(raw_text: str) -> dict:
    """Robustly parse LLM JSON output. Raises ValueError on failure."""
    json_text = _extract_json(raw_text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON: {e}. Raw: {raw_text[:200]}")


# ── Hallucination check ─────────────────────────────────────────────


def quote_appears_in_context(
    quote: str,
    context: str,
    threshold: Optional[float] = None,
) -> bool:
    """Fuzzy substring match — handles minor LLM paraphrasing."""
    threshold = threshold or settings.mcq.hallucination_fuzzy_threshold
    quote_norm = re.sub(r"\s+", " ", quote.strip().lower())
    context_norm = re.sub(r"\s+", " ", context.lower())
    if quote_norm in context_norm:
        return True
    # Fuzzy: longest matching subsequence ratio
    matcher = SequenceMatcher(None, quote_norm, context_norm,
                              autojunk=False)
    longest = matcher.find_longest_match(0, len(quote_norm),
                                          0, len(context_norm))
    coverage = longest.size / max(len(quote_norm), 1)
    return coverage >= threshold


# ── Full validation pipeline ────────────────────────────────────────


def validate_response(
    raw_text: str,
    context: str,
    expected_count: int,
    require_source_quote: Optional[bool] = None,
) -> ValidationReport:
    require_source_quote = (
        require_source_quote
        if require_source_quote is not None
        else settings.mcq.require_source_quote
    )

    report = ValidationReport()

    # Step 1: Parse JSON
    try:
        parsed = parse_llm_json(raw_text)
    except ValueError as e:
        report.add_rejection({}, f"json_parse_error: {e}")
        return report

    raw_questions = parsed.get("questions", [])
    report.raw_parsed_count = len(raw_questions)

    # Step 2-4: Validate each MCQ
    for idx, item in enumerate(raw_questions):
        try:
            mcq = MCQ(**item)
        except Exception as e:
            report.add_rejection(item, f"schema_error: {e}")
            continue

        if not mcq.choices_unique():
            report.add_rejection(item, "duplicate_choices")
            continue

        if require_source_quote:
            if not quote_appears_in_context(mcq.source_quote, context):
                report.add_rejection(item,
                    f"hallucination: source_quote not found "
                    f"(quote: {mcq.source_quote[:80]!r})")
                continue

        report.accepted.append(mcq)

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
      "explanation": "Per §2.13, Echo Lock is triggered by acoustic shock ≥ 145 dB within the targeting window.",
      "source_quote": "Acoustic shock during the targeting window (≥ 145 dB)",
      "topic": "Echo Lock"
    }
  ]
}
```"""
    context = (
        "Echo Lock is a neurological seizure-equivalent observed twice. "
        "Triggering conditions include: Attempted suspension above the 240 kg ceiling. "
        "Acoustic shock during the targeting window (≥ 145 dB). "
        "Severe hypoglycemia (blood glucose < 58 mg/dL)."
    )
    report = validate_response(sample, context, expected_count=1)
    print(f"  Accepted: {len(report.accepted)}, Rejected: {len(report.rejected)}")
    if report.accepted:
        m = report.accepted[0]
        print(f"  MCQ: {m.question_text}")
        print(f"  Correct: {m.correct_answer} | Topic: {m.topic}")
    print()

    print("=== Test 2: Hallucinated quote ===")
    bad_sample = sample.replace(
        'Acoustic shock during the targeting window (≥ 145 dB)',
        'The Calexin-7 antidote prevents Echo Lock entirely'
    )
    report = validate_response(bad_sample, context, expected_count=1)
    print(f"  Accepted: {len(report.accepted)}, Rejected: {len(report.rejected)}")
    for r in report.rejected:
        print(f"  Reason: {r['reason'][:90]}")
    print()

    print("=== Test 3: Duplicate choices ===")
    dup_sample = """{
  "questions": [
    {
      "question_text": "What is the Echo Lock duration?",
      "choice_a": "4 seconds", "choice_b": "4 seconds",
      "choice_c": "4 seconds", "choice_d": "4 seconds",
      "correct_answer": "A",
      "explanation": "Echo Lock lasts four seconds per §2.13.",
      "source_quote": "a 4-second targeting freeze",
      "topic": "Echo Lock"
    }
  ]
}"""
    report = validate_response(dup_sample, "a 4-second targeting freeze", expected_count=1)
    print(f"  Accepted: {len(report.accepted)}, Rejected: {len(report.rejected)}")
    for r in report.rejected:
        print(f"  Reason: {r['reason']}")
    print()

    print("=== Test 4: Invalid JSON ===")
    broken = "Here are your MCQs: { invalid json }"
    report = validate_response(broken, "any context", expected_count=1)
    print(f"  Accepted: {len(report.accepted)}, Rejected: {len(report.rejected)}")
    for r in report.rejected:
        print(f"  Reason: {r['reason'][:80]}")
