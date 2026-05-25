"""Chunk parsed sub-sections into LLM-ready units.

Implements the 4-layer hybrid strategy:
  Layer 1 — primary: each sub-section is a candidate chunk
  Layer 2 — size normalisation: merge tiny, split oversized
  Layer 3 — special handlers: glossary §10.1 → per-entry chunks
  Layer 4 — cross-reference extraction: §X.Y mentions captured as metadata
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion.pdf_parser import ParsedSubSection


_ENCODER = tiktoken.get_encoding("cl100k_base")

CROSS_REF_RE = re.compile(r"§(\d+\.\d+(?:\.\d+)?)")


def _prepend_title(sub_id: str, title: str, content: str) -> str:
    """Prepend sub-section title as first line so vector search hits the
    title text (otherwise the title — held in metadata only — wouldn't be
    embedded). Format: '§2.13 Echo Lock Failure Mode\n\n<content>'."""
    return f"§{sub_id} {title}\n\n{content}"


@dataclass
class Chunk:
    section_id: int
    section_title: str
    sub_section_id: str
    sub_section_title: str
    content: str
    token_count: int
    page_start: int
    page_end: int
    has_table: bool = False
    has_bullets: bool = False
    cross_refs: List[str] = field(default_factory=list)
    chunk_kind: str = "narrative"  # narrative | glossary_entry | split


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _extract_cross_refs(text: str, exclude: Optional[str] = None) -> List[str]:
    """Find §X.Y patterns; optionally drop a self-reference (the chunk's own id)."""
    refs = set(CROSS_REF_RE.findall(text))
    if exclude:
        # Handle merged-chunk ids like "5.10+5.11" — drop all parts
        for part in exclude.split("+"):
            base = part.split("-")[0].split("#")[0]  # drop "-A" or "#1" suffixes
            refs.discard(base)
    return sorted(refs)


def _split_oversized(sub: ParsedSubSection, max_tokens: int) -> List[Chunk]:
    """Apply RecursiveCharacterTextSplitter to an oversized sub-section.

    Sub-divisions get suffixed IDs (e.g., '3.4-A', '3.4-B') so they remain
    traceable to the parent sub-section while being distinct chunks.
    """
    separators = settings.chunking.recursive_splitter_separators
    overlap = settings.chunking.overlap_tokens

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=count_tokens,
        separators=separators,
    )
    pieces = splitter.split_text(sub.content)

    chunks: List[Chunk] = []
    for idx, piece in enumerate(pieces):
        suffix = chr(ord("A") + idx) if idx < 26 else f"{idx + 1}"
        sub_id = f"{sub.sub_section_id}-{suffix}"
        title_with_part = f"{sub.sub_section_title} (part {suffix})"
        content = _prepend_title(sub_id, title_with_part, piece.strip())
        chunks.append(Chunk(
            section_id=sub.section_id,
            section_title=sub.section_title,
            sub_section_id=sub_id,
            sub_section_title=sub.sub_section_title,
            content=content,
            token_count=count_tokens(content),
            page_start=sub.page_start,
            page_end=sub.page_end,
            has_table=sub.has_table,
            has_bullets=sub.has_bullets,
            cross_refs=_extract_cross_refs(content, exclude=sub.sub_section_id),
            chunk_kind="split",
        ))
    return chunks


def _split_glossary(sub: ParsedSubSection) -> List[Chunk]:
    """Glossary entries have format 'Term — definition'. Split per-entry.

    Each entry becomes its own chunk so vector search returns precise
    definitions, not the entire 70-term glossary as a single noisy chunk.
    """
    # SLATEFALL glossary entries use em-dash or hyphen-dash separator
    entry_pattern = re.compile(
        r"(?:^|\n)([A-Z][^\n]*?(?:\s—\s|\s-\s)[^\n]+(?:\n(?!\s*[A-Z][^\n]*?\s—\s)[^\n]*)*)",
    )
    entries = entry_pattern.findall(sub.content)

    if not entries:
        # Fall back to a single chunk if no entries detected
        return [_default_chunk(sub)]

    chunks: List[Chunk] = []
    for idx, entry in enumerate(entries):
        entry = entry.strip()
        if len(entry) < 10:
            continue
        # Glossary entries already begin with "Term — definition", so we
        # don't prepend a generic title that would be redundant.
        chunks.append(Chunk(
            section_id=sub.section_id,
            section_title=sub.section_title,
            sub_section_id=f"{sub.sub_section_id}#{idx + 1}",
            sub_section_title=sub.sub_section_title,
            content=entry,
            token_count=count_tokens(entry),
            page_start=sub.page_start,
            page_end=sub.page_end,
            has_table=False,
            has_bullets=False,
            cross_refs=_extract_cross_refs(entry, exclude=sub.sub_section_id),
            chunk_kind="glossary_entry",
        ))
    return chunks


def _default_chunk(sub: ParsedSubSection) -> Chunk:
    content = _prepend_title(sub.sub_section_id, sub.sub_section_title, sub.content)
    return Chunk(
        section_id=sub.section_id,
        section_title=sub.section_title,
        sub_section_id=sub.sub_section_id,
        sub_section_title=sub.sub_section_title,
        content=content,
        token_count=count_tokens(content),
        page_start=sub.page_start,
        page_end=sub.page_end,
        has_table=sub.has_table,
        has_bullets=sub.has_bullets,
        cross_refs=_extract_cross_refs(content, exclude=sub.sub_section_id),
        chunk_kind="narrative",
    )


def _merge_undersized(chunks: List[Chunk], min_tokens: int) -> List[Chunk]:
    """Merge any chunk under min_tokens with its successor in the same section.

    We don't merge glossary or split chunks — those have intentional small
    sizes (per-entry definitions / fixed-width splits).
    """
    merged: List[Chunk] = []
    pending: Optional[Chunk] = None

    for chunk in chunks:
        if chunk.chunk_kind != "narrative":
            if pending is not None:
                merged.append(pending)
                pending = None
            merged.append(chunk)
            continue

        if pending is None:
            pending = chunk
            continue

        # Decide: merge pending with current?
        same_section = pending.section_id == chunk.section_id
        if pending.token_count < min_tokens and same_section:
            combined = f"{pending.content}\n\n{chunk.content}"
            pending = Chunk(
                section_id=pending.section_id,
                section_title=pending.section_title,
                sub_section_id=f"{pending.sub_section_id}+{chunk.sub_section_id}",
                sub_section_title=f"{pending.sub_section_title} / {chunk.sub_section_title}",
                content=combined,
                token_count=count_tokens(combined),
                page_start=pending.page_start,
                page_end=chunk.page_end,
                has_table=pending.has_table or chunk.has_table,
                has_bullets=pending.has_bullets or chunk.has_bullets,
                cross_refs=sorted(set(pending.cross_refs + chunk.cross_refs)),
                chunk_kind="narrative",
            )
        else:
            merged.append(pending)
            pending = chunk

    if pending is not None:
        merged.append(pending)

    return merged


def chunk_sub_sections(sub_sections: List[ParsedSubSection]) -> List[Chunk]:
    """Apply the 4-layer hybrid strategy and return final chunks."""
    min_tokens = settings.chunking.min_chunk_tokens
    max_tokens = settings.chunking.max_chunk_tokens

    chunks: List[Chunk] = []

    for sub in sub_sections:
        token_count = count_tokens(sub.content)

        # Special case: glossary entries
        if sub.sub_section_id == "10.1":
            chunks.extend(_split_glossary(sub))
            continue

        if token_count > max_tokens:
            chunks.extend(_split_oversized(sub, max_tokens))
        else:
            chunks.append(_default_chunk(sub))

    chunks = _merge_undersized(chunks, min_tokens)
    return chunks


if __name__ == "__main__":
    import sys
    from collections import Counter
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.ingestion.pdf_parser import parse_pdf  # noqa: E402

    pdf_path = settings.paths.pdf_corpus
    sub_sections = parse_pdf(pdf_path)
    chunks = chunk_sub_sections(sub_sections)

    print(f"Input  sub-sections: {len(sub_sections)}")
    print(f"Output chunks:       {len(chunks)}\n")

    kinds = Counter(c.chunk_kind for c in chunks)
    print(f"Chunk kinds: {dict(kinds)}\n")

    token_counts = [c.token_count for c in chunks]
    print("Token-count statistics:")
    print(f"  Min:    {min(token_counts)}")
    print(f"  Max:    {max(token_counts)}")
    print(f"  Mean:   {sum(token_counts) / len(token_counts):.1f}")
    print(f"  Median: {sorted(token_counts)[len(token_counts) // 2]}")
    print()

    # Distribution buckets
    buckets = {
        "1-99":    sum(1 for t in token_counts if t < 100),
        "100-199": sum(1 for t in token_counts if 100 <= t < 200),
        "200-299": sum(1 for t in token_counts if 200 <= t < 300),
        "300-399": sum(1 for t in token_counts if 300 <= t < 400),
        "400-500": sum(1 for t in token_counts if 400 <= t <= 500),
        "501+":    sum(1 for t in token_counts if t > 500),
    }
    print("Distribution:")
    for k, v in buckets.items():
        bar = "█" * v
        print(f"  {k:>8}: {v:3d} {bar}")
    print()

    print("=== Chunk: §2.13 Echo Lock ===")
    for c in chunks:
        if c.sub_section_id == "2.13":
            print(f"Kind:         {c.chunk_kind}")
            print(f"Tokens:       {c.token_count}")
            print(f"Cross-refs:   {c.cross_refs}")
            print(f"Has bullets:  {c.has_bullets}")
            print()
            print(c.content)
            break
    print()

    print("=== Chunk: §3.4 (Cerro Castillo — should be split if oversized) ===")
    for c in chunks:
        if c.sub_section_id.startswith("3.4"):
            print(f"Id: {c.sub_section_id} | Kind: {c.chunk_kind} | Tokens: {c.token_count}")
    print()

    print("=== Glossary entries (§10.1) ===")
    glossary_chunks = [c for c in chunks if c.chunk_kind == "glossary_entry"]
    print(f"Total glossary entries: {len(glossary_chunks)}\n")
    for c in glossary_chunks[:5]:
        print(f"  [{c.sub_section_id}] {c.content[:120]}")
