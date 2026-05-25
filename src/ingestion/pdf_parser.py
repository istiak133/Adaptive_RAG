"""Parse the SLATEFALL_DOSSIER PDF into sub-section units.

We use PyMuPDF as the primary parser because the SLATEFALL tables have no
visible borders, so neither pdfplumber's default line-based detection nor
PyMuPDF's table detector recognise them. PyMuPDF's plain text extraction,
however, preserves the column-aligned layout (left column on one side of the
line, right column on the other), which is readable enough for both
embeddings and LLM consumption.

Page-level noise — page numbers like "4 / 50" and the document footer
"SLATEFALL_DOSSIER.md  2026-05-18" — is filtered out so it doesn't pollute
chunks.

Sub-section IDs of two and three levels are both detected (e.g., §2.13 and
§9.9.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


SECTION_RE = re.compile(r"^Section\s+(\d+)\.\s+(.+?)$", re.MULTILINE)

# Sub-section IDs may have two or three levels: 2.13, 9.9.5
SUBSECTION_RE = re.compile(
    r"^(\d+\.\d+(?:\.\d+)?)\s+([A-Z].+?)$",
    re.MULTILINE,
)

# Page-number lines like "4 / 50"
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.MULTILINE)

# Document footer
FOOTER_RE = re.compile(
    r"^\s*SLATEFALL_DOSSIER\.md\s*\d{4}-\d{2}-\d{2}\s*$",
    re.MULTILINE,
)


@dataclass
class ParsedSubSection:
    section_id: int
    section_title: str
    sub_section_id: str
    sub_section_title: str
    content: str
    page_start: int
    page_end: int
    has_table: bool = False
    has_bullets: bool = False


def _clean_page_text(text: str) -> str:
    """Strip footer + page number lines that PyMuPDF emits per page."""
    text = PAGE_NUM_RE.sub("", text)
    text = FOOTER_RE.sub("", text)
    return text


def _looks_like_table(content: str) -> bool:
    """Heuristic: 3+ consecutive lines where each line ends with a number,
    percentage, or unit — characteristic of 2-column data rows."""
    lines = content.split("\n")
    streak = 0
    for line in lines:
        line = line.strip()
        if not line:
            streak = 0
            continue
        if re.search(r"(\d+(?:\.\d+)?\s*(?:%|kg|s|m|km|°C|kJ|MJ|dB)?\s*\)?)\s*$", line):
            streak += 1
            if streak >= 3:
                return True
        else:
            streak = 0
    return False


def _has_bullets(content: str) -> bool:
    """Detect lines that start with explicit bullet markers, OR list patterns
    typical of SLATEFALL (a colon-ending line followed by ≥ 2 indented items).
    """
    if re.search(r"^\s*[•\-\*]\s+", content, re.MULTILINE):
        return True
    # Pattern: line ending with ":\n" followed by lines starting with capital
    matches = re.findall(
        r":\s*\n((?:\s*[A-Z].+\n){2,})",
        content,
    )
    return bool(matches)


def parse_pdf(pdf_path: Path | str) -> List[ParsedSubSection]:
    """Parse SLATEFALL_DOSSIER.pdf into a list of sub-section units."""
    pdf_path = Path(pdf_path)

    pages_text: List[str] = []
    doc = fitz.open(pdf_path)
    for page in doc:
        raw = page.get_text() or ""
        cleaned = _clean_page_text(raw)
        pages_text.append(cleaned)
    doc.close()

    page_char_offsets: List[int] = []
    cumulative = 0
    separator = "\n"
    for page_text in pages_text:
        page_char_offsets.append(cumulative)
        cumulative += len(page_text) + len(separator)
    full_text = separator.join(pages_text)

    def char_to_page(char_pos: int) -> int:
        page = 0
        for i, off in enumerate(page_char_offsets):
            if off <= char_pos:
                page = i
            else:
                break
        return page

    section_matches = [
        (m.start(), int(m.group(1)), m.group(2).strip())
        for m in SECTION_RE.finditer(full_text)
    ]

    all_sub_matches = [
        (m.start(), m.group(1), m.group(2).strip())
        for m in SUBSECTION_RE.finditer(full_text)
    ]

    if section_matches:
        first_section_pos = section_matches[0][0]
        sub_matches = [s for s in all_sub_matches if s[0] > first_section_pos]
    else:
        sub_matches = all_sub_matches

    sub_sections: List[ParsedSubSection] = []

    for i, (start_pos, sub_id, sub_title) in enumerate(sub_matches):
        section_id = 0
        section_title = ""
        for sec_pos, sec_id, sec_title in section_matches:
            if sec_pos <= start_pos:
                section_id = sec_id
                section_title = sec_title
            else:
                break
        if section_id == 0:
            continue
        # Sub-section id must match parent section (e.g., "2.13" inside Section 2)
        if not sub_id.startswith(f"{section_id}."):
            continue

        if i + 1 < len(sub_matches):
            end_pos = sub_matches[i + 1][0]
        else:
            end_pos = len(full_text)

        header_end_idx = full_text.find("\n", start_pos)
        if header_end_idx == -1 or header_end_idx > end_pos:
            content_start = start_pos
        else:
            content_start = header_end_idx + 1

        content = full_text[content_start:end_pos].strip()

        page_start = char_to_page(start_pos) + 1
        page_end = char_to_page(max(end_pos - 1, start_pos)) + 1

        sub_sections.append(ParsedSubSection(
            section_id=section_id,
            section_title=section_title,
            sub_section_id=sub_id,
            sub_section_title=sub_title,
            content=content,
            page_start=page_start,
            page_end=page_end,
            has_table=_looks_like_table(content),
            has_bullets=_has_bullets(content),
        ))

    return sub_sections


if __name__ == "__main__":
    import sys
    from collections import Counter

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import settings  # noqa: E402

    pdf_path = settings.paths.pdf_corpus
    print(f"Parsing: {pdf_path}\n")

    sub_sections = parse_pdf(pdf_path)

    print(f"Total sub-sections detected: {len(sub_sections)}\n")

    by_section = Counter(ss.section_id for ss in sub_sections)
    print("Sub-sections per section:")
    for sec_id in sorted(by_section.keys()):
        print(f"  Section {sec_id}: {by_section[sec_id]} sub-sections")
    print()

    with_tables = sum(1 for ss in sub_sections if ss.has_table)
    with_bullets = sum(1 for ss in sub_sections if ss.has_bullets)
    print(f"Sub-sections flagged as having tables:  {with_tables}")
    print(f"Sub-sections flagged as having bullets: {with_bullets}")
    print()

    # Show §9 sub-section IDs (verify 9.9.5/6/7 captured)
    s9_ids = [ss.sub_section_id for ss in sub_sections if ss.section_id == 9]
    print(f"Section 9 sub-section IDs: {s9_ids}")
    print()

    print("=== Sample 1: §2.13 Echo Lock ===")
    for ss in sub_sections:
        if ss.sub_section_id == "2.13":
            print(f"Title:        {ss.sub_section_title}")
            print(f"Pages:        {ss.page_start}-{ss.page_end}")
            print(f"Has table:    {ss.has_table}")
            print(f"Has bullets:  {ss.has_bullets}")
            print(f"Length:       {len(ss.content)} chars")
            print()
            print(ss.content)
            break
    print()

    print("=== Sample 2: §2.2 Targeting (table) ===")
    for ss in sub_sections:
        if ss.sub_section_id == "2.2":
            print(f"Has table:    {ss.has_table}")
            print()
            print(ss.content[:1500])
            break
