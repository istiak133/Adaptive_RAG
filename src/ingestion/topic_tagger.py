"""Topic extraction and tagging.

Strategy: glossary-driven. The SLATEFALL PDF §10.1 glossary already contains
69 author-curated term-definition pairs covering every named concept. We use
those as our topic dictionary, then match each topic against chunk content
with case-insensitive substring search.

This avoids LLM-based topic extraction (slower, inconsistent naming, costlier)
while achieving high recall — every formally named concept in the document
is captured.

For each (section, topic) pair, depth is inferred from how many chunks in
that section mention the topic:
  primary    — many chunks (≥3 or ≥20% of section's chunks)
  secondary  — 2 chunks
  mention    — exactly 1 chunk
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.ingestion.chunker import Chunk


# ── Critical concepts (importance = 'critical' override) ─────────────
# These are evaluation-defining concepts — adaptive logic must prioritise them.
CRITICAL_TOPIC_NAMES: Set[str] = {
    "Echo Lock",
    "Tail Momentum",
    "Inertial Suspension",
    "Drift Read",
    "Hold",
    "Three-Two-One Rule",
    "Doctrine of Sequential Suspension",
    "Tail-Strike",
    "Drop-Tune",
    "Suspended Field",
    "Mar Inverso",
    "Cell Halcón",
    "Calvache–Marrón Equations",
    "Calexin-7",
}

# ── Category keyword heuristic ───────────────────────────────────────
CATEGORY_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("power_mechanic", [
        "power", "mechanic", "ability", "suspension", "failure mode",
        "precognitive", "metahuman classification",
    ]),
    ("combat_tactic", [
        "doctrine", "tactic", "rule", "technique", "manoeuvre", "directive",
        "window", "protocol", "carryover doctrine",
    ]),
    ("equipment", [
        "baton", "uniform", "hardshell", "bracer", "kit", "vehicle",
        "radio", "cloak", "sidearm", "pistol", "boot", "harness", "sling",
    ]),
    ("adversary", [
        "adversary", "manipulator", "syndicate", "pressure", "bone-density",
        "augment", "at-large", "bureaucratic-adversary",
    ]),
    ("personnel", [
        "specialist", "tracker", "medic", "officer", "psychologist",
        "spouse", "director", "engineer", "cell technical lead",
        "field medic", "intel asset", "safehouse keeper", "liaison",
    ]),
    ("organization", [
        "cooperative", "pamc", "bureau", "directorate", "academy",
        "metahuman compact", "metahuman reserve", "cell ", "compact",
    ]),
    ("location", [
        "base", "safehouse", "facility", "hq", "depot", "restraint facility",
    ]),
    ("event", [
        "incident", "engagement", "trigger event", "deployment",
        "operación", "case",
    ]),
    ("substance", [
        "electrolyte", "prophylactic", "stimulant", "dose", "blister",
    ]),
    ("document", [
        "agreement", "equations", "protocol",
    ]),
]


@dataclass
class Topic:
    name: str
    slug: str
    description: Optional[str]
    category: str
    importance: str


@dataclass
class ChunkTopicTag:
    """Output: which topics appear in which chunk."""
    chunk_index: int
    section_id: int
    sub_section_id: str
    topic_slugs: List[str] = field(default_factory=list)


@dataclass
class SectionTopicLink:
    section_id: int
    topic_slug: str
    depth: str            # primary | secondary | mention
    mention_count: int
    relevance_score: float


_PAREN_ABBREV_RE = re.compile(r"\s*\([^)]+\)\s*$")


def _strip_paren_abbrev(name: str) -> str:
    """'Doctrine of Sequential Suspension (DSS)' → 'Doctrine of Sequential Suspension'."""
    return _PAREN_ABBREV_RE.sub("", name).strip()


def slugify(name: str) -> str:
    """Convert 'Echo Lock' → 'echo_lock', preserving accented chars."""
    s = name.lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s-]+", "_", s)
    return s.strip("_")


def _infer_category(description: str) -> str:
    desc_lower = description.lower()
    for cat, keywords in CATEGORY_KEYWORDS:
        if any(kw in desc_lower for kw in keywords):
            return cat
    return "general"


def _infer_importance(name: str, description: str) -> str:
    if name in CRITICAL_TOPIC_NAMES:
        return "critical"
    if len(description) < 60:
        return "minor"
    return "normal"


def extract_topics_from_glossary(chunks: List[Chunk]) -> List[Topic]:
    """Parse glossary-entry chunks → Topic objects.

    Glossary format is 'Term — definition (§X.Y)'. We split on the em-dash
    and treat everything before as the term name.
    """
    topics: List[Topic] = []
    seen_slugs: Set[str] = set()

    for chunk in chunks:
        if chunk.chunk_kind != "glossary_entry":
            continue

        # Split on em-dash (— with surrounding spaces) — only the FIRST occurrence
        parts = re.split(r"\s+[—–-]\s+", chunk.content, maxsplit=1)
        if len(parts) != 2:
            continue

        name_raw, description = parts
        # Remove all quotes (handles 'Anchor Stone "Cerro-1"' edge cases)
        name = name_raw.replace('"', "").strip()
        # Strip trailing "(DSS)", "(IS)" etc. — keep the canonical name only
        name = _strip_paren_abbrev(name)

        # Skip if name is empty, contains line breaks (parsing failure),
        # or is unreasonably long
        if not name or "\n" in name or len(name) > 80:
            continue

        slug = slugify(name)
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        topics.append(Topic(
            name=name,
            slug=slug,
            description=description.strip()[:500],
            category=_infer_category(description),
            importance=_infer_importance(name, description),
        ))

    return topics


def _topic_to_regex(name: str) -> re.Pattern:
    """Build a word-bounded, case-insensitive regex for a topic name.

    Avoids false positives like 'CEM' inside 'ceiling' or 'PAMC' inside other
    tokens. Treats em-dash and hyphen interchangeably so 'Calvache–Marrón' in
    content matches 'Calvache-Marrón' in topic, etc.
    """
    name_norm = name.replace("–", "-").replace("—", "-")
    parts = re.split(r"\s+", name_norm.strip())
    quoted = [re.escape(p) for p in parts]
    pattern = r"\b" + r"\s+".join(quoted) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def tag_chunks(
    chunks: List[Chunk],
    topics: List[Topic],
) -> List[ChunkTopicTag]:
    """For each chunk, find which topics appear in its content."""
    compiled: List[Tuple[re.Pattern, str]] = [
        (_topic_to_regex(t.name), t.slug) for t in topics
    ]

    tags: List[ChunkTopicTag] = []
    for i, chunk in enumerate(chunks):
        # Normalize dashes in content too, so 'Calvache–Marrón' is comparable
        content_norm = chunk.content.replace("–", "-").replace("—", "-")
        matched_slugs = [slug for pat, slug in compiled if pat.search(content_norm)]
        tags.append(ChunkTopicTag(
            chunk_index=i,
            section_id=chunk.section_id,
            sub_section_id=chunk.sub_section_id,
            topic_slugs=matched_slugs,
        ))
    return tags


def derive_section_topics(
    chunks: List[Chunk],
    chunk_tags: List[ChunkTopicTag],
) -> List[SectionTopicLink]:
    """For each (section_id, topic_slug), assess depth + relevance.

    Depth:
      primary    — topic appears in ≥3 chunks of the section, OR ≥20% of them
      secondary  — topic appears in 2 chunks
      mention    — topic appears in exactly 1 chunk
    """
    section_topic_count: Dict[Tuple[int, str], int] = defaultdict(int)
    section_chunk_total: Dict[int, int] = defaultdict(int)

    for tag in chunk_tags:
        section_chunk_total[tag.section_id] += 1
        for slug in tag.topic_slugs:
            section_topic_count[(tag.section_id, slug)] += 1

    links: List[SectionTopicLink] = []
    for (section_id, slug), count in section_topic_count.items():
        total = section_chunk_total[section_id]
        ratio = count / total if total else 0.0

        if count >= 3 or ratio >= 0.20:
            depth = "primary"
            relevance = 1.0
        elif count == 2:
            depth = "secondary"
            relevance = 0.6
        else:
            depth = "mention"
            relevance = 0.3

        links.append(SectionTopicLink(
            section_id=section_id,
            topic_slug=slug,
            depth=depth,
            mention_count=count,
            relevance_score=relevance,
        ))
    return links


if __name__ == "__main__":
    import sys
    from collections import Counter
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.ingestion.pdf_parser import parse_pdf  # noqa: E402
    from src.config import settings  # noqa: E402

    chunks = []
    from src.ingestion.chunker import chunk_sub_sections  # noqa: E402
    chunks = chunk_sub_sections(parse_pdf(settings.paths.pdf_corpus))

    topics = extract_topics_from_glossary(chunks)
    chunk_tags = tag_chunks(chunks, topics)
    section_links = derive_section_topics(chunks, chunk_tags)

    print(f"Total topics extracted: {len(topics)}\n")

    cat_counts = Counter(t.category for t in topics)
    print("By category:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<18} {n}")
    print()

    imp_counts = Counter(t.importance for t in topics)
    print("By importance:")
    for imp, n in imp_counts.items():
        print(f"  {imp:<10} {n}")
    print()

    print("=== Sample topics ===")
    for t in topics[:10]:
        print(f"  [{t.category}/{t.importance}] {t.name} ({t.slug})")
        print(f"    {t.description[:100]}")
    print()

    print("=== Critical topics verification ===")
    found_critical = {t.name for t in topics if t.importance == "critical"}
    for name in CRITICAL_TOPIC_NAMES:
        mark = "✓" if name in found_critical else "✗"
        print(f"  {mark} {name}")
    print()

    print("=== Chunk-topic stats ===")
    tags_with_topics = [t for t in chunk_tags if t.topic_slugs]
    print(f"Chunks with ≥1 topic match: {len(tags_with_topics)}/{len(chunk_tags)}")
    avg_per_chunk = sum(len(t.topic_slugs) for t in chunk_tags) / len(chunk_tags)
    print(f"Average topics per chunk:   {avg_per_chunk:.1f}")
    print()

    print("=== Sample chunk tags ===")
    # Show §2.13 Echo Lock chunk's tags
    for tag in chunk_tags:
        if tag.sub_section_id == "2.13":
            print(f"  §2.13 Echo Lock → topics: {tag.topic_slugs}")
            break
    for tag in chunk_tags:
        if tag.sub_section_id == "5.2":
            print(f"  §5.2 Three-Two-One → topics: {tag.topic_slugs}")
            break
    for tag in chunk_tags:
        if tag.sub_section_id == "9.7":
            print(f"  §9.7 Salta showdown → topics: {tag.topic_slugs}")
            break
    print()

    print("=== Section-topic depth distribution ===")
    depth_counts = Counter(link.depth for link in section_links)
    print(f"  Total section-topic links: {len(section_links)}")
    for d, n in depth_counts.items():
        print(f"  {d:<12} {n}")
    print()

    print("=== Section 2 (Powers) — its topics ===")
    s2_links = sorted(
        [l for l in section_links if l.section_id == 2],
        key=lambda l: (-l.mention_count, l.topic_slug),
    )
    for link in s2_links[:15]:
        print(f"  [{link.depth:<10}] {link.topic_slug:<35} ({link.mention_count} chunks)")
