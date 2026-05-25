"""Hybrid retriever: SQL-filter (sections) + ChromaDB semantic ranking.

Phase-2 retrieval path:
  1. SQL — get all chunks for requested section_ids
  2. ChromaDB — semantically rank within those sections
  3. (Optional) Cross-reference resolution — pull §X.Y referenced chunks
  4. Token budget filter — drop chunks that won't fit

Returns ordered Chunk model objects ready for prompt assembly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.ingestion.embedder import get_embedder
from src.kb.chroma_repo import ChromaRepo
from src.kb.models import Chunk as ChunkModel


@dataclass
class RetrievalResult:
    chunks: List[ChunkModel] = field(default_factory=list)
    chroma_hits: int = 0
    cross_refs_pulled: int = 0
    tokens_total: int = 0


_CROSS_REF_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


def _resolve_cross_refs(
    chunks: Sequence[ChunkModel],
    session: Session,
    max_depth: int = 1,
) -> List[ChunkModel]:
    """For each chunk's cross_refs, pull referenced chunks (one level deep)."""
    already_present = {c.sub_section_id for c in chunks}
    referenced_ids: Set[str] = set()
    for c in chunks:
        for ref in (c.cross_refs or []):
            base = ref.split("-")[0].split("#")[0].split("+")[0]
            if base not in already_present:
                referenced_ids.add(base)

    if not referenced_ids:
        return []

    # Match either exact sub_section_id or a merged variant ('5.10+5.11' contains '5.10')
    additional: List[ChunkModel] = []
    for ref_id in referenced_ids:
        match = session.execute(
            select(ChunkModel).where(ChunkModel.sub_section_id == ref_id)
        ).scalar_one_or_none()
        if match is None:
            match = session.execute(
                select(ChunkModel).where(
                    ChunkModel.sub_section_id.like(f"%{ref_id}%")
                )
            ).first()
            match = match[0] if match else None
        if match and match.sub_section_id not in already_present:
            additional.append(match)
            already_present.add(match.sub_section_id)

    return additional


def retrieve(
    session: Session,
    section_ids: Sequence[int],
    query: str,
    top_k: Optional[int] = None,
    include_cross_refs: bool = True,
    exclude_kinds: Optional[List[str]] = None,
) -> RetrievalResult:
    """Retrieve relevant chunks for a query, filtered by section.

    `query` is what we semantically search for inside the requested sections.
    """
    top_k = top_k or settings.retrieval.vector_top_k
    section_ids = list(section_ids)

    # Step 1: Semantic search (Chroma) filtered by section
    embedder = get_embedder()
    query_emb = embedder.embed_one(query)

    repo = ChromaRepo()
    hits = repo.query(
        query_embedding=query_emb,
        top_k=top_k,
        section_ids=section_ids,
        exclude_kinds=exclude_kinds,
    )
    hit_ids = [h["id"] for h in hits]

    # Step 2: Fetch the ORM rows (preserves Chroma's ranking)
    if hit_ids:
        rows = session.scalars(
            select(ChunkModel).where(ChunkModel.sub_section_id.in_(hit_ids))
        ).all()
        # Reorder rows to match Chroma's ranking
        by_sub_id = {c.sub_section_id: c for c in rows}
        chunks = [by_sub_id[i] for i in hit_ids if i in by_sub_id]
    else:
        chunks = []

    result = RetrievalResult(
        chunks=chunks,
        chroma_hits=len(chunks),
    )

    # Step 3: Cross-reference resolution
    if include_cross_refs and settings.retrieval.resolve_cross_references:
        additional = _resolve_cross_refs(
            chunks, session, settings.retrieval.cross_reference_max_depth
        )
        result.chunks.extend(additional)
        result.cross_refs_pulled = len(additional)

    result.tokens_total = sum(c.token_count for c in result.chunks)
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from sqlalchemy import create_engine

    engine = create_engine(settings.secrets.database_url, pool_pre_ping=True)

    with Session(engine) as session:
        print("=== Query 1: 'Echo Lock triggers' in Section 2 ===")
        r = retrieve(session, [2], "Echo Lock failure mode triggers", top_k=5)
        print(f"  Chroma hits: {r.chroma_hits}, cross-refs pulled: {r.cross_refs_pulled}")
        print(f"  Total tokens: {r.tokens_total}")
        for c in r.chunks[:7]:
            print(f"    §{c.sub_section_id:<12} ({c.token_count:>3} tok) "
                  f"{c.sub_section_title[:55]}")
        print()

        print("=== Query 2: 'Salta engagement' in Section 9 ===")
        r = retrieve(session, [9], "Salta engagement catastrophic outcome", top_k=5)
        print(f"  Chroma hits: {r.chroma_hits}, cross-refs pulled: {r.cross_refs_pulled}")
        for c in r.chunks[:7]:
            print(f"    §{c.sub_section_id:<12} ({c.token_count:>3} tok) "
                  f"{c.sub_section_title[:55]}")
        print()

        print("=== Query 3: combined sections [5, 8] ===")
        r = retrieve(session, [5, 8], "operational protocols and safehouses", top_k=8)
        print(f"  Chroma hits: {r.chroma_hits}, cross-refs pulled: {r.cross_refs_pulled}")
        for c in r.chunks:
            print(f"    §{c.sub_section_id:<12} sec={c.section_id} "
                  f"{c.sub_section_title[:50]}")
