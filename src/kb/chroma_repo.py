"""ChromaDB persistent repository for chunk embeddings.

Stores all 195 chunk embeddings + minimal metadata for fast semantic
retrieval. Relational topic memberships and mastery state live in
PostgreSQL — this collection is purely for vector search ranking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.ingestion.chunker import Chunk


# Quiet ChromaDB's verbose telemetry log
logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)


class ChromaRepo:
    """Repository for the SLATEFALL chunk collection."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or settings.paths.chromadb_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.vectordb.collection_name

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": settings.vectordb.similarity_metric},
        )

    # ── Inspection ────────────────────────────────────────────────────

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        result = self._collection.get(ids=[chunk_id])
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "document": result["documents"][0] if result.get("documents") else None,
            "metadata": result["metadatas"][0] if result.get("metadatas") else None,
        }

    # ── Mutation ──────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> int:
        """Upsert chunks (idempotent — safe to re-run).

        Returns the number of chunks written.
        """
        if not chunks:
            return 0
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match "
                f"chunk count ({len(chunks)})"
            )

        ids = [c.sub_section_id for c in chunks]
        if len(set(ids)) != len(ids):
            duplicates = [i for i in ids if ids.count(i) > 1]
            raise ValueError(f"Duplicate sub_section_ids detected: {duplicates}")

        documents = [c.content for c in chunks]
        metadatas = [self._chunk_metadata(c) for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        return len(chunks)

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": settings.vectordb.similarity_metric},
        )

    # ── Query ─────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        section_ids: Optional[List[int]] = None,
        exclude_kinds: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search returning a flat list of result dicts.

        Filters:
          section_ids   — only chunks from these sections
          exclude_kinds — drop chunks with these chunk_kind values
                          (e.g., ['glossary_entry'] to skip dictionary noise)
        """
        top_k = top_k or settings.vectordb.default_top_k

        where_clauses: List[Dict[str, Any]] = []
        if section_ids:
            where_clauses.append({"section_id": {"$in": section_ids}})
        if exclude_kinds:
            where_clauses.append({"chunk_kind": {"$nin": exclude_kinds}})

        where: Optional[Dict[str, Any]] = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
        )

        # ChromaDB returns parallel lists per query; we sent 1 query, so [0]
        n = len(result["ids"][0])
        return [
            {
                "id": result["ids"][0][i],
                "document": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
                "similarity": 1.0 - result["distances"][0][i],
            }
            for i in range(n)
        ]

    def get_by_sections(self, section_ids: List[int]) -> List[Dict[str, Any]]:
        """Return all chunks for the given sections (no semantic ranking)."""
        result = self._collection.get(where={"section_id": {"$in": section_ids}})
        n = len(result["ids"])
        return [
            {
                "id": result["ids"][i],
                "document": result["documents"][i] if result.get("documents") else None,
                "metadata": result["metadatas"][i] if result.get("metadatas") else None,
            }
            for i in range(n)
        ]

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _chunk_metadata(chunk: Chunk) -> Dict[str, Any]:
        """Build the metadata dict ChromaDB stores per chunk.

        ChromaDB only accepts primitive types in metadata; lists are
        JSON-stringified.
        """
        return {
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "sub_section_id": chunk.sub_section_id,
            "sub_section_title": chunk.sub_section_title,
            "token_count": chunk.token_count,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "has_table": chunk.has_table,
            "has_bullets": chunk.has_bullets,
            "chunk_kind": chunk.chunk_kind,
            "cross_refs": json.dumps(chunk.cross_refs),
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.ingestion.chunker import chunk_sub_sections  # noqa: E402
    from src.ingestion.embedder import get_embedder  # noqa: E402
    from src.ingestion.pdf_parser import parse_pdf  # noqa: E402

    print("Building pipeline output (parse → chunk → embed)...")
    chunks = chunk_sub_sections(parse_pdf(settings.paths.pdf_corpus))
    embedder = get_embedder()
    embeddings = embedder.embed([c.content for c in chunks], show_progress=False)
    print(f"  {len(chunks)} chunks ready, embeddings shape {embeddings.shape}")
    print()

    repo = ChromaRepo()
    print(f"ChromaDB collection: '{repo.collection_name}'")
    print(f"Persistence path:    {repo.persist_dir}")
    print(f"Initial count:       {repo.count()}")
    print()

    print("Resetting collection (fresh start for the test)...")
    repo.reset()
    print(f"  After reset: {repo.count()}")
    print()

    print("Adding all chunks...")
    n_added = repo.add_chunks(chunks, embeddings)
    print(f"  Added {n_added} chunks")
    print(f"  Collection count: {repo.count()}")
    print()

    print("=== Test 1: get_by_id('2.13') ===")
    record = repo.get_by_id("2.13")
    print(f"  Found: §{record['metadata']['sub_section_id']}")
    print(f"  Title: {record['metadata']['sub_section_title']}")
    print(f"  Pages: {record['metadata']['page_start']}-{record['metadata']['page_end']}")
    print(f"  Cross-refs (JSON): {record['metadata']['cross_refs']}")
    print()

    print("=== Test 2: semantic query 'Echo Lock failure triggers' ===")
    q_emb = embedder.embed_one("Echo Lock failure triggers neurological seizure")
    hits = repo.query(q_emb, top_k=5)
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"  {i}. sim={h['similarity']:.3f}  §{m['sub_section_id']}  "
              f"{m['sub_section_title'][:50]}")
    print()

    print("=== Test 3: filter by section_ids=[2] ===")
    q_emb = embedder.embed_one("targeting failure rate environmental condition")
    hits = repo.query(q_emb, top_k=5, section_ids=[2])
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"  {i}. sim={h['similarity']:.3f}  §{m['sub_section_id']}  "
              f"{m['sub_section_title'][:50]}")
    print()

    print("=== Test 4: exclude glossary_entry chunks ===")
    q_emb = embedder.embed_one("Quebrantadero adversary engagement")
    hits = repo.query(q_emb, top_k=5, exclude_kinds=["glossary_entry"])
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"  {i}. sim={h['similarity']:.3f}  §{m['sub_section_id']} "
              f"[{m['chunk_kind']}]  {m['sub_section_title'][:40]}")
    print()

    print("=== Test 5: get_by_sections([5]) — all Section 5 chunks ===")
    s5_chunks = repo.get_by_sections([5])
    print(f"  Got {len(s5_chunks)} chunks from Section 5")
    for c in sorted(s5_chunks, key=lambda x: x['metadata']['sub_section_id']):
        m = c["metadata"]
        print(f"    §{m['sub_section_id']:<12} {m['sub_section_title'][:50]}")
    print()

    print("=== Test 6: Persistence — re-open client and verify data survives ===")
    del repo
    repo2 = ChromaRepo()
    surviving_count = repo2.count()
    print(f"  Count after fresh client: {surviving_count}")
    assert surviving_count == n_added, "Persistence broken!"
    print("  ✓ Data persisted correctly")
    print()

    print("All ChromaDB tests passed.")
