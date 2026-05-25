"""Embedding generation via sentence-transformers.

Wraps `all-MiniLM-L6-v2` (384-dim, ~90MB download, runs comfortably on CPU)
behind a lazy-loaded class. Embeddings are normalised to unit length so
cosine similarity reduces to a dot product later in retrieval.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


class Embedder:
    """Lazy-loaded text embedder."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name or settings.embedding.model_name
        self.device = device or settings.embedding.device
        self.batch_size = batch_size or settings.embedding.batch_size
        self.normalize = (
            settings.embedding.normalize if normalize is None else normalize
        )
        self._model: Optional[SentenceTransformer] = None

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Return a numpy array of shape (len(texts), dim)."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        model = self._ensure_loaded()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Return a single embedding of shape (dim,) for a single text."""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return settings.embedding.dimensions


_default_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Return a process-wide singleton Embedder."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.ingestion.chunker import chunk_sub_sections  # noqa: E402
    from src.ingestion.pdf_parser import parse_pdf  # noqa: E402

    print("Loading embedder (may download model on first run)...")
    t0 = time.time()
    embedder = get_embedder()
    embedder._ensure_loaded()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Model: {embedder.model_name}")
    print(f"  Device: {embedder.device}")
    print(f"  Dim:   {embedder.dim}")
    print()

    # Load actual chunks
    print("Parsing + chunking PDF...")
    chunks = chunk_sub_sections(parse_pdf(settings.paths.pdf_corpus))
    print(f"  {len(chunks)} chunks ready")
    print()

    # Embed all chunks
    texts = [c.content for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    t0 = time.time()
    embeddings = embedder.embed(texts, show_progress=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts) / elapsed:.1f} chunks/sec)")
    print()

    # ── Verification ──
    print("=== Shape check ===")
    print(f"  Expected: ({len(chunks)}, {embedder.dim})")
    print(f"  Got:      {embeddings.shape}")
    assert embeddings.shape == (len(chunks), embedder.dim), "Shape mismatch!"
    print("  ✓ Pass")
    print()

    print("=== Normalisation check (L2 norms ≈ 1.0) ===")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Min norm: {norms.min():.4f}")
    print(f"  Max norm: {norms.max():.4f}")
    print(f"  Mean:     {norms.mean():.4f}")
    assert np.allclose(norms, 1.0, atol=1e-3), "Vectors not normalised!"
    print("  ✓ Pass")
    print()

    # Find specific chunks for similarity tests
    def find_chunk(sub_id: str) -> int:
        for i, c in enumerate(chunks):
            if c.sub_section_id == sub_id:
                return i
        raise ValueError(f"Chunk {sub_id} not found")

    echo_idx = find_chunk("2.13")
    range_idx = find_chunk("2.5")
    salta_idx = find_chunk("9.7")

    print("=== Similarity sanity checks ===")
    # Query: 'Echo Lock failure mode' should be most similar to §2.13
    query_emb = embedder.embed_one("Echo Lock failure mode neurological seizure")

    # Compute similarity between query and all chunks
    sims = embeddings @ query_emb  # dot product (normalised vectors → cosine)
    top5_indices = np.argsort(-sims)[:5]

    print('  Query: "Echo Lock failure mode neurological seizure"')
    print("  Top-5 most similar chunks:")
    for rank, idx in enumerate(top5_indices, 1):
        c = chunks[idx]
        marker = " ← expected" if c.sub_section_id == "2.13" else ""
        print(f"    {rank}. {sims[idx]:.4f}  §{c.sub_section_id} {c.sub_section_title[:50]}{marker}")
    print()

    # Direct chunk-to-chunk similarities
    print("  Pairwise sims (chunk vs chunk):")
    print(f"    §2.13 ↔ §2.13 (self):    {embeddings[echo_idx] @ embeddings[echo_idx]:.4f}")
    print(f"    §2.13 ↔ §9.7 (Echo+Salta){embeddings[echo_idx] @ embeddings[salta_idx]:.4f}")
    print(f"    §2.13 ↔ §2.5 (range):    {embeddings[echo_idx] @ embeddings[range_idx]:.4f}")
    print()

    print("=== Sample 10 chunks: section/sub_id and embedding norm ===")
    for i in range(0, 10):
        c = chunks[i]
        print(f"  §{c.sub_section_id:<12} norm={norms[i]:.4f}  '{c.sub_section_title[:40]}'")
