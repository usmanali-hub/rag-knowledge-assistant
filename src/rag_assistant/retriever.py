"""Keyword, vector and hybrid retrieval over a list of chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder, HashingEmbedder
from .text import Chunk, chunk_words


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> dict[int, float]:
    """Merge several best-first rankings of item ids.

    Each item scores sum(1 / (k + rank)) over the rankings it appears in. Using ranks instead of
    raw scores means BM25 and cosine scores never have to be put on the same scale. Optional
    ``weights`` (one per ranking) let a stronger ranker count for more.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return fused


def load_corpus(directory: str | Path, size: int = 80, overlap: int = 20) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        chunks.extend(chunk_words(path.read_text(encoding="utf-8"), path.name, size, overlap))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        if not chunks:
            raise ValueError("need at least one chunk")
        self.chunks = chunks
        self.embedder = embedder or HashingEmbedder()
        texts = [c.text for c in chunks]
        self._bm25 = BM25(texts)
        self._matrix = self.embedder.embed(texts)

    def _rank(self, scores: np.ndarray) -> list[int]:
        # Stable sort so ties keep document order, which makes results deterministic.
        return list(np.argsort(-scores, kind="stable"))

    def search(
        self, query: str, k: int = 3, mode: str = "hybrid", keyword_weight: float = 1.0
    ) -> list[Hit]:
        keyword = np.array(self._bm25.scores(query))
        vector = self._matrix @ self.embedder.embed([query])[0]
        if mode == "bm25":
            order, score = self._rank(keyword), keyword
        elif mode == "dense":
            order, score = self._rank(vector), vector
        elif mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [self._rank(keyword), self._rank(vector)], weights=[keyword_weight, 1.0]
            )
            score = np.array([fused.get(i, 0.0) for i in range(len(self.chunks))])
            order = self._rank(score)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        return [Hit(self.chunks[i], float(score[i])) for i in order[:k]]
