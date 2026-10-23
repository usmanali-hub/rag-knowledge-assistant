"""Embedders turn text into vectors.

``HashingEmbedder`` runs offline with no model download, which keeps tests and CI fast and
deterministic. It is a bag-of-words signal, not a semantic one: it will not know that "car" and
"automobile" are related. Swap in a real model by implementing the ``Embedder`` protocol.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float array of L2-normalised vectors."""
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 512, use_bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim, self.use_bigrams = dim, use_bigrams

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        feats = list(tokens)
        if self.use_bigrams:
            feats += [f"{a}_{b}" for a, b in pairwise(tokens)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                sign = 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collision bias
                out[row, h % self.dim] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)
