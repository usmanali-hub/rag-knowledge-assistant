"""Embedders turn text into vectors.

``HashingEmbedder`` runs offline with no model download, which keeps tests and CI fast and
deterministic. It is a bag-of-words signal, not a semantic one: it will not know that "car" and
"automobile" are related. Swap in a real model by implementing the ``Embedder`` protocol.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
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


class LSAEmbedder:
    """Latent semantic analysis: TF-IDF followed by a truncated SVD fitted on the corpus.

    Words that appear in the same documents end up with similar directions, so a query can match a
    passage that shares no exact word with it. It only knows the vocabulary of the corpus it was
    fitted on, which makes it a small offline stand-in for a trained embedding model.
    """

    def __init__(self, vocab: dict[str, int], idf: np.ndarray, components: np.ndarray) -> None:
        self.vocab, self.idf, self.components = vocab, idf, components

    @classmethod
    def fit(cls, texts: list[str], dim: int = 64) -> LSAEmbedder:
        if not texts:
            raise ValueError("need at least one text to fit")
        tokenized = [tokenize(t) for t in texts]
        vocab = {w: i for i, w in enumerate(sorted({w for toks in tokenized for w in toks}))}
        counts = np.zeros((len(texts), len(vocab)))
        for row, toks in enumerate(tokenized):
            for w in toks:
                counts[row, vocab[w]] += 1
        df = (counts > 0).sum(axis=0)
        idf = np.log((1 + len(texts)) / (1 + df)) + 1.0
        tfidf = np.log1p(counts) * idf
        _, _, vt = np.linalg.svd(tfidf, full_matrices=False)
        return cls(vocab, idf, vt[: min(dim, vt.shape[0])].T)  # (vocab, dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        counts = np.zeros((len(texts), len(self.vocab)))
        for row, text in enumerate(texts):
            for w in tokenize(text):
                if w in self.vocab:
                    counts[row, self.vocab[w]] += 1
        out = (np.log1p(counts) * self.idf) @ self.components
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.asarray(out / np.where(norms == 0, 1.0, norms))


class WordVectorEmbedder:
    """Mean of pretrained word vectors, weighted by inverse document frequency.

    Unlike ``LSAEmbedder`` the word vectors were trained on a large external corpus, so related
    words ("car", "automobile") are close even if they never co-occur in your documents. Pass any
    ``lookup(word) -> vector or None``; ``from_spacy_package`` reads the vectors shipped inside a
    spaCy model package without loading the spaCy pipeline.
    """

    def __init__(
        self,
        lookup: Callable[[str], np.ndarray | None],
        dim: int,
        idf: dict[str, float] | None = None,
    ) -> None:
        self.lookup, self.dim, self.idf = lookup, dim, idf or {}

    def fit_idf(self, texts: list[str]) -> WordVectorEmbedder:
        """Return a copy that down-weights words appearing in many of ``texts``."""
        df: dict[str, int] = {}
        for text in texts:
            for word in set(tokenize(text)):
                df[word] = df.get(word, 0) + 1
        n = len(texts)
        idf = {w: float(np.log((1 + n) / (1 + c)) + 1.0) for w, c in df.items()}
        return WordVectorEmbedder(self.lookup, self.dim, idf)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for word in tokenize(text):
                vec = self.lookup(word)
                if vec is not None:
                    out[row] += self.idf.get(word, 1.0) * vec
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.asarray(out / np.where(norms == 0, 1.0, norms))

    @classmethod
    def from_spacy_package(cls, name: str = "en_core_web_md") -> WordVectorEmbedder:
        import importlib
        from pathlib import Path

        srsly = importlib.import_module("srsly")  # optional dependency, imported lazily
        hash_string = importlib.import_module("spacy.strings").hash_string
        package = importlib.import_module(name)
        vocab = next(Path(str(package.__file__)).parent.glob(f"{name}-*")) / "vocab"
        matrix = np.load(vocab / "vectors", allow_pickle=False)
        with open(vocab / "key2row", "rb") as f:
            key2row = next(iter(srsly.msgpack.Unpacker(f, raw=False, strict_map_key=False)))

        def lookup(word: str) -> np.ndarray | None:
            row = key2row.get(hash_string(word))
            return None if row is None else np.asarray(matrix[row], dtype=np.float64)

        return cls(lookup, matrix.shape[1])
