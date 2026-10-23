"""Tokenizing and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
        "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "which", "who",
        "why", "with", "do", "does", "can", "i", "you",
    }
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS] if drop_stopwords else tokens


@dataclass(frozen=True)
class Chunk:
    """A slice of a source document that can be retrieved and cited."""

    source: str
    index: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"


def chunk_words(text: str, source: str, size: int = 80, overlap: int = 20) -> list[Chunk]:
    """Split ``text`` into windows of ``size`` words that share ``overlap`` words.

    Overlap keeps sentences that straddle a boundary retrievable from at least one chunk.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        chunks.append(Chunk(source=source, index=i, text=" ".join(window)))
        if start + size >= len(words):
            break
    return chunks
