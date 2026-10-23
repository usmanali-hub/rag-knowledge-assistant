"""BM25 keyword ranking, implemented from the formula so it is easy to inspect and test."""

from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(d)) for d in documents]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(documents)
        # The +1 inside the log keeps IDF positive even for terms in most documents.
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, length in zip(self._tf, self._len):
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                norm = 1 - self.b + self.b * length / self._avg_len if self._avg_len else 1.0
                score += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            out.append(score)
        return out
