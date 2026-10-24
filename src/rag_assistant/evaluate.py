"""Retrieval metrics. Measure retrieval before tuning prompts: a bad retriever cannot be saved by the model."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from .retriever import Hit, Retriever


class Case(TypedDict):
    question: str
    sources: list[str]
    type: NotRequired[str]


def first_relevant_rank(hits: list[Hit], relevant_sources: set[str]) -> int | None:
    """1-based rank of the first hit from a relevant source, or None."""
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.source in relevant_sources:
            return rank
    return None


def evaluate(
    search: Callable[[str, int], list[Hit]], cases: list[Case], k: int = 3
) -> dict[str, float]:
    """Compute recall@k and MRR over cases shaped like {"question": str, "sources": [str]}."""
    if not cases:
        raise ValueError("no evaluation cases")
    hits_at_k = 0
    reciprocal = 0.0
    for case in cases:
        rank = first_relevant_rank(search(case["question"], k), set(case["sources"]))
        if rank is not None:
            hits_at_k += 1
            reciprocal += 1.0 / rank
    n = len(cases)
    return {f"recall@{k}": hits_at_k / n, "mrr": reciprocal / n}


def load_cases(path: str | Path) -> list[Case]:
    return cast(list[Case], json.loads(Path(path).read_text(encoding="utf-8")))


def compare_modes(
    retriever: Retriever, cases: list[Case], k: int = 3
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for mode in ("bm25", "dense", "hybrid"):

        def search(query: str, top: int, mode: str = mode) -> list[Hit]:
            return retriever.search(query, top, mode)

        results[mode] = evaluate(search, cases, k)
    return results
