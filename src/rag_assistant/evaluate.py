"""Retrieval metrics. Measure retrieval before tuning prompts: a bad retriever cannot be saved by the model."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np

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


def split_dev_test(cases: list[Case]) -> tuple[list[Case], list[Case]]:
    """Alternate cases between a tuning (dev) set and a held-out test set."""
    return cases[0::2], cases[1::2]


def tune_keyword_weight(
    retriever: Retriever,
    dev: list[Case],
    grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    k: int = 3,
) -> float:
    """Pick the BM25 fusion weight with the best MRR on ``dev`` (first grid value wins ties)."""
    best_weight, best_mrr = grid[0], -1.0
    for weight in grid:

        def search(query: str, top: int, w: float = weight) -> list[Hit]:
            return retriever.search(query, top, "hybrid", keyword_weight=w)

        mrr = evaluate(search, dev, k)["mrr"]
        if mrr > best_mrr + 1e-12:
            best_weight, best_mrr = weight, mrr
    return best_weight


def reciprocal_ranks(
    search: Callable[[str, int], list[Hit]], cases: list[Case], k: int = 3
) -> list[float]:
    """Per-question reciprocal rank (0 when nothing relevant is in the top k)."""
    out = []
    for case in cases:
        rank = first_relevant_rank(search(case["question"], k), set(case["sources"]))
        out.append(0.0 if rank is None else 1.0 / rank)
    return out


def paired_bootstrap_ci(
    a: list[float], b: list[float], n_resamples: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean difference (b - a) with a 95% bootstrap interval, resampling questions in pairs.

    If the interval contains 0 the difference could easily be chance at this sample size.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty score lists")
    diff = np.array(b) - np.array(a)
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_resamples, len(diff)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
