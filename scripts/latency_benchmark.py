"""Time retrieval: index build once, then per-question search latency for each mode, on the 50 labelled questions.

Everything runs offline in-process. Prints a Markdown table (milliseconds per query, p50 and p95) plus the one-off index build time.
Run `python scripts/latency_benchmark.py`; the README quotes one run and says on what machine.
"""

import argparse
import statistics
import time
from dataclasses import replace

from rag_assistant.cli import ROOT, make_embedder
from rag_assistant.evaluate import load_cases
from rag_assistant.retriever import Retriever, load_corpus
from rag_assistant.text import Chunk

MODES = ("bm25", "dense", "hybrid")


def percentiles(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def scaled(chunks: list[Chunk], factor: int) -> list[Chunk]:
    """``factor`` copies of the corpus with distinct chunk ids, to see how search time grows (the content repeats, so this
    measures cost, not retrieval quality)."""
    return [replace(c, index=c.index + copy * 10_000) for copy in range(factor) for c in chunks]


def benchmark(
    embedder: str = "lsa", repeats: int = 5, k: int = 3, scale: int = 1
) -> tuple[float, int, dict[str, tuple[float, float]]]:
    """(index build seconds, number of chunks, {mode: (p50 ms, p95 ms)}) over every labelled question, ``repeats`` times."""
    chunks = scaled(load_corpus(ROOT / "data" / "corpus"), scale)
    questions = [c["question"] for c in load_cases(ROOT / "evals" / "questions.json")]
    start = time.perf_counter()
    retriever = Retriever(chunks, make_embedder(embedder, chunks))
    build = time.perf_counter() - start
    for q in questions[:5]:  # warm-up
        retriever.search(q, k, "hybrid")
    results: dict[str, tuple[float, float]] = {}
    for mode in MODES:
        samples = []
        for _ in range(repeats):
            for q in questions:
                t = time.perf_counter()
                retriever.search(q, k, mode)
                samples.append((time.perf_counter() - t) * 1000)
        results[mode] = percentiles(samples)
    return build, len(chunks), results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=["hashing", "lsa"], default="lsa")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 10, 100], help="corpus size multipliers (40 chunks each)")
    args = parser.parse_args()
    print("| chunks | index build (ms, once) | mode | p50 (ms) | p95 (ms) |\n|---|---|---|---|---|")
    for scale in args.scales:
        build, n_chunks, results = benchmark(args.embedder, args.repeats, scale=scale)
        for mode, (p50, p95) in results.items():
            print(f"| {n_chunks:,} | {build * 1000:.0f} | {mode} | {p50:.2f} | {p95:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
