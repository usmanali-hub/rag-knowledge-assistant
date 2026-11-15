import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("latency_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "latency_benchmark.py")
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

from rag_assistant.retriever import load_corpus


def test_percentiles_are_ordered_and_pick_the_median():
    p50, p95 = bench.percentiles([5.0, 1.0, 3.0, 2.0, 4.0])
    assert p50 == 3.0 and p95 == 5.0
    assert bench.percentiles([2.0]) == (2.0, 2.0)


def test_scaled_corpus_multiplies_chunks_and_keeps_ids_unique():
    chunks = load_corpus(bench.ROOT / "data" / "corpus")
    big = bench.scaled(chunks, 3)
    assert len(big) == 3 * len(chunks) and len({c.id for c in big}) == len(big)
    assert bench.scaled(chunks, 1) == chunks


def test_benchmark_reports_every_mode_with_ordered_percentiles():
    build, n_chunks, results = bench.benchmark("hashing", repeats=1, scale=2)
    assert build > 0 and n_chunks == 80 and set(results) == {"bm25", "dense", "hybrid"}
    assert all(0 < p50 <= p95 for p50, p95 in results.values())
