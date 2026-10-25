import pytest

from rag_assistant.evaluate import split_dev_test, tune_keyword_weight
from rag_assistant.retriever import Retriever, reciprocal_rank_fusion
from rag_assistant.stdlib_eval import build_benchmark, mask_name, split_docstring
from rag_assistant.text import Chunk


def test_weighted_rrf_prefers_the_heavier_ranker():
    fused = reciprocal_rank_fusion([[0, 1], [1, 0]], weights=[3.0, 1.0])
    assert fused[0] > fused[1]
    assert reciprocal_rank_fusion([[0, 1], [1, 0]])[0] == pytest.approx(reciprocal_rank_fusion([[0, 1], [1, 0]])[1])


def test_split_docstring_needs_enough_text():
    long_rest = " ".join(["word"] * 30)
    assert split_docstring(f"Do the thing well. {long_rest}") == ("Do the thing well.", long_rest)
    assert split_docstring("Too short to use.") is None
    assert split_docstring("") is None


def test_mask_name_removes_name_parts_but_keeps_other_words():
    assert mask_name("Fill a single paragraph of text.", "fill") == "a single paragraph of text."
    assert mask_name("Return the heap_pop of items", "heap_pop") == "Return the of items"


def test_benchmark_targets_are_unique_and_self_consistent():
    chunks, cases = build_benchmark(("textwrap", "bisect"))
    assert len(chunks) == len(cases) > 3
    assert {c.source for c in chunks} == {case["sources"][0] for case in cases}
    for chunk, case in zip(chunks, cases):
        assert chunk.source.lower().split(".")[-1] not in case["question"].lower().split()


def test_dev_test_split_is_disjoint_and_complete():
    cases = [{"question": str(i), "sources": ["a.md"]} for i in range(7)]
    dev, test = split_dev_test(cases)  # type: ignore[arg-type]
    assert len(dev) + len(test) == 7
    assert not {c["question"] for c in dev} & {c["question"] for c in test}


def test_tune_keyword_weight_returns_a_grid_value():
    chunks = [Chunk(f"d{i}.md", 0, t) for i, t in enumerate(["red apple pie", "blue ocean wave", "green tea leaf"])]
    r = Retriever(chunks)
    dev = [{"question": "ocean wave", "sources": ["d1.md"]}, {"question": "apple pie", "sources": ["d0.md"]}]
    assert tune_keyword_weight(r, dev, grid=(0.5, 1.0, 2.0)) in (0.5, 1.0, 2.0)  # type: ignore[arg-type]
