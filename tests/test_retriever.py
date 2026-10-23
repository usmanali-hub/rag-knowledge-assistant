import pytest

from rag_assistant.retriever import Retriever, load_corpus, reciprocal_rank_fusion
from rag_assistant.text import Chunk


def make(texts):
    return [Chunk(f"d{i}.md", 0, t) for i, t in enumerate(texts)]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 0, 2]])
    assert fused[2] < fused[0] and fused[2] < fused[1]
    assert fused[0] == pytest.approx(fused[1])


def test_rrf_handles_item_in_one_list_only():
    fused = reciprocal_rank_fusion([[0], [1]])
    assert set(fused) == {0, 1}


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_each_mode_finds_the_obvious_chunk(mode):
    r = Retriever(make(["docker compose starts containers", "cosine similarity of embeddings", "bm25 keyword ranking"]))
    assert r.search("cosine similarity", k=1, mode=mode)[0].chunk.source == "d1.md"


def test_k_limits_results_and_unknown_mode_fails():
    r = Retriever(make(["a b", "c d", "e f"]))
    assert len(r.search("a", k=2)) == 2
    with pytest.raises(ValueError):
        r.search("a", mode="magic")


def test_requires_chunks():
    with pytest.raises(ValueError):
        Retriever([])


def test_load_corpus_reads_markdown(tmp_path):
    (tmp_path / "one.md").write_text("alpha beta gamma", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("ignored", encoding="utf-8")
    chunks = load_corpus(tmp_path)
    assert [c.source for c in chunks] == ["one.md"]
