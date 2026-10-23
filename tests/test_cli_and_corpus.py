from rag_assistant.cli import ROOT, main
from rag_assistant.evaluate import compare_modes, load_cases
from rag_assistant.retriever import Retriever, load_corpus


def test_every_labelled_source_exists_in_corpus():
    sources = {p.name for p in (ROOT / "data" / "corpus").glob("*.md")}
    for case in load_cases(ROOT / "evals" / "questions.json"):
        assert set(case["sources"]) <= sources, case


def test_hybrid_is_not_worse_than_either_single_mode():
    r = Retriever(load_corpus(ROOT / "data" / "corpus"))
    res = compare_modes(r, load_cases(ROOT / "evals" / "questions.json"), k=3)
    assert res["hybrid"]["recall@3"] >= max(res["bm25"]["recall@3"], res["dense"]["recall@3"]) - 1e-9
    assert res["hybrid"]["recall@3"] >= 0.9  # regression guard for the retrieval pipeline


def test_cli_ask_and_eval_run(capsys):
    assert main(["ask", "Why normalise vectors?"]) == 0
    assert "embeddings.md" in capsys.readouterr().out
    assert main(["eval"]) == 0
    assert "hybrid" in capsys.readouterr().out
