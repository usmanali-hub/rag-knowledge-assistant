from rag_assistant.cli import ROOT, main, make_embedder
from rag_assistant.evaluate import compare_modes, load_cases
from rag_assistant.retriever import Retriever, load_corpus


def test_every_labelled_source_exists_in_corpus():
    sources = {p.name for p in (ROOT / "data" / "corpus").glob("*.md")}
    for case in load_cases(ROOT / "evals" / "questions.json"):
        assert set(case["sources"]) <= sources, case


def _retriever(kind: str) -> Retriever:
    chunks = load_corpus(ROOT / "data" / "corpus")
    return Retriever(chunks, make_embedder(kind, chunks))


def _recall(retriever: Retriever, kind: str, mode: str) -> float:
    cases = [c for c in load_cases(ROOT / "evals" / "questions.json") if c.get("type") == kind]
    return compare_modes(retriever, cases, k=3)[mode]["recall@3"]


def test_retrieval_regression_guards():
    r = _retriever("lsa")
    assert _recall(r, "lexical", "hybrid") >= 0.95
    assert _recall(r, "paraphrase", "hybrid") >= 0.8


def test_lsa_beats_hashing_on_paraphrased_questions():
    # The hashing embedder only sees shared words; LSA also uses word co-occurrence.
    assert _recall(_retriever("lsa"), "paraphrase", "dense") > _recall(_retriever("hashing"), "paraphrase", "dense") + 0.3


def test_cli_ask_and_eval_run(capsys):
    assert main(["ask", "Why normalise vectors?"]) == 0
    assert "embeddings.md" in capsys.readouterr().out
    assert main(["eval"]) == 0
    assert "hybrid" in capsys.readouterr().out
