import pytest

from rag_assistant.evaluate import evaluate, first_relevant_rank
from rag_assistant.generate import ExtractiveAnswerer, LLMAnswerer, build_prompt
from rag_assistant.retriever import Hit
from rag_assistant.text import Chunk


def hit(source, text="text", score=1.0):
    return Hit(Chunk(source, 0, text), score)


def test_prompt_contains_context_ids_and_question():
    p = build_prompt("What is X?", [hit("a.md", "X is a thing.")])
    assert "[a.md#0] X is a thing." in p and "Question: What is X?" in p


def test_llm_answerer_passes_prompt_to_callable():
    seen = []
    out = LLMAnswerer(lambda prompt: seen.append(prompt) or "ok").answer("Q?", [hit("a.md")])
    assert out == "ok" and "Q?" in seen[0]


def test_llm_answerer_without_hits_does_not_call_model():
    def boom(_: str) -> str:
        raise AssertionError("model should not be called")

    assert "don't know" in LLMAnswerer(boom).answer("Q?", [])


def test_extractive_answer_cites_best_sentence():
    hits = [hit("a.md", "Cats sleep a lot. Dogs bark at night."), hit("b.md", "Rain falls.")]
    assert ExtractiveAnswerer().answer("when do dogs bark", hits) == "Dogs bark at night. [a.md#0]"


def test_extractive_answer_admits_ignorance():
    assert "don't know" in ExtractiveAnswerer().answer("quantum chromodynamics", [hit("a.md", "Cats sleep.")])


def test_first_relevant_rank():
    hits = [hit("a.md"), hit("b.md"), hit("c.md")]
    assert first_relevant_rank(hits, {"b.md"}) == 2
    assert first_relevant_rank(hits, {"z.md"}) is None


def test_evaluate_metrics_by_hand():
    results = {"q1": [hit("a.md")], "q2": [hit("x.md"), hit("b.md")], "q3": [hit("x.md")]}
    cases = [{"question": q, "sources": [s]} for q, s in [("q1", "a.md"), ("q2", "b.md"), ("q3", "c.md")]]
    m = evaluate(lambda q, k: results[q][:k], cases, k=2)
    assert m["recall@2"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)


def test_evaluate_rejects_empty_cases():
    with pytest.raises(ValueError):
        evaluate(lambda q, k: [], [])
