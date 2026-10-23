"""Answer generation behind a small interface, so the model can be swapped without touching retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .retriever import Hit
from .text import tokenize

PROMPT = """Answer the question using only the context below. If the context does not contain the \
answer, say you don't know. Cite sources as [source#index].

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits)
    return PROMPT.format(context=context, question=question)


class Answerer(Protocol):
    def answer(self, question: str, hits: list[Hit]) -> str: ...


class ExtractiveAnswerer:
    """Offline fallback: return the retrieved sentence that best overlaps the question, with a citation."""

    def answer(self, question: str, hits: list[Hit]) -> str:
        q = set(tokenize(question))
        best: tuple[int, str, str] | None = None
        for h in hits:
            for sentence in h.chunk.text.replace("\n", " ").split(". "):
                overlap = len(q & set(tokenize(sentence)))
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, sentence.strip().rstrip("."), h.chunk.id)
        if best is None:
            return "I don't know: nothing relevant was retrieved."
        return f"{best[1]}. [{best[2]}]"


class LLMAnswerer:
    """Wraps any ``complete(prompt) -> str`` callable (OpenAI, Anthropic, a local model, a test double)."""

    def __init__(self, complete: Callable[[str], str]) -> None:
        self._complete = complete

    def answer(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return "I don't know: nothing relevant was retrieved."
        return self._complete(build_prompt(question, hits))
