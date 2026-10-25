"""A held-out retrieval benchmark built from text I did not write: the Python standard library docs.

For each documented function, the *first sentence* of its docstring is the query (with the
function's own name words removed) and the *rest of the docstring* is the passage to find. Queries
and passages come from the library authors, not from this project, and the target is the one
function whose passage matches. It is synthetic (summary to description) but it removes the
author-written-questions caveat.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Iterable

from .evaluate import Case, evaluate
from .retriever import Hit, Retriever
from .text import Chunk

MODULES = ("textwrap", "shutil", "statistics", "heapq", "bisect", "random", "string", "json", "re", "math", "itertools", "functools", "operator", "os.path", "collections", "pathlib", "csv", "datetime")


def split_docstring(doc: str) -> tuple[str, str] | None:
    """(first sentence, remainder) or None when the docstring is too short to be useful."""
    doc = inspect.cleandoc(doc)
    match = re.match(r"(.+?[.:])(\s|$)", doc.replace("\n", " "))
    if not match:
        return None
    first, rest = match.group(1).strip(), doc.replace("\n", " ")[match.end():].strip()
    return (first, rest) if len(rest.split()) >= 25 and len(first.split()) >= 4 else None


def mask_name(sentence: str, name: str) -> str:
    """Remove the function name and its underscore-separated parts from the query."""
    parts = {name.lower(), *name.lower().split("_")}
    words = [w for w in sentence.split() if re.sub(r"\W", "", w).lower() not in parts]
    return " ".join(words)


def build_benchmark(modules: Iterable[str] = MODULES) -> tuple[list[Chunk], list[Case]]:
    chunks: list[Chunk] = []
    cases: list[Case] = []
    seen: set[str] = set()
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name, obj in sorted(vars(module).items()):
            if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isclass(obj)):
                continue
            parts = split_docstring(obj.__doc__ or "")
            key = f"{module_name}.{name}"
            if parts is None or parts[1] in seen:
                continue
            seen.add(parts[1])
            query = mask_name(parts[0], name)
            if len(query.split()) < 4:
                continue
            chunks.append(Chunk(source=key, index=0, text=parts[1]))
            cases.append({"question": query, "sources": [key], "type": "stdlib"})
    return chunks, cases


def run(
    embedder_name: str, k: int = 3, vector_model: str = "en_core_web_lg"
) -> dict[str, dict[str, float]]:
    """Evaluate every retrieval mode on the standard-library benchmark."""
    from .cli import make_embedder  # local import: cli imports this module's siblings

    chunks, cases = build_benchmark()
    retriever = Retriever(chunks, make_embedder(embedder_name, chunks, vector_model))
    results: dict[str, dict[str, float]] = {}
    for mode in ("bm25", "dense", "hybrid"):

        def search(query: str, top: int, mode: str = mode) -> list[Hit]:
            return retriever.search(query, top, mode)

        results[mode] = evaluate(search, cases, k)
    return results
