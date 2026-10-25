"""Command line: ``python -m rag_assistant ask "question"`` or ``... eval``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .embeddings import Embedder, HashingEmbedder, LSAEmbedder, WordVectorEmbedder
from .evaluate import (
    Case,
    compare_modes,
    evaluate,
    load_cases,
    paired_bootstrap_ci,
    reciprocal_ranks,
    split_dev_test,
    tune_keyword_weight,
)
from .generate import ExtractiveAnswerer
from .retriever import Hit, Retriever, load_corpus
from .text import Chunk

ROOT = Path(__file__).resolve().parents[2]


def make_embedder(name: str, chunks: list[Chunk], vector_model: str = "en_core_web_lg") -> Embedder:
    if name == "lsa":
        return LSAEmbedder.fit([c.text for c in chunks])
    if name == "wordvec":
        return WordVectorEmbedder.from_spacy_package(vector_model).fit_idf([c.text for c in chunks])
    return HashingEmbedder()


def _print_table(title: str, results: dict[str, dict[str, float]]) -> None:
    metrics = list(next(iter(results.values())))
    print(f"\n{title}")
    print(f"{'mode':<8} " + " ".join(f"{m:>10}" for m in metrics))
    for mode, values in results.items():
        print(f"{mode:<8} " + " ".join(f"{v:>10.3f}" for v in values.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag_assistant")
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus"))
    p.add_argument("--embedder", choices=["hashing", "lsa", "wordvec"], default="lsa")
    p.add_argument("--vector-model", default="en_core_web_lg", help="spaCy package for --embedder wordvec")
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="answer a question from the corpus")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=3)
    ask.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ev = sub.add_parser("eval", help="compare retrieval modes on the labelled questions")
    ev.add_argument("--cases", default=str(ROOT / "evals" / "questions.json"))
    ev.add_argument("-k", type=int, default=3)
    std = sub.add_parser("eval-stdlib", help="held-out benchmark built from Python standard library docs")
    std.add_argument("-k", type=int, default=3)
    tune = sub.add_parser("eval-tuned", help="tune the fusion weight on a dev split, report on a held-out split")
    tune.add_argument("--cases", default=str(ROOT / "evals" / "questions.json"))
    args = p.parse_args(argv)

    if args.cmd == "eval-stdlib":
        import sys

        from .stdlib_eval import build_benchmark, run

        n = len(build_benchmark()[0])
        print(f"embedder: {args.embedder}, {n} passages from the standard library (Python {sys.version.split()[0]})")
        _print_table("stdlib summary-to-description questions", run(args.embedder, args.k, args.vector_model))
        return 0

    chunks = load_corpus(args.corpus)
    retriever = Retriever(chunks, make_embedder(args.embedder, chunks, args.vector_model))
    if args.cmd == "ask":
        hits = retriever.search(args.question, args.k, args.mode)
        print(ExtractiveAnswerer().answer(args.question, hits))
        for h in hits:
            print(f"  {h.score:.3f}  {h.chunk.id}")
        return 0

    if args.cmd == "eval-tuned":
        all_cases = load_cases(args.cases)
        dev, test = split_dev_test(all_cases)
        weight = tune_keyword_weight(retriever, dev)
        print(f"embedder: {args.embedder}; tuned BM25 weight on {len(dev)} dev questions: {weight}")
        for label, weights in (("default weight 1.0", 1.0), (f"tuned weight {weight}", weight)):
            def search(q: str, top: int, w: float = weights) -> list[Hit]:
                return retriever.search(q, top, "hybrid", keyword_weight=w)

            _print_table(f"hybrid, {label}, on {len(test)} held-out questions", {"hybrid": evaluate(search, test, 3)})

        def default(q: str, top: int) -> list[Hit]:
            return retriever.search(q, top, "hybrid", keyword_weight=1.0)

        def tuned(q: str, top: int) -> list[Hit]:
            return retriever.search(q, top, "hybrid", keyword_weight=weight)

        mean, low, high = paired_bootstrap_ci(reciprocal_ranks(default, test), reciprocal_ranks(tuned, test))
        verdict = "could be chance" if low <= 0 <= high else "unlikely to be chance"
        print(f"\nMRR change from tuning: {mean:+.3f}, 95% interval [{low:+.3f}, {high:+.3f}] ({verdict})")
        return 0

    cases = load_cases(args.cases)
    print(f"embedder: {args.embedder}, {len(chunks)} chunks, {len(cases)} questions")
    _print_table("all questions", compare_modes(retriever, cases, args.k))
    for kind in sorted({c.get("type", "") for c in cases} - {""}):
        subset: list[Case] = [c for c in cases if c.get("type") == kind]
        _print_table(f"{kind} questions ({len(subset)})", compare_modes(retriever, subset, args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
