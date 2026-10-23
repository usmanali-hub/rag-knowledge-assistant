"""Command line: ``python -m rag_assistant ask "question"`` or ``... eval``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import compare_modes, load_cases
from .generate import ExtractiveAnswerer
from .retriever import Retriever, load_corpus

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag_assistant")
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus"))
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="answer a question from the corpus")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=3)
    ask.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ev = sub.add_parser("eval", help="compare retrieval modes on the labelled questions")
    ev.add_argument("--cases", default=str(ROOT / "evals" / "questions.json"))
    ev.add_argument("-k", type=int, default=3)
    args = p.parse_args(argv)

    retriever = Retriever(load_corpus(args.corpus))
    if args.cmd == "ask":
        hits = retriever.search(args.question, args.k, args.mode)
        print(ExtractiveAnswerer().answer(args.question, hits))
        for h in hits:
            print(f"  {h.score:.3f}  {h.chunk.id}")
        return 0

    results = compare_modes(retriever, load_cases(args.cases), args.k)
    print(f"{'mode':<8} " + " ".join(f"{m:>10}" for m in next(iter(results.values()))))
    for mode, metrics in results.items():
        print(f"{mode:<8} " + " ".join(f"{v:>10.3f}" for v in metrics.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
