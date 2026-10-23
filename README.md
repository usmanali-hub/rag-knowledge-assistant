# rag-knowledge-assistant

![CI](https://github.com/usmanali-hub/rag-knowledge-assistant/actions/workflows/ci.yml/badge.svg)

A small, fully tested retrieval-augmented generation (RAG) toolkit in Python: chunking, BM25 keyword
search, vector search, hybrid fusion, grounded prompts with citations, and an evaluation harness that
measures retrieval quality. It runs offline with only NumPy, and each piece sits behind a small
interface so a real embedding model or LLM can be plugged in.

## Quick start

```bash
pip install -e ".[dev]"
python -m rag_assistant ask "Why normalise vectors to unit length?"
python -m rag_assistant eval
pytest
```

```
$ python -m rag_assistant ask "Why normalise vectors?"
Normalising vectors to unit length turns cosine similarity into a plain dot product, which is fast
to compute for thousands of chunks. [embeddings.md#0]
```

## How it works

| Stage | Module | Notes |
|---|---|---|
| Chunk | `text.chunk_words` | Word windows with overlap; validated arguments |
| Keyword search | `bm25.BM25` | Written from the formula: IDF, saturating term frequency, length normalisation |
| Vector search | `embeddings.HashingEmbedder` | Offline signed feature hashing with bigrams; `Embedder` protocol for real models |
| Hybrid | `retriever.reciprocal_rank_fusion` | Merges ranks, so BM25 and cosine scores need no calibration |
| Generate | `generate.LLMAnswerer`, `ExtractiveAnswerer` | Grounded prompt with `[source#chunk]` citations; wraps any `complete(prompt) -> str` |
| Evaluate | `evaluate.evaluate` | recall@k and MRR over labelled questions |

## Evaluation

`python -m rag_assistant eval` compares the three retrieval modes on 18 labelled questions over 8
short documents in `data/corpus/`:

| mode | recall@1 | recall@3 | MRR (k=3) |
|---|---|---|---|
| BM25 | 0.944 | 1.000 | 0.972 |
| dense (hashing) | 0.944 | 1.000 | 0.972 |
| hybrid (RRF) | 1.000 | 1.000 | 1.000 |

**Read these numbers carefully.** The corpus is tiny and the questions were written against it, so
the scores are close to ceiling and show that the harness works, not that the system is state of the
art. The hashing embedder is lexical, so "dense" here does not capture synonyms the way a trained
model would. The useful part is the method: fixed labelled questions, comparable metrics, and a CI
regression guard (`test_hybrid_is_not_worse_than_either_single_mode`) so a change to chunking or
fusion cannot quietly make retrieval worse.

## Using a real model

```python
from rag_assistant.generate import LLMAnswerer
from rag_assistant.retriever import Retriever, load_corpus

retriever = Retriever(load_corpus("data/corpus"))        # pass embedder=YourEmbedder() for real vectors
answerer = LLMAnswerer(lambda prompt: call_your_llm(prompt))

question = "What does a cross-encoder reranker do?"
print(answerer.answer(question, retriever.search(question, k=3)))
```

The prompt tells the model to answer only from the context, say when it doesn't know, and cite sources.
Retrieved text is treated as untrusted data (see `data/corpus/prompt-safety.md`).

## Design decisions

- **Retrieval is measured before generation is tuned.** If the right chunk never reaches the prompt, no prompt can fix it.
- **Offline by default.** Tests and CI are deterministic and need no API keys or model downloads.
- **Ranks, not scores, for fusion.** Reciprocal rank fusion avoids comparing unrelated score scales.
- **Strict typing.** `mypy --strict`, ruff and pytest all run in CI.

## Roadmap

- Sentence-transformer embedder behind the existing `Embedder` protocol
- Cross-encoder reranking of the top candidates
- Larger corpus with answer-level evaluation (faithfulness, citation accuracy)
- FastAPI service and Docker image
