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
python -m rag_assistant eval                      # LSA embeddings (default)
python -m rag_assistant --embedder hashing eval   # compare
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
| Vector search | `embeddings.LSAEmbedder`, `HashingEmbedder` | Offline LSA (TF-IDF + SVD) and signed feature hashing; `Embedder` protocol for real models |
| Hybrid | `retriever.reciprocal_rank_fusion` | Merges ranks, so BM25 and cosine scores need no calibration |
| Generate | `generate.LLMAnswerer`, `ExtractiveAnswerer` | Grounded prompt with `[source#chunk]` citations; wraps any `complete(prompt) -> str` |
| Evaluate | `evaluate.evaluate` | recall@k and MRR over labelled questions |

## Evaluation

`python -m rag_assistant eval` compares retrieval modes on 50 labelled questions over 20 short
documents (40 chunks) in `data/corpus/`. Questions come in two kinds:

- **lexical** (18): they reuse words from the answer passage
- **paraphrase** (32): written to avoid the passage's key terms, e.g. "How do I stop an autonomous
  assistant from running forever?" for a passage about a *maximum number of steps*

Results at k=3 (recall@3 / MRR):

| questions | embedder | BM25 | dense | hybrid (RRF) |
|---|---|---|---|---|
| lexical | LSA | 1.000 / 0.972 | 1.000 / 0.944 | 1.000 / 0.972 |
| paraphrase | LSA | 0.875 / 0.740 | 0.875 / 0.688 | 0.875 / 0.708 |
| paraphrase | hashing | 0.875 / 0.740 | 0.344 / 0.328 | 0.688 / 0.516 |

What this shows:

- **Paraphrased questions are much harder** than lexical ones: top-1 accuracy on them is about 0.6 for the best mode, against 0.94 for lexical.
- **The hashing embedder is nearly useless on paraphrases** (recall@3 0.344), because it only sees shared words. Fusing it with BM25 makes results *worse* than BM25 alone (0.688 vs 0.875), so a weak ranker can hurt a hybrid.
- **LSA** (TF-IDF plus SVD fitted on the corpus, so it learns which words co-occur) lifts dense search to BM25's level on paraphrases, but hybrid does not beat BM25 here. I did not tune fusion weights to the eval set, because that would overfit 50 questions.
- **Limits:** the corpus is small and I wrote both documents and questions, so treat the numbers as a comparison between methods on one dataset, not as general performance. A trained embedding model and a cross-encoder reranker are the next step; the model weights could not be downloaded in my build environment, so that comparison is still open.

CI runs `test_retrieval_regression_guards` and a check that LSA clearly beats hashing on paraphrases,
so a change to chunking, embeddings or fusion cannot quietly make retrieval worse.

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

- Trained sentence-transformer embedder behind the existing `Embedder` protocol, evaluated on the same questions
- Cross-encoder reranking of the top candidates
- Larger corpus with answer-level evaluation (faithfulness, citation accuracy)
- FastAPI service and Docker image
