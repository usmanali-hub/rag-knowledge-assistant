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
| paraphrase | pretrained word vectors | 0.875 / 0.740 | 0.812 / 0.714 | 0.812 / 0.740 |

What this shows:

- **Paraphrased questions are much harder** than lexical ones: top-1 accuracy on them is about 0.6 for the best mode, against 0.94 for lexical.
- **The hashing embedder is nearly useless on paraphrases** (recall@3 0.344), because it only sees shared words. Fusing it with BM25 makes results *worse* than BM25 alone (0.688 vs 0.875), so a weak ranker can hurt a hybrid.
- **LSA** (TF-IDF plus SVD fitted on the corpus, so it learns which words co-occur) lifts dense search to BM25's level on paraphrases, but hybrid does not beat BM25 here. I did not tune fusion weights to the eval set, because that would overfit 50 questions.
- **Limits:** the corpus is small and I wrote both documents and questions, so treat these numbers as a comparison between methods on one dataset. The standard-library benchmark below reduces that bias. A trained embedding model and a cross-encoder reranker are the next step; the model weights could not be downloaded in my build environment, so that comparison is still open.

CI runs `test_retrieval_regression_guards` and a check that LSA clearly beats hashing on paraphrases,
so a change to chunking, embeddings or fusion cannot quietly make retrieval worse.

### Held-out check on text I did not write

The 50 questions above were written by me, which is a weakness. `python -m rag_assistant eval-stdlib`
builds a second benchmark from the Python standard library docs: for each documented function, the
first sentence of its docstring (with the function's name removed) is the query and the rest of the
docstring is the passage to find. It is synthetic, but the text comes from the library authors.
85 passages from Python 3.13, recall@3 / MRR:

| embedder | BM25 | dense | hybrid |
|---|---|---|---|
| hashing | 0.682 / 0.573 | 0.400 / 0.333 | 0.576 / 0.443 |
| LSA | 0.682 / 0.573 | 0.694 / 0.567 | 0.694 / 0.565 |
| pretrained word vectors | 0.682 / 0.573 | 0.471 / 0.376 | 0.588 / 0.459 |

The ordering matches my own questions: BM25 is a strong baseline, hashing vectors are weak, LSA
matches BM25, and hybrid does not clearly win. Passages differ between Python versions, so numbers
will vary slightly with the interpreter.

### Pretrained vectors (and why they did not win)

`--embedder wordvec` averages real pretrained GloVe word vectors (300-d, the vectors shipped in
spaCy's `en_core_web_lg` package, read straight from the package files), weighted by IDF. These are
trained on a huge external corpus, so "car" and "automobile" are close (cosine 0.75) even though
they never co-occur in my documents. I expected them to beat LSA. They did not: averaging word vectors
blurs short technical passages, so on the paraphrase questions they reach 0.812 recall@3 against 0.875
for LSA and BM25, and on the standard-library benchmark 0.471 against 0.694 for LSA. (The smaller
`en_core_web_md` package keeps only 20,000 vectors, which made things worse, so I use `lg`.) A
sentence-transformer model would likely do better; its weights are not reachable from my build
environment, so that comparison is still open. This mode is not run in CI because the model is
~400 MB.

```bash
pip install spacy
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
python -m rag_assistant --embedder wordvec eval
```

### Tuning the fusion weight without fooling myself

`python -m rag_assistant eval-tuned` picks the BM25 weight in rank fusion on half of the 50 questions
(dev) and reports on the other half (held-out). It also gives a paired bootstrap 95% interval for the
change in MRR, so a gain that is inside the noise is labelled as such:

| embedder | tuned weight | MRR change on held-out questions | 95% interval | verdict |
|---|---|---|---|---|
| hashing | 3.0 | +0.127 | [+0.013, +0.260] | unlikely to be chance |
| LSA | 0.5 | -0.047 | [-0.107, +0.000] | could be chance |
| pretrained vectors | 0.5 | -0.080 | [-0.160, -0.013] | tuning made it worse |

Tuning helps when one ranker is much weaker (hashing: trust BM25 more). For the stronger embedders the
weight picked on 25 dev questions did not transfer, and for pretrained vectors it overfit and hurt the
held-out half. The default weight of 1.0 is the safer choice at this sample size.

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
- Answer-level evaluation (faithfulness, citation accuracy)
- FastAPI service and Docker image
