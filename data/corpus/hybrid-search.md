# Hybrid search and reranking

Keyword and vector search fail in different ways, so combining them is more robust than either alone. Reciprocal rank fusion merges the two ranked lists by adding one divided by a constant plus rank for each list in which a document appears. Because it uses ranks rather than raw scores, the two scoring scales never have to be calibrated against each other. A reranker, usually a cross-encoder, can then rescore the top candidates by reading the query and passage together. Reranking is slower, so it is applied only to a short list.
