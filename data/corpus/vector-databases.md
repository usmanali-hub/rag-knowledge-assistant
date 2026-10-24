# Vector databases

A vector database stores embeddings with metadata and answers nearest-neighbour queries quickly. Instead of comparing the query with every stored vector, it uses an approximate index such as HNSW graphs or inverted file lists, trading a small loss of recall for large speed gains. Metadata filters, for example restricting results to one customer or date range, are applied alongside the vector search. For a few thousand chunks a plain NumPy matrix is enough, and a dedicated database starts to pay off at millions of vectors or when many writers update the index.
