# Evaluating retrieval

Evaluate the retriever before tuning prompts. Build a small set of real questions, and for each one record which document contains the answer. Recall at k measures the share of questions where a relevant document appears in the top k results. Mean reciprocal rank rewards placing the first relevant result higher, scoring one for rank one, a half for rank two and so on. If recall is low the language model never sees the evidence, so no prompt can fix it. Keep the evaluation set fixed so that changes to chunking, embeddings or fusion can be compared fairly.
