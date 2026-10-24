# Caching model calls

Calling a model is slow and billed per token, so repeated identical requests should be answered from a cache. An exact-match cache keys on the full prompt and model settings and returns the stored reply instantly. A semantic cache compares embeddings and reuses an answer for a question that is close enough, which saves more calls but risks returning a wrong answer for a subtly different question. Set a time to live so stale answers expire, and never cache replies that contain personal data. Provider-side prompt caching can also discount a long, repeated prefix.
