# Tokens and context windows

Language models read tokens, which are pieces of words produced by a tokenizer, not whole words or characters. English text averages roughly four characters per token, but code, numbers and non-English text often need more. Every model has a context window, the maximum number of tokens covering the prompt and the reply together. Pricing and latency both grow with token count, so trimming irrelevant context saves money and time. When retrieved passages plus the question exceed the window, the lowest-ranked passages must be dropped first.
