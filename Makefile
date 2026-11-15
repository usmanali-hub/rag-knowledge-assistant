# From a fresh clone to a checked, evaluated and timed retrieval pipeline:  make demo
PY ?= python

.PHONY: install demo test
install:
	$(PY) -m pip install -e ".[dev]"

demo:
	$(PY) -m rag_assistant ask "Why normalise vectors to unit length?"
	$(PY) -m rag_assistant eval
	$(PY) scripts/latency_benchmark.py --repeats 3

test:
	$(PY) -m pytest -q
