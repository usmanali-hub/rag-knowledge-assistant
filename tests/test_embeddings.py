import numpy as np
import pytest

from rag_assistant.embeddings import HashingEmbedder


def test_vectors_are_unit_length_and_deterministic():
    e = HashingEmbedder(dim=128)
    a = e.embed(["vector search with embeddings"])
    b = e.embed(["vector search with embeddings"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, b)


def test_similar_text_is_closer_than_unrelated_text():
    e = HashingEmbedder()
    q, near, far = e.embed(["docker image layers", "docker image caching layers", "cosine similarity"])
    assert q @ near > q @ far


def test_empty_text_gives_zero_vector_not_nan():
    out = HashingEmbedder(dim=16).embed([""])
    assert not np.isnan(out).any()
    assert np.count_nonzero(out) == 0


def test_invalid_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
