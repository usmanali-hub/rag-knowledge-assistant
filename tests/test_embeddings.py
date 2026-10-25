import numpy as np
import pytest

from rag_assistant.embeddings import HashingEmbedder, LSAEmbedder


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


def test_lsa_connects_words_that_share_contexts():
    texts = [
        "car engine fuel", "automobile engine fuel", "car wheel road", "automobile wheel road",
        "banana fruit sweet", "apple fruit sweet",
    ]
    e = LSAEmbedder.fit(texts, dim=3)
    q, other_word_same_topic, unrelated = e.embed(["car", "automobile", "banana"])
    assert q @ other_word_same_topic > q @ unrelated  # no shared word, yet closer


def test_lsa_unit_vectors_and_unknown_words():
    e = LSAEmbedder.fit(["alpha beta", "beta gamma", "gamma delta"], dim=2)
    assert np.allclose(np.linalg.norm(e.embed(["alpha beta"]), axis=1), 1.0)
    assert np.count_nonzero(e.embed(["zzzz"])) == 0  # out-of-vocabulary text gives a zero vector


def test_lsa_requires_text():
    with pytest.raises(ValueError):
        LSAEmbedder.fit([])


def test_word_vector_embedder_with_a_fake_lookup():
    from rag_assistant.embeddings import WordVectorEmbedder

    table = {"car": np.array([1.0, 0.0]), "automobile": np.array([0.9, 0.1]), "banana": np.array([0.0, 1.0])}
    e = WordVectorEmbedder(table.get, dim=2)
    a, b, c = e.embed(["car", "automobile", "banana"])
    assert a @ b > a @ c
    assert np.count_nonzero(e.embed(["unknownword"])) == 0  # no known words gives a zero vector, not NaN


def test_word_vector_idf_downweights_common_words():
    from rag_assistant.embeddings import WordVectorEmbedder

    table = {"common": np.array([1.0, 0.0]), "rare": np.array([0.0, 1.0])}
    plain = WordVectorEmbedder(table.get, dim=2)
    fitted = plain.fit_idf(["common one", "common two", "common three", "rare four"])
    v_plain = plain.embed(["common rare"])[0]
    v_fit = fitted.embed(["common rare"])[0]
    assert v_plain[0] == pytest.approx(v_plain[1])
    assert v_fit[1] > v_fit[0]
