from rag_assistant.bm25 import BM25


def test_matching_document_scores_highest():
    docs = ["docker image layers", "cosine similarity of vectors", "gradient descent step"]
    scores = BM25(docs).scores("cosine vectors")
    assert scores.index(max(scores)) == 1


def test_unknown_terms_score_zero():
    assert BM25(["alpha beta"]).scores("zzz") == [0.0]


def test_rare_term_outweighs_common_term():
    docs = ["data data model", "data pipeline", "data cleaning", "unicorn"]
    bm = BM25(docs)
    assert bm.scores("unicorn")[3] > bm.scores("data")[0]


def test_term_frequency_saturates():
    few = BM25(["spam eggs", "other words"]).scores("spam")[0]
    many = BM25(["spam " * 30 + "eggs", "other words"]).scores("spam")[0]
    assert many < few * 3  # thirty times the frequency is far from thirty times the score


def test_empty_collection():
    assert BM25([]).scores("anything") == []
