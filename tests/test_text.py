import pytest

from rag_assistant.text import chunk_words, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Quick, brown FOX!") == ["quick", "brown", "fox"]
    assert tokenize("The cat", drop_stopwords=False) == ["the", "cat"]


def test_chunk_words_overlap_and_coverage():
    text = " ".join(str(i) for i in range(25))
    chunks = chunk_words(text, "doc.md", size=10, overlap=3)
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[-1].text.split()[-1] == "24"  # nothing dropped at the end


def test_chunk_words_short_and_empty():
    assert len(chunk_words("a b c", "d.md", size=10, overlap=2)) == 1
    assert chunk_words("   ", "d.md") == []


def test_chunk_ids_include_source():
    assert chunk_words("a b", "x.md")[0].id == "x.md#0"


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1)])
def test_chunk_words_validates_arguments(size, overlap):
    with pytest.raises(ValueError):
        chunk_words("a b c", "d.md", size=size, overlap=overlap)
