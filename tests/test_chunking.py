from itertools import pairwise

from backend.app.services.chunking import chunk_text


def test_empty_input():
    assert chunk_text("") == [] and chunk_text("   \n ") == []


def test_short_text_single_chunk():
    assert chunk_text("One. Two. Three.") == ["One. Two. Three."]


def test_splits_on_sentences_with_overlap():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(30)]
    chunks = chunk_text(" ".join(sentences), chunk_size=40, overlap=14)
    assert len(chunks) > 1
    for first, second in pairwise(chunks):
        assert first.split(". ")[-1].rstrip(".") in second  # last sentence is carried over
    assert all(len(c.split()) <= 60 for c in chunks)


def test_sentence_longer_than_chunk_is_split_by_words():
    text = " ".join(f"w{i}" for i in range(500))  # no punctuation at all
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 5
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[0].startswith("w0") and chunks[-1].endswith("w499")


def test_whitespace_is_normalised():
    assert chunk_text("a\n\n b\t c.") == ["a b c."]
