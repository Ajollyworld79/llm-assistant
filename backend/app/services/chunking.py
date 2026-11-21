"""Sentence-aware text chunking with word overlap."""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long(sentence: list[str], size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    pieces = []
    for start in range(0, len(sentence), step):
        pieces.append(" ".join(sentence[start : start + size]))
        if start + size >= len(sentence):
            break
    return pieces


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    """Split ``text`` into chunks of about ``chunk_size`` words.

    Chunks break on sentence boundaries and carry roughly ``overlap`` words of the
    previous chunk. A single sentence longer than ``chunk_size`` words is split by words.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    overlap = min(overlap, chunk_size - 1)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(" ".join(current))
        current, current_words = [], 0

    for sentence in _SENTENCE_SPLIT.split(text):
        words = sentence.split()
        if len(words) > chunk_size:
            # Already windowed with overlap, so each window is a chunk on its own.
            flush()
            chunks.extend(_split_long(words, chunk_size, overlap))
            continue
        if not current or current_words + len(words) <= chunk_size:
            current.append(sentence)
            current_words += len(words)
            continue
        chunks.append(" ".join(current))
        carry: list[str] = []
        carried = 0
        for previous in reversed(current):
            if carried >= overlap:
                break
            carry.insert(0, previous)
            carried += len(previous.split())
        current = [*carry, sentence]
        current_words = sum(len(s.split()) for s in current)
    flush()
    return chunks
