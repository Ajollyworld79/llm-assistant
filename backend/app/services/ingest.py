"""Upload pipeline: parse -> chunk -> embed -> upsert."""

from __future__ import annotations

import asyncio
import datetime
import re
import time
import uuid
from dataclasses import dataclass

from backend.app.services.chunking import chunk_text
from backend.app.services.embeddings import Embedder
from backend.app.services.parser import EmptyDocument, file_type, parse_content_async
from backend.app.services.vector_store import VectorStore

_FILENAME_STRIP = re.compile(r"[\x00-\x1f\x7f\"'<>&`]")


def sanitize_filename(name: object) -> str:
    """Make a client-supplied filename safe to store and render."""
    if isinstance(name, str):
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
        name = " ".join(_FILENAME_STRIP.sub("", name).split())[:120].strip()
        if name and name.strip("."):
            return name
    return f"upload-{int(time.time())}"


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    filename: str
    chunks: int


class IngestService:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    async def ingest_file(self, filename: str, content: bytes) -> IngestResult:
        text = await parse_content_async(filename, content)
        return await self.ingest_text(filename, text)

    async def ingest_text(self, filename: str, text: str) -> IngestResult:
        chunks = await asyncio.to_thread(chunk_text, text)
        if not chunks:
            raise EmptyDocument("no extractable text")
        vectors = await self.embedder.embed(chunks)
        doc_id = str(uuid.uuid4())
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        count = await self.store.upsert_chunks(doc_id, filename, file_type(filename), stamp, chunks, vectors)
        return IngestResult(doc_id, filename, count)
