"""Text embeddings. DEMO uses deterministic hash vectors; everything else uses Gemini.

There is deliberately no silent fallback outside DEMO: a provider failure raises
``EmbeddingError`` so garbage vectors can never reach the index.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import Any

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)

_BATCH = 100


class EmbeddingError(RuntimeError):
    """Embeddings could not be produced."""


def hash_embedding(text: str, dim: int) -> list[float]:
    """Deterministic unit vector derived from md5 digests (DEMO only)."""
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).digest()
    values: list[float] = []
    while len(values) < dim:
        values.extend(b / 255.0 for b in digest)
        digest = hashlib.md5(digest, usedforsecurity=False).digest()
    values = values[:dim]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0.0 else [v / norm for v in vec]


class Embedder:
    def __init__(self, settings: Settings, client: Any | None) -> None:
        self.settings = settings
        self._client = client

    @property
    def provider(self) -> str:
        return self.settings.embedding_provider

    @property
    def ready(self) -> bool:
        return self.settings.demo or self._client is not None

    async def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.demo:
            dim = self.settings.embedding_dim
            return await asyncio.to_thread(lambda: [hash_embedding(t, dim) for t in texts])
        if self._client is None:
            raise EmbeddingError("Gemini API key is not configured")
        return await self._embed_gemini(texts, is_query)

    async def _embed_gemini(self, texts: list[str], is_query: bool) -> list[list[float]]:
        from google.genai import types

        client = self._client
        if client is None:
            raise EmbeddingError("Gemini API key is not configured")
        s = self.settings
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
            output_dimensionality=s.embedding_dim,
        )
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            batch = texts[start : start + _BATCH]
            try:
                resp = await asyncio.wait_for(
                    client.aio.models.embed_content(model=s.gemini_embedding_model, contents=batch, config=config),
                    timeout=s.embedding_timeout,
                )
            except Exception as exc:
                logger.error("Gemini embedding call failed: %s", type(exc).__name__)
                raise EmbeddingError("embedding provider request failed") from exc
            embeddings = resp.embeddings or []
            if len(embeddings) != len(batch):
                raise EmbeddingError("embedding provider returned an unexpected number of vectors")
            for item in embeddings:
                vec = list(item.values or [])
                if len(vec) != s.embedding_dim:
                    raise EmbeddingError(f"embedding has {len(vec)} dimensions, expected {s.embedding_dim}")
                out.append(_l2_normalize(vec) if s.embedding_dim < 3072 else vec)
        return out
