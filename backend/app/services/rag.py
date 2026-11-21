"""Retrieval-augmented answering. ``RagService.search`` is the only builder of the /search response."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.config import Settings
from backend.app.infra.monitor import Monitor
from backend.app.infra.privacy import redact_pii
from backend.app.infra.rate_limit import RateLimiter
from backend.app.infra.usage_db import UsageDB
from backend.app.services.embeddings import Embedder
from backend.app.services.llm import DemoLLM, ImageInput, LLMService, Usage
from backend.app.services.vector_store import Hit, VectorStore

logger = logging.getLogger(__name__)

DEMO_BANNER = "Demo mode — results are simulated; no active LLM connection."
DEMO_PREFIX = "(Demo mode - no chat model available) "
DEMO_NO_MATCH = f"{DEMO_PREFIX}I couldn't find any confident matches in the documents (no documents matched the query)."
SNIPPET_CHARS = 400
# Tokens reserved against the daily budget before a model call (prompt + output estimate).
PROMPT_TOKEN_ESTIMATE = 2000


class BudgetExhausted(RuntimeError):
    """The daily token budget has been used up."""


@dataclass
class SearchQuery:
    query: str
    top_k: int
    thread_id: str
    images: list[ImageInput] = field(default_factory=list)
    file_type: str | None = None


def _snippet(text: str) -> str:
    return text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else "")


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMService,
        limiter: RateLimiter,
        monitor: Monitor,
        usage: UsageDB,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.limiter = limiter
        self.monitor = monitor
        self.usage = usage

    async def _demo_answer(self, hits: list[Hit]) -> str:
        s = self.settings
        best = max((h.score for h in hits), default=0.0)
        threshold = max(best * s.demo_match_ratio, s.demo_min_match_score)
        kept = [h for h in hits if h.score >= threshold]
        if not kept:
            return DEMO_NO_MATCH
        context = "\n\n".join(f"Source ({h.filename}): {_snippet(h.text)}" for h in kept)
        return DEMO_PREFIX + DemoLLM().generate(context)

    async def _llm_answer(self, query: SearchQuery, hits: list[Hit]) -> tuple[str, Usage]:
        if not self.llm.configured:
            return "Gemini API not configured (GEMINI_API_KEY missing).", Usage()
        reserve = self.settings.max_tokens + PROMPT_TOKEN_ESTIMATE
        reason = await self.limiter.reserve(reserve)
        if reason:
            raise BudgetExhausted(reason)
        context = "\n\n".join(f"Source ({h.filename}): {h.text}" for h in hits)
        try:
            result = await self.llm.answer(query.query, context, query.images)
        except BaseException:
            await self.limiter.settle(reserve, 0)
            raise
        await self.limiter.settle(reserve, result.usage.total_tokens)
        u = result.usage
        self.monitor.record_llm(u.prompt_tokens, u.cached_tokens, u.fallback)
        return result.answer, u

    async def search(self, query: SearchQuery) -> dict[str, Any]:
        s = self.settings
        started = time.perf_counter()

        t = time.perf_counter()
        vectors = await self.embedder.embed([query.query], is_query=True)
        embed_ms = int((time.perf_counter() - t) * 1000)
        hits = await self.store.search(vectors[0], query.top_k, query.file_type)

        t = time.perf_counter()
        if s.demo:
            answer = await self._demo_answer(hits)
            usage = Usage(model="demo")
        else:
            answer, usage = await self._llm_answer(query, hits)
        ai_ms = int((time.perf_counter() - t) * 1000)
        total_ms = int((time.perf_counter() - started) * 1000)
        model = usage.model or s.gemini_model or ""

        self._record(query, answer, hits, usage, model, embed_ms, ai_ms, total_ms)

        results = [
            {"doc_id": h.doc_id, "filename": h.filename, "chunk_text": _snippet(h.text), "score": h.score} for h in hits
        ]
        return {
            "results": results,
            "used_documents": [dict(r) for r in results],
            "demo": s.demo,
            "message": DEMO_BANNER if s.demo else "",
            "answer": answer,
            "sources": list(dict.fromkeys(h.filename for h in hits)),
            "embedding_provider": s.embedding_provider,
            "thread_id": query.thread_id,
            "embed_ms": embed_ms,
            "ai_ms": ai_ms,
            "total_ms": total_ms,
            "model": model,
        }

    def _record(
        self,
        query: SearchQuery,
        answer: str,
        hits: list[Hit],
        usage: Usage,
        model: str,
        embed_ms: int,
        ai_ms: int,
        total_ms: int,
    ) -> None:
        try:
            self.usage.log_chat(
                thread_id=query.thread_id,
                query=redact_pii(query.query),
                answer=redact_pii(answer),
                embed_ms=embed_ms,
                ai_ms=ai_ms,
                total_ms=total_ms,
                model=model,
                tokens_input=usage.prompt_tokens,
                tokens_output=usage.completion_tokens,
                doc_count=len(hits),
            )
            self.usage.log_event(
                session_id=None,
                endpoint="/search",
                model=model,
                tokens_input=usage.prompt_tokens,
                tokens_output=usage.completion_tokens,
                tokens_cached=usage.cached_tokens,
                latency_ms=total_ms,
                status_code=200,
                error_kind="fallback" if usage.fallback else None,
            )
        except Exception:
            logger.warning("usage logging failed", exc_info=True)
