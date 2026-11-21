"""The per-app service container, reachable from request handlers via ``services()``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quart import current_app

if TYPE_CHECKING:
    from backend.app.core.config import Settings
    from backend.app.infra.crypto import PayloadCipher
    from backend.app.infra.monitor import Monitor
    from backend.app.infra.rate_limit import FailureThrottle, RateLimiter
    from backend.app.infra.usage_db import UsageDB
    from backend.app.services.embeddings import Embedder
    from backend.app.services.ingest import IngestService
    from backend.app.services.llm import LLMService
    from backend.app.services.rag import RagService
    from backend.app.services.vector_store import VectorStore


@dataclass
class Services:
    settings: Settings
    cipher: PayloadCipher
    limiter: RateLimiter
    login_throttle: FailureThrottle
    usage: UsageDB
    monitor: Monitor
    store: VectorStore
    embedder: Embedder
    llm: LLMService
    ingest: IngestService
    rag: RagService
    asset_version: int


def services() -> Services:
    svc: Services = current_app.extensions["helios"]
    return svc
