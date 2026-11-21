"""Helios: a Quart RAG assistant. Use :func:`create_app` to build the application."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from quart import Quart, Response, g, jsonify, request
from quart_cors import cors
from werkzeug.exceptions import HTTPException

from backend.app.api import admin, chat, console, documents, health, session
from backend.app.core.config import ConfigError, Settings
from backend.app.core.container import Services
from backend.app.core.logging import setup_logging
from backend.app.core.security import add_security_headers, is_search_bearer
from backend.app.infra.crypto import CryptoError, PayloadCipher
from backend.app.infra.monitor import Monitor
from backend.app.infra.rate_limit import FailureThrottle, RateLimiter
from backend.app.infra.tasks import drain
from backend.app.infra.usage_db import UsageDB
from backend.app.services.embeddings import Embedder
from backend.app.services.gemini import make_client
from backend.app.services.ingest import IngestService
from backend.app.services.llm import LLMService
from backend.app.services.rag import RagService
from backend.app.services.vector_store import VectorStore

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

RATE_LIMITED_PATHS = ("/search", "/upload")
DEFAULT_BODY_LIMIT = 64 * 1024
SEARCH_BODY_LIMIT = chat.MAX_IMAGES * (chat.MAX_IMAGE_BYTES * 4 // 3 + 64) + DEFAULT_BODY_LIMIT
UPLOAD_PATHS = ("/upload", "/admin/upload_vectors")
DEMO_SESSION_COOKIE = "demo_session_id"
DEMO_SESSION_MAX_AGE = 7 * 24 * 3600
LIMIT_DETAIL = "Demo-loftet er ramt. Kontakt os for fuld adgang."
BUDGET_DETAIL = "Demoens daglige token-budget er opbrugt. Prøv igen i morgen."


def _build_cipher(settings: Settings) -> PayloadCipher:
    try:
        return PayloadCipher.from_setting(settings.payload_encryption_key)
    except CryptoError as exc:
        if not settings.demo:
            raise ConfigError(str(exc)) from exc
        logger.warning("%s; payload encryption is disabled in DEMO", exc)
        return PayloadCipher(None)


def _budget_file(settings: Settings) -> Path:
    if settings.qdrant_path == ":memory:":
        return Path(tempfile.gettempdir()) / "helios-budget.json"
    return Path(settings.qdrant_path) / "budget.json"


def create_app(
    settings: Settings | None = None,
    *,
    qdrant_client: QdrantClient | None = None,
    gemini_client: Any | None = None,
    budget_file: str | Path | None = None,
) -> Quart:
    settings = settings or Settings.from_env()
    setup_logging(settings)
    settings.validate()
    if os.environ.get("QDRANT_FORCE_RECREATE"):
        logger.warning("QDRANT_FORCE_RECREATE is ignored: the collection is never recreated automatically")

    app = Quart("backend.app", template_folder="../templates", static_folder="../static")
    app.config["MAX_CONTENT_LENGTH"] = max(settings.max_upload_bytes, SEARCH_BODY_LIMIT)
    if settings.cors_allow_origins:
        app = cors(app, allow_origin=list(settings.cors_allow_origins))

    cipher = _build_cipher(settings)
    client = gemini_client if gemini_client is not None else make_client(settings)
    store = VectorStore(settings, cipher, qdrant_client)
    embedder = Embedder(settings, client)
    llm = LLMService(settings, client)
    limiter = RateLimiter(
        settings.rate_limit_per_min,
        settings.rate_limit_per_day,
        settings.rate_limit_per_session,
        settings.daily_token_budget,
        budget_file or _budget_file(settings),
    )
    monitor = Monitor()
    usage = UsageDB(settings.database_url)
    app.extensions["helios"] = Services(
        settings=settings,
        cipher=cipher,
        limiter=limiter,
        login_throttle=FailureThrottle(),
        usage=usage,
        monitor=monitor,
        store=store,
        embedder=embedder,
        llm=llm,
        ingest=IngestService(store, embedder),
        rag=RagService(settings, embedder, store, llm, limiter, monitor, usage),
        asset_version=int(time.time()),
    )

    for blueprint in (chat.bp, documents.bp, admin.bp, health.bp, session.bp, console.bp):
        app.register_blueprint(blueprint)
    _register_hooks(app, settings, limiter, monitor)
    _register_errors(app)

    @app.before_serving
    async def _startup() -> None:
        await usage.initialize()
        import asyncio

        await asyncio.to_thread(store.connect)
        await store.encrypt_legacy_payloads()

    @app.after_serving
    async def _shutdown() -> None:
        await drain()
        await usage.shutdown()
        await store.close()

    return app


def _too_many(error: str, detail: str, limit: str) -> tuple[Response, int]:
    return jsonify({"error": error, "detail": detail, "limit": limit}), 429


def _register_hooks(app: Quart, settings: Settings, limiter: RateLimiter, monitor: Monitor) -> None:
    from backend.app.core.security import client_ip

    @app.before_request
    async def guard() -> tuple[Response, int] | None:
        g.start = time.perf_counter()
        path = request.path
        if path in UPLOAD_PATHS:
            cap = settings.max_upload_bytes
        elif path == "/search":
            cap = SEARCH_BODY_LIMIT
        else:
            cap = DEFAULT_BODY_LIMIT
        if request.content_length is not None and request.content_length > cap:
            return jsonify({"error": "payload_too_large"}), 413

        g.bearer = is_search_bearer()
        existing = request.cookies.get(DEMO_SESSION_COOKIE, "")
        g.session_id = existing or uuid.uuid4().hex
        g.new_session = not existing and not g.bearer and path in ("/", *RATE_LIMITED_PATHS)

        if path not in RATE_LIMITED_PATHS:
            return None
        if not g.bearer:  # trusted API consumers (the portfolio proxy) share one IP and session
            reason = await limiter.check_ip(client_ip()) or await limiter.check_session(g.session_id)
            if reason:
                return _too_many("rate_limited", LIMIT_DETAIL, reason)
        if path == "/search":
            reason = await limiter.check_budget()
            if reason:
                return _too_many("budget_exhausted", BUDGET_DETAIL, reason)
        return None

    @app.after_request
    async def finish(response: Response) -> Response:
        response = await add_security_headers(response)
        if getattr(g, "new_session", False):
            response.set_cookie(
                DEMO_SESSION_COOKIE,
                g.session_id,
                max_age=DEMO_SESSION_MAX_AGE,
                httponly=True,
                secure=True,
                samesite="Lax",
            )
        start = getattr(g, "start", None)
        if start is not None:
            latency = (time.perf_counter() - start) * 1000
            rule = request.url_rule.rule if request.url_rule else "unmatched"
            monitor.record_request(rule, latency, response.status_code)
            logger.info("%s %s %d - %.1fms", request.method, rule, response.status_code, latency)
        return response


def _register_errors(app: Quart) -> None:
    @app.errorhandler(HTTPException)
    async def http_error(exc: HTTPException) -> tuple[Response, int]:
        messages = {404: "Resource not found", 401: "Unauthorized", 413: "payload_too_large"}
        code = exc.code or 500
        return jsonify({"error": messages.get(code, exc.name)}), code

    @app.errorhandler(Exception)
    async def unhandled(exc: Exception) -> tuple[Response, int]:
        logger.error("Unhandled %s on %s %s", type(exc).__name__, request.method, request.path, exc_info=exc)
        return jsonify({"error": "internal_error"}), 500
