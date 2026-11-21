"""Operator console: HTML shell plus JSON endpoints for status, metrics and chat history."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from quart import Blueprint, Response, jsonify, render_template, request

from backend.app.core.container import services
from backend.app.core.security import require_admin
from backend.app.infra.demo_lock import reject_if_locked
from backend.app.infra.model_catalog import catalog_for_viewer
from backend.app.infra.monitor import rss_mb
from backend.app.services.prompt import CHAT_ANSWER_SCHEMA, SYSTEM_PROMPT

logger = logging.getLogger(__name__)
bp = Blueprint("console", __name__, url_prefix="/backend")

_START = time.time()
_COUNT_TTL = 30.0
_count_cache: dict[str, tuple[float, int | None]] = {}


@bp.get("/")
async def index() -> str:
    svc = services()
    return await render_template("backend.html", locked=svc.settings.demo_locked, asset_v=svc.asset_version)


# Overview data is public by design (the console's signed-out view shows system status);
# it is allow-listed and aggregate only. The prompt and conversations stay operator-only.
@bp.get("/api/system")
async def api_system() -> Response:
    s = services().settings
    git_sha = os.environ.get("GIT_SHA") or os.environ.get("RAILWAY_GIT_COMMIT_SHA") or None
    return jsonify(
        {
            "model_primary": s.gemini_model or "not_configured",
            "model_fallback": s.gemini_model_fallback or "not_configured",
            "embedding_provider": s.embedding_provider,
            "embedding_model": s.gemini_embedding_model,
            "embedding_dim": s.embedding_dim,
            "llm_provider": "Google Gemini",
            "git_sha": git_sha[:12] if git_sha else None,
            "deployed_at": os.environ.get("RAILWAY_DEPLOYMENT_STARTED_AT", ""),
            "uptime_seconds": int(time.time() - _START),
            "demo_locked": s.demo_locked,
            "config": s.public_view,
        }
    )


async def _cached(key: str, fn: Callable[[], Awaitable[int | None]]) -> int | None:
    now = time.monotonic()
    hit = _count_cache.get(key)
    if hit and now - hit[0] < _COUNT_TTL:
        return hit[1]
    try:
        value = await fn()
    except Exception:
        logger.debug("count %s failed", key, exc_info=True)
        value = None
    _count_cache[key] = (now, value)
    return value


@bp.get("/api/connections")
async def api_connections() -> Response:
    svc = services()
    s = svc.settings
    now = datetime.now(UTC).isoformat()

    async def qdrant_points() -> int | None:
        points = (await svc.store.status()).get("points_count")
        return int(points) if points is not None else None

    qdrant_up = svc.store.available and svc.store.collection_ready
    postgres = svc.usage.backend_kind == "postgres"
    return jsonify(
        {
            "qdrant": {
                "status": "up" if qdrant_up else "down",
                "points_count": await _cached("qdrant", qdrant_points) if qdrant_up else None,
                "last_check_at": now,
            },
            "gemini": {"status": "up" if s.gemini_api_key and s.gemini_model else "unconfigured", "last_check_at": now},
            "embedding_model": {
                "status": "configured" if svc.embedder.ready else "unconfigured",
                "provider": s.embedding_provider,
                "model": s.gemini_embedding_model,
                "last_check_at": now,
            },
            "logger": {"status": "writing" if s.log_to_file else "stdout_only", "last_check_at": now},
            "usage_db": {
                "status": "up" if postgres else "memory_fallback",
                "backend": svc.usage.backend_kind,
                "database_url_set": svc.usage.configured,
                "last_check_at": now,
            },
            "chat_db": {
                "status": "up" if postgres else "memory_fallback",
                "backend": svc.usage.backend_kind,
                "count": await _cached("chat_events", _count_chats),
                "last_check_at": now,
            },
        }
    )


async def _count_chats() -> int:
    return await services().usage.count_chats()


@bp.get("/api/metrics")
async def api_metrics() -> Response:
    svc = services()
    stats = svc.monitor.stats()
    return jsonify(
        {
            "memory_mb": rss_mb(),
            "requests_total": stats["requests"]["total"],
            "requests_success": stats["requests"]["success"],
            "requests_failed": stats["requests"]["failed"],
            "avg_latency_ms": stats["performance"]["avg_latency_ms"],
            "uptime_seconds": stats["uptime_seconds"],
            "cache_hit_ratio_pct": round(svc.monitor.cache_hit_ratio * 100, 1),
            "fallback_invocations": svc.monitor.fallback_invocations,
            "rate_limit": svc.limiter.snapshot(),
        }
    )


@bp.get("/api/prompt")
@require_admin
async def api_prompt() -> Response:
    return jsonify(
        {
            "system_prompt": SYSTEM_PROMPT,
            "schema": CHAT_ANSWER_SCHEMA,
            "example_context": "(handbook.pdf) Adgangskoder skal ændres hver 90. dag og må ikke genbruges.",
            "example_query": "Hvor ofte skal jeg skifte adgangskode?",
        }
    )


@bp.get("/api/models")
async def api_models() -> Response:
    s = services().settings
    return jsonify(
        {
            "models": catalog_for_viewer(s.gemini_model, s.gemini_model_fallback),
            "active_primary": s.gemini_model,
            "active_fallback": s.gemini_model_fallback,
        }
    )


@bp.get("/api/usage/summary")
async def api_usage_summary() -> Response:
    usage = services().usage
    return jsonify({**await usage.query_summary(), "backend": usage.backend_kind})


@bp.get("/api/usage/timeseries")
async def api_usage_timeseries() -> Response | tuple[Response, int]:
    metric = request.args.get("metric", "requests")
    range_label = request.args.get("range", "24h")
    if metric not in ("requests", "tokens", "cache_hit_ratio", "fallback_rate"):
        return jsonify({"error": "invalid_metric"}), 400
    if range_label not in ("24h", "7d", "30d"):
        return jsonify({"error": "invalid_range"}), 400
    series = await services().usage.query_timeseries(metric, range_label)
    return jsonify({"metric": metric, "range": range_label, "series": series})


@bp.get("/api/conversations")
@require_admin
async def api_conversations_list() -> Response | tuple[Response, int]:
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    before_raw = request.args.get("before_id")
    try:
        before_id = int(before_raw) if before_raw else None
    except ValueError:
        return jsonify({"error": "invalid_before_id"}), 400
    q = (request.args.get("q") or "").strip() or None
    usage = services().usage
    try:
        page: dict[str, Any] = await usage.list_chats(limit=limit, before_id=before_id, q=q)
    except Exception:
        logger.error("conversations list failed", exc_info=True)
        return jsonify({"error": "conversations_unavailable"}), 500
    return jsonify({**page, "backend": usage.backend_kind})


@bp.delete("/api/conversations")
@reject_if_locked
@require_admin
async def api_conversations_clear() -> Response:
    return jsonify({"deleted": await services().usage.delete_all_chats()})


@bp.delete("/api/conversations/<int:chat_id>")
@reject_if_locked
@require_admin
async def api_conversations_delete(chat_id: int) -> Response:
    return jsonify({"deleted": 1 if await services().usage.delete_chat(chat_id) else 0})
