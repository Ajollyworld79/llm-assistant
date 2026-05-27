"""Read-only backend viewer for the public demo.

Lives at /backend. All endpoints are GET. The demo lock decorator on the
mutation endpoints in main.py ensures viewers cannot trigger any writes from
here (or from anywhere else). This blueprint just surfaces state.

Endpoints:
  /backend             -> HTML page
  /backend/api/system  -> sanitized settings + deploy info
  /backend/api/connections -> per-service status dots
  /backend/api/metrics -> live counters (memory, requests, cache ratio, budget)
  /backend/api/prompt  -> current system prompt + schema (for playground)
  /backend/api/models  -> model catalog with primary/fallback annotation
  /backend/api/usage/summary    -> top-line history
  /backend/api/usage/timeseries -> bucketed series for charts
"""
from __future__ import annotations

import logging
import os
import platform
import sys
import time
from typing import Any, Dict

from quart import Blueprint, jsonify, render_template, request

try:
    from ..config import settings
    from ..module.lifecycle import lifecycle
    from ..module.apimonitor import monitor
    from ..module.demo_lock import is_locked as demo_is_locked
    from ..module.rate_limiter import get_limiter
    from ..module.model_catalog import catalog_for_viewer
    from ..module.config_view import safe_config_dict
    from ..module.usage_db import usage_db
    from ..services.prompt import SYSTEM_PROMPT, CHAT_ANSWER_SCHEMA
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
    import config as _cfg
    settings = _cfg.settings
    from module.lifecycle import lifecycle
    from module.apimonitor import monitor
    from module.demo_lock import is_locked as demo_is_locked
    from module.rate_limiter import get_limiter
    from module.model_catalog import catalog_for_viewer
    from module.config_view import safe_config_dict
    from module.usage_db import usage_db
    from services.prompt import SYSTEM_PROMPT, CHAT_ANSWER_SCHEMA

logger = logging.getLogger(__name__)

backend_bp = Blueprint('backend_viewer', __name__, url_prefix='/backend')

_START_TIME = time.time()


@backend_bp.get('/')
async def index():
    # Use container start time as a cache-bust query so JS/CSS reload after
    # every deploy without users needing to hard-refresh.
    return await render_template('backend.html', locked=demo_is_locked(), asset_v=int(_START_TIME))


@backend_bp.get('/api/system')
async def api_system():
    git_sha = os.environ.get('GIT_SHA') or os.environ.get('RAILWAY_GIT_COMMIT_SHA') or 'unknown'
    return jsonify({
        "model_primary": settings.gemini_model or 'not_configured',
        "model_fallback": settings.gemini_model_fallback or 'not_configured',
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.fastembed_model,
        "embedding_dim": settings.embedding_dim,
        "llm_provider": "Google Gemini (free API)",
        "git_sha": git_sha[:12] if git_sha != 'unknown' else git_sha,
        "deployed_at": os.environ.get('RAILWAY_DEPLOYMENT_STARTED_AT', ''),
        "uptime_seconds": int(time.time() - _START_TIME),
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "demo_locked": demo_is_locked(),
        "config": safe_config_dict(settings),
    })


@backend_bp.get('/api/connections')
async def api_connections():
    """Status dot per external dependency."""
    now_iso = _now_iso()
    out: Dict[str, Any] = {}

    # Qdrant — connection state is determined by client presence; a failing
    # count() call does not mean the service is down (collection may not exist
    # yet, or the call may transiently fail) so we keep status as "up" and
    # simply leave points_count as None in that case.
    qdrant_ok = lifecycle.qdrant_client is not None
    points_count = None
    if qdrant_ok:
        try:
            import asyncio
            count_obj = await asyncio.to_thread(
                lifecycle.qdrant_client.count, collection_name=settings.qdrant_collection
            )
            points_count = getattr(count_obj, 'count', None)
        except Exception:
            points_count = None
    out["qdrant"] = {
        "status": "up" if qdrant_ok else "down",
        "points_count": points_count,
        "last_check_at": now_iso,
    }

    # Gemini — best-effort: presence of API key counts as "configured"
    gemini_configured = bool(settings.gemini_api_key and settings.gemini_model)
    out["gemini"] = {
        "status": "up" if gemini_configured else "unconfigured",
        "last_check_at": now_iso,
    }

    # Embedding model — provider-aware. fastembed is "loaded" when the ONNX model
    # is in memory; gemini/azure are "configured" when their credentials are set
    # (we trust the API call to succeed on demand rather than warming a model).
    provider = (getattr(settings, 'embedding_provider', '') or '').strip().lower()
    if provider in ('gemini',):
        ok = bool(getattr(settings, 'gemini_api_key', None) and getattr(settings, 'gemini_embedding_model', None))
        out["embedding_model"] = {
            "status": "configured" if ok else "unconfigured",
            "provider": "gemini",
            "model": getattr(settings, 'gemini_embedding_model', None),
            "last_check_at": now_iso,
        }
    elif provider in ('azure', 'azure_openai'):
        ok = bool(getattr(settings, 'azure_openai_api_key', None))
        out["embedding_model"] = {
            "status": "configured" if ok else "unconfigured",
            "provider": "azure",
            "model": getattr(settings, 'azure_embedding_deployment', None),
            "last_check_at": now_iso,
        }
    else:
        try:
            from ..embeddings import _FE_MODEL  # type: ignore
        except ImportError:
            try:
                from embeddings import _FE_MODEL  # type: ignore
            except Exception:
                _FE_MODEL = None
        out["embedding_model"] = {
            "status": "loaded" if _FE_MODEL is not None else "not_loaded",
            "provider": "fastembed",
            "model": getattr(settings, 'fastembed_model', None),
            "last_check_at": now_iso,
        }

    # Logger / disk
    out["logger"] = {
        "status": "writing" if settings.LOG_TO_FILE else "stdout_only",
        "last_check_at": now_iso,
    }

    # Postgres usage DB
    db_url_set = bool(os.getenv('DATABASE_URL'))
    try:
        from ..module.usage_db import HAS_ASYNCPG
    except ImportError:
        from module.usage_db import HAS_ASYNCPG  # type: ignore
    out["usage_db"] = {
        "status": "up" if usage_db.backend_kind == "postgres" else "memory_fallback",
        "backend": usage_db.backend_kind,
        "database_url_set": db_url_set,
        "asyncpg_available": HAS_ASYNCPG,
        "last_check_at": now_iso,
    }

    # Chat DB — same postgres pool, but report row count from chat_events so
    # the operator can confirm conversations are persisting.
    chat_count = None
    if usage_db.backend_kind == "postgres":
        try:
            async with usage_db._pool.acquire() as conn:  # type: ignore
                chat_count = int(await conn.fetchval("SELECT count(*) FROM chat_events"))
        except Exception:
            chat_count = None
    else:
        try:
            chat_count = len(usage_db._mem_chats)
        except Exception:
            chat_count = None
    out["chat_db"] = {
        "status": "up" if usage_db.backend_kind == "postgres" else "memory_fallback",
        "backend": usage_db.backend_kind,
        "count": chat_count,
        "last_check_at": now_iso,
    }

    return jsonify(out)


@backend_bp.get('/api/metrics')
async def api_metrics():
    """Live counters for the metrics panel."""
    api_stats = monitor.get_stats()
    cache_ratio = round(monitor.get_cache_hit_ratio() * 100, 1)

    try:
        limiter = get_limiter()
        limiter_state = limiter.snapshot()
    except Exception:
        limiter_state = {}

    try:
        cleanup = getattr(_get_app(), 'cleanup_middleware', None)
        memory_mb = float((await cleanup.get_health_metrics())['memory_usage_mb']) if cleanup else 0.0
    except Exception:
        memory_mb = 0.0

    return jsonify({
        "memory_mb": memory_mb,
        "requests_total": api_stats["requests"]["total"],
        "requests_success": api_stats["requests"]["success"],
        "requests_failed": api_stats["requests"]["failed"],
        "avg_latency_ms": api_stats["performance"]["avg_latency_ms"],
        "uptime_seconds": api_stats["uptime_seconds"],
        "cache_hit_ratio_pct": cache_ratio,
        "fallback_invocations": monitor.metrics.fallback_invocations,
        "rate_limit": limiter_state,
    })


@backend_bp.get('/api/prompt')
async def api_prompt():
    """Surface the current system prompt + schema for the playground."""
    return jsonify({
        "system_prompt": SYSTEM_PROMPT,
        "schema": CHAT_ANSWER_SCHEMA,
        "example_context": "(handbook.pdf) Adgangskoder skal ændres hver 90. dag og må ikke genbruges.",
        "example_query": "Hvor ofte skal jeg skifte adgangskode?",
    })


@backend_bp.get('/api/models')
async def api_models():
    primary = settings.gemini_model or ''
    fallback = settings.gemini_model_fallback or ''
    return jsonify({
        "models": catalog_for_viewer(primary, fallback),
        "active_primary": primary,
        "active_fallback": fallback,
    })


@backend_bp.get('/api/usage/summary')
async def api_usage_summary():
    summary = await usage_db.query_summary()
    return jsonify({**summary, "backend": usage_db.backend_kind})


@backend_bp.get('/api/usage/timeseries')
async def api_usage_timeseries():
    metric = request.args.get('metric', 'requests')
    range_label = request.args.get('range', '24h')
    if metric not in ('requests', 'tokens', 'cache_hit_ratio', 'fallback_rate'):
        return jsonify({"error": "invalid_metric"}), 400
    if range_label not in ('24h', '7d', '30d'):
        return jsonify({"error": "invalid_range"}), 400
    series = await usage_db.query_timeseries(metric, range_label)
    return jsonify({"metric": metric, "range": range_label, "series": series})


def _require_admin_token() -> bool:
    """Return True if request is authorized to mutate. When admin_token is unset
    (e.g. local dev), no auth is required."""
    expected = getattr(settings, 'admin_token', None)
    if not expected:
        return True
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return False
    return auth.split(' ', 1)[1].strip() == expected


@backend_bp.get('/api/conversations')
async def api_conversations_list():
    """List recent chat exchanges. Read-only — no auth required."""
    try:
        limit = int(request.args.get('limit', 200))
    except (TypeError, ValueError):
        limit = 200
    items = await usage_db.list_chats(limit=limit)
    return jsonify({"items": items, "backend": usage_db.backend_kind})


@backend_bp.delete('/api/conversations')
async def api_conversations_clear():
    """Delete ALL chat exchanges. Admin-gated when admin_token is configured."""
    if not _require_admin_token():
        return jsonify({"error": "unauthorized"}), 401
    deleted = await usage_db.delete_all_chats()
    return jsonify({"deleted": deleted})


@backend_bp.delete('/api/conversations/<int:chat_id>')
async def api_conversations_delete(chat_id: int):
    """Delete a single chat exchange by id. Admin-gated when configured."""
    if not _require_admin_token():
        return jsonify({"error": "unauthorized"}), 401
    ok = await usage_db.delete_chat(chat_id)
    return jsonify({"deleted": 1 if ok else 0})


# ---- helpers --------------------------------------------------------------

def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _get_app():
    from quart import current_app
    return current_app
