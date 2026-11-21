"""Liveness and readiness endpoints (public)."""

from __future__ import annotations

import datetime

from quart import Blueprint, Response, jsonify

from backend.app.core.container import services
from backend.app.core.security import is_operator_request, require_admin

bp = Blueprint("health", __name__)


@bp.get("/livez")
async def livez() -> Response:
    return jsonify({"status": "ok"})


async def _database_status() -> dict[str, str]:
    usage = services().usage
    if not usage.configured:
        return {"status": "memory_fallback"}
    return {"status": "healthy" if await usage.ping() else "error"}


@bp.get("/health")
async def health() -> tuple[Response, int]:
    svc = services()
    s = svc.settings

    if s.gemini_api_key:
        gemini = "healthy"
    else:
        gemini = "simulated" if s.demo else "unavailable"

    components = {
        "qdrant": await svc.store.status(),
        "gemini": {"status": gemini},
        "database": await _database_status(),
    }
    summary = svc.monitor.health_summary()
    metrics = dict(summary["metrics"])
    if not is_operator_request():
        errors = metrics.pop("recent_errors", [])
        metrics["recent_errors_count"] = len(errors)

    degraded = any(c["status"] in ("error", "unavailable") for c in components.values())
    body = {
        "status": "degraded" if degraded else "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "components": components,
        "metrics": metrics,
        "monitor": {"status": summary["status"], "issues": summary["issues"]},
    }
    return jsonify(body), 503 if degraded else 200


@bp.get("/monitor")
@require_admin
async def monitor() -> Response:
    return jsonify(services().monitor.stats())
