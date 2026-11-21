"""Operator login: exchanges ADMIN_TOKEN for an HttpOnly, signed session cookie."""

from __future__ import annotations

import hmac

from quart import Blueprint, Response, jsonify, request

from backend.app.core.container import services
from backend.app.core.security import (
    OPERATOR_COOKIE,
    OPERATOR_MAX_AGE,
    client_ip,
    has_operator_cookie,
    is_admin_bearer,
    issue_operator_cookie,
    set_secure_cookie,
)

bp = Blueprint("session", __name__, url_prefix="/backend/api")


@bp.get("/session")
async def status() -> Response:
    return jsonify({"authenticated": is_admin_bearer() or has_operator_cookie()})


@bp.post("/session")
async def login() -> Response | tuple[Response, int]:
    svc = services()
    ip = client_ip()
    if svc.login_throttle.blocked(ip):
        return jsonify({"error": "too_many_attempts"}), 429
    payload = await request.get_json(silent=True)
    token = payload.get("token") if isinstance(payload, dict) else None
    expected = svc.settings.admin_token
    if not (isinstance(token, str) and token and expected and hmac.compare_digest(token, expected)):
        svc.login_throttle.record_failure(ip)
        return jsonify({"error": "unauthorized"}), 401
    svc.login_throttle.reset(ip)
    response = Response(b"", status=204)
    set_secure_cookie(response, OPERATOR_COOKIE, issue_operator_cookie(), OPERATOR_MAX_AGE)
    return response


@bp.delete("/session")
async def logout() -> Response:
    response = Response(b"", status=204)
    response.delete_cookie(OPERATOR_COOKIE, path="/", secure=True, httponly=True, samesite="Strict")
    return response
