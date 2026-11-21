"""Authentication, signed cookies, request caps and response headers.

Every check fails closed: when the relevant secret is not configured the request is refused.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from quart import Response, jsonify, request

from backend.app.core.container import services

SESSION_COOKIE = "helios_session"
SESSION_MAX_AGE = 24 * 3600
OPERATOR_COOKIE = "helios_operator"
OPERATOR_MAX_AGE = 12 * 3600
CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "helios"

CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)


def _matches(candidate: str, expected: str | None) -> bool:
    return bool(expected) and bool(candidate) and hmac.compare_digest(candidate, expected or "")


def bearer_token() -> str:
    auth = request.headers.get("Authorization", "")
    return auth[7:].strip() if auth.startswith("Bearer ") else ""


def is_admin_bearer() -> bool:
    return _matches(bearer_token(), services().settings.admin_token)


def is_search_bearer() -> bool:
    s = services().settings
    token = bearer_token()
    return _matches(token, s.search_api_token) or _matches(token, s.admin_token)


def client_ip() -> str:
    """Client address from X-Real-IP (set by Railway's proxy), never from X-Forwarded-For."""
    real = request.headers.get("X-Real-IP", "").strip()
    return real or request.remote_addr or "unknown"


# -- signed timestamp cookies -------------------------------------------------


def _sign(secret: str, purpose: str, issued: str) -> str:
    key = hashlib.sha256(f"helios-{purpose}:{secret}".encode()).digest()
    return hmac.new(key, issued.encode(), hashlib.sha256).hexdigest()


def _issue(secret: str | None, purpose: str) -> str:
    if not secret:
        raise RuntimeError(f"no secret configured for {purpose} cookies")
    issued = str(int(time.time()))
    return f"{issued}.{_sign(secret, purpose, issued)}"


def _valid(secret: str | None, purpose: str, value: str, max_age: int) -> bool:
    issued, _, sig = value.partition(".")
    if not secret or not issued.isdigit() or not sig:
        return False
    if not hmac.compare_digest(sig, _sign(secret, purpose, issued)):
        return False
    return 0 <= time.time() - int(issued) <= max_age


def _session_secret() -> str | None:
    s = services().settings
    return s.search_api_token or s.admin_token


def issue_session_cookie() -> str:
    return _issue(_session_secret(), "session")


def has_valid_session() -> bool:
    return _valid(_session_secret(), "session", request.cookies.get(SESSION_COOKIE, ""), SESSION_MAX_AGE)


def issue_operator_cookie() -> str:
    return _issue(services().settings.admin_token, "operator")


def has_operator_cookie() -> bool:
    return _valid(
        services().settings.admin_token, "operator", request.cookies.get(OPERATOR_COOKIE, ""), OPERATOR_MAX_AGE
    )


def has_csrf_header() -> bool:
    return request.headers.get(CSRF_HEADER, "") == CSRF_VALUE


def is_operator_request() -> bool:
    """Admin Bearer token, or operator cookie accompanied by the CSRF header."""
    return is_admin_bearer() or (has_operator_cookie() and has_csrf_header())


def set_secure_cookie(response: Response, name: str, value: str, max_age: int) -> None:
    response.set_cookie(name, value, max_age=max_age, httponly=True, secure=True, samesite="Strict", path="/")


# -- decorators ---------------------------------------------------------------

Handler = Callable[..., Awaitable[Any]]


def require_admin(handler: Handler) -> Handler:
    @wraps(handler)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if is_operator_request():
            return await handler(*args, **kwargs)
        if has_operator_cookie() and not has_csrf_header():
            return jsonify({"error": "csrf_header_required"}), 403
        return jsonify({"error": "unauthorized"}), 401

    return wrapped


def require_search_access(handler: Handler) -> Handler:
    """Bearer SEARCH_API_TOKEN/ADMIN_TOKEN, or the signed session cookie issued by ``GET /``."""

    @wraps(handler)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if is_search_bearer() or has_valid_session():
            return await handler(*args, **kwargs)
        return jsonify({"error": "unauthorized"}), 401

    return wrapped


async def add_security_headers(response: Response) -> Response:
    headers = response.headers
    headers.setdefault("Content-Security-Policy", CSP)
    headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response
