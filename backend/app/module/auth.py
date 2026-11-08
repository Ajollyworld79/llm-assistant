"""Shared request authentication helpers.

Every check fails closed: when the relevant secret is not configured, the
request is refused rather than allowed.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Optional

from quart import request

# Signed cookie that lets the app's own chat page call /search without exposing
# a token to the browser. SameSite=Strict keeps other sites from riding on it.
SESSION_COOKIE = "helios_session"
SESSION_MAX_AGE = 24 * 3600


def _bearer() -> str:
    auth = request.headers.get("Authorization", "")
    return auth[7:].strip() if auth.startswith("Bearer ") else ""


def _matches(candidate: str, expected: Optional[str]) -> bool:
    return bool(expected) and bool(candidate) and hmac.compare_digest(candidate, expected)


def admin_token() -> Optional[str]:
    return os.getenv("ADMIN_TOKEN") or None


def search_token() -> Optional[str]:
    return os.getenv("SEARCH_API_TOKEN") or None


def is_admin_request() -> bool:
    """True only when ADMIN_TOKEN is configured and the Bearer token matches it."""
    return _matches(_bearer(), admin_token())


def is_search_request() -> bool:
    """Bearer SEARCH_API_TOKEN or ADMIN_TOKEN."""
    token = _bearer()
    return _matches(token, search_token()) or _matches(token, admin_token())


def _session_secret() -> bytes:
    secret = search_token() or admin_token()
    if not secret:
        raise RuntimeError("SEARCH_API_TOKEN or ADMIN_TOKEN must be set to issue chat sessions")
    return hashlib.sha256(b"helios-session:" + secret.encode()).digest()


def issue_session_cookie() -> str:
    issued = str(int(time.time()))
    sig = hmac.new(_session_secret(), issued.encode(), hashlib.sha256).hexdigest()
    return f"{issued}.{sig}"


def has_valid_session() -> bool:
    value = request.cookies.get(SESSION_COOKIE, "")
    issued, _, sig = value.partition(".")
    if not issued.isdigit() or not sig:
        return False
    try:
        expected = hmac.new(_session_secret(), issued.encode(), hashlib.sha256).hexdigest()
    except RuntimeError:
        return False
    if not hmac.compare_digest(sig, expected):
        return False
    return 0 <= time.time() - int(issued) <= SESSION_MAX_AGE
