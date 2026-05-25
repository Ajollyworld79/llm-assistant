"""Hard lock against write/mutation in the public demo.

When BACKEND_DEMO_LOCKED is true, any endpoint decorated with @reject_if_locked
returns 403 immediately. This is the safety net that lets us expose /admin/*
without an auth token in the demo — the viewer can inspect everything but the
underlying operations refuse to run.

The GDPR endpoints (forget/export) are exempt: they only touch the caller's
own session, which is the whole point of having them.
"""
from __future__ import annotations

import os
from functools import wraps
from typing import Awaitable, Callable

from quart import jsonify


def is_locked() -> bool:
    return os.getenv('BACKEND_DEMO_LOCKED', 'false').lower() in ('1', 'true', 'yes')


_LOCKED_RESPONSE = {
    "error": "read_only_demo",
    "detail": "Demo-miljø — skrive-operationer er deaktiveret. Kontakt os for fuld adgang.",
}


def reject_if_locked(handler: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
    """Wrap an async route handler so it returns 403 when the demo lock is on."""

    @wraps(handler)
    async def wrapped(*args, **kwargs):
        if is_locked():
            return jsonify(_LOCKED_RESPONSE), 403
        return await handler(*args, **kwargs)

    return wrapped
