"""Read-only switch for the public demo: mutating routes answer 403 when locked."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from quart import jsonify

from backend.app.core.container import services

LOCKED_BODY = {"error": "read_only_demo", "detail": "Write operations are disabled in this demo."}


def reject_if_locked(handler: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    @wraps(handler)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if services().settings.demo_locked:
            return jsonify(LOCKED_BODY), 403
        return await handler(*args, **kwargs)

    return wrapped
