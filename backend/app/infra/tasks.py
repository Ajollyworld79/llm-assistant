"""Fire-and-forget asyncio tasks that stay referenced until they finish."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_pending: set[asyncio.Task[Any]] = set()


def spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    task = asyncio.ensure_future(coro)
    _pending.add(task)
    task.add_done_callback(_pending.discard)
    return task


async def drain() -> None:
    """Wait for outstanding background tasks (used on shutdown and in tests)."""
    if _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)
