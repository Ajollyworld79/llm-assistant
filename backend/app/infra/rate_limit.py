"""Per-IP / per-session limits, a persistent daily token budget and a login throttle.

IP and session counters live in memory (reset on restart). The daily token budget is
persisted to a JSON file so a restart loop cannot be used to reset it.
Tokens are *reserved* before an LLM call and settled afterwards, so concurrent
requests cannot overshoot the budget by more than their estimation error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


class RateLimiter:
    def __init__(
        self,
        per_min: int,
        per_day: int,
        per_session: int,
        daily_token_budget: int,
        budget_file: str | Path,
    ) -> None:
        self.per_min = per_min
        self.per_day = per_day
        self.per_session = per_session
        self.daily_token_budget = daily_token_budget
        self.budget_file = Path(budget_file)

        self._ip_minute: dict[str, deque[float]] = defaultdict(deque)
        self._ip_day: dict[str, deque[float]] = defaultdict(deque)
        self._sessions: dict[str, int] = defaultdict(int)
        self._sessions_date = _today()
        self._last_prune = 0.0
        self._lock = asyncio.Lock()
        self._budget = self._load_budget()

    # -- budget persistence -------------------------------------------------

    def _load_budget(self) -> dict[str, Any]:
        try:
            data = json.loads(self.budget_file.read_text())
            if data.get("date") == _today():
                return {"date": data["date"], "tokens_used": int(data.get("tokens_used", 0))}
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError):
            logger.warning("RateLimiter: could not load budget state; starting at zero")
        return {"date": _today(), "tokens_used": 0}

    def _save_budget(self, state: dict[str, Any]) -> None:
        try:
            self.budget_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.budget_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(self.budget_file)
        except OSError:
            logger.warning("RateLimiter: could not persist budget state")

    async def _persist(self) -> None:
        await asyncio.to_thread(self._save_budget, dict(self._budget))

    def _rotate_budget(self) -> bool:
        if self._budget["date"] != _today():
            self._budget = {"date": _today(), "tokens_used": 0}
            return True
        return False

    # -- IP / session -------------------------------------------------------

    def _prune(self, now: float) -> None:
        for ip in list(self._ip_day):
            day = self._ip_day[ip]
            while day and day[0] < now - 86400:
                day.popleft()
            minute = self._ip_minute.get(ip)
            while minute and minute[0] < now - 60:
                minute.popleft()
            if minute is not None and not minute:
                del self._ip_minute[ip]
            if not day:
                del self._ip_day[ip]
                self._ip_minute.pop(ip, None)
        if self._sessions_date != _today():
            self._sessions_date = _today()
            self._sessions.clear()

    async def check_ip(self, ip: str) -> str | None:
        async with self._lock:
            now = time.time()
            if now - self._last_prune >= 60:
                self._last_prune = now
                self._prune(now)
            minute, day = self._ip_minute[ip], self._ip_day[ip]
            while minute and minute[0] < now - 60:
                minute.popleft()
            while day and day[0] < now - 86400:
                day.popleft()
            if len(minute) >= self.per_min:
                return f"per-IP minute limit ({self.per_min}/min)"
            if len(day) >= self.per_day:
                return f"per-IP daily limit ({self.per_day}/day)"
            minute.append(now)
            day.append(now)
        return None

    async def check_session(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        async with self._lock:
            if self._sessions_date != _today():
                self._sessions_date = _today()
                self._sessions.clear()
            if self._sessions[session_id] >= self.per_session:
                return f"per-session message limit ({self.per_session})"
            self._sessions[session_id] += 1
        return None

    # -- token budget -------------------------------------------------------

    async def check_budget(self) -> str | None:
        async with self._lock:
            if self._rotate_budget():
                await self._persist()
            if self._budget["tokens_used"] >= self.daily_token_budget:
                return f"daily token budget exhausted ({self.daily_token_budget})"
        return None

    async def reserve(self, tokens: int) -> str | None:
        """Atomically check the budget and reserve ``tokens`` against it."""
        async with self._lock:
            self._rotate_budget()
            if self._budget["tokens_used"] >= self.daily_token_budget:
                return f"daily token budget exhausted ({self.daily_token_budget})"
            self._budget["tokens_used"] += max(0, tokens)
            await self._persist()
        return None

    async def settle(self, reserved: int, actual: int) -> None:
        """Replace a reservation with the real usage."""
        async with self._lock:
            self._rotate_budget()
            used = self._budget["tokens_used"] + max(0, actual) - max(0, reserved)
            self._budget["tokens_used"] = max(0, used)
            await self._persist()

    def snapshot(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        reset_at = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).isoformat()
        return {
            "per_min_limit": self.per_min,
            "per_day_limit": self.per_day,
            "per_session_limit": self.per_session,
            "daily_token_budget": self.daily_token_budget,
            "tokens_used_today": int(self._budget["tokens_used"]),
            "budget_resets_at": reset_at,
            "active_ips": len(self._ip_minute),
            "active_sessions": len(self._sessions),
        }


class FailureThrottle:
    """Blocks a client after too many failures within a window (operator login)."""

    def __init__(self, max_failures: int = 10, window_seconds: int = 300) -> None:
        self.max_failures = max_failures
        self.window = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _trim(self, key: str, now: float) -> deque[float]:
        q = self._failures[key]
        while q and q[0] < now - self.window:
            q.popleft()
        if not q:
            self._failures.pop(key, None)
            return self._failures[key]
        return q

    def blocked(self, key: str) -> bool:
        return len(self._trim(key, time.monotonic())) >= self.max_failures

    def record_failure(self, key: str) -> None:
        self._trim(key, time.monotonic()).append(time.monotonic())

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)
