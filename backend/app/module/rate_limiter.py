"""Rate limiter for public demo: per-IP, per-session, and persistent daily token budget.

In-memory IP/session counters reset on app restart. The daily token budget is
persisted to a JSON file under the Qdrant data dir so it survives Railway restarts —
this prevents an attacker from triggering a restart loop to bypass the budget.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, per_min: int, per_day: int, per_session: int,
                 daily_token_budget: int, budget_file_path: str):
        self.per_min = per_min
        self.per_day = per_day
        self.per_session = per_session
        self.daily_token_budget = daily_token_budget
        self.budget_file_path = budget_file_path

        self._ip_minute: dict[str, deque] = defaultdict(deque)
        self._ip_day: dict[str, deque] = defaultdict(deque)
        self._session_count: dict[str, int] = defaultdict(int)

        self._lock = asyncio.Lock()
        self._budget_state = self._load_budget()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime('%Y-%m-%d')

    def _load_budget(self) -> dict:
        try:
            if os.path.exists(self.budget_file_path):
                with open(self.budget_file_path, 'r') as f:
                    data = json.load(f)
                if data.get('date') == self._today():
                    return data
        except Exception as e:
            logger.warning("RateLimiter: failed to load budget state: %s", e)
        return {'date': self._today(), 'tokens_used': 0}

    def _save_budget(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.budget_file_path) or '.', exist_ok=True)
            tmp = self.budget_file_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self._budget_state, f)
            os.replace(tmp, self.budget_file_path)
        except Exception as e:
            logger.warning("RateLimiter: failed to persist budget state: %s", e)

    def _rotate_budget_if_new_day(self) -> None:
        if self._budget_state.get('date') != self._today():
            self._budget_state = {'date': self._today(), 'tokens_used': 0}
            self._save_budget()

    async def check_ip(self, ip: str) -> Optional[str]:
        async with self._lock:
            now = time.time()
            minute_ago = now - 60
            day_ago = now - 86400

            mq = self._ip_minute[ip]
            while mq and mq[0] < minute_ago:
                mq.popleft()
            if len(mq) >= self.per_min:
                return f"per-IP minute limit ({self.per_min}/min)"

            dq = self._ip_day[ip]
            while dq and dq[0] < day_ago:
                dq.popleft()
            if len(dq) >= self.per_day:
                return f"per-IP daily limit ({self.per_day}/day)"

            mq.append(now)
            dq.append(now)
        return None

    async def check_session(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        async with self._lock:
            if self._session_count[session_id] >= self.per_session:
                return f"per-session message limit ({self.per_session})"
            self._session_count[session_id] += 1
        return None

    async def check_token_budget(self) -> Optional[str]:
        async with self._lock:
            self._rotate_budget_if_new_day()
            if self._budget_state['tokens_used'] >= self.daily_token_budget:
                return f"daily token budget exhausted ({self.daily_token_budget})"
        return None

    async def record_tokens(self, tokens: int) -> None:
        if tokens <= 0:
            return
        async with self._lock:
            self._rotate_budget_if_new_day()
            self._budget_state['tokens_used'] = int(self._budget_state.get('tokens_used', 0)) + int(tokens)
            self._save_budget()

    def snapshot(self) -> dict:
        """State for /backend/api/metrics. Safe to call from sync context."""
        now_utc = datetime.now(timezone.utc)
        reset_at = (now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta(days=1)).isoformat()
        return {
            'per_min_limit': self.per_min,
            'per_day_limit': self.per_day,
            'per_session_limit': self.per_session,
            'daily_token_budget': self.daily_token_budget,
            'tokens_used_today': int(self._budget_state.get('tokens_used', 0)),
            'budget_resets_at': reset_at,
            'active_ips': len(self._ip_minute),
            'active_sessions': len(self._session_count),
        }


_limiter: Optional[RateLimiter] = None


def get_limiter():
    global _limiter
    if _limiter is None:
        try:
            from ..config import settings as _settings
        except ImportError:
            import config as _config_mod
            _settings = _config_mod.settings
        budget_path = os.path.join(_settings.qdrant_path, 'budget.json')
        _limiter = RateLimiter(
            per_min=_settings.RATE_LIMIT_PER_MIN,
            per_day=_settings.RATE_LIMIT_PER_DAY,
            per_session=_settings.RATE_LIMIT_PER_SESSION,
            daily_token_budget=_settings.DAILY_TOKEN_BUDGET,
            budget_file_path=budget_path,
        )
    return _limiter
