"""In-process request metrics for /health and the operator console."""

from __future__ import annotations

import resource
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def rss_mb() -> float:
    """Resident memory of this process in MB (peak RSS where current RSS is unavailable)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


class Monitor:
    def __init__(self) -> None:
        self.started = time.time()
        self.total = 0
        self.failed = 0
        self.total_latency_ms = 0.0
        self.endpoints: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "errors": 0, "avg_time": 0.0})
        self.errors: deque[dict[str, Any]] = deque(maxlen=50)
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.fallback_invocations = 0

    def record_request(self, endpoint: str, latency_ms: float, status: int, error: str | None = None) -> None:
        ok = status < 400
        self.total += 1
        self.total_latency_ms += latency_ms
        stats = self.endpoints[endpoint]
        stats["count"] += 1
        stats["avg_time"] += (latency_ms - stats["avg_time"]) / stats["count"]
        if not ok:
            self.failed += 1
            stats["errors"] += 1
            self.errors.append({"endpoint": endpoint, "status": status, "error": error or f"HTTP {status}"})

    def record_llm(self, prompt_tokens: int, cached_tokens: int, fallback: bool) -> None:
        self.prompt_tokens += max(0, prompt_tokens)
        self.cached_tokens += max(0, cached_tokens)
        self.fallback_invocations += int(fallback)

    @property
    def cache_hit_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "uptime_seconds": round(time.time() - self.started, 2),
            "requests": {"total": self.total, "success": self.total - self.failed, "failed": self.failed},
            "performance": {"avg_latency_ms": round(self.total_latency_ms / self.total, 2) if self.total else 0.0},
            "endpoints": {k: dict(v) for k, v in self.endpoints.items()},
            "recent_errors": list(self.errors)[-10:],
        }

    def health_summary(self) -> dict[str, Any]:
        stats = self.stats()
        issues: list[str] = []
        if self.total and self.failed / self.total > 0.5 and self.total >= 20:
            issues.append(f"High error rate: {self.failed / self.total:.0%}")
        if stats["uptime_seconds"] < 60:
            issues.append("Recent restart detected")
        return {"status": "degraded" if issues else "healthy", "issues": issues, "metrics": stats}
