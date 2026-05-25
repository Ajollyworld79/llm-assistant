# LLM Assistant - API Monitoring
# Migrated into module package
import re
import time
import asyncio
import dataclasses
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4


# PII patterns scrubbed before any prompt is logged or returned via /gdpr/audit.
# Ordered: most-specific first so e.g. CPR is not partially matched as a phone fragment.
_PII_PATTERNS = [
    (re.compile(r"\b\d{6}[-\s]?\d{4}\b"), "[cpr-redacted]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[email-redacted]"),
    (re.compile(r"(?:\+45[\s-]?)?(?:\d{2}[\s-]?){3}\d{2}\b"), "[phone-redacted]"),
]


def redact_pii(text: str) -> str:
    """Replace emails, Danish CPR numbers, and phone numbers with placeholders.

    Used before logging or surfacing user prompts in /gdpr/audit. Never call this
    on retrieved document chunks — those should remain intact for the model.
    """
    if not text:
        return text
    out = text
    for pattern, replacement in _PII_PATTERNS:
        out = pattern.sub(replacement, out)
    return out

# Setup logging
try:
    from .Functions_module import setup_logger
    logger = setup_logger()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

@dataclasses.dataclass
class APIMetrics:
    requests_total: int = 0
    requests_failed: int = 0
    requests_success: int = 0
    total_latency_ms: float = 0.0
    last_request_time: float = 0.0
    start_time: float = dataclasses.field(default_factory=time.time)
    endpoint_stats: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: {"count": 0, "errors": 0, "total_time": 0.0, "avg_time": 0.0}))
    errors: deque = dataclasses.field(default_factory=lambda: deque(maxlen=100))
    long_operations: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    slow_requests_threshold_ms: float = 10000.0  # 10 seconds
    error_rate_threshold: float = 0.1  # 10%
    expected_long_operations: Dict[str, float] = dataclasses.field(default_factory=lambda: {
        "/upload": 120000.0,  # 2 minutes for uploads (parsing, chunking, embedding)
        "/admin/upload_vectors": 300000.0,  # 5 minutes for vector uploads (large batches)
    })
    # GDPR audit log — recent (PII-redacted) prompts, capped at 50 entries
    audit_events: deque = dataclasses.field(default_factory=lambda: deque(maxlen=50))
    # Prompt cache tracking
    cached_tokens_total: int = 0
    prompt_tokens_total: int = 0
    fallback_invocations: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

class APIMonitor:
    _instance = None
    metrics: APIMetrics

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIMonitor, cls).__new__(cls)
            cls._instance.metrics = APIMetrics()
        return cls._instance

    def record_request(self, latency_ms: float, success: bool = True):
        self.metrics.requests_total += 1
        self.metrics.total_latency_ms += latency_ms
        self.metrics.last_request_time = time.time()

        if success:
            self.metrics.requests_success += 1
        else:
            self.metrics.requests_failed += 1

    def start_request(self, endpoint: str, method: str = "GET") -> str:
        """Start tracking a request and return tracking ID"""
        request_id = str(uuid4())
        start_time = time.monotonic()
        
        # Store request info
        if not hasattr(self, '_active_requests'):
            self._active_requests = {}
        
        self._active_requests[request_id] = {
            "endpoint": endpoint,
            "method": method,
            "start_time": start_time,
        }
        
        logger.debug(f"Started request {request_id} for {method} {endpoint}")
        return request_id

    def end_request(self, request_id: str, success: bool = True, error: Optional[str] = None):
        """End tracking a request"""
        if request_id not in getattr(self, '_active_requests', {}):
            logger.warning(f"Request {request_id} not found in active requests")
            return
        
        request_info = self._active_requests.pop(request_id)
        duration_ms = (time.monotonic() - request_info["start_time"]) * 1000
        
        # Update global metrics
        self.record_request(duration_ms, success)
        
        # Update endpoint stats
        endpoint = request_info["endpoint"]
        self.metrics.endpoint_stats[endpoint]["count"] += 1
        self.metrics.endpoint_stats[endpoint]["total_time"] += duration_ms
        self.metrics.endpoint_stats[endpoint]["avg_time"] = (
            self.metrics.endpoint_stats[endpoint]["total_time"] / self.metrics.endpoint_stats[endpoint]["count"]
        )
        
        if not success:
            self.metrics.endpoint_stats[endpoint]["errors"] += 1
            self.metrics.errors.append({
                "endpoint": endpoint,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": duration_ms,
            })
        
        # Check for performance alerts
        self._check_performance_alerts(endpoint, duration_ms, success, error)
        
        status = "success" if success else f"failed: {error}"
        logger.info(f"Request {request_id} {status} in {duration_ms:.1f}ms for {request_info['method']} {endpoint}")

    def start_long_operation(self, operation_name: str, details: str = "") -> str:
        """Start tracking a long-running operation"""
        operation_id = str(uuid4())
        start_time = time.monotonic()
        
        self.metrics.long_operations[operation_id] = {
            "name": operation_name,
            "details": details,
            "start_time": start_time,
            "progress": 0,
            "total": 0,
        }
        
        logger.info(f"Started long operation {operation_name} (ID: {operation_id})")
        return operation_id

    def end_long_operation(self, operation_id: str, success: bool = True, error: Optional[str] = None):
        """End tracking a long-running operation"""
        if operation_id not in self.metrics.long_operations:
            logger.warning(f"Long operation {operation_id} not found")
            return
        
        operation = self.metrics.long_operations.pop(operation_id)
        duration = time.monotonic() - operation["start_time"]
        
        status = "completed" if success else f"failed: {error}"
        logger.info(f"Long operation {operation['name']} {status} in {duration:.1f}s")
        
        # Log to endpoint stats as well
        endpoint = f"LONG_OP {operation['name']}"
        self.metrics.endpoint_stats[endpoint]["count"] += 1
        self.metrics.endpoint_stats[endpoint]["total_time"] += duration * 1000  # Convert to ms
        self.metrics.endpoint_stats[endpoint]["avg_time"] = (
            self.metrics.endpoint_stats[endpoint]["total_time"] / self.metrics.endpoint_stats[endpoint]["count"]
        )
        
        if not success:
            self.metrics.endpoint_stats[endpoint]["errors"] += 1

    def cleanup_stale_requests(self, max_age_seconds: float = 10800) -> int:
        """Remove requests that have been active too long (default: 3 hours)"""
        if not hasattr(self, '_active_requests') or not self._active_requests:
            return 0
        
        current_time = time.monotonic()
        stale_requests = [
            req_id for req_id, req_info in self._active_requests.items()
            if (current_time - req_info["start_time"]) > max_age_seconds
        ]
        
        for req_id in stale_requests:
            del self._active_requests[req_id]
            logger.warning(f"Cleaned up stale request {req_id}")
        
        return len(stale_requests)

    async def cleanup_stale_requests_async(self, max_age_seconds: float = 10800) -> int:
        """Async version of cleanup_stale_requests"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.cleanup_stale_requests, max_age_seconds)

    def _check_performance_alerts(self, endpoint: str, duration_ms: float, success: bool, error: Optional[str]):
        """Check for performance issues and log alerts"""
        expected_threshold = self.metrics.expected_long_operations.get(endpoint)
        
        if expected_threshold:
            if duration_ms > expected_threshold:
                logger.warning(f"Expected long operation {endpoint} took {duration_ms:.1f}ms (threshold: {expected_threshold}ms)")
        else:
            if duration_ms > self.metrics.slow_requests_threshold_ms:
                logger.warning(f"Slow request: {endpoint} took {duration_ms:.1f}ms")
        
        # High error rate alert
        if not success:
            error_rate = self.metrics.requests_failed / max(1, self.metrics.requests_total)
            if error_rate > self.metrics.error_rate_threshold:
                logger.error(f"High error rate: {error_rate:.1%} ({self.metrics.requests_failed}/{self.metrics.requests_total})")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": round(self.metrics.uptime_seconds, 2),
            "requests": {
                "total": self.metrics.requests_total,
                "success": self.metrics.requests_success,
                "failed": self.metrics.requests_failed,
            },
            "performance": {
                "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
            },
            "endpoints": dict(self.metrics.endpoint_stats),
            "recent_errors": list(self.metrics.errors)[-10:],  # Last 10 errors
            "active_requests": len(getattr(self, '_active_requests', {})),
            "long_operations": len(self.metrics.long_operations),
        }

    async def get_stats_async(self) -> Dict[str, Any]:
        """Async version for non-blocking stats"""
        return self.get_stats()

    def get_health_summary(self) -> Dict[str, Any]:
        stats = self.get_stats()
        issues = []
        
        # Check error rate
        if stats["requests"]["total"] > 0:
            error_rate = stats["requests"]["failed"] / stats["requests"]["total"]
            if error_rate > self.metrics.error_rate_threshold:
                issues.append(f"High error rate: {error_rate:.1%}")
        
        # Check for stale requests
        if stats["active_requests"] > 10:  # Arbitrary threshold
            issues.append(f"Many active requests: {stats['active_requests']}")
        
        # Check uptime (if very short, might indicate restart)
        if stats["uptime_seconds"] < 60:  # Less than 1 minute
            issues.append("Recent restart detected")
        
        status = "degraded" if issues else "healthy"
        
        return {
            "status": status,
            "issues": issues,
            "metrics": stats,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_health_summary_async(self) -> Dict[str, Any]:
        """Async version for health checks"""
        return self.get_health_summary()

    # ---- GDPR audit + cache tracking ----

    def record_audit_event(self, session_id: str, endpoint: str, prompt: str,
                           model: Optional[str] = None) -> None:
        """Append a PII-redacted prompt record. Used for /gdpr/audit view."""
        self.metrics.audit_events.append({
            "ts": datetime.now().isoformat(),
            "session_id": session_id or "anon",
            "endpoint": endpoint,
            "prompt_redacted": redact_pii(prompt or "")[:500],
            "model": model or "",
        })

    def get_audit_events(self, limit: int = 50, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        events = list(self.metrics.audit_events)
        if session_id:
            events = [e for e in events if e.get("session_id") == session_id]
        return events[-limit:]

    def clear_session_audit(self, session_id: str) -> int:
        """Remove audit entries belonging to a session. Returns count removed."""
        if not session_id:
            return 0
        before = len(self.metrics.audit_events)
        kept = [e for e in self.metrics.audit_events if e.get("session_id") != session_id]
        self.metrics.audit_events.clear()
        self.metrics.audit_events.extend(kept)
        return before - len(self.metrics.audit_events)

    def record_cache_stats(self, prompt_tokens: int, cached_tokens: int) -> None:
        self.metrics.prompt_tokens_total += max(0, int(prompt_tokens or 0))
        self.metrics.cached_tokens_total += max(0, int(cached_tokens or 0))

    def record_fallback_invocation(self) -> None:
        self.metrics.fallback_invocations += 1

    def get_cache_hit_ratio(self) -> float:
        if self.metrics.prompt_tokens_total == 0:
            return 0.0
        return self.metrics.cached_tokens_total / self.metrics.prompt_tokens_total


monitor = APIMonitor()