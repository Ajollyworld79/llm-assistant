# LLM Assistant - Middleware (migrated into module package)
# Content mostly preserved from top-level middleware.py
import time
import logging
import traceback
import asyncio
import gc
from datetime import datetime
from quart import Request, Response, jsonify, g
from werkzeug.exceptions import HTTPException
from typing import Dict, Any

try:
    import psutil
except Exception:
    psutil = None

try:
    from .apimonitor import monitor
except ImportError:
    from backend.app.module.apimonitor import monitor

# Optional centralized GC manager
try:
    from .garbage_collection import gc_manager
except Exception:
    gc_manager = None

log = logging.getLogger(__name__)


class EnhancedCleanupMiddleware:
    """ASGI middleware to monitor and cleanup resources for Quart app.

    Adapted from the Support_agent project; integrated here to avoid an extra file.
    """

    def __init__(self, app_asgi, cleanup_interval: int = 300, max_tracked_connections: int = 200):
        self.app = app_asgi
        self.start_time = time.monotonic()
        self.last_cleanup = self.start_time
        self.cleanup_interval = cleanup_interval
        self.max_tracked_connections = max_tracked_connections

        # metrics and tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.peak_memory_mb = 0.0

        self.health_stats = {
            "total_requests": 0,
            "failed_requests": 0,
            "avg_response_time_ms": 0.0,
            "memory_usage_mb": 0.0,
            "active_connections": 0,
            "cleanup_runs": 0,
            "last_cleanup": datetime.now().isoformat(),
            "uptime_seconds": 0,
        }

        self.last_memory_check = 0.0
        self.memory_cache_duration = 5.0
        self.cached_memory_mb = 0.0

        self.active_connections = set()
        self.connection_timestamps = {}

        self.memory_warning_threshold_mb = 512
        self.memory_critical_threshold_mb = 1024

        self.gc_threshold_requests = 100
        self.requests_since_gc = 0

        log.info("EnhancedCleanupMiddleware initialized")

    async def __call__(self, scope, receive, send):
        # Only handle http requests
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_start = time.monotonic()
        conn_id = id(scope)

        # Pre-request tracking
        if len(self.active_connections) >= self.max_tracked_connections:
            log.warning("Max tracked connections reached, triggering cleanup")
            await self._cleanup_resources()

        self.active_connections.add(conn_id)
        self.connection_timestamps[conn_id] = request_start

        try:
            self.health_stats["total_requests"] += 1

            mem = await self._get_memory_usage_mb()
            self.health_stats["memory_usage_mb"] = mem
            if mem > self.peak_memory_mb:
                self.peak_memory_mb = mem

            if mem > self.memory_critical_threshold_mb:
                log.warning(f"CRITICAL memory usage: {mem:.1f} MB")
                await self._emergency_cleanup()
            elif mem > self.memory_warning_threshold_mb:
                log.warning(f"High memory usage: {mem:.1f} MB")

            if self._should_run_cleanup():
                await self._cleanup_resources()

            # call inner app
            await self.app(scope, receive, send)

            # response tracking: we cannot measure response duration precisely here without wrapper
            resp_time_ms = (time.monotonic() - request_start) * 1000
            self.total_response_time += resp_time_ms
            self.health_stats["avg_response_time_ms"] = (
                self.total_response_time / max(1, self.health_stats["total_requests"])  # type: ignore[index]
            )

            # GC management
            self.requests_since_gc += 1
            if self.requests_since_gc >= self.gc_threshold_requests:
                self._run_garbage_collection()

        except Exception as e:
            self.health_stats["failed_requests"] += 1
            log.exception(f"Middleware error for connection {conn_id}: {e}")
            raise
        finally:
            self.active_connections.discard(conn_id)
            self.connection_timestamps.pop(conn_id, None)

    async def _get_memory_usage_mb(self) -> float:
        now = time.monotonic()
        if (now - self.last_memory_check) < self.memory_cache_duration:
            return self.cached_memory_mb

        try:
            if psutil is None:
                return self.cached_memory_mb

            loop = asyncio.get_event_loop()
            ps = psutil  # type: ignore

            def _mem():
                proc = ps.Process()
                return proc.memory_info().rss / 1024 / 1024

            mem = await loop.run_in_executor(None, _mem)
            self.cached_memory_mb = mem
            self.last_memory_check = now
            return mem
        except Exception as e:
            log.debug(f"_get_memory_usage_mb failed: {e}")
            return self.cached_memory_mb

    def _should_run_cleanup(self) -> bool:
        return (time.monotonic() - self.last_cleanup) > self.cleanup_interval

    async def _cleanup_resources(self) -> None:
        try:
            start = time.monotonic()
            # Remove old connections
            now = time.monotonic()
            old_conns = [cid for cid, ts in self.connection_timestamps.items() if (now - ts) > 300]
            for cid in old_conns:
                self.active_connections.discard(cid)
                self.connection_timestamps.pop(cid, None)

            # Prefer gc_manager for cleanup if available
            if gc_manager is not None:
                await gc_manager.periodic_cleanup()
            else:
                self._run_garbage_collection()

            self.last_cleanup = time.monotonic()
            self.health_stats["cleanup_runs"] += 1
            self.health_stats["last_cleanup"] = datetime.now().isoformat()
            self.health_stats["uptime_seconds"] = int(time.monotonic() - self.start_time)

            duration = time.monotonic() - start
            log.info(f"Periodic resource cleanup completed in {duration:.3f}s")
        except Exception as e:
            log.exception(f"_cleanup_resources failed: {e}")

    async def _emergency_cleanup(self) -> None:
        try:
            log.warning("Running emergency cleanup")
            # Aggressive GC
            for _ in range(3):
                c = gc.collect()
                if c > 0:
                    log.info(f"Emergency GC collected {c} objects")

            # Clear tracked connections
            old = len(self.active_connections)
            self.active_connections.clear()
            self.connection_timestamps.clear()
            if old > 0:
                log.warning(f"Emergency: cleared {old} tracked connections")

            # Use centralized gc_manager emergency if available
            if gc_manager is not None:
                await gc_manager.emergency_cleanup()

            # Update memory sample after cleanup
            await self._get_memory_usage_mb()
        except Exception as e:
            log.exception(f"_emergency_cleanup failed: {e}")

    def _run_garbage_collection(self) -> None:
        try:
            collected = gc.collect()
            self.requests_since_gc = 0
            if collected > 0:
                log.debug(f"Middleware GC: collected {collected} objects")
        except Exception as e:
            log.warning(f"_run_garbage_collection error: {e}")

    async def get_health_metrics(self) -> Dict[str, Any]:
        mem = await self._get_memory_usage_mb()
        now = time.monotonic()
        return {
            "memory_usage_mb": mem,
            "peak_memory_mb": self.peak_memory_mb,
            "active_connections": len(self.active_connections),
            "total_requests": self.health_stats["total_requests"],
            "failed_requests": self.health_stats["failed_requests"],
            "success_rate": (
                (self.health_stats["total_requests"] - self.health_stats["failed_requests"]) / max(1, self.health_stats["total_requests"]) * 100
            ),
            "avg_response_time_ms": self.health_stats["avg_response_time_ms"],
            "uptime_seconds": int(now - self.start_time),
            "cleanup_runs": self.health_stats["cleanup_runs"],
            "last_cleanup": self.health_stats["last_cleanup"],
            "requests_since_gc": self.requests_since_gc,
        }

    async def cleanup_on_shutdown(self) -> None:
        try:
            log.info("Middleware shutdown cleanup starting")
            await self._cleanup_resources()
            self.active_connections.clear()
            self.connection_timestamps.clear()
            collected = gc.collect()
            log.info(f"Middleware shutdown cleanup completed ({collected} objects collected)")
        except Exception as e:
            log.exception(f"cleanup_on_shutdown failed: {e}")


def setup_middleware(app):
    """Register request timing handlers and wrap the ASGI app with EnhancedCleanupMiddleware."""
    # Attach simple before/after request timing
    @app.before_request
    async def before_request():
        # Attach start time to global request context
        g.start_time = time.time()

    @app.after_request
    async def after_request(response: Response):
        from quart import request
        # Check if start_time is set in g
        start_time = getattr(g, 'start_time', None)
        if start_time:
            duration_ms = (time.time() - start_time) * 1000
            status_code = response.status_code

            # Log basic info
            log.info(f"{request.method} {request.path} {status_code} - {duration_ms:.2f}ms")

            # Record metrics
            is_success = 200 <= status_code < 400
            monitor.record_request(latency_ms=duration_ms, success=is_success)

        return response

    @app.errorhandler(Exception)
    async def handle_exception(e):
        # Pass through HTTP exceptions
        if isinstance(e, HTTPException):
            return e

        # Log unexpected errors
        log.error(f"Unhandled exception: {str(e)}")
        log.debug(traceback.format_exc())

        return jsonify({
            "error": "Internal Server Error",
            "details": str(e),
            "type": type(e).__name__,
        }), 500

    @app.errorhandler(404)
    async def not_found(e):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(401)
    async def unauthorized(e):
        return jsonify({"error": "Unauthorized"}), 401

    # Wrap ASGI app with enhanced cleanup middleware (if available)
    try:

        existing_asgi = app.asgi_app
        cleanup = EnhancedCleanupMiddleware(existing_asgi)
        app.asgi_app = cleanup
        app.cleanup_middleware = cleanup
        log.info("EnhancedCleanupMiddleware registered and ASGI app wrapped")
    except Exception as e:
        log.warning(f"EnhancedCleanupMiddleware not available: {e}")