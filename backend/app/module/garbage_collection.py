# Garbage Collection Manager (merged features)
# Adds configuration, memory tracking, adaptive intervals, and robust async hooks
import gc
import asyncio
import logging
import os
import sys
import time
import weakref
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
try:
    from .Functions_module import setup_logger
    logger = setup_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

# Attempt to read runtime settings if available
try:
    from backend.app.config import settings
except Exception:
    try:
        from ..config import settings
    except Exception:
        settings = None


class GarbageCollectionManager:
    """Garbage collection manager with memory tracking and adaptive GC.

    - async get_memory_usage()
    - async periodic_cleanup()
    - async emergency_cleanup()
    - async cleanup_on_shutdown()
    - async get_statistics()

    Designed to be robust when psutil is missing (returns empty memory dict).
    """

    def __init__(self):
        self.start_time = time.monotonic()
        self._load_config()

        # Stats
        self.gc_runs = 0
        self.total_collected = 0
        self.last_gc_time = self.start_time

        # Memory tracking
        self.memory_samples: list[Dict[str, Any]] = []
        self.max_memory_samples = int(getattr(self, 'gc_max_memory_samples', 100))

        # Object tracking
        self.tracked_objects = weakref.WeakSet()
        self.object_type_counts: Dict[str, int] = {}

        # Control
        self.auto_gc_enabled = True

        # Performance
        self.gc_performance = {
            "total_time_seconds": 0.0,
            "average_time_ms": 0.0,
            "last_collection_objects": 0,
            "last_collection_time_ms": 0.0,
        }

        # Caching for memory info
        self.last_memory_check = 0
        self.cached_memory_info: Dict[str, Any] = {}

        # Initialize logger
        self.logger = logger

        # Tune GC thresholds
        try:
            self._configure_gc()
        except Exception as e:
            self.logger.debug('GC configure failed: %s', e)

    def _load_config(self):
        defaults = {
            "gc_interval": int(getattr(settings, 'GC_INTERVAL', 900) if settings else 900),
            "min_gc_interval": int(getattr(settings, 'MIN_GC_INTERVAL', 300) if settings else 300),
            "max_gc_interval": int(getattr(settings, 'MAX_GC_INTERVAL', 1800) if settings else 1800),
            "memory_warning_threshold": float(getattr(settings, 'MEMORY_WARNING_THRESHOLD', 512) if settings else 512),
            "memory_critical_threshold": float(getattr(settings, 'MEMORY_CRITICAL_THRESHOLD', 1024) if settings else 1024),
            "max_memory_samples": int(getattr(settings, 'GC_MAX_MEMORY_SAMPLES', 100) if settings else 100),
            "emergency_gc_threshold": float(getattr(settings, 'GC_EMERGENCY_THRESHOLD', 0.85) if settings else 0.85),
        }

        self.gc_interval = defaults["gc_interval"]
        self.min_gc_interval = defaults["min_gc_interval"]
        self.max_gc_interval = defaults["max_gc_interval"]
        self.memory_warning_threshold = defaults["memory_warning_threshold"]
        self.memory_critical_threshold = defaults["memory_critical_threshold"]
        self.gc_max_memory_samples = defaults["max_memory_samples"]
        self.emergency_gc_threshold = defaults["emergency_gc_threshold"]

    def _configure_gc(self):
        try:
            cur = gc.get_threshold()
            new = (500, 10, 10)
            gc.set_threshold(*new)
            self.logger.info("[GC] thresholds: %s -> %s", cur, new)
            gc.enable()
        except Exception as e:
            self.logger.debug("_configure_gc error: %s", e)

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Return memory usage; uses psutil when available and caches short-term."""
        now = time.monotonic()
        # cache for a short duration
        if (now - self.last_memory_check) < getattr(self, 'memory_cache_duration', 3.0):
            return dict(self.cached_memory_info) if self.cached_memory_info else {}

        try:
            import psutil
            loop = asyncio.get_running_loop()

            def _mem():
                p = psutil.Process()
                mi = p.memory_info()
                vm = psutil.virtual_memory()
                return {
                    'rss_mb': mi.rss / 1024 / 1024,
                    'vms_mb': mi.vms / 1024 / 1024,
                    'percent': p.memory_percent(),
                    'available_mb': vm.available / 1024 / 1024,
                    'total_mb': vm.total / 1024 / 1024,
                }

            mem = await loop.run_in_executor(None, _mem)
            self.cached_memory_info = mem
            self.last_memory_check = now
            return mem
        except Exception as e:
            self.logger.debug("get_memory_usage failed: %s", e)
            return dict(self.cached_memory_info) if self.cached_memory_info else {}

    async def track_memory(self):
        """Track memory and record samples; trigger emergency cleanup if critical."""
        mem = await self.get_memory_usage()
        if not mem:
            return
        mem['timestamp'] = time.monotonic()
        self.memory_samples.append(mem)
        if len(self.memory_samples) > self.gc_max_memory_samples:
            self.memory_samples.pop(0)

        rss = mem.get('rss_mb', 0)
        if rss and rss > self.memory_critical_threshold:
            self.logger.warning("[GC] CRITICAL memory usage: %.1f MB", rss)
            if self.auto_gc_enabled:
                asyncio.create_task(self.emergency_cleanup())
        elif rss and rss > self.memory_warning_threshold:
            self.logger.info("[GC] High memory usage: %.1f MB", rss)

    async def run_garbage_collection(self, force: bool = False) -> Dict[str, Any]:
        """Run GC in a thread and record timing and counts."""
        start = time.monotonic()
        try:
            collected = await asyncio.to_thread(gc.collect)
        except Exception as e:
            self.logger.error("run_garbage_collection failed: %s", e)
            return {"error": str(e)}

        duration_ms = (time.monotonic() - start) * 1000
        self.gc_runs += 1
        self.total_collected += collected
        self.last_gc_time = time.monotonic()

        self.gc_performance['total_time_seconds'] += duration_ms / 1000.0
        self.gc_performance['average_time_ms'] = (self.gc_performance['total_time_seconds'] * 1000.0) / max(1, self.gc_runs)
        self.gc_performance['last_collection_objects'] = collected
        self.gc_performance['last_collection_time_ms'] = duration_ms

        # Adaptive interval adjustment
        try:
            self._adjust_gc_interval(collected, len(gc.get_objects()))
        except Exception:
            pass

        self.logger.info("[GC] collected %d objects in %.1fms (force=%s)", collected, duration_ms, force)
        return {"collected_objects": collected, "duration_ms": duration_ms, "forced": force}

    async def emergency_cleanup(self):
        """Aggressive emergency cleanup."""
        self.logger.warning("[GC] Starting emergency cleanup")
        # Single GC pass to avoid slow-task warnings
        try:
            collected = await asyncio.to_thread(gc.collect)
            if collected > 0:
                self.logger.info("[GC] Emergency collected %d objects", collected)
        except Exception as e:
            self.logger.debug("Emergency GC failed: %s", e)
        
        # Yield to event loop
        await asyncio.sleep(0)
        
        try:
            await self._clear_caches()
        except Exception:
            pass

    async def _clear_caches(self):
        """Clear import cache and other ephemeral caches."""
        try:
            modules_to_clear = []
            for name in list(sys.modules.keys()):
                if name.startswith('__pycache__') or name.startswith('_pytest') or 'temp' in name.lower():
                    modules_to_clear.append(name)
            for name in modules_to_clear:
                try:
                    del sys.modules[name]
                except Exception:
                    pass
            if modules_to_clear:
                self.logger.debug("[GC] Cleared %d cached modules", len(modules_to_clear))
        except Exception as e:
            self.logger.debug("_clear_caches failed: %s", e)

    def _adjust_gc_interval(self, collected: int, total_objects: int):
        try:
            if total_objects == 0:
                return
            efficiency = collected / max(1, total_objects)
            if efficiency < 0.0005:
                self.gc_interval = min(self.gc_interval * 1.5, self.max_gc_interval)
            elif efficiency < 0.002:
                self.gc_interval = min(self.gc_interval * 1.15, self.max_gc_interval)
            elif efficiency > 0.01:
                self.gc_interval = max(self.gc_interval * 0.7, self.min_gc_interval)
        except Exception as e:
            self.logger.debug("_adjust_gc_interval failed: %s", e)

    async def should_run_gc(self) -> bool:
        now = time.monotonic()
        if (now - self.last_gc_time) > self.gc_interval:
            return True
        if (now - self.last_gc_time) > (self.max_gc_interval * 2):
            return True
        mem = await self.get_memory_usage()
        if mem:
            perc = mem.get('percent', 0)
            if perc and perc > (self.emergency_gc_threshold * 100):
                return True
        return False

    async def periodic_cleanup(self):
        try:
            await self.track_memory()
            if await self.should_run_gc():
                await self.run_garbage_collection()
        except Exception as e:
            self.logger.error("periodic_cleanup failed: %s", e)

    async def get_statistics(self) -> Dict[str, Any]:
        mem = await self.get_memory_usage()
        uptime = time.monotonic() - self.start_time
        return {
            "uptime_seconds": uptime,
            "gc_runs": self.gc_runs,
            "total_objects_collected": self.total_collected,
            "gc_performance": dict(self.gc_performance),
            "current_memory": mem,
            "memory_samples_count": len(self.memory_samples),
            "auto_gc_enabled": self.auto_gc_enabled,
        }

    async def cleanup_on_shutdown(self):
        try:
            self.logger.info("[GC] Final cleanup on shutdown")
            await self.run_garbage_collection(force=True)
            self.tracked_objects.clear()
            stats = await self.get_statistics()
            self.logger.info("[GC] Final stats: %s", stats)
        except Exception as e:
            self.logger.error("cleanup_on_shutdown failed: %s", e)


# Single instance
gc_manager = GarbageCollectionManager()