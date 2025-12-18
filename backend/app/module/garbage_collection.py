# Migrated garbage collection manager into module package
import gc
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

log = logging.getLogger(__name__)

# Attempt to read runtime settings if available
try:
    from backend.app.config import settings
except Exception:
    try:
        from ..config import settings
    except Exception:
        settings = None

class GarbageCollectionManager:
    def __init__(self):
        self.last_run = None
        self.runs = 0
        # Configurable tuning
        self.gc_max_memory_samples = int(getattr(settings, 'GC_MAX_MEMORY_SAMPLES', 100)) if settings else 100
        self.gc_emergency_threshold = float(getattr(settings, 'GC_EMERGENCY_THRESHOLD', 0.85)) if settings else 0.85

    async def get_memory_usage(self) -> Dict[str, Any]:
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info().rss / 1024 / 1024
            return {"rss_mb": mem}
        except Exception:
            return {}

    def run_garbage_collection(self, force: bool = False) -> Dict[str, Any]:
        collected = gc.collect()
        self.runs += 1
        self.last_run = datetime.now().isoformat()
        return {"collected_objects": collected, "last_run": self.last_run}

    async def periodic_cleanup(self) -> None:
        self.run_garbage_collection()

    async def emergency_cleanup(self) -> None:
        # Respect configured sample limit, but keep a small upper bound to avoid stalls
        sample_count = min(3, max(1, getattr(self, 'gc_max_memory_samples', 3)))
        for _ in range(sample_count):
            gc.collect()

    async def get_statistics(self) -> Dict[str, Any]:
        return {
            "runs": self.runs,
            "last_run": self.last_run,
            "gc_max_memory_samples": self.gc_max_memory_samples,
            "gc_emergency_threshold": self.gc_emergency_threshold,
        }


# Single instance
gc_manager = GarbageCollectionManager()