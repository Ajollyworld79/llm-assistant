# Migrated garbage collection manager into module package
import gc
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

log = logging.getLogger(__name__)

class GarbageCollectionManager:
    def __init__(self):
        self.last_run = None
        self.runs = 0

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
        for _ in range(3):
            gc.collect()

    async def get_statistics(self) -> Dict[str, Any]:
        return {"runs": self.runs, "last_run": self.last_run}


# Single instance
gc_manager = GarbageCollectionManager()