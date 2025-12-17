"""Subpackage to hold internal modules (apimonitor, lifecycle, middleware, garbage_collection).
This keeps implementation modules isolated from `main.py` for clarity.
"""

from .apimonitor import monitor
from .lifecycle import lifecycle
from .middleware import setup_middleware, EnhancedCleanupMiddleware

__all__ = ["monitor", "lifecycle", "setup_middleware", "EnhancedCleanupMiddleware"]