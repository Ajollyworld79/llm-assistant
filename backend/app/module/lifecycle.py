# Migrated lifecycle into module package
import logging
import asyncio
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:
    # Prefer absolute package import when available
    from backend.app.config import settings
    from backend.app import embeddings
except Exception:
    try:
        # Fallback to relative import when used as a package
        from ..config import settings
        from .. import embeddings
    except Exception:
        # Last-resort fallback when running `main.py` directly;
        # ensure repo root is on sys.path and import top-level modules
        import os, sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import config as config_mod
        settings = config_mod.settings
        import embeddings as embeddings_mod
        embeddings = embeddings_mod

log = logging.getLogger(__name__)

# Optional GC manager import for periodic cleanup
try:
    from .garbage_collection import gc_manager  # type: ignore
except Exception:
    gc_manager = None

class LifeCycleManager:
    _instance = None
    _qdrant_client: Optional[QdrantClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LifeCycleManager, cls).__new__(cls)
        return cls._instance

    @property
    def qdrant_client(self) -> Optional[QdrantClient]:
        return self._qdrant_client

    async def startup(self):
        """Initialize resources on application startup and schedule background tasks."""
        log.info("Starting up application resources...")
        # Run Qdrant connection in a separate thread to avoid blocking startup
        await asyncio.to_thread(self._connect_qdrant)
        await embeddings.preload_embedding_models()

        # Start periodic GC cleanup task if gc_manager available
        try:
            if gc_manager is not None:
                # Respect configured default/min/max interval values
                default = int(getattr(settings, 'GC_DEFAULT_INTERVAL', 1800))
                min_i = int(getattr(settings, 'GC_MIN_INTERVAL', 1800))
                max_i = int(getattr(settings, 'GC_MAX_INTERVAL', 3600))
                interval = max(min_i, min(default, max_i))
                # Expose chosen interval for observability and tests
                self.gc_interval = interval
                _local_gc = gc_manager

                async def _gc_loop():
                    log.info('GC periodic task started (interval=%s sec)', interval)
                    try:
                        while True:
                            try:
                                await asyncio.sleep(interval)
                                await _local_gc.periodic_cleanup()
                            except asyncio.CancelledError:
                                break
                            except Exception as e:
                                log.exception(f'Error in GC periodic task: {e}')
                    finally:
                        log.info('GC periodic task stopped')

                self._gc_task = asyncio.create_task(_gc_loop())
        except Exception as e:
            log.warning(f'Failed to start GC periodic task: {e}')

    async def shutdown(self):
        """Cleanup resources on application shutdown."""
        log.info("Shutting down application resources...")

        # Cancel GC task if running
        try:
            if getattr(self, '_gc_task', None):
                self._gc_task.cancel()
                try:
                    await self._gc_task
                except asyncio.CancelledError:
                    log.info('GC periodic task cancelled successfully')
        except Exception as e:
            log.warning(f'Error cancelling GC task: {e}')

        # Shutdown FastEmbed process pool if any
        try:
            await embeddings.shutdown_fe_executor()
        except Exception:
            log.debug('No FE executor to shut down or shutdown failed')

        if self._qdrant_client:
            # Qdrant client doesn't strictly require close() but good practice if available
            # self._qdrant_client.close() 
            pass

    def _connect_qdrant(self):
        """Establish connection to Qdrant (local or remote)."""
        if settings.qdrant_url:
            try:
                self._qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
                log.info(f"Connected to Qdrant at {settings.qdrant_url}")
            except Exception as e:
                log.error(f"Failed to connect to Qdrant URL: {e}")
                self._qdrant_client = None
        elif settings.qdrant_path:
            try:
                self._qdrant_client = QdrantClient(path=settings.qdrant_path)
                log.info(f"Connected to local Qdrant at {settings.qdrant_path}")
            except Exception as e:
                log.error(f"Failed to initialize local Qdrant: {e}")
                self._qdrant_client = None

        if self._qdrant_client:
            self._ensure_collection()
            # Log current collection size to verify persistence across restarts
            try:
                count = self._qdrant_client.count(collection_name=settings.qdrant_collection)
                log.info("Collection '%s' contains %s points", settings.qdrant_collection, count.count)
            except Exception:
                log.debug('Could not retrieve collection count; qdrant may not expose count')

    def _ensure_collection(self):
        """Ensure the configured collection exists."""
        if self._qdrant_client is None:
            return

        try:
            self._qdrant_client.get_collection(settings.qdrant_collection)
        except Exception:
            log.info(f"Collection '{settings.qdrant_collection}' not found. Creating...")
            try:
                # Use create_collection (does not delete existing data). Only use
                # recreate_collection when explicitly requested by config to avoid
                # wiping data on every startup.
                if getattr(settings, 'qdrant_force_recreate', False):
                    log.info('Force recreate enabled; recreating collection (this will delete existing data)')
                    self._qdrant_client.recreate_collection(
                        collection_name=settings.qdrant_collection,
                        vectors_config=qmodels.VectorParams(
                            size=settings.embedding_dim, 
                            distance=qmodels.Distance.COSINE
                        )
                    )
                else:
                    self._qdrant_client.create_collection(
                        collection_name=settings.qdrant_collection,
                        vectors_config=qmodels.VectorParams(
                            size=settings.embedding_dim, 
                            distance=qmodels.Distance.COSINE
                        )
                    )
                log.info(f"Collection '{settings.qdrant_collection}' created.")
            except Exception as e:
                log.error(f"Failed to create collection: {e}")

lifecycle = LifeCycleManager()