# LLM Assistant - Lifecycle Management
# Created by Gustav Christensen
# Date: December 2025
# Description: Application lifecycle management for startup, shutdown, and Qdrant initialization

import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:
    from .config import settings
    from . import embeddings
except ImportError:
    import config
    settings = config.settings
    import embeddings

log = logging.getLogger(__name__)

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
        """Initialize resources on application startup."""
        log.info("Starting up application resources...")
        self._connect_qdrant()
        await embeddings.preload_embedding_models()

    async def shutdown(self):
        """Cleanup resources on application shutdown."""
        log.info("Shutting down application resources...")
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
