# LLM Assistant - Main Application
# Created by Gustav Christensen
# Date: December 2025
# Description: Quart-based backend for document upload, search, and RAG chat using Qdrant and Azure OpenAI

"""Quart-based demo backend for Qdrant-style search + FastEmbed.
- Async endpoints
- Upload and parse files (txt, csv, pdf, docx)
- Chunking and embedding (FastEmbed)
- Search using cosine similarity
- RAG chat backed by the free Gemini API
"""
from quart import Quart, request, jsonify, render_template, g
from quart_cors import cors
from pydantic import BaseModel
from typing import Dict, Any, Optional, cast
import uuid
import time
import asyncio
import random
import logging
import datetime
import inspect
from functools import wraps

try:
    from .config import settings
    from .module.middleware import setup_middleware
    from .module.lifecycle import lifecycle
    from .module.apimonitor import monitor
    from .module.crypto import encrypt_text, decrypt_text
    from .module.demo_lock import reject_if_locked, is_locked as demo_is_locked
    from .module.usage_db import usage_db
    try:
        from .module.garbage_collection import gc_manager
    except Exception:
        gc_manager = None
except ImportError:
    # Fallback for non-packaged execution. NOTE: this branch loads modules at
    # top-level paths (e.g. ``module.lifecycle``) which Python treats as a
    # different module from the relative form (``backend.app.module.lifecycle``)
    # used by sibling routes. Mixing the two creates duplicate singletons —
    # avoid hitting this branch in production by running under uvicorn with
    # ``backend.app.main:app`` so the try branch above succeeds.
    import os, sys
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    import config as config_mod
    settings = config_mod.settings
    from module.middleware import setup_middleware
    from module.lifecycle import lifecycle
    from module.apimonitor import monitor
    from module.crypto import encrypt_text, decrypt_text
    from module.demo_lock import reject_if_locked, is_locked as demo_is_locked
    from module.usage_db import usage_db
    try:
        from module.garbage_collection import gc_manager
    except Exception:
        gc_manager = None

# Import refactored services
try:
    from .services.parser import parse_content, parser_service
    from .services.ai import AIManager, DemoLLM
    from .services.vector_store import (
        QdrantService, QdrantAdapter, embedding_adapter, embedding_service, DOCUMENTS
    )
except ImportError as e:
    try:
        from services.parser import parse_content, parser_service
        from services.ai import AIManager, DemoLLM
        from services.vector_store import (
            QdrantService, QdrantAdapter, embedding_adapter, embedding_service, DOCUMENTS
        )
    except ImportError as e2:
        raise

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Setup logging
try:
    from .module.Functions_module import setup_logger
    logger = setup_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)


app = Quart(__name__, template_folder='../templates', static_folder='../static')
app = cors(app, allow_origin="*")

# Register Middleware
setup_middleware(app)

# Register backend-viewer blueprint
def _register_blueprints():
    import logging as _l
    _log = _l.getLogger(__name__)
    try:
        try:
            from .routes.backend_viewer import backend_bp
        except ImportError:
            from routes.backend_viewer import backend_bp
        app.register_blueprint(backend_bp)
        _log.info("Registered blueprints: backend_viewer")
    except Exception as _e:
        _log.warning("Could not register blueprints: %s", _e)

_register_blueprints()

DEMO_BANNER = "Demo mode — results are simulated; no active LLM connection."
EMBED_DIM = settings.embedding_dim

CONTENT_FILTER_RESPONSE = (
    "The assistant could not comply with the request due to content policy restrictions. "
    "If you believe this is an error, please rephrase the request."
)


def get_welcome_prompt() -> str:
    """Return the welcome message shown on the index page (demo-aware, bilingual DA/EN)."""
    da = "Hej. Jeg er din AI-supportassistent til IT, fejlfinding og praktiske spørgsmål. Hvis der findes relevante dokumenter i RAG-indekset, bruger jeg dem som kilder — ellers svarer jeg ud fra generel viden."
    en = "Hi. I'm your AI support assistant for IT, troubleshooting, and practical questions. When relevant documents exist in the RAG index, I use them as sources — otherwise I answer from general knowledge."
    base = f"{da}\n\n{en}"
    if getattr(settings, "demo", False):
        return f"{base}\n\n(Demo mode er aktiv — svarene er simulerede. / Demo mode is active — answers are simulated.)"
    return base


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    simulate_latency: bool = True
    filters: Optional[Dict[str, Any]] = None # Added for Metadata Filtering


class AuthManager:
    """Simple decorator for admin endpoints using a Bearer token defined in settings."""

    def __init__(self, settings) -> None:
        self.settings = settings

    def require_admin(self, f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            token_expected = getattr(self.settings, 'admin_token', None)
            if not token_expected:
                return jsonify({'error': 'unauthorized', 'detail': 'Admin token not configured'}), 401
            auth = request.headers.get('Authorization', '')
            if not auth.startswith('Bearer '):
                return jsonify({'error': 'unauthorized'}), 401
            token = auth.split(' ', 1)[1].strip()
            if token != token_expected:
                return jsonify({'error': 'unauthorized'}), 401
            return await f(*args, **kwargs)

        return decorated_function

    def require_search_token(self, f):
        """Gate /search smartly: same-origin browser requests pass freely, while
        external API consumers must present a Bearer token (SEARCH_API_TOKEN or
        ADMIN_TOKEN).

        - If SEARCH_API_TOKEN is not configured: endpoint stays fully open.
        - If configured: requests from our own web UI (detected via
          Sec-Fetch-Site=same-origin/same-site or matching Origin/Referer host)
          are allowed without a token. Everything else must carry a valid Bearer.

        Note: Origin/Sec-Fetch headers can be spoofed by determined attackers —
        this is a usability heuristic, not strict security. Use rate limiting
        (already installed) for actual abuse protection.
        """
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            search_token = getattr(self.settings, 'search_api_token', None)
            if not search_token:
                # Not configured — endpoint stays open.
                return await f(*args, **kwargs)

            # Browser same-origin heuristic
            sec_fetch_site = request.headers.get('Sec-Fetch-Site', '')
            host = request.headers.get('Host', '')
            origin = request.headers.get('Origin', '') or request.headers.get('Referer', '')
            is_browser_same_origin = (
                sec_fetch_site in ('same-origin', 'same-site')
                or (host and origin and host in origin)
            )
            if is_browser_same_origin:
                return await f(*args, **kwargs)

            # External request — Bearer required
            auth = request.headers.get('Authorization', '')
            if not auth.startswith('Bearer '):
                return jsonify({'error': 'unauthorized', 'message': 'External calls require Authorization: Bearer <token>'}), 401
            token = auth.split(' ', 1)[1].strip()
            admin_token = getattr(self.settings, 'admin_token', None) or ''
            if token != search_token and token != admin_token:
                return jsonify({'error': 'unauthorized'}), 401
            return await f(*args, **kwargs)

        return decorated_function

# Instantiate helpers
auth_manager = AuthManager(settings)

# Single instance for endpoints to use (dependency injection manually)
qdrant_service = QdrantService(lifecycle)
# QdrantAdapter also needs lifecycle
qdrant_adapter = QdrantAdapter(lifecycle, settings.qdrant_collection)


class ChatAgent:
    """Aggregator that holds the shared service singletons used by route handlers."""

    def __init__(self):
        self.parser = parser_service
        self.embedding = embedding_service
        self.qdrant = qdrant_service
        self.settings = settings
        self.lifecycle = lifecycle
        self.ai_manager = AIManager(self.settings, self.embedding) # Re-instantiate or we could have moved single instance to ai.py

# Single global agent
chat_agent = ChatAgent()


@app.route('/health')
async def health():
    """Enhanced health check with component status, memory and middleware metrics."""
    try:
        basic_health = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "components": {},
        }

        # API monitor summary if available. Kept under its own key — the
        # monitor flags soft issues (recent restart, error-rate spikes) that
        # must not overwrite the overall service status.
        try:
            if hasattr(monitor, 'get_health_summary'):
                monitor_summary = monitor.get_health_summary()
                basic_health['metrics'] = monitor_summary.get('metrics')
                basic_health['monitor'] = {
                    'status': monitor_summary.get('status'),
                    'issues': monitor_summary.get('issues', []),
                }
            else:
                basic_health['metrics'] = monitor.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get api monitor health: {e}")

        # Qdrant: client presence and collection readiness. No client means the
        # in-memory fallback store is active — functional, just not persistent.
        try:
            qdrant_info = {
                "status": "healthy" if lifecycle.qdrant_client else "fallback_in_memory",
                "collection_ready": getattr(qdrant_service, 'collection_ready', False),
            }
            if qdrant_info["collection_ready"] and lifecycle.qdrant_client:
                try:
                    coll_call = lifecycle.qdrant_client.get_collection(settings.qdrant_collection)
                    # handle both sync and async client implementations
                    if inspect.isawaitable(coll_call):
                        coll = await coll_call  # type: ignore
                    else:
                        coll = coll_call
                    qdrant_info["points_count"] = getattr(coll, 'points_count', None)
                except Exception:
                    pass
            basic_health["components"]["qdrant"] = qdrant_info
        except Exception as e:
            basic_health["components"]["qdrant"] = {"status": "error", "error": str(e)}

        # Gemini availability (best-effort). In demo mode the simulated LLM is
        # the expected engine, so a missing key is not a failure there.
        try:
            if getattr(settings, 'gemini_api_key', None):
                gemini_status = "healthy"
            elif settings.demo:
                gemini_status = "simulated"
            else:
                gemini_status = "unavailable"
            basic_health["components"]["gemini"] = {"status": gemini_status}
        except Exception as e:
            basic_health["components"]["gemini"] = {"status": "error", "error": str(e)}

        try:
            basic_health["components"]["database"] = {"status": "not_configured"}
        except Exception as e:
            basic_health["components"]["database"] = {"status": "error", "error": str(e)}

        # Determine overall status: degrade only on real failures, not on
        # designed fallback modes (in-memory store, demo LLM) or soft monitor
        # warnings like a recent restart.
        component_statuses = [c.get("status") for c in basic_health["components"].values()]
        if "error" in component_statuses or "unavailable" in component_statuses:
            basic_health["status"] = "degraded"

        # Add GC memory info if available
        try:
            if 'gc_manager' in globals() and gc_manager is not None:
                mem = await gc_manager.get_memory_usage()
                if mem:
                    basic_health['memory_mb'] = mem.get('rss_mb', 0)
        except Exception as e:
            logger.warning(f"Failed to get gc_manager memory info: {e}")

        # Add middleware health if available
        try:
            cleanup_mw = getattr(app, 'cleanup_middleware', None)
            if cleanup_mw is not None:
                middleware_health = await cleanup_mw.get_health_metrics()
                basic_health['middleware'] = {
                    'memory_usage_mb': middleware_health.get('memory_usage_mb', 0),
                    'active_connections': middleware_health.get('active_connections', 0),
                    'cleanup_runs': middleware_health.get('cleanup_runs', 0),
                }
        except Exception as e:
            logger.warning(f"Failed to get cleanup_middleware health: {e}")

        # Response code
        status_code = 200 if basic_health["status"] == "healthy" else 503
        return jsonify(basic_health), status_code

    except Exception as e:
        logger.exception('Health check error: %s', e)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/health/gc')
async def health_gc():
    """Return garbage collection manager statistics if available."""
    try:
        if 'gc_manager' not in globals() or gc_manager is None:
            return jsonify({"error": "gc_manager not available"}), 404
        stats = await gc_manager.get_statistics()
        return jsonify({"status": "ok", "gc": stats})
    except Exception as e:
        logger.exception('Failed to get GC stats: %s', e)
        return jsonify({"error": str(e)}), 500


@app.route('/health/middleware')
async def health_middleware():
    """Return health metrics from the EnhancedCleanupMiddleware if installed."""
    try:
        middleware = getattr(app, 'cleanup_middleware', None)
        if middleware is None:
            return jsonify({"error": "middleware metrics not available"}), 404
        stats = await middleware.get_health_metrics()
        return jsonify({"status": "ok", "middleware": stats})
    except Exception as e:
        logger.exception('Failed to get middleware stats: %s', e)
        return jsonify({"error": str(e)}), 500


@app.route('/')
async def index():
    # Pass a dynamic welcome prompt and demo flag to the template
    welcome_prompt = get_welcome_prompt()
    demo_mode = bool(getattr(settings, 'demo', False))
    locked = bool(demo_is_locked())
    return await render_template('index.html', welcome_prompt=welcome_prompt, demo_mode=demo_mode, locked=locked)


@app.route('/monitor')
async def monitor_stats():
    """API monitoring statistics endpoint"""
    try:
        stats = await monitor.get_stats_async()
        return jsonify(stats)
    except Exception as e:
        logger.exception(f"Failed to get monitor stats: {e}")
        return jsonify({"error": "Failed to get stats", "details": str(e)}), 500


@app.route('/upload', methods=['POST'])
@reject_if_locked
async def upload_file():
    request_id = monitor.start_request('/upload', 'POST')
    # Accept either multipart file uploads or JSON with 'text' and optional 'filename'
    form = await request.files
    if 'file' in form:
        file = form['file']
        content = await asyncio.to_thread(file.read)
        filename = file.filename or f"upload-{int(time.time())}"
        text = await parse_content(filename, content)
    else:
        payload = await request.get_json(silent=True)
        if payload and 'text' in payload:
            filename = payload.get('filename', f"upload-{int(time.time())}")
            text = payload['text']
        else:
            monitor.end_request(request_id, success=False, error="file field or JSON text required")
            return jsonify({"error": "file field or JSON text required"}), 400

    # Chunk text using service
    chunks = await embedding_service.chunk_text_async(text)
    doc_id = str(uuid.uuid4())

    # Admin auth (if configured)
    if settings.admin_token:
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            monitor.end_request(request_id, success=False, error="unauthorized")
            return jsonify({'error': 'unauthorized'}), 401
        token = auth.split(' ', 1)[1].strip()
        if token != settings.admin_token:
            monitor.end_request(request_id, success=False, error="unauthorized")
            return jsonify({'error': 'unauthorized'}), 401

    # Store in Qdrant if available, otherwise in-memory
    if not lifecycle.qdrant_client:
        # Batch process embeddings even in demo mode to avoid log spam and overhead
        embeddings_batch = await embedding_adapter.embed_texts(chunks, provider=settings.embedding_provider, context=filename)
        
        chunk_objs = []
        for c, emb in zip(chunks, embeddings_batch):
            chunk_objs.append({"text": c, "embed": emb})

        DOCUMENTS.append({
            "id": doc_id,
            "filename": filename,
            "text": text,
            "chunks": chunk_objs,
            "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "type": filename.split('.')[-1].upper() if '.' in filename else 'TXT'
        })
        embedded = sum(1 for c in chunk_objs if c.get("embed") is not None)
        monitor.end_request(request_id, success=True)
        return jsonify({
            "id": doc_id,
            "filename": filename,
            "status": "uploaded (fallback)",
            "requested_chunks": len(chunks),
            "embedded_chunks": embedded,
            "embedding_provider": getattr(settings, "embedding_provider", None),
        })

    # Index into Qdrant (both demo and prod modes)
    # Create points and upsert
    points = []
    embeddings_batch = await embedding_adapter.embed_texts(chunks, provider=settings.embedding_provider, context=filename)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    file_type = filename.split('.')[-1].upper() if '.' in filename else 'TXT'
    
    session_id = getattr(g, 'session_id', 'anon')
    for idx, (c, emb) in enumerate(zip(chunks, embeddings_batch)):
        if emb is None:
             continue
        points.append(qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "filename": filename,
                "chunk_index": idx,
                "text": encrypt_text(c),
                "uploaded_at": timestamp,
                "type": file_type,
                "session_id": session_id,
            }
        ))

    # Before upsert: verify Qdrant collection vector size matches embedding dim
    try:
        coll_dim = None
        try:
            coll_info = await asyncio.to_thread(lifecycle.qdrant_client.get_collection, settings.qdrant_collection)
            if isinstance(coll_info, dict):
                vectors = coll_info.get('vectors') or {}
                coll_dim = vectors.get('size') if isinstance(vectors, dict) else None
            else:
                vectors = getattr(coll_info, 'vectors', None)
                coll_dim = getattr(vectors, 'size', None) if vectors is not None else None
        except Exception as _:
            logger.debug('Could not fetch collection metadata before upsert; proceeding to upsert and handling errors if any')

        emb_dim = None
        if embeddings_batch and len(embeddings_batch) > 0 and embeddings_batch[0] is not None:
            emb_dim = len(embeddings_batch[0])

        if coll_dim is not None and emb_dim is not None and coll_dim != emb_dim:
            # Dimension mismatch — reject with helpful message
            msg = (f"Embedding dimension mismatch: collection vectors are size {coll_dim} but current embeddings are size {emb_dim}. "
                   f"This prevents upsert. To resolve, recreate the collection with size={emb_dim} or create a new collection. "
                   "Use the admin endpoint POST /admin/collection (mode=recreate or mode=new) to proceed.")
            logger.warning(msg)
            monitor.end_request(request_id, success=False, error=msg)
            return jsonify({"error": "embedding_dim_mismatch", "details": msg}), 409

        logger.info('Upserting %d points into collection %s', len(points), settings.qdrant_collection)
        try:
            await asyncio.to_thread(lifecycle.qdrant_client.upsert, collection_name=settings.qdrant_collection, points=points)
        except ValueError as e:
            # Likely a numpy broadcast / dim mismatch error from local Qdrant implementation
            logger.exception('Failed to upsert vectors (likely dim mismatch): %s', e)
            
            detail_msg = str(e)
            if coll_dim is not None:
                detail_msg = f"Collection vector size is {coll_dim}; attempted upsert vector had incompatible shape. Error: {str(e)}"
            monitor.end_request(request_id, success=False, error=detail_msg)
            return jsonify({"error": "embedding_dim_mismatch", "details": detail_msg,
                            "hint": "Recreate or create a collection with EMBEDDING_DIM=3072 using /admin/collection then reindex."}), 409
        except Exception as e:
            raise
        
        valid_embeds = sum(1 for e in embeddings_batch if e is not None)
        failed = max(0, len(embeddings_batch) - valid_embeds)
        monitor.end_request(request_id, success=True)
        return jsonify({
            "id": doc_id,
            "filename": filename,
            "status": "indexed",
            "requested_chunks": len(embeddings_batch),
            "embedded_chunks": valid_embeds,
            "failed_chunks": failed,
            "embedding_provider": getattr(settings, "embedding_provider", None),
        })
    except Exception as e:
        logger.exception('Failed to upsert vectors: %s', e)
        monitor.end_request(request_id, success=False, error=str(e))
        return jsonify({"error": "Failed to index document", "details": str(e)}), 500


@app.route('/admin/collection', methods=['POST'])
@reject_if_locked
@auth_manager.require_admin
async def admin_manage_collection():
    """Admin endpoint to recreate or create a Qdrant collection using current EMBEDDING_DIM."""
    if not lifecycle.qdrant_client:
        return jsonify({"error": "qdrant_unavailable", "details": "No Qdrant client is connected."}), 503

    payload = await request.get_json(silent=True) or {}
    mode = payload.get('mode', 'new')
    if mode not in ('recreate', 'new'):
        return jsonify({"error": "invalid_mode", "details": "mode must be 'recreate' or 'new'"}), 400

    if mode == 'recreate':
        if not payload.get('confirm'):
            return jsonify({"error": "confirm_required", "details": "Set 'confirm': true to recreate (destructive)."}), 400
        try:
            logger.info('Admin requested recreate of collection %s with size=%s', settings.qdrant_collection, settings.embedding_dim)
            await asyncio.to_thread(lifecycle.qdrant_client.recreate_collection, collection_name=settings.qdrant_collection,
                                     vectors_config=qmodels.VectorParams(size=settings.embedding_dim, distance=qmodels.Distance.COSINE))
            return jsonify({"status": "recreated", "collection": settings.qdrant_collection, "size": settings.embedding_dim})
        except Exception as e:
            logger.exception('Failed to recreate collection: %s', e)
            return jsonify({"error": "failed", "details": str(e)}), 500

    # mode == 'new'
    new_name = payload.get('new_collection_name')
    if not new_name:
        return jsonify({"error": "missing_new_collection_name", "details": "Specify new_collection_name for mode 'new'."}), 400
    try:
        logger.info('Admin requested create of new collection %s with size=%s', new_name, settings.embedding_dim)
        await asyncio.to_thread(lifecycle.qdrant_client.create_collection, collection_name=new_name,
                                 vectors_config=qmodels.VectorParams(size=settings.embedding_dim, distance=qmodels.Distance.COSINE))
        return jsonify({"status": "created", "collection": new_name, "size": settings.embedding_dim})
    except Exception as e:
        logger.exception('Failed to create collection: %s', e)
        return jsonify({"error": "failed", "details": str(e)}), 500


@app.route('/admin/reindex_in_memory', methods=['POST'])
@reject_if_locked
@auth_manager.require_admin
async def admin_reindex_in_memory():
    """Upsert all DOCUMENTS currently stored in-memory into the configured collection."""
    if not lifecycle.qdrant_client:
        return jsonify({"error": "qdrant_unavailable", "details": "No Qdrant client is connected."}), 503

    # Build list of points from DOCUMENTS
    total = 0
    failed = 0
    try:
        for d in DOCUMENTS:
            filename = d.get('filename')
            chunks = d.get('chunks', [])
            # Ensure embeddings for chunks are present and correct dim; compute where missing
            texts_to_compute = []
            idxs_to_compute = []
            for i, c in enumerate(chunks):
                vec = c.get('embed') or c.get('vector')
                if not isinstance(vec, list) or len(vec) != settings.embedding_dim:
                    texts_to_compute.append(c.get('text',''))
                    idxs_to_compute.append(i)
            if texts_to_compute:
                try:
                    new_embs = await embedding_adapter.embed_texts(texts_to_compute, provider=settings.embedding_provider, context=filename)
                except Exception as e:
                    logger.exception('Failed to compute embeddings during reindex: %s', e)
                    failed += len(texts_to_compute)
                    continue
                # assign back
                for idx, vec in zip(idxs_to_compute, new_embs):
                    chunks[idx]['embed'] = vec

            # Prepare points
            points = []
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            file_type = filename.split('.')[-1].upper() if filename and '.' in filename else 'TXT'
            for idx, c in enumerate(chunks):
                vec = c.get('embed') or c.get('vector')
                if not isinstance(vec, list) or len(vec) != settings.embedding_dim:
                    failed += 1
                    continue
                points.append(qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "filename": filename,
                        "chunk_index": idx,
                        "text": encrypt_text(c.get('text','')),
                        "uploaded_at": d.get('uploaded_at') or timestamp,
                        "type": file_type,
                        "session_id": getattr(g, 'session_id', 'anon'),
                    }
                ))
            if not points:
                continue
            # upsert in reasonable batches
            batch_size = 500
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                await asyncio.to_thread(lifecycle.qdrant_client.upsert, collection_name=settings.qdrant_collection, points=batch)
                total += len(batch)
        return jsonify({"status": "reindexed", "upserted_points": total, "failed_chunks": failed})
    except Exception as e:
        logger.exception('Reindex in-memory failed: %s', e)
        return jsonify({"error": "failed", "details": str(e)}), 500 


@app.route('/admin/upload_vectors', methods=['POST'])
@reject_if_locked
@auth_manager.require_admin
async def admin_upload_vectors():
    """Admin endpoint to upload pre-computed vectors in JSON format."""
    payload = await request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON payload required"}), 400

    docs = payload.get('documents') or ( [payload.get('document')] if payload.get('document') else None )
    if not docs:
        return jsonify({"error": "documents field missing or empty"}), 400

    results = []
    for doc in docs:
        try:
            doc_id = doc.get('id') or str(uuid.uuid4())
            filename = doc.get('filename') or f"upload-{int(time.time())}"
            chunks = doc.get('chunks') or []
            # Basic validation
            if not isinstance(chunks, list) or not chunks:
                results.append({"id": doc_id, "status": "skipped", "reason": "no chunks provided"})
                continue

            # Ensure each chunk has vector
            ok_chunks = []
            for i, c in enumerate(chunks):
                txt = c.get('text','')
                vec = c.get('vector') or c.get('embed')
                if not isinstance(vec, list) or len(vec) == 0:
                    if getattr(settings, 'demo', False):
                        vec = embedding_service.deterministic_embed(txt)
                    else:
                        results.append({"id": doc_id, "status": "skipped", "reason": f"missing vector for chunk {i}"})
                        ok_chunks = None
                        break
                ok_chunks.append({"text": txt, "vector": vec, "chunk_index": c.get('chunk_index', i)})

            if ok_chunks is None:
                continue

            # Upsert via adapter
            try:
                upserted = await qdrant_adapter.upsert_document_vectors(doc_id, filename, ok_chunks)
                results.append({"id": doc_id, "status": "upserted", "upserted": upserted})
            except Exception as e:
                logger.exception('Failed to upsert document %s: %s', doc_id, e)
                results.append({"id": doc_id, "status": "error", "detail": str(e)})

        except Exception as e:
            logger.exception('Error processing uploaded doc: %s', e)
            results.append({"id": doc.get('id', 'unknown'), "status": "error", "detail": str(e)})

    return jsonify({"results": results})


@app.route('/documents')
async def list_documents():
    if not lifecycle.qdrant_client:
        return jsonify([{
            "id": d["id"], 
            "filename": d["filename"], 
            "chunks": len(d["chunks"]),
            "uploaded_at": d.get("uploaded_at", "N/A"),
            "type": d.get("type", "FILE")
        } for d in DOCUMENTS])
    
    try:
        docs = await qdrant_service.list_documents()
        return jsonify(docs)
    except Exception:
        return jsonify([])


@app.route('/document/<doc_id>')
async def get_document(doc_id):
    if not lifecycle.qdrant_client:
        doc = next((d for d in DOCUMENTS if d["id"] == doc_id), None)
        if not doc:
            return jsonify({"error": "document not found"}), 404
        return jsonify({"id": doc["id"], "filename": doc["filename"], "text": doc["text"], "chunks": len(doc["chunks"])})
    
    chunks = await qdrant_service.get_document_chunks(doc_id)
    if not chunks:
        return jsonify({"error": "document not found"}), 404

    # Reconstruct text
    chunks.sort(key=lambda x: x['chunk_index'])
    full_text = '\n'.join(c['text'] for c in chunks)
    return jsonify({"id": doc_id, "filename": doc_id, "text": full_text, "chunks": len(chunks)})


@app.route('/search', methods=['POST'])
@auth_manager.require_search_token
async def search():
    request_id = monitor.start_request('/search', 'POST')
    payload = await request.get_json()
    if not payload or 'query' not in payload:
        monitor.end_request(request_id, success=False, error="query required")
        return jsonify({"error": "query required"}), 400
    
    query = payload.get('query', '').strip()
    top_k = int(payload.get('top_k', 5))
    simulate_latency = payload.get('simulate_latency', True)
    filters = payload.get('filters', None) # Extract filters
    images = payload.get('images', None)  # Optional [{data_base64, mime_type}, ...] for multimodal Gemini
    thread_id = (payload.get('thread_id') or '').strip() or uuid.uuid4().hex[:16]

    if not query:
        monitor.end_request(request_id, success=False, error="query is required")
        return jsonify({"error": "query is required"}), 400

    if simulate_latency:
        await asyncio.sleep(random.uniform(0.05, 0.4))

    # Timing accumulators for the conversations log
    t_request_start = time.perf_counter()
    embed_ms_total = 0
    ai_ms_total = 0

    # Use QdrantService for search (which now handles filtering)
    # The new search_index method does embedding internally too, making this route simpler.
    # However, for consistency with original logic which handled in-memory search separately,
    # I should check if I should delegate completely or keep some logic here.
    # VectorStoreService handles both in-memory and Qdrant in theory if I had moved in-memory logic there fully.
    # But QdrantService in vector_store.py currently assumes Qdrant is primary and returns empty if unavailable.
    
    # If using in-memory only:
    if not lifecycle.qdrant_client:
        # manual embedding
        _t_emb = time.perf_counter()
        q_embs = await embedding_adapter.embed_texts([query], provider=settings.embedding_provider, context=f"search:{query[:80]}", is_query=True)
        embed_ms_total += int((time.perf_counter() - _t_emb) * 1000)
        q_emb = q_embs[0]
        results = []
        for d in DOCUMENTS:
            for chunk in d['chunks']:
                score = sum(x * y for x, y in zip(q_emb, chunk['embed']))
                results.append({
                    'doc_id': d['id'],
                    'filename': d['filename'],
                    'chunk_text': (chunk['text'][:400] + ('...' if len(chunk['text']) > 400 else '')),
                    'score': float(score),
                })
        results.sort(key=lambda r: r['score'], reverse=True)
        top = results[:top_k]
        unique_files = set(r['filename'] for r in top)
    else:
        # Qdrant search with filters (Point 3 compliance)
        # Use qdrant_service helper to build filters
        query_filter = qdrant_service.build_filters(filters)

        _t_emb = time.perf_counter()
        q_embs = await embedding_adapter.embed_texts([query], provider=settings.embedding_provider, context=f"search:{query[:80]}", is_query=True)
        embed_ms_total += int((time.perf_counter() - _t_emb) * 1000)
        q_emb = q_embs[0]

        results = []
        try:
            client = lifecycle.qdrant_client
            if hasattr(client, 'search'):
                search_res = await asyncio.to_thread(cast(Any, client).search, collection_name=settings.qdrant_collection, query_vector=q_emb, query_filter=query_filter, limit=top_k)
            else:
                res = await asyncio.to_thread(client.query_points, collection_name=settings.qdrant_collection, query=q_emb, query_filter=query_filter, limit=top_k)
                search_res = getattr(res, 'results', res)
            
            logger.info(f"DEBUG: search_res type: {type(search_res)}, content: {search_res}")
            
            # Handle QueryResponse object from newer Qdrant clients
            if hasattr(search_res, 'points'):
                search_res = search_res.points
            elif isinstance(search_res, tuple):
                 # If it returned a tuple (e.g. (points, offset)), unpack it
                 search_res = search_res[0]

            for hit in search_res:
                payload = getattr(hit, 'payload', {}) or {}
                score = getattr(hit, 'score', None) or 0.0
                results.append({
                    'doc_id': getattr(hit, 'id'),
                    'filename': payload.get('filename', 'unknown'),
                    'chunk_text': (lambda t: t[:400] + ('...' if len(t) > 400 else ''))(decrypt_text(payload.get('text', ''))),
                    'score': float(score),
                })
        except Exception as e:
            logger.error(f"Search failed: {e}")
            
        top = results # already limited by Qdrant
        unique_files = set(r['filename'] for r in top)

    answer = None
    logger.info("AI branch check: demo=%s, gemini_key_set=%s, model=%s",
                settings.demo, bool(getattr(settings, 'gemini_api_key', None)), getattr(settings, 'gemini_model', None))

    if settings.demo:
        try:
            best_score = max((r.get('score', 0.0) for r in top), default=0.0)
            ratio = float(getattr(settings, 'DEMO_MATCH_RATIO', 0.4))
            min_score = float(getattr(settings, 'DEMO_MIN_MATCH_SCORE', 0.0))
            threshold = max(best_score * ratio, min_score)

            filtered = [r for r in top if r.get('score', 0.0) >= threshold]

            if not filtered:
                answer = "(Demo mode - no chat model available) I couldn't find any confident matches in the documents (no documents matched the query)."
            else:
                context_parts = [f"Source ({r['filename']}): {r['chunk_text']}" for r in filtered]
                context = "\n\n".join(context_parts)
                demo = DemoLLM()
                _t_ai = time.perf_counter()
                cleaned, source_indicator = await demo.generate(query, context)
                ai_ms_total += int((time.perf_counter() - _t_ai) * 1000)
                answer = f"(Demo mode - no chat model available) {cleaned}"
        except Exception as e:
            logger.exception("Demo LLM failed: %s", e)
            answer = "(Demo mode) Sorry, demo generation failed."

    else:
        try:
            if not getattr(settings, 'gemini_api_key', None):
                answer = "Gemini API not configured (GEMINI_API_KEY missing)."
            else:
                context_parts = [f"Source ({r['filename']}): {r['chunk_text']}" for r in top]
                context = "\n\n".join(context_parts)

                _t_ai = time.perf_counter()
                cleaned, source_indicator = await chat_agent.ai_manager.ask_agent_with_context(query, context, conversation_history=[], images=images)
                ai_ms_total += int((time.perf_counter() - _t_ai) * 1000)

                if cleaned == "__CONTENT_FILTER_BLOCKED__":
                    answer = "Content blocked by safety filters."
                else:
                    answer = cleaned
        except Exception as e:
            logger.exception("RAG Generation failed")
            answer = f"Error generating answer: {str(e)}"

    monitor.end_request(request_id, success=True)

    total_ms = int((time.perf_counter() - t_request_start) * 1000)
    # Pull token counts from the AI manager's last call (if Gemini was used)
    last_model = getattr(chat_agent.ai_manager, 'last_used_model', None) or (
        getattr(settings, 'gemini_model', None) or ('demo' if settings.demo else '')
    )
    last_tokens_total = int(getattr(chat_agent.ai_manager, 'last_total_tokens', 0) or 0)
    last_cached = int(getattr(chat_agent.ai_manager, 'last_cached_tokens', 0) or 0)

    try:
        usage_db.log_chat(
            thread_id=thread_id,
            query=query,
            answer=answer or "",
            embed_ms=embed_ms_total,
            ai_ms=ai_ms_total,
            total_ms=total_ms,
            model=last_model,
            tokens_input=last_tokens_total,
            tokens_output=0,
            doc_count=len(top) if 'top' in locals() else 0,
        )
    except Exception as e:
        logger.warning("usage_db.log_chat failed: %s", e, exc_info=True)

    # Also log a usage_event independently so the Usage History tab fills even
    # if ai.py's _record_usage path misses (e.g., no usage_metadata on fallback).
    try:
        usage_db.log_event(
            session_id=None,
            endpoint="/search",
            model=last_model,
            tokens_input=last_tokens_total,
            tokens_output=0,
            tokens_cached=last_cached,
            latency_ms=total_ms,
            status_code=200,
            error_kind=None,
        )
    except Exception as e:
        logger.warning("usage_db.log_event from /search failed: %s", e, exc_info=True)

    return jsonify({
        "results": top,
        "used_documents": [{"doc_id": r.get("doc_id"), "filename": r.get("filename"), "score": r.get("score"), "chunk_text": r.get("chunk_text")} for r in top],
        "demo": settings.demo,
        "message": DEMO_BANNER if settings.demo else "",
        "answer": answer,
        "sources": list(unique_files) if 'unique_files' in locals() else [],
        "embedding_provider": getattr(settings, "embedding_provider", None),
        "thread_id": thread_id,
        "embed_ms": embed_ms_total,
        "ai_ms": ai_ms_total,
        "total_ms": total_ms,
    })


@app.route('/optimize', methods=['POST'])
@reject_if_locked
async def optimize_store():
    """Optimize Qdrant collection.

    Requires admin token when configured. Triggers collection optimization in Qdrant
    unless running in demo mode or Qdrant client is unavailable. Returns JSON with
    operation result or error details.
    """
    # Admin auth required
    if settings.admin_token:
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'unauthorized'}), 401
        token = auth.split(' ', 1)[1].strip()
        if token != settings.admin_token:
            return jsonify({'error': 'unauthorized'}), 401

    if lifecycle.qdrant_client and not settings.demo:
        try:
            # Trigger optimization
            result = await qdrant_service.optimize_collection()
            return jsonify(result)
        except Exception:
            logger.exception('Failed to optimize Qdrant collection')
            return jsonify({"error": "optimization failed"}), 500
            
    return jsonify({"status": "optimized"})

@app.route('/reset', methods=['POST'])
@reject_if_locked
async def reset_store():
    """Reset in-memory documents and clear Qdrant collection.

    Requires admin token when configured. Clears the in-memory `DOCUMENTS` list and,
    if a Qdrant client is connected and not in demo mode, attempts to clear the configured
    collection and returns deletion details.
    """
    # Admin auth required
    if settings.admin_token:
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'unauthorized'}), 401
        token = auth.split(' ', 1)[1].strip()
        if token != settings.admin_token:
            return jsonify({'error': 'unauthorized'}), 401

    DOCUMENTS.clear()
    if lifecycle.qdrant_client and not settings.demo:
        try:
            result = await qdrant_service.clear_collection()
            if result.get('success'):
                deleted = result.get('deleted', 0)
                return jsonify({"status": "cleared", "deleted": deleted})
            else:
                return jsonify({"error": result.get('error', 'failed')}), 500
        except Exception as e:
            logger.exception('Failed to clear Qdrant collection')
            return jsonify({"error": "failed", "details": str(e)}), 500
            
    return jsonify({"status": "cleared"})


@app.route('/document/<doc_id>', methods=['DELETE'])
@reject_if_locked
async def delete_document(doc_id):
    # Admin auth required
    if settings.admin_token:
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'unauthorized'}), 401
        token = auth.split(' ', 1)[1].strip()
        if token != settings.admin_token:
            return jsonify({'error': 'unauthorized'}), 401

    if not lifecycle.qdrant_client:
        # Remove from DOCUMENTS (global fallback)
        # Note: DOCUMENTS is imported from vector_store now, so modifying it here modifies the shared list
        # But we need to iterate and remove.
        # Modifying list in place is safer by index or rebuilding.
        # Rebuilding:
        global DOCUMENTS
        len_before = len(DOCUMENTS)
        DOCUMENTS[:] = [d for d in DOCUMENTS if d["id"] != doc_id]
        if len(DOCUMENTS) < len_before:
             return jsonify({"status": "deleted"})
        return jsonify({"error": "not found in memory"}), 404

    # From Qdrant, use QdrantService to delete by filename
    try:
        deleted_count = await qdrant_service.delete_by_filename(doc_id)
        if deleted_count > 0:
            return jsonify({"status": "deleted", "deleted": deleted_count})

        # If deletion via scanning didn't find anything, try field filters as last resort (direct client call fallback)
        try:
            await asyncio.to_thread(
                lifecycle.qdrant_client.delete,
                collection_name=settings.qdrant_collection,
                points_selector=qmodels.Filter(must=[qmodels.FieldCondition(key="filename", match=qmodels.MatchValue(value=doc_id))])
            )
            return jsonify({"status": "deleted"})
        except Exception:
            try:
                await asyncio.to_thread(
                    lifecycle.qdrant_client.delete,
                    collection_name=settings.qdrant_collection,
                    points_selector=qmodels.Filter(must=[qmodels.FieldCondition(key="file_name", match=qmodels.MatchValue(value=doc_id))])
                )
                return jsonify({"status": "deleted"})
            except Exception as e:
                logger.exception('Failed to delete document with filters: %s', e)
                return jsonify({"error": "failed to delete"}), 500

    except Exception as e:
        logger.exception('Failed to delete document')
        return jsonify({"error": "failed to delete"}), 500


@app.before_serving
async def _on_startup():
    # connect qdrant if configured
    await lifecycle.startup()

@app.after_serving
async def _on_shutdown():
    await lifecycle.shutdown()

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8002))
    app.run(host='0.0.0.0', port=port, debug=True)
