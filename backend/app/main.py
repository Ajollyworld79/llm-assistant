# LLM Assistant - Main Application
# Created by Gustav Christensen
# Date: December 2025
# Description: Quart-based backend for document upload, search, and RAG chat using Qdrant and Azure OpenAI

"""Quart-based demo backend for Qdrant-style search + FastEmbed simulation.
- Async endpoints
- Upload and parse files (txt, csv, pdf, docx)
- Chunking and deterministic embedding (hash-based)
- Search using cosine similarity
- Demo-mode flag and simulated latency
"""
from quart import Quart, request, jsonify, render_template
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
import re

try:
    from .config import settings
    import embeddings as embeddings
    from .module.middleware import setup_middleware
    from .module.lifecycle import lifecycle
    from .module.apimonitor import monitor
except ImportError:
    import os, sys
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    import config as config_mod
    settings = config_mod.settings
    import embeddings as embeddings
    from module.middleware import setup_middleware
    from module.lifecycle import lifecycle
    from module.apimonitor import monitor
    # Garbage collection manager (optional psutil dependency)
    try:
        from .module.garbage_collection import gc_manager
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

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None

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

DEMO_BANNER = "Demo mode — results are simulated; no active LLM connection."
EMBED_DIM = settings.embedding_dim

# Friendly user-facing responses used by validation and safety checks
NON_IT_RESPONSE = (
    "That question doesn't appear related to IT or work tasks (recipes, sports, travel, etc.). "
    "I can help with IT topics such as Microsoft 365, Windows, networking, printers, and related business questions. "
    "Please rephrase your question with an IT focus so I can assist."
)

CONTENT_FILTER_RESPONSE = (
    "The assistant could not comply with the request due to content policy restrictions. "
    "If you believe this is an error, please rephrase the request or contact support."
)


def get_welcome_prompt() -> str:
    """Return the welcome message shown on the index page (demo-aware)."""
    base = "Hello! I'm your AI assistant powered by Qdrant. Upload documents to the Knowledge Base to get started, or ask me questions about existing data."
    if getattr(settings, "demo", False):
        return f"{base} Note: demo mode is active and responses are simulated."
    return base


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    simulate_latency: bool = True
    filters: Optional[Dict[str, Any]] = None # Added for Metadata Filtering


# ------------------------- Helper classes (portfolio demo) -------------------------
class DomainClassifier:
    """Lightweight domain classifier for demo: IT vs non-IT vs math."""

    def __init__(self) -> None:
        self.non_it_keywords = set([
            'recipe', 'recipes', 'food', 'cooking', 'travel', 'movie', 'music', 'book', 'books', 'sport', 'sports',
        ])
        self.it_keywords = set([
            'windows', 'excel', 'outlook', 'teams', 'sharepoint', 'onedrive', 'word', 'powerpoint', 'vpn', 'printer', 'network', 'wifi', 'email', 'password', 'printer', 'linux', 'azure', 'microsoft'
        ])

    def is_math_query(self, text: str) -> bool:
        if not text:
            return False
        clean_text = text.replace(' ', '').strip()
        math_pattern = re.compile(r"^[\d\+\-\*\/\(\)\=\.\,\s]*[\+\-\*\/][\d\+\-\*\/\(\)\=\.\,\s]*$")
        if math_pattern.match(clean_text):
            return True
        math_keywords = ['plus', 'minus', 'times', 'divide', 'percentage', '%']
        text_lower = text.lower()
        has_math_word = any(word in text_lower for word in math_keywords)
        has_numbers = any(char.isdigit() for char in text)
        return has_math_word and has_numbers

    def classify_query_domain(self, text: str) -> dict:
        if not text or not text.strip():
            return {'domain': 'unknown', 'is_it': False}
        if self.is_math_query(text):
            return {'domain': 'math', 'is_it': False}
        q = text.lower()
        for k in self.non_it_keywords:
            if k in q:
                return {'domain': 'non_it', 'is_it': False}
        for k in self.it_keywords:
            if k in q:
                return {'domain': 'it', 'is_it': True}
        return {'domain': 'unknown', 'is_it': True}


class ValidationManager:
    """Validate user input and provide friendly messages."""

    def __init__(self, domain_classifier: DomainClassifier) -> None:
        self.domain_classifier = domain_classifier

    def validate_user_query(self, query: str, max_chars: int = 2000) -> tuple[bool, Optional[str]]:
        if not query or not query.strip():
            return False, "Please provide a non-empty query."
        if len(query) > max_chars:
            return False, "The query is too long; please shorten it."
        classification = self.domain_classifier.classify_query_domain(query)
        if classification.get('domain') == 'non_it':
            return False, NON_IT_RESPONSE
        return True, None


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

# Instantiate helpers
domain_classifier = DomainClassifier()
validation_manager = ValidationManager(domain_classifier)
auth_manager = AuthManager(settings)

# Single instance for endpoints to use (dependency injection manually)
qdrant_service = QdrantService(lifecycle)
# QdrantAdapter also needs lifecycle
qdrant_adapter = QdrantAdapter(lifecycle, settings.qdrant_collection)


class SupportAgent:
    """Aggregator that keeps references to the main services used by routes."""

    def __init__(self):
        self.parser = parser_service
        self.embedding = embedding_service
        self.qdrant = qdrant_service
        self.settings = settings
        self.lifecycle = lifecycle
        self.ai_manager = AIManager(self.settings, self.embedding) # Re-instantiate or we could have moved single instance to ai.py

# Single global agent
support_agent = SupportAgent()


@app.route('/health')
async def health():
    """Enhanced health check with component status, memory and middleware metrics."""
    try:
        basic_health = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "components": {},
        }

        # API monitor summary if available
        try:
            if hasattr(monitor, 'get_health_summary'):
                monitor_summary = monitor.get_health_summary()
                basic_health.update(monitor_summary)
            else:
                basic_health['metrics'] = monitor.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get api monitor health: {e}")

        # Qdrant: client presence and collection readiness
        try:
            qdrant_info = {
                "status": "healthy" if lifecycle.qdrant_client else "unavailable",
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

        # OpenAI availability (best-effort)
        try:
            openai_status = "healthy" if AsyncAzureOpenAI and getattr(settings, 'azure_openai_api_key', None) else "unavailable"
            basic_health["components"]["openai"] = {"status": openai_status}
        except Exception as e:
            basic_health["components"]["openai"] = {"status": "error", "error": str(e)}

        try:
            basic_health["components"]["database"] = {"status": "not_configured"}
        except Exception as e:
            basic_health["components"]["database"] = {"status": "error", "error": str(e)}

        # Determine overall status
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
    return await render_template('index.html', welcome_prompt=welcome_prompt, demo_mode=demo_mode)


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
    
    for idx, (c, emb) in enumerate(zip(chunks, embeddings_batch)):
        if emb is None:
             # Skip empty embeddings
             continue
        points.append(qmodels.PointStruct(
            id=str(uuid.uuid4()), 
            vector=emb, 
            payload={
                "filename": filename, 
                "chunk_index": idx, 
                "text": c,
                "uploaded_at": timestamp,
                "type": file_type
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
                        "text": c.get('text',''),
                        "uploaded_at": d.get('uploaded_at') or timestamp,
                        "type": file_type,
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

    if not query:
        monitor.end_request(request_id, success=False, error="query is required")
        return jsonify({"error": "query is required"}), 400

    if simulate_latency:
        await asyncio.sleep(random.uniform(0.05, 0.4))

    # Use QdrantService for search (which now handles filtering)
    # The new search_index method does embedding internally too, making this route simpler.
    # However, for consistency with original logic which handled in-memory search separately,
    # I should check if I should delegate completely or keep some logic here.
    # VectorStoreService handles both in-memory and Qdrant in theory if I had moved in-memory logic there fully.
    # But QdrantService in vector_store.py currently assumes Qdrant is primary and returns empty if unavailable.
    
    # If using in-memory only:
    if not lifecycle.qdrant_client:
        # manual embedding
        q_embs = await embedding_adapter.embed_texts([query], provider=settings.embedding_provider, context=f"search:{query[:80]}")
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

        q_embs = await embedding_adapter.embed_texts([query], provider=settings.embedding_provider, context=f"search:{query[:80]}")
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
                    'chunk_text': (payload.get('text', '')[:400] + ('...' if len(payload.get('text', '')) > 400 else '')),
                    'score': float(score),
                })
        except Exception as e:
            logger.error(f"Search failed: {e}")
            
        top = results # already limited by Qdrant
        unique_files = set(r['filename'] for r in top)

    answer = None
    # Debug: log AI branch decisions
    logger.info("AI branch check: demo=%s, AsyncAzureOpenAI=%s, key=%s, endpoint=%s, deployment=%s", settings.demo, bool(AsyncAzureOpenAI), getattr(settings, 'azure_openai_api_key', None), getattr(settings, 'azure_openai_endpoint', None), getattr(settings, 'azure_deployment_chat', None))

    # Demo-mode branch: use deterministic DemoLLM and clearly note that no chat model is available
    if settings.demo:
        try:
            # Apply demo matching thresholds: only include chunks that pass confidence checks
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
                cleaned, source_indicator = await demo.generate(query, context)
                answer = f"(Demo mode - no chat model available) {cleaned}"
        except Exception as e:
            logger.exception("Demo LLM failed: %s", e)
            answer = "(Demo mode) Sorry, demo generation failed."

    elif not settings.demo and AsyncAzureOpenAI:
        try:
            if not settings.azure_openai_api_key or not settings.azure_openai_endpoint or not settings.azure_deployment_chat:
                answer = "Azure OpenAI not configured (Key, Endpoint, or Chat Deployment missing)."
            else:
                # Build context from top search results
                context_parts = [f"Source ({r['filename']}): {r['chunk_text']}" for r in top]
                context = "\n\n".join(context_parts)

                # Ask via AIManager (returns cleaned text and source indicator)
                cleaned, source_indicator = await support_agent.ai_manager.ask_agent_with_context(query, context, conversation_history=[])

                if cleaned == "__CONTENT_FILTER_BLOCKED__":
                    answer = "Content blocked by safety filters."
                else:
                    answer = cleaned
        except Exception as e:
            logger.exception("RAG Generation failed")
            answer = f"Error generating answer: {str(e)}"

    monitor.end_request(request_id, success=True)
    return jsonify({
        "results": top, 
        "used_documents": [{"doc_id": r.get("doc_id"), "filename": r.get("filename"), "score": r.get("score"), "chunk_text": r.get("chunk_text")} for r in top],
        "demo": settings.demo, 
        "message": DEMO_BANNER if settings.demo else "",
        "answer": answer,
        "sources": list(unique_files) if 'unique_files' in locals() else [],
        "embedding_provider": getattr(settings, "embedding_provider", None)
    })


@app.route('/optimize', methods=['POST'])
async def optimize_store():
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
async def reset_store():
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
