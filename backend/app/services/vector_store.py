import logging
import asyncio
import uuid
import datetime
import random
import math
import hashlib
import re
import gc
from typing import List, Dict, Any, Optional, cast

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:
    from backend.app.config import settings
except ImportError:
    try:
        import config
        settings = config.settings # Fallback for script execution
    except ImportError:
        pass

try:
    from backend.app import embeddings
except ImportError:
    try:
        import embeddings # Fallback
    except ImportError:
        pass

# Setup logging
try:
    from backend.app.module.Functions_module import setup_logger
    logger = setup_logger()
except ImportError:
    try:
        from module.Functions_module import setup_logger
        logger = setup_logger()
    except ImportError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
        logger = logging.getLogger(__name__)

# In-memory store (for demo or fallback)
DOCUMENTS: List[Dict[str, Any]] = []

class EmbeddingService:
    """Helper for deterministic embeddings and chunking logic."""

    def __init__(self, dim: int = 384): # Default dim, overridden by settings usually
        self.dim = getattr(settings, 'embedding_dim', dim)

    def deterministic_embed(self, text: str) -> List[float]:
        # Deterministic embedding using md5 digest split into numbers
        h = hashlib.md5(text.encode('utf-8')).digest()
        vals = []
        # expand digest to dim by repeating and combining
        while len(vals) < self.dim:
            for b in h:
                vals.append(b / 255.0)
                if len(vals) >= self.dim:
                    break
            h = hashlib.md5(h).digest()
        # normalize vector
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    def cosine_sim(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    async def chunk_text_async(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.chunk_text, text, chunk_size, overlap)

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """Chunk text by sentences into near chunk_size words with overlap.
        This gives more natural breaks than raw word slicing."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        # Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_words = 0
        for s in sentences:
            sw = len(s.split())
            if current_words + sw <= chunk_size or not current:
                current.append(s)
                current_words += sw
            else:
                chunks.append(' '.join(current))
                # start next chunk with overlap sentences (approx by words)
                # carry over last sentences until overlap words satisfied
                carry = []
                carry_words = 0
                j = len(current) - 1
                while j >= 0 and carry_words < overlap:
                    sentence_j = current[j]
                    carry.insert(0, sentence_j)
                    carry_words += len(sentence_j.split())
                    j -= 1
                current = carry + [s]
                current_words = sum(len(x.split()) for x in current)
        if current:
            chunks.append(' '.join(current))
        return chunks

# Create a single global instance
embedding_service = EmbeddingService()

class EmbeddingAdapter:
    """Adapter to produce embeddings: deterministic demo or call project's `embeddings` module."""

    def __init__(self, settings_obj, service):
        self.settings = settings_obj
        self.embedding_service = service

    async def embed_texts(self, texts: list[str], provider: Optional[str] = None, context: Optional[str] = None) -> list[list[float]]:
        # Demo mode: deterministic embedding
        if getattr(self.settings, 'demo', False):
            # run deterministic embedding synchronously but in thread to avoid blocking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: [self.embedding_service.deterministic_embed(t) for t in texts])

        try:
            # Prefer project's embeddings.embed_texts if available
            prov = provider if provider is not None else getattr(self.settings, 'embedding_provider', "")
            prov = str(prov) if prov is not None else ""
            return await embeddings.embed_texts(texts, provider=prov, context=context)
        except Exception as e:
            logger.warning('EmbeddingAdapter falling back to deterministic due to: %s', e)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: [self.embedding_service.deterministic_embed(t) for t in texts])

# Instantiate adapter
embedding_adapter = EmbeddingAdapter(settings, embedding_service)


class QdrantService:
    """Wrapper for common Qdrant operations used by endpoints."""

    def __init__(self, lifecycle_obj):
        self.lifecycle = lifecycle_obj
        self.collection = settings.qdrant_collection
        self.loading_state = {}

    def is_available(self) -> bool:
        return bool(self.lifecycle.qdrant_client)

    async def scroll(self, limit: int = 2000):
        try:
            return await asyncio.to_thread(self.lifecycle.qdrant_client.scroll, collection_name=self.collection, limit=limit)
        except Exception:
            return ([], None)

    async def list_documents(self):
        if not self.is_available():
            return []
        try:
            res, _ = await self.scroll(limit=2000)
        except Exception:
            return []

        unique_files = {}
        for hit in res or []:
            payload = getattr(hit, 'payload', {}) or {}
            fname = payload.get('filename') or payload.get('file_name') or 'unknown'
            if fname not in unique_files:
                unique_files[fname] = {
                    "id": fname,
                    "filename": fname,
                    "chunks": 0,
                    "uploaded_at": payload.get('uploaded_at', 'N/A'),
                    "type": payload.get('type', 'FILE')
                }
            unique_files[fname]["chunks"] += 1

        return list(unique_files.values())

    async def get_document_chunks(self, doc_id: str):
        if not self.is_available():
            return []
        try:
            res, _ = await asyncio.to_thread(self.lifecycle.qdrant_client.scroll, collection_name=self.collection, limit=1000)
        except Exception:
            return []

        chunks = []
        for hit in res or []:
            payload = getattr(hit, 'payload', {}) or {}
            if payload.get('filename') == doc_id or payload.get('file_name') == doc_id:
                chunks.append({"text": payload.get('text', ''), "chunk_index": payload.get('chunk_index', 0)})

        return chunks

    async def delete_by_filename(self, doc_id: str) -> int:
        """Delete all points whose payload filename/file_name matches doc_id.
        Returns number of deleted points (best-effort)."""
        if not self.is_available():
            return 0

        try:
            scroll_res = await asyncio.to_thread(self.lifecycle.qdrant_client.scroll, collection_name=self.collection, limit=10000)
            all_points = scroll_res[0] if scroll_res else []
        except Exception:
            all_points = []

        ids_to_delete = []
        for point in all_points:
            payload = getattr(point, 'payload', {}) or {}
            if payload.get('filename') == doc_id or payload.get('file_name') == doc_id:
                ids_to_delete.append(point.id)

        if ids_to_delete:
            await asyncio.to_thread(self.lifecycle.qdrant_client.delete, collection_name=self.collection, points_selector=ids_to_delete)

        return len(ids_to_delete)

    async def setup_collection(self) -> bool:
        """Ensure the collection exists and mark it as ready. Returns True if ready."""
        try:
            client = getattr(self.lifecycle, 'qdrant_client', None)
            if client is None:
                logger.warning("Qdrant client not available - cannot setup collection")
                return False

            try:
                collection_info = await asyncio.to_thread(client.get_collection, self.collection)
                logger.info(f"Qdrant collection '{self.collection}' ready: {collection_info.points_count} points")
                self.collection_ready = True
                return True
            except Exception as e:
                logger.info(f"Qdrant collection '{self.collection}' not present or unusable: {e}")
                self.collection_ready = False
                return False
        except Exception as e:
            logger.error(f"setup_collection error: {e}")
            return False

    def preprocess_query(self, query_text: str) -> str:
        """Preprocess query to expand common terms for better matching (English)."""
        try:
            query = " ".join(query_text.strip().split())

            synonyms = {
                "office": "Office 365 Microsoft 365",
                "teams": "Microsoft Teams",
                "outlook": "Microsoft Outlook",
                "word": "Microsoft Word",
                "excel": "Microsoft Excel",
                "powerpoint": "Microsoft PowerPoint",
                "sharepoint": "Microsoft SharePoint",
                "onedrive": "OneDrive",
                "login": "log in login access password account",
                "printer": "printer print printing",
                "internet": "network wifi connection",
                "mail": "email outlook",
                "password": "password change reset update",
                "reset": "reset restart reboot",
            }

            q_lower = query.lower()
            for term, extras in synonyms.items():
                if term in q_lower:
                    query += f" {extras}"

            return query
        except Exception:
            return query_text

    def build_filters(self, filters: Optional[dict]) -> Optional[Any]:
        """Build Qdrant Filter object from a dictionary of simple filters."""
        if not filters:
            return None
        
        must_conditions = []
        
        # 1. Filter by file type (e.g. 'pdf', 'docx')
        if 'file_type' in filters and filters['file_type']:
            ftype = filters['file_type'].upper()
            must_conditions.append(qmodels.FieldCondition(
                key="type", 
                match=qmodels.MatchValue(value=ftype)
            ))
            
        # 2. Filter by date range (uploaded_at >= date_from)
        # Format assumed: YYYY-MM-DD or YYYY-MM-DD HH:MM
        if 'date_from' in filters and filters['date_from']:
            must_conditions.append(qmodels.FieldCondition(
                key="uploaded_at", 
                range=qmodels.Range(gte=filters['date_from'])
            ))
            
        if 'date_to' in filters and filters['date_to']:
             must_conditions.append(qmodels.FieldCondition(
                key="uploaded_at", 
                range=qmodels.Range(lte=filters['date_to'])
            ))

        if not must_conditions:
            return None
            
        return qmodels.Filter(must=must_conditions)

    async def search_index(self, query_text: str, top_k: int | None = None, filters: Optional[dict] = None) -> dict:
        """Search the vector index with retries, returns context and sources.
        Supports metadata filtering (Point 3).
        """
        try:
            if top_k is None:
                top_k = int(getattr(settings, 'TOP_K_RESULTS', 5))

            await self.setup_collection()

            processed_query = self.preprocess_query(query_text)

            # Generate embedding
            emb_list = await embedding_adapter.embed_texts([processed_query], provider=settings.embedding_provider, context=f"search:{processed_query[:80]}")
            if not emb_list:
                return {"context": "", "sources": [], "embedding_time": 0.0, "search_time": 0.0}
            vector = emb_list[0]

            client = getattr(self.lifecycle, 'qdrant_client', None)
            if client is None:
                logger.error("Qdrant client not available - cannot search")
                return {"context": "", "sources": [], "embedding_time": 0.0, "search_time": 0.0}

            # Build filters
            query_filter = self.build_filters(filters)

            # Retry with exponential backoff
            max_retries = 3
            base_delay = 0.5
            results = None
            for attempt in range(max_retries):
                try:
                    # Use search API if present or fallback to query_points
                    if hasattr(client, 'search'):
                        results = await asyncio.to_thread(
                            cast(Any, client).search, 
                            collection_name=self.collection, 
                            query_vector=vector, 
                            query_filter=query_filter, # Apply filters
                            limit=min(top_k*2, getattr(settings, 'MAX_SEARCH_LIMIT', 100)), 
                            score_threshold=getattr(settings, 'QDRANT_SCORE_THRESHOLD', 0.0)
                        )
                    else:
                        # query_points for older client
                        res = await asyncio.to_thread(
                            client.query_points, 
                            collection_name=self.collection, 
                            query=vector, 
                            query_filter=query_filter, # Apply filters
                            limit=top_k*2
                        )
                        results = getattr(res, 'results', res)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Qdrant search attempt {attempt+1} failed: {e}. Retrying in {delay}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Qdrant search failed after {max_retries} attempts: {e}")
                        return {"context": "", "sources": [], "embedding_time": 0.0, "search_time": 0.0}

            # Format results into context and sources
            chunks = []
            source_docs = []
            seen_titles = set()

            if results:
                for hit in results:
                    payload = getattr(hit, 'payload', {}) or {}
                    score = getattr(hit, 'score', None) or 0.0
                    text = payload.get('text', '')
                    title_src = payload.get('file_name') or payload.get('filename') or 'Document'
                    title = title_src.replace('.pdf','').replace('.docx','').replace('_',' ')

                    # Apply simple score threshold if present
                    threshold = float(getattr(settings, 'SIMILARITY_THRESHOLD', 0.0))
                    try:
                        if float(score) <= threshold:
                            continue
                    except Exception:
                        pass

                    if text:
                        if title not in seen_titles:
                            source_docs.append({"title": title, "similarity_score": round(float(score), 3)})
                            seen_titles.add(title)

                        if title and title.lower() not in text[:100].lower():
                            chunks.append(f"**{title}**\n\n{text}")
                        else:
                            chunks.append(text)

                    if len(source_docs) >= top_k:
                        break

            return {"context": "\n\n".join(chunks), "sources": source_docs, "embedding_time": 0.0, "search_time": 0.0}
        except Exception as e:
            logger.error(f"search_index error: {e}")
            return {"context": "", "sources": [], "embedding_time": 0.0, "search_time": 0.0}

    def set_loading_state(self, is_loading: bool, operation: str = "", total_docs: int = 0) -> None:
        if is_loading:
            self.loading_state = {
                "is_loading": True,
                "start_time": datetime.datetime.now().timestamp(),
                "total_documents": total_docs,
                "processed_documents": 0,
                "current_operation": operation,
                # "estimated_completion": ... 
            }
        else:
            self.loading_state = {
                "is_loading": False
            }
            gc.collect()

    async def clear_collection(self) -> dict:
        """Delete all points from the collection by scrolling and deleting in batches."""
        try:
            client = getattr(self.lifecycle, 'qdrant_client', None)
            if client is None:
                return {"success": False, "error": "Qdrant client not available"}

            self.set_loading_state(True, "Clearing collection", 0)
            total_deleted = 0
            offset = None
            batch_size = 100

            while True:
                scroll_result = await asyncio.to_thread(client.scroll, collection_name=self.collection, limit=batch_size, offset=offset, with_payload=False, with_vectors=False)
                points, next_offset = scroll_result
                if not points:
                    break
                ids = [p.id for p in points]
                await asyncio.to_thread(client.delete, collection_name=self.collection, points_selector=ids)
                total_deleted += len(ids)
                offset = next_offset
                if next_offset is None:
                    break

            self.set_loading_state(False)
            return {"success": True, "deleted": total_deleted}
        except Exception as e:
            self.set_loading_state(False)
            logger.error(f"clear_collection error: {e}")
            return {"success": False, "error": str(e)}

    async def optimize_collection(self) -> dict:
        """Trigger collection optimization (best-effort)."""
        try:
            client = getattr(self.lifecycle, 'qdrant_client', None)
            if client is None:
                return {"success": False, "error": "Qdrant client not available"}

            try:
                await asyncio.to_thread(client.update_collection, collection_name=self.collection, optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=0))
                return {"success": True, "message": "Optimization triggered"}
            except Exception as e:
                logger.warning(f"optimize_collection failed: {e}")
                return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"optimize_collection error: {e}")
            return {"success": False, "error": str(e)}


class QdrantAdapter:
    """Thin adapter to upsert/delete/list vectors in Qdrant or use in-memory fallback."""

    def __init__(self, lifecycle_obj, collection_name: str):
        self.lifecycle = lifecycle_obj
        self.collection = collection_name

    async def upsert_document_vectors(self, doc_id: str, filename: str, chunks: list[dict]):
        """Chunks: list of {'text': str, 'vector': list[float], 'chunk_index': int}
        Returns number of upserted points or raises.
        """
        client = getattr(self.lifecycle, 'qdrant_client', None)
        if not client:
            # fallback: store in in-memory DOCUMENTS
            existing = next((d for d in DOCUMENTS if d['id'] == doc_id), None)
            if existing:
                existing['chunks'] = [{'text': c['text'], 'embed': c['vector']} for c in chunks]
                return len(chunks)
            DOCUMENTS.append({
                'id': doc_id,
                'filename': filename,
                'text': ' '.join([c.get('text','') for c in chunks]),
                'chunks': [{'text': c['text'], 'embed': c['vector']} for c in chunks],
                'uploaded_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'type': filename.split('.')[-1].upper() if '.' in filename else 'FILE'
            })
            return len(chunks)

        # Prepare points
        points = []
        skipped = 0
        try:
            for c in chunks:
                vec = c.get('vector') or c.get('embed')
                txt = c.get('text','')

                if vec is None:
                    try:
                        vec = embedding_service.deterministic_embed(txt)
                    except Exception:
                        skipped += 1
                        continue

                # Ensure vector is a plain Python list of floats
                try:
                    if not isinstance(vec, list):
                        vec = list(vec)
                except Exception:
                    skipped += 1
                    continue

                idx = c.get('chunk_index', 0)
                payload = {
                    'filename': filename,
                    'chunk_index': idx,
                    'text': txt,
                    'uploaded_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'type': filename.split('.')[-1].upper() if '.' in filename else 'FILE'
                }

                if qmodels:
                    try:
                        points.append(qmodels.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
                    except Exception as e:
                        logger.warning('qmodels.PointStruct construction failed, falling back to dict point: %s', e)
                        points.append({'id': str(uuid.uuid4()), 'vector': vec, 'payload': payload})
                else:
                    points.append({'id': str(uuid.uuid4()), 'vector': vec, 'payload': payload})

            if not points:
                logger.warning('QdrantAdapter: no valid points to upsert (all chunks skipped) for filename=%s', filename)
                return 0

            # Upsert using client in a thread
            await asyncio.to_thread(client.upsert, collection_name=self.collection, points=points)
            return len(points)
        except Exception as e:
            logger.exception('Failed to upsert vectors: %s', e)
            raise
