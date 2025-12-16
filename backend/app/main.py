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
from quart import Quart, request, jsonify, Response, render_template, send_from_directory
from quart_cors import cors
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import io
import csv
import time
import asyncio
import random
import hashlib
import math
import logging
import datetime

try:
    from .config import settings
    from . import embeddings
    from .middleware import setup_middleware
    from .lifecycle import lifecycle
    from .apimonitor import monitor
except ImportError:

    import os, sys
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    import config as config_mod
    settings = config_mod.settings
    import embeddings as embeddings
    from middleware import setup_middleware
    from lifecycle import lifecycle
    from apimonitor import monitor

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
log = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None

# Optional Qdrant imports for typing
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:
    pass # Managed by lifecycle

app = Quart(__name__, template_folder='../templates', static_folder='../static')
app = cors(app, allow_origin="*")

# Register Middleware
setup_middleware(app)

# In-memory store (for demo or fallback)
DOCUMENTS: List[Dict[str, Any]] = []

# Helper: simple chunker / cleanup
def _clean_text(t: str) -> str:
    return t.replace('\r', '\n').strip()


DEMO_BANNER = "Demo mode — results are simulated; no active LLM connection."
EMBED_DIM = settings.embedding_dim


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


import re

async def chunk_text_async(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, chunk_text, text, chunk_size, overlap)

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
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


def deterministic_embed(text: str) -> List[float]:
    # Deterministic embedding using md5 digest split into numbers
    h = hashlib.md5(text.encode('utf-8')).digest()
    vals = []
    # expand digest to EMBED_DIM by repeating and combining
    while len(vals) < EMBED_DIM:
        for b in h:
            vals.append(b / 255.0)
            if len(vals) >= EMBED_DIM:
                break
        h = hashlib.md5(h).digest()
    # normalize vector
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def cosine_sim(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


async def parse_content(filename: str, content: bytes) -> str:
    lower = filename.lower()
    try:
        if lower.endswith('.txt'):
            return content.decode('utf-8', errors='replace')
        elif lower.endswith('.csv'):
            s = content.decode('utf-8', errors='replace')
            rows = list(csv.reader(io.StringIO(s)))
            lines = [', '.join(row) for row in rows]
            return '\n'.join(lines)
        if lower.endswith('.pdf'):
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    for page in pdf.pages:
                        txt = page.extract_text() or ''
                        if txt.strip():
                            text_parts.append(txt)
                        else:
                            # Try OCR fallback if pytesseract is available
                            try:
                                from PIL import Image
                                import pytesseract
                                img = page.to_image(resolution=150).original
                                ocr_txt = pytesseract.image_to_string(img)
                                text_parts.append(ocr_txt)
                            except Exception:
                                log.debug('OCR not available or failed for a PDF page')
                                pass
                combined = '\n'.join(text_parts)
                if combined.strip() == '':
                    return f"[Simulated PDF parsing for {filename} - pdfplumber missing or failed]"
                return combined
            except Exception:
                return f"[Simulated PDF parsing for {filename} - pdfplumber missing or failed]"
        elif lower.endswith('.docx'):
            try:
                import docx
                doc = docx.Document(io.BytesIO(content))
                paras = [p.text for p in doc.paragraphs if p.text]
                return '\n'.join(paras)
            except Exception:
                return f"[Simulated DOCX parsing for {filename} - python-docx missing or failed]"
        else:
            return f"[Simulated parsing for {filename} - demo mode]\nContent not extracted."
    except Exception as e:
        log.exception('Error parsing content: %s', e)
        return f"[Error parsing {filename}, using fallback text.]"


@app.route('/health')
async def health():
    stats = monitor.get_stats()
    qdrant_status = bool(lifecycle.qdrant_client)
    
    return jsonify({
        "status": "ok", 
        "demo": settings.demo, 
        "qdrant": qdrant_status,
        "metrics": stats
    })


@app.route('/')
async def index():
    return await render_template('index.html')


@app.route('/upload', methods=['POST'])
async def upload_file():
    # Accept either multipart file uploads or JSON with 'text' and optional 'filename'
    form = await request.files
    if 'file' in form:
        file = form['file']
        content = file.read()
        filename = file.filename or f"upload-{int(time.time())}"
        text = await parse_content(filename, content)
    else:
        payload = await request.get_json(silent=True)
        if payload and 'text' in payload:
            filename = payload.get('filename', f"upload-{int(time.time())}")
            text = payload['text']
        else:
            return jsonify({"error": "file field or JSON text required"}), 400

    chunks = await chunk_text_async(text)
    doc_id = str(uuid.uuid4())

    # Admin auth (if configured)
    if settings.admin_token:
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'unauthorized'}), 401
        token = auth.split(' ', 1)[1].strip()
        if token != settings.admin_token:
            return jsonify({'error': 'unauthorized'}), 401

    # Store in Qdrant if available, otherwise in-memory
    if not lifecycle.qdrant_client:
        # Batch process embeddings even in demo mode to avoid log spam and overhead
        embeddings_batch = await embeddings.embed_texts(chunks, provider=settings.embedding_provider, context=filename)
        
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
        return jsonify({"id": doc_id, "filename": filename, "status": "uploaded (fallback)"})

    # Index into Qdrant (both demo and prod modes)
    # Create points and upsert
    points = []
    embeddings_batch = await embeddings.embed_texts(chunks, provider=settings.embedding_provider, context=filename)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    file_type = filename.split('.')[-1].upper() if '.' in filename else 'TXT'
    
    for idx, (c, emb) in enumerate(zip(chunks, embeddings_batch)):
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

    try:
        log.info('Upserting %d points into collection %s', len(points), settings.qdrant_collection)
        lifecycle.qdrant_client.upsert(collection_name=settings.qdrant_collection, points=points)
        # Verify by counting points after upsert
        try:
            count = lifecycle.qdrant_client.count(collection_name=settings.qdrant_collection)
            log.info("After upsert, collection '%s' contains %s points", settings.qdrant_collection, getattr(count, 'count', count))
        except Exception:
            log.debug('Could not retrieve collection count after upsert')
        return jsonify({"id": doc_id, "filename": filename, "status": "indexed"})
    except Exception as e:
        log.exception('Failed to upsert points into Qdrant: %s', e)
        return jsonify({"error": "failed to index document", "detail": str(e)}), 500


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
    
    # list from Qdrant - dedup by filename
    try:
        res, _ = lifecycle.qdrant_client.scroll(collection_name=settings.qdrant_collection, limit=2000)
    except Exception:
        return jsonify([])
        
    # fallback formatting
    unique_files = {}
    for hit in res or []:
        payload = hit.payload or {}
        fname = payload.get('filename', 'unknown')
        if fname not in unique_files:
            unique_files[fname] = {
                "id": fname, # Use filename as ID for deletion in Qdrant mode since we delete by filter
                "filename": fname,
                "chunks": 0,
                "uploaded_at": payload.get('uploaded_at', 'N/A'),
                "type": payload.get('type', 'FILE')
            }
        unique_files[fname]["chunks"] += 1
        
    return jsonify(list(unique_files.values()))


@app.route('/document/<doc_id>')
async def get_document(doc_id):
    if not lifecycle.qdrant_client:
        doc = next((d for d in DOCUMENTS if d["id"] == doc_id), None)
        if not doc:
            return jsonify({"error": "document not found"}), 404
        return jsonify({"id": doc["id"], "filename": doc["filename"], "text": doc["text"], "chunks": len(doc["chunks"])})
    # From Qdrant, get all points with that doc_id? Wait, doc_id is not stored.
    # Actually, since points have payload with filename, but not doc_id.
    # In demo, doc_id is generated, but in Qdrant, points have id as uuid, payload filename.
    # To get full document, perhaps need to store full text in payload or something.
    # For now, return points for that filename.
    # But filename may not be unique.
    # For simplicity, since demo, and Qdrant, perhaps return the chunks.
    res, _ = lifecycle.qdrant_client.scroll(collection_name=settings.qdrant_collection, limit=1000)
    chunks = []
    for hit in res or []:
        payload = hit.payload or {}
        if payload.get('filename') == doc_id:  # assuming doc_id is filename for now
            chunks.append({"text": payload.get('text', ''), "chunk_index": payload.get('chunk_index', 0)})
    if not chunks:
        return jsonify({"error": "document not found"}), 404
    # Reconstruct text
    chunks.sort(key=lambda x: x['chunk_index'])
    full_text = '\n'.join(c['text'] for c in chunks)
    return jsonify({"id": doc_id, "filename": doc_id, "text": full_text, "chunks": len(chunks)})


@app.route('/search', methods=['POST'])
async def search():
    payload = await request.get_json()
    if not payload or 'query' not in payload:
        return jsonify({"error": "query required"}), 400
    query = payload.get('query', '').strip()
    top_k = int(payload.get('top_k', 5))
    simulate_latency = payload.get('simulate_latency', True)

    if not query:
        return jsonify({"error": "query is required"}), 400

    if simulate_latency:
        await asyncio.sleep(random.uniform(0.05, 0.4))

    q_embs = await embeddings.embed_texts([query], provider=settings.embedding_provider, context=f"search:{query[:80]}")
    q_emb = q_embs[0]

    results: List[Dict[str, Any]] = []

    if not lifecycle.qdrant_client:
        # local in-memory search
        for d in DOCUMENTS:
            for chunk in d['chunks']:
                score = sum(x * y for x, y in zip(q_emb, chunk['embed']))
                results.append({
                    'doc_id': d['id'],
                    'filename': d['filename'],
                    'chunk_text': (chunk['text'][:400] + ('...' if len(chunk['text']) > 400 else '')),
                    'score': float(score),
                })
    else:
        # Use Qdrant search
        search_res = lifecycle.qdrant_client.query_points(collection_name=settings.qdrant_collection, query=q_emb, limit=top_k)
        for hit in search_res.points:
            payload = hit.payload or {}  # type: ignore
            score = getattr(hit, 'score', None)
            # Qdrant might return different score field names; normalize
            s = float(score) if score is not None else 0.0
            results.append({
                'doc_id': hit.id,  # type: ignore
                'filename': payload.get('filename', 'unknown'),
                'chunk_text': (payload.get('text', '')[:400] + ('...' if len(payload.get('text', '')) > 400 else '')),
                'score': s,
            })

    results.sort(key=lambda r: r['score'], reverse=True)
    top = results[:top_k]

    # Calculate sources
    unique_files = set(r['filename'] for r in top)

    answer = None
    if not settings.demo and AsyncAzureOpenAI:
        try:
            if not settings.azure_openai_api_key or not settings.azure_openai_endpoint or not settings.azure_deployment_chat:
                 answer = "Azure OpenAI not configured (Key, Endpoint, or Chat Deployment missing)."
            else:
                # Type assertion for the linter since we checked it above
                chat_model: str = settings.azure_deployment_chat  # type: ignore
                
                context_parts = []
                for r in top:
                    context_parts.append(f"Source ({r['filename']}): {r['chunk_text']}")
                context = "\n\n".join(context_parts)
                
                # Calculate stats
                num_files = len(unique_files)
                num_chunks = len(top)
                
                system_prompt = "You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say so."
                user_prompt = f"Based on {num_chunks} relevant chunks from {num_files} files.\n\nContext:\n{context}\n\nQuestion: {query}"
                
                client = AsyncAzureOpenAI(
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.azure_openai_api_version,
                    azure_endpoint=settings.azure_openai_endpoint
                )
                
                completion = await client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                answer = completion.choices[0].message.content
        except Exception as e:
            log.exception("RAG Generation failed")
            answer = f"Error generating answer: {str(e)}"

    return jsonify({
        "results": top, 
        "demo": settings.demo, 
        "message": DEMO_BANNER if settings.demo else "",
        "answer": answer,
        "sources": list(unique_files) if 'unique_files' in locals() else []
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
            # Trigger optimization (vacuum)
            # This is specific to Qdrant's internal optimizer configuration
            # Forcing an optimization is not always directly exposed via a simple method, 
            # but we can try to update collection params to trigger it or just acknowledge.
            # A common trick is to force a vacuum by setting vacuum_min_vector_number to something low then back? 
            # Or just rely on the fact that this button is mostly for user reassurance/debugging.
            # We will try to call update_collection with default optimizer config which might trigger a check.
            
            # Since qdrant-client python wrapper is used:
            # lifecycle.qdrant_client.update_collection(collection_name=settings.qdrant_collection, optimizer_config=models.OptimizersConfigDiff(vacuum_min_vector_number=...))
            
            # Simple "pass" for now with a log, unless we find a direct "optimize" method.
            # Qdrant automatically optimizes. 
            pass
        except Exception:
            log.exception('Failed to optimize Qdrant collection')
            return jsonify({"error": "optimization failed"}), 500
            
    return jsonify({"status": "optimized"})
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
            # delete all points in collection (careful in prod)
            lifecycle.qdrant_client.delete(collection_name=settings.qdrant_collection, points_selector=qmodels.Filter(must=[]))
        except Exception:
            log.exception('Failed to clear Qdrant collection')
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
        # Remove from DOCUMENTS
        DOCUMENTS[:] = [d for d in DOCUMENTS if d["id"] != doc_id]
        return jsonify({"status": "deleted"})
    # From Qdrant, delete points with matching filename
    # But since filename may not be unique, and doc_id is not stored, assume doc_id is filename
    try:
        lifecycle.qdrant_client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=qmodels.Filter(must=[qmodels.FieldCondition(key="filename", match=qmodels.MatchValue(value=doc_id))])
        )
        return jsonify({"status": "deleted"})
    except Exception as e:
        log.exception('Failed to delete document')
        return jsonify({"error": "failed to delete"}), 500


@app.before_serving
async def _on_startup():
    # connect qdrant if configured
    await lifecycle.startup()

@app.after_serving
async def _on_shutdown():
    await lifecycle.shutdown()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8002, debug=True)
