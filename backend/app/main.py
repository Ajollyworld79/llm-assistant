"""Quart-based demo backend for Qdrant-style search + FastEmbed simulation.
- Async endpoints
- Upload and parse files (txt, csv, pdf, docx)
- Chunking and deterministic embedding (hash-based)
- Search using cosine similarity
- Demo-mode flag and simulated latency
"""
from quart import Quart, request, jsonify, Response
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

# Support running module directly (script) or as a package
try:
    from .config import settings
    from . import embeddings
except Exception:
    # When run as a script (python backend/app/main.py) relative imports fail;
    # add the app directory to sys.path and import modules directly.
    import os, sys
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    import config as config_mod
    settings = config_mod.settings
    import embeddings as embeddings

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
log = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

# Optional Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    HAS_QDRANT = True
except Exception:
    if not TYPE_CHECKING:
        QdrantClient = None
        qmodels = None
    HAS_QDRANT = False

app = Quart(__name__)
app = cors(app, allow_origin="*")

# In-memory store (for demo or fallback)
DOCUMENTS: List[Dict[str, Any]] = []

# Helper: simple chunker / cleanup
def _clean_text(t: str) -> str:
    return t.replace('\r', '\n').strip()


DEMO_BANNER = "Demo mode — results are simulated; no active LLM connection."
EMBED_DIM = settings.embedding_dim
QDRANT_CLIENT: QdrantClient | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


import re

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
    return jsonify({"status": "ok", "demo": settings.demo, "qdrant": bool(QDRANT_CLIENT)})


@app.route('/upload', methods=['POST'])
async def upload_file():
    # Accept either multipart file uploads or JSON with 'text' and optional 'filename'
    form = await request.files
    if 'file' in form:
        file = form['file']
        content = await file.read()
        filename = file.filename or f"upload-{int(time.time())}"
        text = await parse_content(filename, content)
    else:
        payload = await request.get_json(silent=True)
        if payload and 'text' in payload:
            filename = payload.get('filename', f"upload-{int(time.time())}")
            text = payload['text']
        else:
            return jsonify({"error": "file field or JSON text required"}), 400

    chunks = chunk_text(text)
    doc_id = str(uuid.uuid4())

    # Admin auth (if configured)
    if settings.admin_token:
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'unauthorized'}), 401
        token = auth.split(' ', 1)[1].strip()
        if token != settings.admin_token:
            return jsonify({'error': 'unauthorized'}), 401

    # In demo mode, store in-memory with deterministic embeddings
    if settings.demo or not QDRANT_CLIENT:
        chunk_objs = []
        for c in chunks:
            emb = await embeddings.embed_texts([c], provider=settings.embedding_provider)
            chunk_objs.append({"text": c, "embed": emb[0]})

        DOCUMENTS.append({
            "id": doc_id,
            "filename": filename,
            "text": text,
            "chunks": chunk_objs,
            "uploaded_at": time.time(),
        })
        return jsonify({"id": doc_id, "filename": filename, "status": "uploaded (demo)"})

    # Production mode: index into Qdrant
    # Create points and upsert
    points = []
    embeddings_batch = await embeddings.embed_texts(chunks, provider=settings.embedding_provider)
    for idx, (c, emb) in enumerate(zip(chunks, embeddings_batch)):
        points.append(qmodels.PointStruct(id=str(uuid.uuid4()), vector=emb, payload={"filename": filename, "chunk_index": idx, "text": c}))

    QDRANT_CLIENT.upsert(collection_name=settings.qdrant_collection, points=points)
    return jsonify({"id": doc_id, "filename": filename, "status": "indexed"})


@app.route('/documents')
async def list_documents():
    if settings.demo or not QDRANT_CLIENT:
        return jsonify([{"id": d["id"], "filename": d["filename"], "chunks": len(d["chunks"])} for d in DOCUMENTS])
    # list from Qdrant
    res, _ = QDRANT_CLIENT.scroll(collection_name=settings.qdrant_collection, limit=100)
    # fallback formatting
    docs = []
    for hit in res or []:
        payload = hit.payload or {}
        docs.append({"id": hit.id, "filename": payload.get('filename', 'unknown')})
    return jsonify(docs)


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

    q_embs = await embeddings.embed_texts([query], provider=settings.embedding_provider)
    q_emb = q_embs[0]

    results: List[Dict[str, Any]] = []

    if settings.demo or not QDRANT_CLIENT:
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
        search_res = QDRANT_CLIENT.query_points(collection_name=settings.qdrant_collection, query=q_emb, limit=top_k)
        for hit in search_res:
            payload = hit.payload or {}
            score = getattr(hit, 'score', None)
            # Qdrant might return different score field names; normalize
            s = float(score) if score is not None else 0.0
            results.append({
                'doc_id': hit.id,
                'filename': payload.get('filename', 'unknown'),
                'chunk_text': (payload.get('text', '')[:400] + ('...' if len(payload.get('text', '')) > 400 else '')),
                'score': s,
            })

    results.sort(key=lambda r: r['score'], reverse=True)
    top = results[:top_k]

    return jsonify({"results": top, "demo": settings.demo, "message": DEMO_BANNER})


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
    if QDRANT_CLIENT and not settings.demo:
        try:
            # delete all points in collection (careful in prod)
            QDRANT_CLIENT.delete(collection_name=settings.qdrant_collection, filter={})
        except Exception:
            log.exception('Failed to clear Qdrant collection')
    return jsonify({"status": "cleared"})


async def _startup_connect_qdrant():
    global QDRANT_CLIENT
    if settings.qdrant_url:
        try:
            QDRANT_CLIENT = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            # Create collection if not exists
            try:
                QDRANT_CLIENT.get_collection(settings.qdrant_collection)
            except Exception:
                QDRANT_CLIENT.recreate_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=qmodels.VectorParams(size=settings.embedding_dim, distance=qmodels.Distance.COSINE)
                )
            # If not demo, we can also ensure indices etc.
        except Exception as e:
            QDRANT_CLIENT = None

@app.before_serving
async def _on_startup():
    # connect qdrant if configured
    await _startup_connect_qdrant()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='127.0.0.1', port=8000, reload=True)
