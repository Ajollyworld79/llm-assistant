"""Embedding providers abstraction.
Tries to use FastEmbed if available, otherwise falls back to sentence-transformers.
Provides async `embed_texts` which returns list[list[float]]
"""
from typing import List
import asyncio
import logging

log = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except Exception:
    if not TYPE_CHECKING:
        TextEmbedding = None
    HAS_FASTEMBED = False

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE = True
except Exception:
    if not TYPE_CHECKING:
        SentenceTransformer = None
    HAS_SENTENCE = False

_CLASS_ST = None

async def _run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def _normalize_vectors(obj):
    """Normalize different vector types (numpy, list, dict wrappers, etc.) into list[list[float]]."""
    try:
        import numpy as _np
    except Exception:
        _np = None

    # If obj is a dict that wraps embeddings, try to extract common fields
    if isinstance(obj, dict):
        # common shapes: {'embeddings': [...]} or {'data': [{'embedding': [...]}, ...]}
        if 'embeddings' in obj and isinstance(obj['embeddings'], (list, tuple)):
            obj = obj['embeddings']
        elif 'vectors' in obj and isinstance(obj['vectors'], (list, tuple)):
            obj = obj['vectors']
        elif 'data' in obj and isinstance(obj['data'], list):
            extracted = []
            for item in obj['data']:
                if isinstance(item, dict):
                    if 'embedding' in item:
                        extracted.append(item['embedding'])
                    elif 'embeddings' in item:
                        extracted.append(item['embeddings'])
                    elif 'vector' in item:
                        extracted.append(item['vector'])
            if extracted:
                obj = extracted
        elif 'embedding' in obj and isinstance(obj['embedding'], (list, tuple)):
            obj = obj['embedding']

    # If it's a single vector (1D), wrap it
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, (int, float)) for x in obj):
        return [list(obj)]

    # If it's a list of vectors
    if isinstance(obj, list) and obj and all(isinstance(x, (list, tuple)) for x in obj):
        return [list(map(float, v)) for v in obj]

    # numpy array
    if _np is not None and isinstance(obj, _np.ndarray):
        if obj.ndim == 1:
            return [obj.astype(float).tolist()]
        return [row.astype(float).tolist() for row in obj]

    # Fallback: try to iterate
    try:
        return [list(map(float, row)) for row in obj]
    except Exception:
        raise RuntimeError('Unable to normalize embedding output')


def _init_sentence_model(model_name: str = 'all-MiniLM-L6-v2'):
    global _CLASS_ST
    if not HAS_SENTENCE:
        raise RuntimeError('sentence-transformers is not installed')
    if _CLASS_ST is None:
        _CLASS_ST = SentenceTransformer(model_name)
    return _CLASS_ST


async def embed_texts_sentencetransformers(texts: List[str]) -> List[List[float]]:
    model = _init_sentence_model()
    # Prefer numpy output for consistent conversion
    try:
        arr = await _run_blocking(model.encode, texts, show_progress_bar=False, convert_to_numpy=True)
        return _normalize_vectors(arr)
    except TypeError:
        # Older/newer APIs may not accept convert_to_numpy; fall back
        arr = await _run_blocking(model.encode, texts, show_progress_bar=False)
        return _normalize_vectors(arr)


async def embed_texts_fastembed(texts: List[str]) -> List[List[float]]:
    # fastembed API may be synchronous; wrap in thread
    if not HAS_FASTEMBED:
        raise RuntimeError('fastembed is not available')

    try:
        model = TextEmbedding()
        out = await _run_blocking(model.embed, texts)
        vecs = _normalize_vectors(out)
        if vecs:
            return vecs
    except Exception as e:
        log.debug('FastEmbed embedding failed: %s', e)
        raise RuntimeError('FastEmbed embedding failed') from e

async def embed_texts(texts: List[str], provider: str = 'fastembed') -> List[List[float]]:
    # Prefer explicit provider if requested
    if provider == 'fastembed' and HAS_FASTEMBED:
        try:
            log.info('Attempting FastEmbed for embeddings')
            return await embed_texts_fastembed(texts)
        except Exception as e:
            log.exception('FastEmbed failed, falling back: %s', e)
    if HAS_SENTENCE:
        log.info('Using SentenceTransformers for embeddings')
        return await embed_texts_sentencetransformers(texts)
    # If fastembed is available but sentence-transformers is not, and fastembed API differs, attempt a few known call patterns
    if HAS_FASTEMBED:
        try:
            log.info('Trying alternate fastembed API call')
            model = TextEmbedding()
            return await _run_blocking(model.embed, texts)
        except Exception:
            log.warning('FastEmbed call failed')
    # fallback: deterministic simple embeddings
    log.warning('No embedding backend available; using deterministic fallback')
    from hashlib import md5
    import math

    def simple_embed(s: str, dim=64):
        h = md5(s.encode('utf-8')).digest()
        vals = [(b / 255.0) for b in h]
        # expand
        while len(vals) < dim:
            h = md5(h).digest()
            vals.extend([(b / 255.0) for b in h])
        vals = vals[:dim]
        norm = math.sqrt(sum(v*v for v in vals)) or 1.0
        return [v / norm for v in vals]

    return [simple_embed(t, dim=64) for t in texts]
