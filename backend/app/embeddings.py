"""Embedding providers — fastembed (ONNX) is primary, Azure OpenAI is alternate.

The previous sentence-transformers backend pulled torch (~800MB) which broke Railway
deploys. fastembed uses onnxruntime and stays well under 200MB resident.
"""
from typing import List, Optional
import asyncio
import logging
import os

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

try:
    from .module.Functions_module import setup_logger
    logger = setup_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

try:
    from .config import settings
except Exception:
    import config as _config
    settings = _config.settings

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except Exception:
    TextEmbedding = None
    HAS_FASTEMBED = False

try:
    from openai import AsyncAzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


_FE_MODEL: Optional["TextEmbedding"] = None


def _is_e5_model(name: str) -> bool:
    return 'e5' in (name or '').lower()


def _apply_e5_prefix(texts: List[str], is_query: bool) -> List[str]:
    prefix = "query: " if is_query else "passage: "
    return [prefix + t for t in texts]


async def _run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def _normalize_vectors(obj) -> List[List[float]]:
    try:
        import numpy as _np
    except Exception:
        _np = None

    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, (int, float)) for x in obj):
        return [list(map(float, obj))]
    if isinstance(obj, list) and obj and all(isinstance(x, (list, tuple)) for x in obj):
        return [list(map(float, v)) for v in obj]
    if _np is not None and isinstance(obj, _np.ndarray):
        if obj.ndim == 1:
            return [obj.astype(float).tolist()]
        return [row.astype(float).tolist() for row in obj]
    try:
        return [list(map(float, row)) for row in obj]
    except Exception:
        raise RuntimeError('Unable to normalize embedding output')


def _init_fastembed_model() -> "TextEmbedding":
    global _FE_MODEL
    if not HAS_FASTEMBED:
        raise RuntimeError('fastembed is not installed')
    if _FE_MODEL is None:
        model_name = getattr(settings, 'fastembed_model', 'intfloat/multilingual-e5-small')
        _FE_MODEL = TextEmbedding(model_name=model_name)
    return _FE_MODEL


async def preload_embedding_models() -> None:
    """Load the fastembed model and run a warm-up embed at startup."""
    if not HAS_FASTEMBED:
        logger.warning('fastembed not available; embeddings will use deterministic fallback')
        return
    try:
        await _run_blocking(_init_fastembed_model)
        await _run_blocking(lambda: list(_FE_MODEL.embed(["warm-up"])))  # type: ignore
        logger.info('fastembed model preloaded and warmed up')
    except Exception as e:
        logger.warning('Failed to preload fastembed model: %s', e)


async def shutdown_fe_executor() -> None:
    """Kept for compatibility with lifecycle.shutdown — no longer needs to do anything."""
    return None


async def embed_texts_fastembed(texts: List[str], is_query: bool = False) -> List[List[float]]:
    model = await _run_blocking(_init_fastembed_model)
    model_name = getattr(settings, 'fastembed_model', '')
    inputs = _apply_e5_prefix(texts, is_query) if _is_e5_model(model_name) else texts

    batch_size = 100
    all_vecs: List[List[float]] = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        raw = await _run_blocking(lambda b=batch: list(model.embed(b)))
        all_vecs.extend(_normalize_vectors(raw))
    return all_vecs


async def embed_texts_azure(texts: List[str]) -> List[List[float]]:
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed")
    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint or not settings.azure_deployment_embed:
        raise RuntimeError("Azure OpenAI credentials not configured")

    client = AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )
    resp = await client.embeddings.create(input=texts, model=settings.azure_deployment_embed)
    return [d.embedding for d in resp.data]


def _deterministic_fallback(texts: List[str], dim: int = 64) -> List[List[float]]:
    from hashlib import md5
    import math

    out: List[List[float]] = []
    for t in texts:
        h = md5(t.encode('utf-8')).digest()
        vals = [(b / 255.0) for b in h]
        while len(vals) < dim:
            h = md5(h).digest()
            vals.extend([(b / 255.0) for b in h])
        vals = vals[:dim]
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        out.append([v / norm for v in vals])
    return out


async def embed_texts(texts: List[str], provider: str = 'fastembed', context: Optional[str] = None,
                     is_query: bool = False) -> List[List[float]]:
    """Embed a batch of texts. `is_query=True` applies e5 query-prefix."""
    p = (provider or settings.embedding_provider or 'fastembed').strip().lower()
    logger.info('embed_texts provider=%s count=%d context=%s', p, len(texts), context or '<none>')

    if p in ('azure', 'azure_openai'):
        return await embed_texts_azure(texts)

    if p in ('fastembed', 'fast', 'fe') and HAS_FASTEMBED:
        try:
            return await embed_texts_fastembed(texts, is_query=is_query)
        except Exception as e:
            logger.exception('fastembed failed, using deterministic fallback: %s', e)

    logger.warning('No embedding backend available; using deterministic fallback (dim=64)')
    return _deterministic_fallback(texts)
