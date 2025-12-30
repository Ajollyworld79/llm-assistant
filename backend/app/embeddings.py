# LLM Assistant - Embeddings
# Created by Gustav Christensen
# Date: December 2025
# Description: Embedding providers abstraction for SentenceTransformers, Azure OpenAI, and FastEmbed

"""Embedding providers abstraction.
Prefers `sentence-transformers` (higher semantic quality) and falls back to FastEmbed when needed.
Provides async `embed_texts` which returns list[list[float]]
"""
from typing import List
import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor

# Avoid tokenizers parallelism warning/deadlock when process is forked (e.g., gunicorn)
# See: https://github.com/huggingface/tokenizers/issues/xxx
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ProcessPool executor for CPU-bound FastEmbed work. We use a small pool and
# run the actual embedding inside a worker process to avoid blocking or leaking
# threads when timeouts occur.
_FE_PROCESS_POOL: ProcessPoolExecutor | None = None
# Per-process local model instance inside worker processes (set lazily)
_FE_PROCESS_LOCAL_MODEL = None

# Setup logging
try:
    from .module.Functions_module import setup_logger
    logger = setup_logger()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

try:
    from .config import settings
except Exception:
    import config as _config
    settings = _config.settings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from fastembed import TextEmbedding

# Prefer sentence-transformers if present
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE = True
except Exception:
    if not TYPE_CHECKING:
        SentenceTransformer = None
    HAS_SENTENCE = False

# FastEmbed may be an optional fast backend
try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except Exception:
    if not TYPE_CHECKING:
        TextEmbedding = None
    HAS_FASTEMBED = False

try:
    from openai import AsyncAzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

_CLASS_ST = None
_CLASS_FE = None

async def preload_embedding_models():
    """Preload embedding models to avoid first-request latency."""
    global _CLASS_FE, _CLASS_ST
    # Prefer preloading SentenceTransformers (default preferred backend)
    if HAS_SENTENCE and _CLASS_ST is None:
        try:
            def init_st():
                return SentenceTransformer('all-MiniLM-L6-v2')
            _CLASS_ST = await _run_blocking(init_st)
            # Warm up the model
            if _CLASS_ST is not None:
                def warm_up():
                    _CLASS_ST.encode(["test"])  # type: ignore
                try:
                    await _run_blocking(warm_up)
                except Exception:
                    logger.warning('SentenceTransformer warm-up had an error')
            logger.info('Preloaded and warmed up SentenceTransformer model')
        except Exception as e:
            logger.warning('Failed to preload SentenceTransformer: %s', e)

    # Only preload FastEmbed if SentenceTransformer is NOT loaded, or if specifically requested
    # Check if we successfully loaded ST above
    st_loaded = (_CLASS_ST is not None)
    
    # If ST is loaded, we skip FastEmbed preload to save resources, 
    # unless FastEmbed is the primary configured provider (unlikely given defaults, but possible).
    should_preload_fe = HAS_FASTEMBED and _CLASS_FE is None
    if st_loaded and settings.embedding_provider != 'fastembed':
        should_preload_fe = False

    if should_preload_fe:
        try:
            # initialize with timeout to avoid blocking startup indefinitely
            def init_fe():
                return TextEmbedding()
            try:
                _CLASS_FE = await asyncio.wait_for(_run_blocking(init_fe), timeout=settings.embedding_timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(f'FastEmbed init timed out after {settings.embedding_timeout}s')

            # Warm up the model with a dummy embed (also bounded by timeout)
            if _CLASS_FE is not None:
                def warm_up():
                    list(_CLASS_FE.embed(["test"]))  # type: ignore
                try:
                    await asyncio.wait_for(_run_blocking(warm_up), timeout=max(1, settings.embedding_timeout // 2))
                    logger.info('Preloaded and warmed up FastEmbed model')
                except asyncio.TimeoutError:
                    logger.warning('FastEmbed warm-up timed out')
        except Exception as e:
            logger.warning('Failed to preload FastEmbed: %s', e)

async def _run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# --- FastEmbed process worker helpers ---

def _fe_embed_process(batch: List[str]):
    """Executed inside worker process. Lazily instantiate a local TextEmbedding there
    and return raw embedding output (to be normalized by the parent)."""
    # Local import inside worker process
    from fastembed import TextEmbedding
    global _FE_PROCESS_LOCAL_MODEL
    # Simple lazy initialization; avoid try/except NameError and unused expression
    if _FE_PROCESS_LOCAL_MODEL is None:
        _FE_PROCESS_LOCAL_MODEL = TextEmbedding()
    raw_out = list(_FE_PROCESS_LOCAL_MODEL.embed(batch))
    return raw_out


async def _run_in_fe_process(batch: List[str]) -> List[List[float]]:
    """Run the process worker and normalize output."""
    global _FE_PROCESS_POOL
    loop = asyncio.get_running_loop()
    if _FE_PROCESS_POOL is None:
        _FE_PROCESS_POOL = ProcessPoolExecutor(max_workers=min(4, (os.cpu_count() or 1)))
    raw_out = await loop.run_in_executor(_FE_PROCESS_POOL, _fe_embed_process, batch)
    return _normalize_vectors(raw_out)


async def shutdown_fe_executor():
    """Shutdown the FE process pool (called from lifecycle shutdown)."""
    global _FE_PROCESS_POOL
    if _FE_PROCESS_POOL:
        pool = _FE_PROCESS_POOL
        _FE_PROCESS_POOL = None
        # run shutdown in a thread to avoid blocking the event loop
        await _run_blocking(pool.shutdown, False)


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
    # Batch to avoid large memory usage
    batch_size = 100
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Prefer numpy output for consistent conversion
        try:
            arr = await _run_blocking(model.encode, batch, show_progress_bar=False, convert_to_numpy=True)
            vecs = _normalize_vectors(arr)
        except TypeError:
            # Older/newer APIs may not accept convert_to_numpy; fall back
            arr = await _run_blocking(model.encode, batch, show_progress_bar=False)
            vecs = _normalize_vectors(arr)
        all_vecs.extend(vecs)
    return all_vecs


async def embed_texts_fastembed(texts: List[str]) -> List[List[float]]:
    # fastembed API may be synchronous; wrap in thread
    if not HAS_FASTEMBED:
        raise RuntimeError('fastembed is not available')

    global _CLASS_FE
    
    # Run initialization in thread to avoid blocking if it loads model
    if _CLASS_FE is None:
        def init_fe():
            return TextEmbedding()
        _CLASS_FE = await _run_blocking(init_fe)

    try:
        model = _CLASS_FE
        # FastEmbed .embed() returns a generator. We must consume it in the thread 
        # to ensure the heavy CPU work happens off the main event loop.
        # Batch to avoid large memory usage and long blocking times
        batch_size = 100
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                # Execute CPU-bound work in a process worker to avoid blocking the main
                # thread and to reduce resource leakage when timeouts occur.
                vecs = await asyncio.wait_for(_run_in_fe_process(batch), timeout=settings.embedding_timeout)
            except asyncio.TimeoutError:
                logger.warning('FastEmbed batch embed timed out after %s seconds', settings.embedding_timeout)
                raise RuntimeError('FastEmbed embedding timed out')
            if vecs:
                all_vecs.extend(vecs)
            else:
                raise RuntimeError('FastEmbed returned empty embeddings for batch')
        return all_vecs
    except Exception as e:
        logger.debug('FastEmbed embedding failed: %s', e)
        raise RuntimeError('FastEmbed embedding failed') from e


async def embed_texts_azure(texts: List[str]) -> List[List[float]]:
    try:
        from .config import settings
    except ImportError:
        import config
        settings = config.settings
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed")
    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint or not settings.azure_deployment_embed:
        raise RuntimeError("Azure OpenAI credentials not configured")

    client = AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint
    )
    
    resp = await client.embeddings.create(
        input=texts,
        model=settings.azure_deployment_embed
    )
    return [d.embedding for d in resp.data]


async def _monitor_slow(task: asyncio.Task, provider: str, context: str | None, delay: int = 20):
    """Log when an embedding Task has been running longer than `delay` seconds."""
    try:
        await asyncio.sleep(delay)
        if not task.done():
            logger.warning("Embedding task running > %s seconds (provider=%s, context=%s)", delay, provider, context or "<no-context>")
    except asyncio.CancelledError:
        return


async def embed_texts(texts: List[str], provider: str = 'sentence', context: str | None = None) -> List[List[float]]:
    """Main entrypoint for embeddings. Accepts optional `context` (e.g., filename)
    which will be included in slow-task logs to help debugging."""
    p = (provider or settings.embedding_provider or '').strip().lower()
    # log start with context to make requests visible
    logger.info('Starting embed_texts provider=%s count=%d context=%s', p, len(texts), context or '<none>')

    # Create a placeholder task variable for monitoring; we'll schedule the actual
    # work as a subtask so we can monitor it for slow execution.

    # Helper to run a coroutine and monitor it for slowness
    async def _run_and_monitor(coro):
        task = asyncio.create_task(coro)
        monitor = asyncio.create_task(_monitor_slow(task, p, context, delay=max(20, settings.embedding_timeout)))
        try:
            result = await task
            return result
        finally:
            # Cancel monitor if still pending
            if not monitor.done():
                monitor.cancel()

    # Azure provider explicit
    if p in ('azure', 'azure_openai'):
        try:
            return await _run_and_monitor(embed_texts_azure(texts))
        except Exception as e:
            logger.exception('Azure embedding failed: %s', e)
            raise e

    # Explicit FastEmbed selection
    if p in ('fastembed', 'fast', 'fe'):
        if not HAS_FASTEMBED:
            raise RuntimeError('FastEmbed requested but not available')
        try:
            logger.info('Attempting FastEmbed for embeddings')
            return await _run_and_monitor(embed_texts_fastembed(texts))
        except Exception as e:
            logger.exception('FastEmbed failed when explicitly requested: %s', e)
            raise e

    # SentenceTransformers preferred (default) with fallback
    if p in ('sentence', 'sentence-transformers', 'sentence_transformers', 'st') or p == '':
        if HAS_SENTENCE:
            try:
                logger.info('Using SentenceTransformers for embeddings')
                return await _run_and_monitor(embed_texts_sentencetransformers(texts))
            except Exception as e:
                logger.exception('SentenceTransformers failed, will try FastEmbed as fallback: %s', e)
        # fallback to FastEmbed if available
        if HAS_FASTEMBED:
            try:
                logger.info('Falling back to FastEmbed for embeddings')
                return await _run_and_monitor(embed_texts_fastembed(texts))
            except Exception as e:
                logger.exception('FastEmbed fallback failed: %s', e)

    # If both model types are unavailable or failed, try alternate fastembed call (best-effort)
    if HAS_FASTEMBED:
        try:
            logger.info('Trying alternate fastembed API call')
            model = TextEmbedding()
            return await _run_and_monitor(_run_blocking(model.embed, texts))
        except Exception:
            logger.warning('FastEmbed call failed')

    # fallback: deterministic simple embeddings
    logger.warning('No embedding backend available; using deterministic fallback')
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
