"""Operator-only maintenance endpoints."""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any

from quart import Blueprint, Response, jsonify, request

from backend.app.api.documents import Reply, store_error
from backend.app.core.container import services
from backend.app.core.security import require_admin
from backend.app.infra.demo_lock import reject_if_locked
from backend.app.services.embeddings import EmbeddingError
from backend.app.services.ingest import sanitize_filename
from backend.app.services.parser import file_type
from backend.app.services.vector_store import DimensionMismatch, StoreUnavailable

logger = logging.getLogger(__name__)
bp = Blueprint("admin", __name__)

MAX_VECTOR_DOCUMENTS = 100
MAX_DOC_ID_CHARS = 128


@bp.post("/optimize")
@reject_if_locked
@require_admin
async def optimize() -> Reply:
    try:
        await services().store.optimize()
    except StoreUnavailable as exc:
        return store_error(exc)
    except Exception:
        logger.exception("optimize failed")
        return jsonify({"error": "optimization failed"}), 500
    return jsonify({"success": True, "message": "Optimization triggered"})


@bp.post("/reset")
@reject_if_locked
@require_admin
async def reset() -> Reply:
    try:
        deleted = await services().store.clear()
    except StoreUnavailable as exc:
        return store_error(exc)
    except Exception:
        logger.exception("reset failed")
        return jsonify({"error": "failed"}), 500
    return jsonify({"status": "cleared", "deleted": deleted})


def _is_vector(vec: Any) -> bool:
    return (
        isinstance(vec, list) and bool(vec) and all(isinstance(v, int | float) and not isinstance(v, bool) for v in vec)
    )


async def _upload_one(doc: Any) -> dict[str, Any]:
    svc = services()
    raw_id = doc.get("id") if isinstance(doc, dict) else None
    doc_id = raw_id if isinstance(raw_id, str) and 0 < len(raw_id) <= MAX_DOC_ID_CHARS else str(uuid.uuid4())
    if not isinstance(doc, dict) or not isinstance(doc.get("chunks"), list) or not doc["chunks"]:
        return {"id": doc_id, "status": "skipped", "reason": "no chunks provided"}

    filename = sanitize_filename(doc.get("filename"))
    texts: list[str] = []
    vectors: list[list[float]] = []
    indices: list[int] = []
    for i, chunk in enumerate(doc["chunks"]):
        text = chunk.get("text", "") if isinstance(chunk, dict) else ""
        vec = (chunk.get("vector") or chunk.get("embed")) if isinstance(chunk, dict) else None
        if not isinstance(text, str):
            return {"id": doc_id, "status": "skipped", "reason": f"invalid text for chunk {i}"}
        if not _is_vector(vec):
            if not svc.settings.demo:
                return {"id": doc_id, "status": "skipped", "reason": f"missing vector for chunk {i}"}
            vec = (await svc.embedder.embed([text]))[0]
        index = chunk.get("chunk_index", i) if isinstance(chunk, dict) else i
        texts.append(text)
        vectors.append(vec)  # type: ignore[arg-type]
        indices.append(index if isinstance(index, int) and not isinstance(index, bool) else i)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        count = await svc.store.upsert_chunks(doc_id, filename, file_type(filename), stamp, texts, vectors, indices)
    except DimensionMismatch:
        return {"id": doc_id, "status": "error", "detail": "embedding_dim_mismatch"}
    except (StoreUnavailable, EmbeddingError):
        return {"id": doc_id, "status": "error", "detail": "store_unavailable"}
    return {"id": doc_id, "status": "upserted", "upserted": count}


@bp.post("/admin/upload_vectors")
@reject_if_locked
@require_admin
async def upload_vectors() -> tuple[Response, int] | Response:
    payload = await request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON payload required"}), 400
    docs = payload.get("documents") or ([payload["document"]] if payload.get("document") else None)
    if not isinstance(docs, list) or not docs:
        return jsonify({"error": "documents field missing or empty"}), 400
    if len(docs) > MAX_VECTOR_DOCUMENTS:
        return jsonify({"error": "too_many_documents"}), 400
    return jsonify({"results": [await _upload_one(doc) for doc in docs]})
