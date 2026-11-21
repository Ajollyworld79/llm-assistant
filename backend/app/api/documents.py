"""Operator-only document management: upload, list, read, delete."""

from __future__ import annotations

import asyncio
import logging

from quart import Blueprint, Response, jsonify, request

from backend.app.core.container import services
from backend.app.core.security import require_admin
from backend.app.infra.demo_lock import reject_if_locked
from backend.app.services.embeddings import EmbeddingError
from backend.app.services.ingest import sanitize_filename
from backend.app.services.parser import SUPPORTED_EXTENSIONS, EmptyDocument, ParseError, UnsupportedFileType
from backend.app.services.vector_store import DimensionMismatch, StoreUnavailable

logger = logging.getLogger(__name__)
bp = Blueprint("documents", __name__)

Reply = tuple[Response, int] | Response


def store_error(exc: Exception) -> tuple[Response, int]:
    """Map domain failures to HTTP responses without echoing exception text."""
    if isinstance(exc, UnsupportedFileType):
        return jsonify({"error": "unsupported_file_type", "supported": list(SUPPORTED_EXTENSIONS)}), 415
    if isinstance(exc, EmptyDocument):
        return jsonify({"error": "empty_document"}), 422
    if isinstance(exc, ParseError):
        return jsonify({"error": "unreadable_file"}), 422
    if isinstance(exc, DimensionMismatch):
        detail = f"Collection vectors have size {exc.expected} but the embeddings have size {exc.actual}."
        return jsonify({"error": "embedding_dim_mismatch", "details": detail}), 409
    if isinstance(exc, EmbeddingError):
        return jsonify({"error": "embedding_unavailable"}), 502
    if isinstance(exc, StoreUnavailable):
        return jsonify({"error": "store_unavailable"}), 503
    raise exc


@bp.post("/upload")
@reject_if_locked
@require_admin
async def upload() -> Reply:
    svc = services()
    files = await request.files
    try:
        if "file" in files:
            upload_file = files["file"]
            content = await asyncio.to_thread(upload_file.read)
            if len(content) > svc.settings.max_upload_bytes:
                return jsonify({"error": "payload_too_large"}), 413
            result = await svc.ingest.ingest_file(sanitize_filename(upload_file.filename), content)
        else:
            payload = await request.get_json(silent=True)
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                return jsonify({"error": "file field or JSON text required"}), 400
            result = await svc.ingest.ingest_text(sanitize_filename(payload.get("filename")), payload["text"])
    except (ParseError, EmbeddingError, DimensionMismatch, StoreUnavailable) as exc:
        return store_error(exc)
    return jsonify(
        {
            "id": result.doc_id,
            "filename": result.filename,
            "status": "indexed",
            "requested_chunks": result.chunks,
            "embedded_chunks": result.chunks,
            "failed_chunks": 0,
            "embedding_provider": svc.settings.embedding_provider,
        }
    )


@bp.get("/documents")
@require_admin
async def list_documents() -> Reply:
    try:
        docs = await services().store.list_documents()
    except StoreUnavailable as exc:
        return store_error(exc)
    return jsonify([d.as_dict() for d in docs])


@bp.get("/document/<doc_id>")
@require_admin
async def get_document(doc_id: str) -> Reply:
    try:
        found = await services().store.get_document(doc_id)
    except StoreUnavailable as exc:
        return store_error(exc)
    if found is None:
        return jsonify({"error": "document not found"}), 404
    filename, text, chunks = found
    return jsonify({"id": doc_id, "filename": filename, "text": text, "chunks": chunks})


@bp.delete("/document/<doc_id>")
@reject_if_locked
@require_admin
async def delete_document(doc_id: str) -> Reply:
    try:
        deleted = await services().store.delete_document(doc_id)
    except StoreUnavailable as exc:
        return store_error(exc)
    if not deleted:
        return jsonify({"error": "document not found"}), 404
    return jsonify({"status": "deleted", "deleted": deleted})
