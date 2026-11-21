"""The public chat page and the /search endpoint."""

from __future__ import annotations

import base64
import binascii
import logging
import uuid
from typing import Any

from quart import Blueprint, Response, jsonify, render_template, request

from backend.app.core.container import services
from backend.app.core.security import (
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    issue_session_cookie,
    require_search_access,
    set_secure_cookie,
)
from backend.app.services.embeddings import EmbeddingError
from backend.app.services.llm import ImageInput
from backend.app.services.rag import BudgetExhausted, SearchQuery
from backend.app.services.vector_store import StoreUnavailable

logger = logging.getLogger(__name__)
bp = Blueprint("chat", __name__)

MAX_QUERY_CHARS = 4000
MAX_THREAD_ID_CHARS = 128
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
ALLOWED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")
DEFAULT_TOP_K = 5

WELCOME_DA = (
    "Hej. Jeg er din AI-supportassistent til IT, fejlfinding og praktiske spørgsmål. Hvis der findes relevante "
    "dokumenter i RAG-indekset, bruger jeg dem som kilder — ellers svarer jeg ud fra generel viden."
)
WELCOME_EN = (
    "Hi. I'm your AI support assistant for IT, troubleshooting, and practical questions. When relevant documents "
    "exist in the RAG index, I use them as sources — otherwise I answer from general knowledge."
)
WELCOME_DEMO = "(Demo mode er aktiv — svarene er simulerede. / Demo mode is active — answers are simulated.)"


class InvalidRequest(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def welcome_prompt(demo: bool) -> str:
    base = f"{WELCOME_DA}\n\n{WELCOME_EN}"
    return f"{base}\n\n{WELCOME_DEMO}" if demo else base


def _parse_images(raw: Any) -> list[ImageInput]:
    if raw in (None, []):
        return []
    if not isinstance(raw, list) or len(raw) > MAX_IMAGES:
        raise InvalidRequest("too_many_images")
    images = []
    for item in raw:
        if not isinstance(item, dict):
            raise InvalidRequest("invalid_image")
        mime = item.get("mime_type") or "image/png"
        data = item.get("data_base64")
        if mime not in ALLOWED_IMAGE_TYPES:
            raise InvalidRequest("unsupported_image_type")
        if not isinstance(data, str) or not data:
            raise InvalidRequest("invalid_image")
        if len(data) > MAX_IMAGE_BYTES * 4 // 3 + 8:
            raise InvalidRequest("image_too_large")
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidRequest("invalid_image") from exc
        if len(decoded) > MAX_IMAGE_BYTES:
            raise InvalidRequest("image_too_large")
        images.append(ImageInput(mime, decoded))
    return images


def parse_search_payload(payload: Any, max_top_k: int) -> SearchQuery:
    if not isinstance(payload, dict) or "query" not in payload:
        raise InvalidRequest("query required")
    query = payload["query"]
    if not isinstance(query, str):
        raise InvalidRequest("query must be a string")
    query = query.strip()
    if not query:
        raise InvalidRequest("query is required")
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidRequest("query_too_long")

    raw_top_k = payload.get("top_k", DEFAULT_TOP_K)
    if isinstance(raw_top_k, bool):
        raise InvalidRequest("invalid top_k")
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidRequest("invalid top_k") from exc
    top_k = max(1, min(top_k, max(1, max_top_k)))

    raw_thread = payload.get("thread_id")
    if raw_thread is None or raw_thread == "":
        thread_id = uuid.uuid4().hex[:16]
    elif isinstance(raw_thread, str) and len(raw_thread.strip()) <= MAX_THREAD_ID_CHARS:
        thread_id = raw_thread.strip() or uuid.uuid4().hex[:16]
    else:
        raise InvalidRequest("invalid thread_id")

    filters = payload.get("filters")
    file_type = None
    if filters is not None:
        if not isinstance(filters, dict):
            raise InvalidRequest("invalid filters")
        ft = filters.get("file_type")
        if ft is not None and (not isinstance(ft, str) or len(ft) > 16):
            raise InvalidRequest("invalid filters")
        file_type = ft or None

    return SearchQuery(query, top_k, thread_id, _parse_images(payload.get("images")), file_type)


@bp.get("/")
async def index() -> Response:
    svc = services()
    s = svc.settings
    html = await render_template(
        "index.html",
        welcome_prompt=welcome_prompt(s.demo),
        demo_mode=s.demo,
        locked=s.demo_locked,
        asset_v=svc.asset_version,
    )
    response = Response(html, mimetype="text/html")
    try:
        set_secure_cookie(response, SESSION_COOKIE, issue_session_cookie(), SESSION_MAX_AGE)
    except RuntimeError:
        logger.error("Chat session cookie not issued: neither SEARCH_API_TOKEN nor ADMIN_TOKEN is configured")
    return response


@bp.post("/search")
@require_search_access
async def search() -> tuple[Response, int] | Response:
    svc = services()
    try:
        query = parse_search_payload(await request.get_json(silent=True), svc.settings.max_search_limit)
    except InvalidRequest as exc:
        return jsonify({"error": exc.code}), 400
    try:
        return jsonify(await svc.rag.search(query))
    except BudgetExhausted:
        return jsonify({"error": "budget_exhausted", "detail": "The daily token budget is used up."}), 429
    except EmbeddingError:
        return jsonify({"error": "embedding_unavailable"}), 503
    except StoreUnavailable:
        return jsonify({"error": "search_unavailable"}), 503
