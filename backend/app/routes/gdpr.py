"""GDPR-facing endpoints: forget, export, audit, policy.

Forget and export operate on the *current* session cookie only — so they remain
safe to expose in the locked public demo without letting visitors touch each
other's data.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import List, Dict, Any

from quart import Blueprint, jsonify, request, g
from qdrant_client.http import models as qmodels

try:
    from ..config import settings
    from ..module.lifecycle import lifecycle
    from ..module.apimonitor import monitor
    from ..module.crypto import decrypt_text, is_enabled as crypto_enabled
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
    import config as _cfg
    settings = _cfg.settings
    from module.lifecycle import lifecycle
    from module.apimonitor import monitor
    from module.crypto import decrypt_text, is_enabled as crypto_enabled

logger = logging.getLogger(__name__)

gdpr_bp = Blueprint('gdpr', __name__, url_prefix='/gdpr')

_forget_count = 0
_export_count = 0


def _session_id() -> str:
    return getattr(g, 'session_id', '') or request.cookies.get('demo_session_id', '')


async def _scroll_session_points(session_id: str) -> List[Any]:
    """Return all Qdrant points whose payload.session_id matches."""
    client = lifecycle.qdrant_client
    if client is None:
        return []
    collection = settings.qdrant_collection

    points: List[Any] = []
    offset = None
    try:
        flt = qmodels.Filter(must=[
            qmodels.FieldCondition(key='session_id', match=qmodels.MatchValue(value=session_id))
        ])
        while True:
            result = await asyncio.to_thread(
                client.scroll,
                collection_name=collection,
                scroll_filter=flt,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            batch, next_offset = result if isinstance(result, tuple) else (result, None)
            if not batch:
                break
            points.extend(batch)
            if not next_offset:
                break
            offset = next_offset
    except Exception as e:
        logger.exception("Failed to scroll session points for %s: %s", session_id, e)
    return points


@gdpr_bp.post('/forget')
async def forget():
    """Art. 17 — Right to be forgotten. Deletes all data tied to current session cookie."""
    global _forget_count

    session_id = _session_id()
    if not session_id:
        return jsonify({"error": "no_session", "detail": "No session cookie present."}), 400

    points = await _scroll_session_points(session_id)
    point_ids = [p.id for p in points]

    deleted_points = 0
    if point_ids and lifecycle.qdrant_client is not None:
        try:
            await asyncio.to_thread(
                lifecycle.qdrant_client.delete,
                collection_name=settings.qdrant_collection,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )
            deleted_points = len(point_ids)
        except Exception as e:
            logger.exception("Forget: delete failed for session %s: %s", session_id, e)
            return jsonify({"error": "delete_failed", "detail": str(e)}), 500

    audit_removed = monitor.clear_session_audit(session_id)
    _forget_count += 1

    return jsonify({
        "status": "ok",
        "session_id": session_id,
        "deleted_points": deleted_points,
        "audit_entries_removed": audit_removed,
        "message": "Alle data tilknyttet din session er slettet.",
    })


@gdpr_bp.get('/export')
async def export():
    """Art. 20 — Data portability. Returns all data the system holds for this session."""
    global _export_count

    session_id = _session_id()
    if not session_id:
        return jsonify({"error": "no_session", "detail": "No session cookie present."}), 400

    points = await _scroll_session_points(session_id)
    documents: Dict[str, Dict[str, Any]] = {}
    for p in points:
        payload = getattr(p, 'payload', {}) or {}
        fname = payload.get('filename', 'unknown')
        doc = documents.setdefault(fname, {
            "filename": fname,
            "type": payload.get('type', 'FILE'),
            "uploaded_at": payload.get('uploaded_at', ''),
            "chunks": [],
        })
        doc["chunks"].append({
            "chunk_index": payload.get('chunk_index', 0),
            "text": decrypt_text(payload.get('text', '')),
        })

    for d in documents.values():
        d["chunks"].sort(key=lambda c: c.get('chunk_index', 0))

    audit_entries = monitor.get_audit_events(session_id=session_id, limit=50)
    _export_count += 1

    return jsonify({
        "session_id": session_id,
        "exported_at": datetime.datetime.now().isoformat(),
        "documents": list(documents.values()),
        "audit_log": audit_entries,
    })


@gdpr_bp.get('/audit')
async def audit():
    """Recent PII-redacted prompts. Public so kunden kan se hvad der logges."""
    return jsonify({
        "events": monitor.get_audit_events(limit=50),
        "note": "Alle prompts er PII-scrubbed (email, CPR, telefon) før logging.",
    })


@gdpr_bp.get('/policy')
async def policy():
    """Public GDPR policy summary — drives the /backend GDPR panel."""
    retention_state = {}
    try:
        worker = getattr(lifecycle, '_retention_worker', None)
        if worker:
            retention_state = worker.state()
    except Exception:
        pass

    abuse_opt_out = (os.environ.get('AZURE_ABUSE_MONITORING_OPT_OUT', '').lower() in ('1', 'true', 'yes'))

    return jsonify({
        "data_residency": {
            "qdrant": "local volume on Railway EU",
            "azure_openai_region": os.environ.get('AZURE_REGION', 'swedencentral'),
            "sub_processors": [],
        },
        "encryption_at_rest": {
            "qdrant_payload_text": "AES-256-GCM (pycryptodome)" if crypto_enabled() else "disabled (no PAYLOAD_ENCRYPTION_KEY)",
            "qdrant_vectors": "not applicable — not human readable",
        },
        "retention": retention_state,
        "azure_abuse_monitoring_opt_out": abuse_opt_out,
        "rights": {
            "forget": "POST /gdpr/forget (uses session cookie)",
            "export": "GET /gdpr/export (uses session cookie)",
            "audit_log": "GET /gdpr/audit",
        },
        "stats": {
            "forget_requests_total": _forget_count,
            "export_requests_total": _export_count,
        },
    })


# os is needed for env vars in policy()
import os  # noqa: E402
