"""Sanitized view of the settings object for the public /backend dashboard.

Uses an allowlist (not denylist) so brand-new env vars don't accidentally leak.
If you add a new public-safe setting, add it to ALLOWED_KEYS explicitly.
"""
from __future__ import annotations

from typing import Dict, Any


# Fields that are safe to surface on the public viewer.
ALLOWED_KEYS = {
    # Mode + storage
    "demo", "qdrant_path", "qdrant_collection",
    # Embeddings
    "embedding_provider", "embedding_dim", "fastembed_model", "embedding_timeout",
    # Gemini (model NAMES are not secrets — only the API key is)
    "gemini_model", "gemini_model_fallback",
    # Search behavior
    "TOP_K_RESULTS", "SIMILARITY_THRESHOLD", "DOMAIN_GUARD_THRESHOLD",
    "QDRANT_SCORE_THRESHOLD", "MAX_SEARCH_LIMIT",
    # AI model params
    "MAX_TOKENS", "TEMPERATURE",
    # Demo match thresholds
    "DEMO_MATCH_RATIO", "DEMO_MIN_MATCH_SCORE",
    # Rate limits + budget
    "RATE_LIMIT_PER_MIN", "RATE_LIMIT_PER_DAY", "RATE_LIMIT_PER_SESSION",
    "DAILY_TOKEN_BUDGET",
    # Logging
    "LOG_TO_FILE",
}

# These exist on settings but must NEVER be exposed.
NEVER_EXPOSE = {
    "gemini_api_key",
    "admin_token",
    "qdrant_api_key",
}


def safe_config_dict(settings_obj) -> Dict[str, Any]:
    """Return only allowlisted fields. Unknown fields are silently skipped."""
    out: Dict[str, Any] = {}
    for key in ALLOWED_KEYS:
        if key in NEVER_EXPOSE:
            continue
        if hasattr(settings_obj, key):
            out[key] = getattr(settings_obj, key)
    return out
