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
    # Azure (deployment NAMES are not secrets — endpoint URL and key are)
    "azure_openai_api_version", "azure_deployment_chat", "azure_deployment_chat_fallback",
    "azure_deployment_embed",
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
    # GDPR retention
    "DOCUMENT_TTL_DAYS", "LOG_RETENTION_DAYS", "RETENTION_SCAN_INTERVAL_SECONDS",
    # Logging
    "LOG_TO_FILE",
}

# These exist on settings but must NEVER be exposed.
NEVER_EXPOSE = {
    "azure_openai_api_key",
    "azure_openai_endpoint",  # URL leaks tenant
    "admin_token",
    "qdrant_api_key",
}


def safe_config_dict(settings_obj) -> Dict[str, Any]:
    """Return only allowlisted fields. Unknown fields are silently skipped."""
    out: Dict[str, Any] = {}
    for key in ALLOWED_KEYS:
        if key in NEVER_EXPOSE:
            # Defensive — shouldn't be in allowlist anyway.
            continue
        if hasattr(settings_obj, key):
            value = getattr(settings_obj, key)
            # Mask deployment names that contain secret-looking suffixes? No —
            # deployment names are not secrets. The KEY is.
            out[key] = value
    return out
