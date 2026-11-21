"""Single construction point for the Gemini client shared by embeddings and chat."""

from __future__ import annotations

import logging
from typing import Any

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)


def make_client(settings: Settings) -> Any | None:
    """Return a google-genai client, or None when no API key is configured."""
    if not settings.gemini_api_key:
        return None
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)
