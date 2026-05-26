"""Static model catalog used by the backend viewer.

Pricing is $0 because we are on the free Google AI Studio tier. The viewer
reads `role` from this list and from settings.gemini_model to mark the active
primary/fallback pair.
"""
from __future__ import annotations

from typing import List, Dict, Any


MODELS: List[Dict[str, Any]] = [
    {
        "id": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Free tier (10 RPM / 250 RPD). Stable Flash — fast, multimodal, supports JSON schema.",
    },
    {
        "id": "gemini-2.5-flash-lite",
        "label": "Gemini 2.5 Flash Lite",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Free tier (15 RPM / 1000 RPD). Highest throughput — used as fallback.",
    },
    {
        "id": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Free tier (5 RPM). Strongest reasoning, but tight rate limit.",
    },
    {
        "id": "gemini-3.5-flash",
        "label": "Gemini 3.5 Flash",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Newer frontier Flash. Available on free tier — swap in via GEMINI_MODEL if you want more intelligence.",
    },
]


def catalog_for_viewer(active_primary: str, active_fallback: str) -> List[Dict[str, Any]]:
    """Return MODELS with `role` annotation based on the configured model ids."""
    primary_l = (active_primary or '').strip().lower()
    fallback_l = (active_fallback or '').strip().lower()

    out = []
    for m in MODELS:
        mid = m["id"].lower()
        if primary_l and (mid in primary_l or primary_l in mid):
            role = "primary"
        elif fallback_l and (mid in fallback_l or fallback_l in mid):
            role = "fallback"
        else:
            role = "alternative"
        out.append({**m, "role": role})
    return out
