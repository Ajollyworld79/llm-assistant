# ruff: noqa: E501
"""Static model catalog used by the operator console.

Pricing is $0 because we are on the free Google AI Studio tier. The viewer
reads `role` from this list and from settings.gemini_model to mark the active
primary/fallback pair.
"""

from __future__ import annotations

from typing import Any

MODELS: list[dict[str, Any]] = [
    {
        "id": "gemini-3.6-flash",
        "label": "Gemini 3.6 Flash",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Stable Flash with thinking. Default primary — no shutdown date announced.",
    },
    {
        "id": "gemini-3.5-flash-lite",
        "label": "Gemini 3.5 Flash-Lite",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Stable Lite. Default fallback — replaces 3.1-flash-lite (shutdown May 7, 2027).",
    },
    {
        "id": "gemini-flash-latest",
        "label": "Gemini Flash (latest)",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Auto-alias to the newest Flash release. May point to a preview/experimental version; hot-swaps with 2-week email notice.",
    },
    {
        "id": "gemini-3.5-flash",
        "label": "Gemini 3.5 Flash",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Stable frontier Flash. Free tier available — use if you want fresh quota-bucket on a newer model.",
    },
    {
        "id": "gemini-3.1-flash-lite",
        "label": "Gemini 3.1 Flash-Lite",
        "vendor": "Google",
        "input_per_m_usd": 0.0,
        "output_per_m_usd": 0.0,
        "context_window": 1_000_000,
        "description": "Stable Lite (released May 2026). Superseded by 3.5-flash-lite; shuts down May 7, 2027.",
    },
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
        "description": "Free tier (15 RPM / 1000 RPD). Older Lite — superseded by 3.1-flash-lite.",
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
]


def catalog_for_viewer(active_primary: str, active_fallback: str) -> list[dict[str, Any]]:
    """Return MODELS with `role` annotation based on the configured model ids."""
    primary_l = (active_primary or "").strip().lower()
    fallback_l = (active_fallback or "").strip().lower()

    out = []
    for m in MODELS:
        mid = m["id"].lower()
        if primary_l and mid == primary_l:
            role = "primary"
        elif fallback_l and mid == fallback_l:
            role = "fallback"
        else:
            role = "alternative"
        out.append({**m, "role": role})
    return out
