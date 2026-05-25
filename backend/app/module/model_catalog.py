"""Static model catalog used by the backend viewer.

Pricing reflects Azure AI Foundry as of May 2026. Numbers are USD per 1M tokens.
The viewer reads `role` from this list and from settings.azure_deployment_chat
to mark the active primary/fallback pair.
"""
from __future__ import annotations

from typing import List, Dict, Any


MODELS: List[Dict[str, Any]] = [
    {
        "id": "gpt-5-nano",
        "label": "GPT-5 nano",
        "vendor": "OpenAI",
        "input_per_m_usd": 0.05,
        "output_per_m_usd": 0.40,
        "context_window": 128_000,
        "description": "Billigste i Azure OpenAI-familien. Hurtig, god til RAG-syntese.",
    },
    {
        "id": "gpt-5-mini",
        "label": "GPT-5 mini",
        "vendor": "OpenAI",
        "input_per_m_usd": 0.15,
        "output_per_m_usd": 0.60,
        "context_window": 128_000,
        "description": "Stærkere reasoning end nano. Anbefales som fallback.",
    },
    {
        "id": "mistral-small-3.2",
        "label": "Mistral Small 3.2",
        "vendor": "Mistral (EU)",
        "input_per_m_usd": 0.10,
        "output_per_m_usd": 0.30,
        "context_window": 128_000,
        "description": "EU-native (fransk firma). Stærkere GDPR-fortælling, multilingual.",
    },
    {
        "id": "gpt-4o-mini",
        "label": "GPT-4o mini",
        "vendor": "OpenAI",
        "input_per_m_usd": 0.15,
        "output_per_m_usd": 0.60,
        "context_window": 128_000,
        "description": "Tidligere baseline. Stadig god, men dyrere end gpt-5-nano.",
    },
    {
        "id": "deepseek-v3.2",
        "label": "DeepSeek V3.2",
        "vendor": "DeepSeek (CN)",
        "input_per_m_usd": 0.14,
        "output_per_m_usd": 0.28,
        "context_window": 64_000,
        "description": "Kvalitet/$-leder. Kinesisk firma — svagere GDPR-vinkel.",
    },
    {
        "id": "grok-3-mini",
        "label": "Grok 3 mini",
        "vendor": "xAI",
        "input_per_m_usd": 0.30,
        "output_per_m_usd": 0.50,
        "context_window": 128_000,
        "description": "xAI på Azure Foundry. Dyrere — kun relevant hvis kunden spørger efter Grok.",
    },
]


def catalog_for_viewer(active_primary: str, active_fallback: str) -> List[Dict[str, Any]]:
    """Return MODELS with `role` annotation based on the deployment names.

    The deployment names are matched case-insensitively against model ids — a
    deployment is typically named after the model, but operators can rename, so
    fall back to `id` matching as a substring check.
    """
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
