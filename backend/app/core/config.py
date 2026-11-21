"""Application settings, read once from the environment.

Environment variable names are matched case-insensitively (Railway stores the Gemini
key as ``gemini_api_key``); an exact-case match wins when both exist.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_TRUE = ("1", "true", "yes", "on")
_BACKEND_DIR = Path(__file__).resolve().parents[2]
THINKING_LEVELS = ("", "minimal", "low", "medium", "high")


class ConfigError(RuntimeError):
    """Raised at startup when the configuration is unsafe or inconsistent."""


class _Env:
    def __init__(self, env: Mapping[str, str]) -> None:
        self._exact = dict(env)
        self._lower = {k.lower(): v for k, v in env.items()}

    def get(self, name: str, default: str = "") -> str:
        value = self._exact.get(name)
        if value is None or value == "":
            value = self._lower.get(name.lower())
        return value if value not in (None, "") else default

    def flag(self, name: str, default: bool = False) -> bool:
        raw = self.get(name)
        return raw.strip().lower() in _TRUE if raw else default

    def integer(self, name: str, default: int) -> int:
        raw = self.get(name)
        try:
            return int(raw) if raw else default
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer") from exc

    def number(self, name: str, default: float) -> float:
        raw = self.get(name)
        try:
            return float(raw) if raw else default
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class Settings:
    demo: bool = True
    demo_locked: bool = False

    # Vector store
    qdrant_url: str | None = None
    qdrant_path: str = str(_BACKEND_DIR.parent / "local_qdrant_db")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    # Embeddings
    embedding_provider: str = "gemini"
    embedding_dim: int = 768
    embedding_timeout: int = 30
    gemini_embedding_model: str = "gemini-embedding-001"

    # Gemini
    gemini_api_key: str | None = None
    # Lite models answer grounded RAG prompts in seconds on the free tier; full Flash models
    # queued for 50-90 s or returned 429/503 when measured (Nov 2026).
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_model_fallback: str = "gemini-3.1-flash-lite"
    # Seconds before a model call is abandoned in favour of the fallback model.
    gemini_timeout: float = 20.0
    max_tokens: int = 2000
    temperature: float = 0.1
    # Reasoning effort for Gemini 3.x ("minimal", "low", "medium", "high"; empty = model default).
    # Grounded answers from retrieved text need little reasoning; higher levels cost seconds and tokens.
    gemini_thinking_level: str = "low"
    max_context_chars: int = 6000

    # Secrets
    admin_token: str | None = None
    search_api_token: str | None = None
    payload_encryption_key: str | None = None
    database_url: str | None = None

    # Search
    max_search_limit: int = 15
    demo_match_ratio: float = 0.6
    demo_min_match_score: float = 0.0

    # Limits
    rate_limit_per_min: int = 10
    rate_limit_per_day: int = 100
    rate_limit_per_session: int = 20
    daily_token_budget: int = 50000
    max_upload_bytes: int = 25 * 1024 * 1024

    # Misc
    cors_allow_origins: tuple[str, ...] = ()
    log_to_file: bool = False
    log_path: str = str(_BACKEND_DIR / "logs")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, load_dotenv_file: bool = True) -> Settings:
        if env is None:
            if load_dotenv_file:
                load_dotenv(find_dotenv(usecwd=False) or None)
            env = os.environ
        e = _Env(env)
        defaults = cls()
        origins = tuple(o.strip() for o in e.get("CORS_ALLOW_ORIGINS").split(",") if o.strip())
        return cls(
            demo=e.flag("DEMO", True),
            demo_locked=e.flag("BACKEND_DEMO_LOCKED", False),
            qdrant_url=e.get("QDRANT_URL") or None,
            qdrant_path=e.get("QDRANT_PATH", defaults.qdrant_path),
            qdrant_api_key=e.get("QDRANT_API_KEY") or None,
            qdrant_collection=e.get("QDRANT_COLLECTION", defaults.qdrant_collection),
            embedding_provider=e.get("EMBEDDING_PROVIDER", defaults.embedding_provider).strip().lower(),
            embedding_dim=e.integer("EMBEDDING_DIM", defaults.embedding_dim),
            embedding_timeout=e.integer("EMBEDDING_TIMEOUT", defaults.embedding_timeout),
            gemini_embedding_model=e.get("GEMINI_EMBEDDING_MODEL", defaults.gemini_embedding_model),
            gemini_api_key=e.get("GEMINI_API_KEY") or e.get("GOOGLE_API_KEY") or None,
            gemini_model=e.get("GEMINI_MODEL", defaults.gemini_model),
            gemini_model_fallback=e.get("GEMINI_MODEL_FALLBACK", defaults.gemini_model_fallback),
            gemini_timeout=e.number("GEMINI_TIMEOUT", defaults.gemini_timeout),
            max_tokens=e.integer("MAX_TOKENS", defaults.max_tokens),
            temperature=e.number("TEMPERATURE", defaults.temperature),
            gemini_thinking_level=e.get("GEMINI_THINKING_LEVEL", defaults.gemini_thinking_level).strip().lower(),
            max_context_chars=e.integer("MAX_CONTEXT_CHARS", defaults.max_context_chars),
            admin_token=e.get("ADMIN_TOKEN") or None,
            search_api_token=e.get("SEARCH_API_TOKEN") or None,
            payload_encryption_key=e.get("PAYLOAD_ENCRYPTION_KEY").strip() or None,
            database_url=e.get("DATABASE_URL") or None,
            max_search_limit=e.integer("MAX_SEARCH_LIMIT", defaults.max_search_limit),
            demo_match_ratio=e.number("DEMO_MATCH_RATIO", defaults.demo_match_ratio),
            demo_min_match_score=e.number("DEMO_MIN_MATCH_SCORE", defaults.demo_min_match_score),
            rate_limit_per_min=e.integer("RATE_LIMIT_PER_MIN", defaults.rate_limit_per_min),
            rate_limit_per_day=e.integer("RATE_LIMIT_PER_DAY", defaults.rate_limit_per_day),
            rate_limit_per_session=e.integer("RATE_LIMIT_PER_SESSION", defaults.rate_limit_per_session),
            daily_token_budget=e.integer("DAILY_TOKEN_BUDGET", defaults.daily_token_budget),
            max_upload_bytes=e.integer("MAX_UPLOAD_BYTES", defaults.max_upload_bytes),
            cors_allow_origins=origins,
            log_to_file=e.flag("LOG_TO_FILE", False),
            log_path=e.get("LOG_PATH", defaults.log_path),
        )

    def validate(self) -> None:
        """Refuse to start with an unsafe production configuration."""
        if self.embedding_dim < 1:
            raise ConfigError("EMBEDDING_DIM must be positive")
        if self.gemini_thinking_level not in THINKING_LEVELS:
            raise ConfigError(f"GEMINI_THINKING_LEVEL must be one of: {', '.join(THINKING_LEVELS[1:])} (or empty)")
        if self.demo:
            return
        if self.embedding_provider != "gemini":
            raise ConfigError("EMBEDDING_PROVIDER must be 'gemini' when DEMO=false")
        if not self.payload_encryption_key:
            raise ConfigError("PAYLOAD_ENCRYPTION_KEY is required when DEMO=false")

    @property
    def public_view(self) -> dict[str, object]:
        """Allow-listed, secret-free settings for the operator console."""
        names = (
            "demo",
            "qdrant_collection",
            "embedding_provider",
            "embedding_dim",
            "embedding_timeout",
            "gemini_model",
            "gemini_model_fallback",
            "max_search_limit",
            "max_tokens",
            "temperature",
            "gemini_thinking_level",
            "max_context_chars",
            "demo_match_ratio",
            "demo_min_match_score",
            "rate_limit_per_min",
            "rate_limit_per_day",
            "rate_limit_per_session",
            "daily_token_budget",
            "log_to_file",
        )
        known = {f.name for f in fields(self)}
        return {n: getattr(self, n) for n in names if n in known}
