"""Shared fixtures: an app built by ``create_app`` with test settings and in-memory Qdrant."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import QdrantClient
from quart import Quart
from quart.testing import QuartClient

from backend.app import create_app
from backend.app.core.config import Settings

ADMIN_TOKEN = "test-admin"
SEARCH_TOKEN = "test-search"
ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
SEARCH = {"Authorization": f"Bearer {SEARCH_TOKEN}"}
OPERATOR_HEADERS = {"X-Requested-With": "helios"}
DIM = 32


@pytest.fixture
def settings() -> Settings:
    return Settings(
        demo=True,
        admin_token=ADMIN_TOKEN,
        search_api_token=SEARCH_TOKEN,
        qdrant_path=":memory:",
        embedding_dim=DIM,
        log_to_file=False,
        rate_limit_per_min=1000,
        rate_limit_per_day=1000,
        rate_limit_per_session=1000,
    )


class FakeEmbeddings:
    def __init__(self, dim: int, fail: bool = False) -> None:
        self.dim, self.fail, self.calls = dim, fail, []

    async def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
        self.calls.append((model, len(contents), config.task_type))
        if self.fail:
            raise RuntimeError("quota exceeded")
        vec = [1.0] + [0.0] * (self.dim - 1)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=list(vec)) for _ in contents])


class FakeModels(FakeEmbeddings):
    def __init__(self, dim: int, behaviours: dict[str, Any]) -> None:
        super().__init__(dim)
        self.behaviours = behaviours
        self.models_called: list[str] = []
        self.configs: list[Any] = []

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.models_called.append(model)
        self.configs.append(config)
        outcome = self.behaviours.get(model, "ok")
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "hang":  # a free-tier model stuck in Google's queue
            await asyncio.sleep(3600)
        text = '{"source":"documents","language":"en","answer_markdown":"fake answer","refusal_reason":null}'
        usage = SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20, total_token_count=120, cached_content_token_count=10
        )
        return SimpleNamespace(text=text if outcome == "ok" else outcome, usage_metadata=usage)


def fake_gemini(dim: int = DIM, behaviours: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(aio=SimpleNamespace(models=FakeModels(dim, behaviours or {})))


@pytest.fixture
def make_app(tmp_path: Any):
    def factory(settings: Settings, **kwargs: Any) -> Quart:
        kwargs.setdefault("qdrant_client", QdrantClient(":memory:"))
        return create_app(settings, budget_file=tmp_path / "budget.json", **kwargs)

    return factory


@pytest.fixture
async def app(settings: Settings, make_app: Any) -> AsyncIterator[Quart]:
    application = make_app(settings)
    async with application.test_app():
        yield application


@pytest.fixture
async def client(app: Quart) -> QuartClient:
    return app.test_client()


@pytest.fixture
def prod_settings(settings: Settings) -> Settings:
    return replace(
        settings,
        demo=False,
        gemini_api_key="k",
        payload_encryption_key="AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
    )


def cookies_from(response: Any) -> dict[str, str]:
    """Cookie name -> value from a response's Set-Cookie headers."""
    out = {}
    for header in response.headers.getlist("Set-Cookie"):
        name, _, rest = header.partition("=")
        out[name] = rest.split(";", 1)[0]
    return out


def cookie_header(**cookies: str) -> dict[str, str]:
    return {"Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items())}


async def ingest(client: QuartClient, text: str, filename: str = "notes.txt") -> dict[str, Any]:
    resp = await client.post("/upload", json={"text": text, "filename": filename}, headers=ADMIN)
    assert resp.status_code == 200, await resp.get_data()
    return await resp.get_json()
