import importlib
import sys
from dataclasses import replace

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from backend.app import create_app
from backend.app.core.config import ConfigError, Settings
from backend.app.core.security import CSP
from tests.conftest import ADMIN, DIM, SEARCH, cookies_from, fake_gemini

EXPECTED_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)


def test_csp_matches_contract():
    assert CSP == EXPECTED_CSP


async def test_security_headers_on_every_kind_of_response(client):
    for response in (
        await client.get("/"),
        await client.get("/livez"),
        await client.get("/nope"),
        await client.post("/search", json={}),
        await client.get("/static/styles.css"),
    ):
        h = response.headers
        assert h["Content-Security-Policy"] == EXPECTED_CSP
        assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"
        assert "max-age=" in h["Strict-Transport-Security"] and h["Referrer-Policy"]


async def test_livez_has_no_dependency_checks(settings, make_app):
    app = make_app(settings)  # not started: no store, no database
    r = await app.test_client().get("/livez")
    assert r.status_code == 200 and await r.get_json() == {"status": "ok"}


async def test_health_shape_in_demo(client):
    r = await client.get("/health")
    body = await r.get_json()
    assert r.status_code == 200 and body["status"] == "healthy"
    assert body["components"]["qdrant"] == {"status": "healthy", "collection_ready": True, "points_count": 0}
    assert body["components"]["gemini"] == {"status": "simulated"}
    assert body["components"]["database"] == {"status": "memory_fallback"}
    assert "recent_errors" not in body["metrics"] and body["metrics"]["recent_errors_count"] == 0
    assert set(body["monitor"]) == {"status", "issues"} and "timestamp" in body


async def test_health_reports_points_count_and_operator_errors(client):
    await client.post("/upload", json={"text": "a b c.", "filename": "a.txt"}, headers=ADMIN)
    await client.get("/documents")  # a 401 to be recorded
    body = await (await client.get("/health", headers=ADMIN)).get_json()
    assert body["components"]["qdrant"]["points_count"] == 1
    assert body["metrics"]["recent_errors"][0]["status"] == 401


async def test_health_degrades_without_gemini_key_in_production(prod_settings, make_app):
    app = make_app(replace(prod_settings, gemini_api_key=None))
    async with app.test_app():
        r = await app.test_client().get("/health")
        assert r.status_code == 503 and (await r.get_json())["components"]["gemini"] == {"status": "unavailable"}


async def test_health_reports_collection_dimension_mismatch(settings, make_app):
    qc = QdrantClient(":memory:")
    qc.create_collection("documents", vectors_config=qm.VectorParams(size=DIM + 1, distance=qm.Distance.COSINE))
    app = make_app(settings, qdrant_client=qc)
    async with app.test_app():
        c = app.test_client()
        r = await c.get("/health")
        assert r.status_code == 503
        assert (await r.get_json())["components"]["qdrant"] == {"status": "error", "collection_ready": False}
        assert (await c.post("/upload", json={"text": "x.", "filename": "a.txt"}, headers=ADMIN)).status_code == 503


async def test_health_database_error_when_postgres_unreachable(settings, make_app):
    app = make_app(replace(settings, database_url="postgresql://u:p@127.0.0.1:1/x"))
    async with app.test_app():
        r = await app.test_client().get("/health")
        assert (await r.get_json())["components"]["database"] == {"status": "error"}
        assert r.status_code == 503


async def test_store_unavailable_outside_demo_does_not_start_in_memory(prod_settings, tmp_path):
    blocker = tmp_path / "plain-file"
    blocker.write_text("x")
    app = create_app(
        replace(prod_settings, qdrant_path=str(blocker / "sub")),
        gemini_client=fake_gemini(),
        budget_file=tmp_path / "b",
    )
    async with app.test_app():
        c = app.test_client()
        assert (await c.get("/health")).status_code == 503
        assert (await c.get("/documents", headers=ADMIN)).status_code == 503
        assert (await c.post("/upload", json={"text": "x.", "filename": "a.txt"}, headers=ADMIN)).status_code == 503
        assert (await c.post("/reset", headers=ADMIN)).status_code == 503
        assert (await c.post("/optimize", headers=ADMIN)).status_code == 503
        assert (await c.get("/backend/api/connections", headers=ADMIN)).status_code == 200


async def test_demo_falls_back_to_in_memory_store(settings, tmp_path):
    blocker = tmp_path / "plain-file"
    blocker.write_text("x")
    app = create_app(replace(settings, qdrant_path=str(blocker / "sub")), budget_file=tmp_path / "b")
    async with app.test_app():
        assert (await app.test_client().get("/health")).status_code == 200


async def test_collection_is_never_recreated_on_startup(settings, make_app, monkeypatch):
    monkeypatch.setenv("QDRANT_FORCE_RECREATE", "true")
    qc = QdrantClient(":memory:")
    qc.create_collection("documents", vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))
    qc.upsert(
        "documents", points=[qm.PointStruct(id="6f1c2c52-6a0e-4c33-9b0f-3a0c7d1c9a11", vector=[1.0] * DIM, payload={})]
    )
    app = make_app(settings, qdrant_client=qc)
    async with app.test_app():
        assert qc.count("documents").count == 1


async def test_legacy_plaintext_payloads_are_encrypted_on_startup(prod_settings, make_app):
    qc = QdrantClient(":memory:")
    qc.create_collection("documents", vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))
    pid = "6f1c2c52-6a0e-4c33-9b0f-3a0c7d1c9a11"
    qc.upsert(
        "documents", points=[qm.PointStruct(id=pid, vector=[1.0] * DIM, payload={"filename": "x.txt", "text": "plain"})]
    )
    app = make_app(prod_settings, qdrant_client=qc, gemini_client=fake_gemini())
    async with app.test_app():
        stored = qc.retrieve("documents", [pid])[0].payload["text"]
        assert stored.startswith("enc:v1:")
        got = await (await app.test_client().get("/document/x.txt", headers=ADMIN)).get_json()
        assert got["text"] == "plain"


def test_production_refuses_to_start_without_valid_encryption_key(prod_settings):
    with pytest.raises(ConfigError, match="PAYLOAD_ENCRYPTION_KEY"):
        create_app(replace(prod_settings, payload_encryption_key=None))
    with pytest.raises(ConfigError, match="base64"):
        create_app(replace(prod_settings, payload_encryption_key="***"))
    with pytest.raises(ConfigError, match="32 bytes"):
        create_app(replace(prod_settings, payload_encryption_key="c2hvcnQ="))


def test_demo_tolerates_an_invalid_encryption_key(settings):
    app = create_app(replace(settings, payload_encryption_key="***"))
    assert not app.extensions["helios"].cipher.enabled


async def test_index_page_variables_and_cookie(client):
    r = await client.get("/")
    html = await r.get_data(as_text=True)
    assert r.status_code == 200 and "Demo mode" in html
    assert set(cookies_from(r)) == {"helios_session", "demo_session_id"}


async def test_console_endpoints_with_operator_token(client):
    system = await (await client.get("/backend/api/system", headers=ADMIN)).get_json()
    assert system["embedding_dim"] == DIM and "qdrant_path" not in system["config"]
    assert "gemini_api_key" not in str(system) and "test-admin" not in str(system)
    conns = await (await client.get("/backend/api/connections", headers=ADMIN)).get_json()
    assert conns["qdrant"]["status"] == "up" and conns["chat_db"]["status"] == "memory_fallback"
    assert conns["embedding_model"]["status"] == "configured"
    metrics = await (await client.get("/backend/api/metrics", headers=ADMIN)).get_json()
    assert metrics["rate_limit"]["daily_token_budget"] == 50000 and metrics["memory_mb"] > 0
    models = await (await client.get("/backend/api/models")).get_json()
    assert models["active_primary"] == Settings().gemini_model
    assert system["git_sha"] is None or isinstance(system["git_sha"], str)
    assert system["uptime_seconds"] >= 0
    series = await client.get("/backend/api/usage/timeseries?metric=tokens&range=7d")
    assert (await series.get_json())["range"] == "7d"
    assert (await client.get("/backend/api/usage/timeseries?metric=x")).status_code == 400
    assert (await client.get("/backend/api/usage/timeseries?range=x")).status_code == 400
    monitor = await (await client.get("/monitor", headers=ADMIN)).get_json()
    assert monitor["requests"]["total"] >= 1


async def test_error_responses_are_json_and_do_not_leak(app, client):
    r = await client.get("/nope")
    assert r.status_code == 404 and await r.get_json() == {"error": "Resource not found"}
    r = await client.delete("/search")
    assert r.status_code == 405

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret internals")

    r = await client.get("/boom")
    assert r.status_code == 500 and await r.get_json() == {"error": "internal_error"}


async def test_cors_only_when_configured(settings, make_app):
    app = make_app(replace(settings, cors_allow_origins=("https://site.example",)))
    async with app.test_app():
        r = await app.test_client().get("/livez", headers={"Origin": "https://site.example"})
        assert r.headers.get("Access-Control-Allow-Origin") == "https://site.example"
    plain = make_app(settings)
    async with plain.test_app():
        r = await plain.test_client().get("/livez", headers={"Origin": "https://site.example"})
        assert "Access-Control-Allow-Origin" not in r.headers


def test_main_exposes_asgi_app(monkeypatch):
    monkeypatch.setenv("DEMO", "true")
    monkeypatch.setenv("QDRANT_PATH", ":memory:")
    monkeypatch.setenv("LOG_TO_FILE", "false")
    sys.modules.pop("backend.app.main", None)
    main = importlib.import_module("backend.app.main")
    assert main.app.name == "backend.app"
    assert isinstance(Settings.from_env({}), Settings)


async def test_session_search_contract_through_real_stack(client):
    page = await client.get("/")
    cookie = {"Cookie": f"helios_session={cookies_from(page)['helios_session']}"}
    assert (await client.post("/search", json={"query": "q"}, headers=cookie)).status_code == 200
    assert (await client.post("/search", json={"query": "q"}, headers=SEARCH)).status_code == 200
