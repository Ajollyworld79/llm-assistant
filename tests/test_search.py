import base64
from dataclasses import replace

import pytest

from tests.conftest import ADMIN, SEARCH, cookie_header, cookies_from, fake_gemini, ingest

CONTRACT_FIELDS = {
    "answer": str,
    "results": list,
    "used_documents": list,
    "demo": bool,
    "message": str,
    "sources": list,
    "embedding_provider": str,
    "thread_id": str,
    "embed_ms": int,
    "ai_ms": int,
    "total_ms": int,
    "model": str,
}
RESULT_FIELDS = {"doc_id": str, "filename": str, "chunk_text": str, "score": float}
TEXT = "To reset your VPN password open the self service portal and follow the steps."


async def test_contract_shape_with_documents(client):
    doc = await ingest(client, TEXT)
    r = await client.post("/search", json={"query": TEXT, "thread_id": "thread-1"}, headers=SEARCH)
    assert r.status_code == 200
    body = await r.get_json()
    for name, typ in CONTRACT_FIELDS.items():
        assert isinstance(body[name], typ), name
    assert body["demo"] is True and body["thread_id"] == "thread-1"
    assert body["message"].startswith("Demo mode") and body["model"] == "demo"
    assert body["sources"] == ["notes.txt"]
    assert body["results"] and body["used_documents"] == body["results"]
    for name, typ in RESULT_FIELDS.items():
        assert isinstance(body["results"][0][name], typ), name
    assert body["results"][0]["doc_id"] == doc["id"]
    assert body["answer"].startswith("(Demo mode - no chat model available)")
    assert "VPN password" in body["answer"]  # markdown body, not raw JSON
    assert not body["answer"].removeprefix("(Demo mode - no chat model available) ").startswith("{")


async def test_demo_no_documents_message(client):
    body = await (await client.post("/search", json={"query": "anything"}, headers=SEARCH)).get_json()
    assert "no documents matched" in body["answer"].lower()
    assert body["results"] == [] and body["sources"] == [] and len(body["thread_id"]) == 16


async def test_long_chunk_text_is_truncated_in_results(client):
    await ingest(client, "word " * 150 + "end.")
    body = await (await client.post("/search", json={"query": "word"}, headers=SEARCH)).get_json()
    assert body["results"][0]["chunk_text"].endswith("...") and len(body["results"][0]["chunk_text"]) == 403


async def test_top_k_is_clamped(client, settings):
    for i in range(20):
        await ingest(client, f"Document number {i} about topic {i}.", f"d{i}.txt")

    async def count(top_k):
        r = await client.post("/search", json={"query": "topic", "top_k": top_k}, headers=SEARCH)
        return len((await r.get_json())["results"])

    assert await count(999) == settings.max_search_limit
    assert await count(0) == 1 and await count(-5) == 1
    assert await count(3) == 3 and await count("2") == 2
    assert (
        len((await (await client.post("/search", json={"query": "topic"}, headers=SEARCH)).get_json())["results"]) == 5
    )


async def test_file_type_filter(client):
    await ingest(client, "alpha beta gamma.", "a.txt")
    await ingest(client, "alpha beta gamma.", "a.md")
    r = await client.post("/search", json={"query": "alpha", "filters": {"file_type": "md"}}, headers=SEARCH)
    assert (await r.get_json())["sources"] == ["a.md"]


PNG = base64.b64encode(b"\x89PNG....").decode()


@pytest.mark.parametrize(
    "payload,error",
    [
        (None, "query required"),
        ({}, "query required"),
        ({"query": 5}, "query must be a string"),
        ({"query": "   "}, "query is required"),
        ({"query": "x" * 4001}, "query_too_long"),
        ({"query": "q", "top_k": True}, "invalid top_k"),
        ({"query": "q", "top_k": "many"}, "invalid top_k"),
        ({"query": "q", "top_k": None}, "invalid top_k"),
        ({"query": "q", "thread_id": "t" * 129}, "invalid thread_id"),
        ({"query": "q", "thread_id": 7}, "invalid thread_id"),
        ({"query": "q", "filters": "pdf"}, "invalid filters"),
        ({"query": "q", "filters": {"file_type": 3}}, "invalid filters"),
        ({"query": "q", "images": "x"}, "too_many_images"),
        ({"query": "q", "images": [{"data_base64": PNG}] * 5}, "too_many_images"),
        ({"query": "q", "images": ["x"]}, "invalid_image"),
        ({"query": "q", "images": [{"mime_type": "image/gif", "data_base64": PNG}]}, "unsupported_image_type"),
        ({"query": "q", "images": [{"mime_type": "image/png"}]}, "invalid_image"),
        ({"query": "q", "images": [{"data_base64": "***"}]}, "invalid_image"),
        ({"query": "q", "images": [{"data_base64": "A" * (6 * 1024 * 1024)}]}, "image_too_large"),
        (
            {"query": "q", "images": [{"data_base64": base64.b64encode(bytes(4 * 1024 * 1024 + 1)).decode()}]},
            "image_too_large",
        ),
    ],
)
async def test_invalid_input_is_rejected(client, payload, error):
    r = (
        await client.post("/search", json=payload, headers=SEARCH)
        if payload is not None
        else await client.post("/search", data="not json", headers=SEARCH)
    )
    assert r.status_code == 400
    assert (await r.get_json())["error"] == error


async def test_valid_images_are_accepted(client):
    r = await client.post(
        "/search",
        json={"query": "q", "images": [{"data_base64": PNG, "mime_type": "image/webp"}], "thread_id": "  "},
        headers=SEARCH,
    )
    assert r.status_code == 200 and len((await r.get_json())["thread_id"]) == 16


# -- authentication -----------------------------------------------------------


async def test_search_requires_credentials(client):
    assert (await client.post("/search", json={"query": "hi"})).status_code == 401
    spoof = {"Origin": "https://x.example", "Referer": "https://x.example/", "Sec-Fetch-Site": "same-origin"}
    assert (await client.post("/search", json={"query": "hi"}, headers=spoof)).status_code == 401
    assert (
        await client.post("/search", json={"query": "hi"}, headers={"Authorization": "Bearer nope"})
    ).status_code == 401


async def test_search_accepts_session_cookie_from_chat_page(client):
    page = await client.get("/")
    cookie = next(h for h in page.headers.getlist("Set-Cookie") if h.startswith("helios_session="))
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Secure" in cookie
    r = await client.post("/search", json={"query": "hi"}, headers=cookie_header(**cookies_from(page)))
    assert r.status_code == 200


async def test_forged_session_cookie_is_rejected(client):
    r = await client.post("/search", json={"query": "hi"}, headers={"Cookie": "helios_session=9999999999.deadbeef"})
    assert r.status_code == 401
    r = await client.post("/search", json={"query": "hi"}, headers={"Cookie": "helios_session=garbage"})
    assert r.status_code == 401


async def test_expired_session_cookie_is_rejected(client, monkeypatch):
    page = await client.get("/")
    cookies = cookies_from(page)
    import time

    monkeypatch.setattr(time, "time", lambda: time.monotonic() + 10**10)
    r = await client.post("/search", json={"query": "hi"}, headers=cookie_header(**cookies))
    assert r.status_code == 401


async def test_admin_bearer_may_search(client):
    assert (await client.post("/search", json={"query": "hi"}, headers=ADMIN)).status_code == 200


async def test_no_tokens_configured_fails_closed(settings, make_app):
    app = make_app(replace(settings, admin_token=None, search_api_token=None))
    async with app.test_app():
        c = app.test_client()
        assert (await c.post("/search", json={"query": "q"}, headers={"Authorization": "Bearer "})).status_code == 401
        page = await c.get("/")
        assert page.status_code == 200 and "helios_session" not in cookies_from(page)
        assert (await c.get("/documents", headers={"Authorization": "Bearer "})).status_code == 401


# -- rate limiting ------------------------------------------------------------


async def test_bearer_requests_are_exempt_from_ip_and_session_limits(settings, make_app):
    app = make_app(replace(settings, rate_limit_per_min=2, rate_limit_per_day=3, rate_limit_per_session=2))
    async with app.test_app():
        c = app.test_client()
        codes = [(await c.post("/search", json={"query": "q"}, headers=SEARCH)).status_code for _ in range(30)]
        assert codes == [200] * 30
        cookie = cookie_header(demo_session_id="same-session")
        codes = [
            (await c.post("/search", json={"query": "q"}, headers={**SEARCH, **cookie})).status_code for _ in range(5)
        ]
        assert codes == [200] * 5


async def test_cookie_requests_are_limited_per_ip_keyed_on_x_real_ip(settings, make_app):
    app = make_app(replace(settings, rate_limit_per_min=2))
    async with app.test_app():
        c = app.test_client()
        session = cookies_from(await c.get("/"))

        async def hit(ip, **extra):
            headers = {**cookie_header(**session), "X-Real-IP": ip, **extra}
            return (await c.post("/search", json={"query": "q"}, headers=headers)).status_code

        assert [await hit("10.0.0.1") for _ in range(3)] == [200, 200, 429]
        assert await hit("10.0.0.2") == 200  # a different client is unaffected
        # a rotating X-Forwarded-For must not buy extra requests
        assert await hit("10.0.0.1", **{"X-Forwarded-For": "6.6.6.6"}) == 429
        r = await c.post("/search", json={"query": "q"}, headers={**cookie_header(**session), "X-Real-IP": "10.0.0.1"})
        assert (await r.get_json())["error"] == "rate_limited"


async def test_per_ip_falls_back_to_remote_addr(settings, make_app):
    app = make_app(replace(settings, rate_limit_per_min=1))
    async with app.test_app():
        c = app.test_client()
        session = cookies_from(await c.get("/"))
        codes = [
            (
                await c.post(
                    "/search",
                    json={"query": "q"},
                    headers={**cookie_header(**session), "X-Forwarded-For": f"1.1.1.{i}"},
                )
            ).status_code
            for i in range(3)
        ]
        assert codes == [200, 429, 429]


async def test_per_session_limit(settings, make_app):
    app = make_app(replace(settings, rate_limit_per_session=2))
    async with app.test_app():
        c = app.test_client()
        session = {**cookies_from(await c.get("/")), "demo_session_id": "abc"}
        codes = [
            (await c.post("/search", json={"query": "q"}, headers=cookie_header(**session))).status_code
            for _ in range(3)
        ]
        assert codes == [200, 200, 429]


async def test_demo_session_cookie_is_secure_and_skipped_for_bearer(client):
    r = await client.post("/search", json={"query": "q"}, headers=SEARCH)
    assert "demo_session_id" not in cookies_from(r)
    page = await client.get("/")
    header = next(h for h in page.headers.getlist("Set-Cookie") if h.startswith("demo_session_id="))
    assert "Secure" in header and "HttpOnly" in header


async def test_daily_budget_blocks_search_even_for_bearer(settings, make_app):
    app = make_app(replace(settings, daily_token_budget=0))
    async with app.test_app():
        r = await app.test_client().post("/search", json={"query": "q"}, headers=SEARCH)
        assert r.status_code == 429 and (await r.get_json())["error"] == "budget_exhausted"


async def test_budget_survives_restart(settings, prod_settings, make_app):
    prod = replace(prod_settings, daily_token_budget=200, max_tokens=100)
    first = make_app(prod, gemini_client=fake_gemini())
    async with first.test_app():
        c = first.test_client()
        assert (await c.post("/search", json={"query": "q"}, headers=SEARCH)).status_code == 200
        assert (await c.post("/search", json={"query": "q"}, headers=SEARCH)).status_code == 200
    second = make_app(prod, gemini_client=fake_gemini())  # same budget file in tmp_path
    async with second.test_app():
        r = await second.test_client().post("/search", json={"query": "q"}, headers=SEARCH)
        assert r.status_code == 429


# -- production (non-demo) path -----------------------------------------------


async def test_production_answer_uses_llm_and_records_usage(prod_settings, make_app):
    gem = fake_gemini()
    app = make_app(prod_settings, gemini_client=gem)
    async with app.test_app():
        c = app.test_client()
        await ingest(c, TEXT)
        body = await (await c.post("/search", json={"query": TEXT}, headers=SEARCH)).get_json()
        assert body["answer"] == "fake answer" and body["demo"] is False and body["message"] == ""
        assert body["model"] == prod_settings.gemini_model
        used = (await (await c.get("/backend/api/metrics", headers=ADMIN)).get_json())["rate_limit"][
            "tokens_used_today"
        ]
        assert used == 120  # reservation replaced by the real usage
        chats = await (await c.get("/backend/api/conversations", headers=ADMIN)).get_json()
        assert chats["items"][0]["answer"] == "fake answer"


async def test_production_without_api_key_reports_not_configured(prod_settings, make_app):
    app = make_app(replace(prod_settings, gemini_api_key=None))
    async with app.test_app():
        r = await app.test_client().post("/search", json={"query": "q"}, headers=SEARCH)
        assert r.status_code == 503 and (await r.get_json())["error"] == "embedding_unavailable"


async def test_embedding_failure_is_a_503_not_fake_results(prod_settings, make_app):
    gem = fake_gemini()
    gem.aio.models.fail = True
    app = make_app(prod_settings, gemini_client=gem)
    async with app.test_app():
        r = await app.test_client().post("/search", json={"query": "q"}, headers=SEARCH)
        assert r.status_code == 503 and "quota" not in (await r.get_data(as_text=True))


async def test_search_unavailable_when_store_missing(prod_settings, make_app, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    app = create_with_bad_store(prod_settings, blocker, tmp_path)
    async with app.test_app():
        r = await app.test_client().post("/search", json={"query": "q"}, headers=SEARCH)
        assert r.status_code == 503 and (await r.get_json())["error"] == "search_unavailable"


def create_with_bad_store(settings, path, tmp_path):
    from backend.app import create_app

    return create_app(
        replace(settings, qdrant_path=str(path / "sub")), gemini_client=fake_gemini(), budget_file=tmp_path / "b.json"
    )


async def test_llm_failure_message_and_budget_release(prod_settings, make_app):
    boom = RuntimeError("down")
    gem = fake_gemini(behaviours={prod_settings.gemini_model: boom, prod_settings.gemini_model_fallback: boom})
    app = make_app(prod_settings, gemini_client=gem)
    async with app.test_app():
        c = app.test_client()
        body = await (await c.post("/search", json={"query": "q"}, headers=SEARCH)).get_json()
        assert "unavailable" in body["answer"]
        metrics = await (await c.get("/backend/api/metrics", headers=ADMIN)).get_json()
        assert metrics["rate_limit"]["tokens_used_today"] == 0
