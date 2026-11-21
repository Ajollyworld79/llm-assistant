from dataclasses import replace

import pytest

from tests.conftest import ADMIN, OPERATOR_HEADERS, cookie_header, cookies_from

ADMIN_ROUTES = [
    ("GET", "/documents"),
    ("GET", "/document/x"),
    ("DELETE", "/document/x"),
    ("POST", "/upload"),
    ("POST", "/reset"),
    ("POST", "/optimize"),
    ("POST", "/admin/upload_vectors"),
    ("GET", "/monitor"),
    ("GET", "/backend/api/system"),
    ("GET", "/backend/api/connections"),
    ("GET", "/backend/api/metrics"),
    ("GET", "/backend/api/prompt"),
    ("GET", "/backend/api/conversations"),
    ("DELETE", "/backend/api/conversations"),
    ("DELETE", "/backend/api/conversations/1"),
]


async def login(client, token="test-admin"):  # noqa: S107
    return await client.post("/backend/api/session", json={"token": token})


@pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
async def test_admin_routes_reject_missing_and_wrong_token(client, method, path):
    assert (await client.open(path, method=method)).status_code == 401
    assert (await client.open(path, method=method, headers={"Authorization": "Bearer nope"})).status_code == 401
    assert (await client.open(path, method=method, headers=OPERATOR_HEADERS)).status_code == 401


@pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
async def test_admin_routes_fail_closed_without_admin_token(settings, make_app, method, path):
    app = make_app(replace(settings, admin_token=None))
    async with app.test_app():
        r = await app.test_client().open(path, method=method, headers={"Authorization": "Bearer "})
        assert r.status_code == 401


async def test_public_console_endpoints_stay_open(client):
    for path in ("/backend/", "/backend/api/models", "/backend/api/usage/summary", "/backend/api/usage/timeseries"):
        assert (await client.get(path)).status_code == 200, path


async def test_system_prompt_is_not_published_to_anonymous_users(client):
    assert (await client.get("/backend/api/prompt")).status_code == 401
    body = await (await client.get("/backend/api/prompt", headers=ADMIN)).get_json()
    assert "system_prompt" in body


async def test_upload_rejected_before_parsing(client, monkeypatch):
    from backend.app.services import ingest

    async def boom(*_a, **_k):
        raise AssertionError("body parsed before auth")

    monkeypatch.setattr(ingest.IngestService, "ingest_text", boom)
    assert (await client.post("/upload", json={"text": "secret", "filename": "x.txt"})).status_code == 401


async def test_bearer_admin_token_authenticates(client):
    assert (await client.get("/backend/api/conversations", headers=ADMIN)).status_code == 200


async def test_login_sets_hardened_cookie(client):
    r = await login(client)
    assert r.status_code == 204
    header = next(h for h in r.headers.getlist("Set-Cookie") if h.startswith("helios_operator="))
    for attr in ("HttpOnly", "Secure", "SameSite=Strict", "Path=/", "Max-Age=43200"):
        assert attr in header, attr


@pytest.mark.parametrize("payload", [{"token": "wrong"}, {"token": ""}, {"token": 5}, {}, None])
async def test_login_rejects_bad_tokens(client, payload):
    r = (
        await client.post("/backend/api/session", json=payload)
        if payload is not None
        else await client.post("/backend/api/session", data="x")
    )
    assert r.status_code == 401 and "helios_operator" not in cookies_from(r)


async def test_login_disabled_without_admin_token(settings, make_app):
    app = make_app(replace(settings, admin_token=None))
    async with app.test_app():
        assert (await login(app.test_client(), "anything")).status_code == 401


async def test_cookie_authenticates_only_with_csrf_header(client):
    cookie = cookie_header(helios_operator=cookies_from(await login(client))["helios_operator"])
    with_csrf = await client.get("/backend/api/conversations", headers={**cookie, **OPERATOR_HEADERS})
    assert with_csrf.status_code == 200
    no_csrf = await client.get("/backend/api/conversations", headers=cookie)
    assert no_csrf.status_code == 403 and (await no_csrf.get_json())["error"] == "csrf_header_required"
    wrong = await client.get("/backend/api/conversations", headers={**cookie, "X-Requested-With": "other"})
    assert wrong.status_code == 403
    delete = await client.post("/reset", headers={**cookie, **OPERATOR_HEADERS})
    assert delete.status_code == 200


async def test_forged_or_expired_operator_cookie_is_rejected(client, monkeypatch):
    for value in ("9999999999.deadbeef", "garbage", ""):
        r = await client.get("/documents", headers={**cookie_header(helios_operator=value), **OPERATOR_HEADERS})
        assert r.status_code == 401
    value = cookies_from(await login(client))["helios_operator"]
    import time

    monkeypatch.setattr(time, "time", lambda: time.monotonic() + 10**10)
    r = await client.get("/documents", headers={**cookie_header(helios_operator=value), **OPERATOR_HEADERS})
    assert r.status_code == 401


async def test_chat_session_cookie_is_not_an_operator_cookie(client):
    session = cookies_from(await client.get("/"))["helios_session"]
    r = await client.get("/documents", headers={**cookie_header(helios_operator=session), **OPERATOR_HEADERS})
    assert r.status_code == 401


async def test_session_status_and_logout(client):
    assert await (await client.get("/backend/api/session")).get_json() == {"authenticated": False}
    cookie = cookie_header(helios_operator=cookies_from(await login(client))["helios_operator"])
    assert await (await client.get("/backend/api/session", headers=cookie)).get_json() == {"authenticated": True}
    assert await (await client.get("/backend/api/session", headers=ADMIN)).get_json() == {"authenticated": True}
    out = await client.delete("/backend/api/session")
    assert out.status_code == 204
    cleared = next(h for h in out.headers.getlist("Set-Cookie") if h.startswith("helios_operator="))
    assert "Max-Age=0" in cleared or "1970" in cleared


async def test_login_is_throttled_after_repeated_failures(client):
    for _ in range(10):
        assert (await login(client, "bad")).status_code == 401
    assert (await login(client, "bad")).status_code == 429
    assert (await login(client)).status_code == 429  # even the right token waits out the window


async def test_cookie_for_one_secret_is_invalid_after_rotation(settings, make_app):
    app = make_app(settings)
    async with app.test_app():
        value = cookies_from(await login(app.test_client()))["helios_operator"]
    rotated = make_app(replace(settings, admin_token="new-token"))
    async with rotated.test_app():
        r = await rotated.test_client().get(
            "/documents", headers={**cookie_header(helios_operator=value), **OPERATOR_HEADERS}
        )
        assert r.status_code == 401
