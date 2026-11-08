"""Access-control regressions: every protected route must fail closed."""
import os

os.environ.setdefault("DEMO", "true")
os.environ.setdefault("LOG_TO_FILE", "0")
os.environ["ADMIN_TOKEN"] = "test-admin"
os.environ["SEARCH_API_TOKEN"] = "test-search"

import pytest  # noqa: E402

from backend.app import main  # noqa: E402

ADMIN = {"Authorization": "Bearer test-admin"}
SPOOFED_BROWSER = {
    "Origin": "https://llm-assistant-production-aa36.up.railway.app",
    "Referer": "https://llm-assistant-production-aa36.up.railway.app/",
    "Sec-Fetch-Site": "same-origin",
    "Host": "llm-assistant-production-aa36.up.railway.app",
}


@pytest.fixture
def client():
    return main.app.test_client()


async def test_search_rejects_spoofed_same_origin_headers(client):
    r = await client.post("/search", json={"query": "hello"}, headers=SPOOFED_BROWSER)
    assert r.status_code == 401


async def test_search_accepts_session_cookie_from_chat_page(client):
    page = await client.get("/")
    cookie = page.headers.get("Set-Cookie", "")
    assert "helios_session=" in cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    r = await client.post("/search", json={"query": "hello"})
    assert r.status_code == 200


async def test_search_rejects_forged_session_cookie(client):
    r = await client.post("/search", json={"query": "hello"},
                          headers={"Cookie": "helios_session=9999999999.deadbeef"})
    assert r.status_code == 401


@pytest.mark.parametrize("method,path", [
    ("GET", "/documents"),
    ("GET", "/document/cv.pdf"),
    ("POST", "/reset"),
    ("POST", "/optimize"),
    ("DELETE", "/document/cv.pdf"),
    ("GET", "/backend/api/conversations"),
    ("DELETE", "/backend/api/conversations"),
])
async def test_protected_routes_reject_missing_and_wrong_token(client, method, path):
    assert (await client.open(path, method=method)).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert (await client.open(path, method=method, headers=wrong)).status_code == 401


async def test_admin_routes_fail_closed_when_token_unset(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN")
    r = await client.post("/reset", headers={"Authorization": "Bearer "})
    assert r.status_code == 401
    r = await client.get("/backend/api/conversations", headers=ADMIN)
    assert r.status_code == 401


async def test_upload_is_rejected_before_parsing(client, monkeypatch):
    async def must_not_parse(*_a, **_k):
        raise AssertionError("upload body parsed before auth")
    monkeypatch.setattr(main, "parse_content", must_not_parse)
    r = await client.post("/upload", json={"text": "secret", "filename": "x.txt"})
    assert r.status_code == 401


async def test_conversations_readable_with_admin_token(client):
    r = await client.get("/backend/api/conversations", headers=ADMIN)
    assert r.status_code == 200
    data = await r.get_json()
    assert {"items", "total", "next_before_id"} <= set(data)
