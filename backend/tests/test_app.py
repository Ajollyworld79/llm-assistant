import sys
import os
import pytest
from quart.testing import QuartClient
# Ensure project root is on path for test import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app.main import app

@pytest.mark.asyncio
async def test_health():
    client = QuartClient(app)
    r = await client.get('/health')
    assert r.status_code == 200
    js = await r.get_json()
    assert 'demo' in js

@pytest.mark.asyncio
async def test_upload_and_search():
    client = QuartClient(app)
    import io
    # Ensure no admin token set for public uploads in this test
    from backend.app import config
    config.settings.admin_token = None

    # Post as JSON text (tests avoid multipart complexity)
    r = await client.post('/upload', json={'text': 'hello world demo document', 'filename': 'test.txt'})
    assert r.status_code == 200
    js = await r.get_json()
    assert 'id' in js

    r2 = await client.post('/search', json={'query': 'hello', 'top_k': 3})
    assert r2.status_code == 200
    js2 = await r2.get_json()
    assert 'results' in js2

@pytest.mark.asyncio
async def test_admin_auth_required(monkeypatch):
    # Set admin token and ensure unauthorized without it
    from backend.app import config
    config.settings.admin_token = 'devtoken'
    client = QuartClient(app)

    r = await client.post('/upload', json={'text': 'p test', 'filename': 'p.txt'})
    assert r.status_code == 401

    # With header
    r2 = await client.post('/upload', json={'text': 'p test', 'filename': 'p.txt'}, headers={'Authorization': 'Bearer devtoken'})
    assert r2.status_code == 200

    # Reset endpoint should also require auth
    r3 = await client.post('/reset')
    assert r3.status_code == 401
    r4 = await client.post('/reset', headers={'Authorization': 'Bearer devtoken'})
    assert r4.status_code == 200

    # Clear admin token for subsequent tests
    config.settings.admin_token = None