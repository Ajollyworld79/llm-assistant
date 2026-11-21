import io
import uuid
from dataclasses import replace

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from werkzeug.datastructures import FileStorage

from tests.conftest import ADMIN, DIM, ingest
from tests.test_parser import make_docx, make_pdf


async def test_upload_list_get_delete_round_trip(client):
    doc = await ingest(client, "Printer setup guide. Install the driver first.", "printer.txt")
    assert doc["status"] == "indexed" and doc["requested_chunks"] == doc["embedded_chunks"] == 1
    assert doc["failed_chunks"] == 0 and doc["filename"] == "printer.txt"

    listed = await (await client.get("/documents", headers=ADMIN)).get_json()
    assert listed == [
        {
            "id": doc["id"],
            "filename": "printer.txt",
            "chunks": 1,
            "uploaded_at": listed[0]["uploaded_at"],
            "type": "TXT",
        }
    ]
    got = await (await client.get(f"/document/{doc['id']}", headers=ADMIN)).get_json()
    assert got["id"] == doc["id"] and got["filename"] == "printer.txt" and "Install the driver" in got["text"]

    deleted = await client.delete(f"/document/{doc['id']}", headers=ADMIN)
    assert deleted.status_code == 200 and (await deleted.get_json())["deleted"] == 1
    assert (await client.get(f"/document/{doc['id']}", headers=ADMIN)).status_code == 404
    assert (await client.delete(f"/document/{doc['id']}", headers=ADMIN)).status_code == 404
    assert await (await client.get("/documents", headers=ADMIN)).get_json() == []


async def test_delete_unknown_document_is_404(client):
    r = await client.delete("/document/never-existed", headers=ADMIN)
    assert r.status_code == 404 and (await r.get_json())["error"] == "document not found"


async def test_same_filename_uploaded_twice_has_distinct_ids(client):
    a = await ingest(client, "First version.", "same.txt")
    b = await ingest(client, "Second version.", "same.txt")
    assert a["id"] != b["id"]
    assert len(await (await client.get("/documents", headers=ADMIN)).get_json()) == 2
    await client.delete(f"/document/{a['id']}", headers=ADMIN)
    remaining = await (await client.get("/documents", headers=ADMIN)).get_json()
    assert [d["id"] for d in remaining] == [b["id"]]


async def test_multipart_upload_of_pdf_and_docx(client):
    for name, content in (
        ("guide.pdf", make_pdf("Reset the router by holding the button")),
        ("notes.docx", make_docx(["Docx paragraph about routers"])),
    ):
        r = await client.post("/upload", files={"file": FileStorage(io.BytesIO(content), filename=name)}, headers=ADMIN)
        assert r.status_code == 200, name
    types = {d["type"] for d in await (await client.get("/documents", headers=ADMIN)).get_json()}
    assert types == {"PDF", "DOCX"}


async def test_filename_is_sanitized(client):
    doc = await ingest(client, "text here.", '../../etc/<b>pass"wd.txt')
    assert doc["filename"] == "bpasswd.txt"


async def test_unsupported_empty_and_corrupt_files_are_rejected_not_indexed(client):
    async def upload(name, content):
        return await client.post(
            "/upload", files={"file": FileStorage(io.BytesIO(content), filename=name)}, headers=ADMIN
        )

    r = await upload("slides.pptx", b"PK")
    assert r.status_code == 415 and ".pdf" in (await r.get_json())["supported"]
    assert (await upload("empty.txt", b"   \n")).status_code == 422
    assert (await upload("scan.pdf", make_pdf(None))).status_code == 422
    assert (await upload("broken.pdf", b"%PDF-1.4 nope")).status_code == 422
    assert (await upload("broken.docx", b"nope")).status_code == 422
    assert (await client.post("/upload", json={"text": "  ", "filename": "a.txt"}, headers=ADMIN)).status_code == 422
    assert await (await client.get("/documents", headers=ADMIN)).get_json() == []


async def test_upload_requires_file_or_text(client):
    assert (await client.post("/upload", json={"nope": 1}, headers=ADMIN)).status_code == 400
    assert (await client.post("/upload", json={"text": 5}, headers=ADMIN)).status_code == 400


async def test_oversized_bodies_are_rejected(settings, make_app):
    app = make_app(replace(settings, max_upload_bytes=1000))
    async with app.test_app():
        c = app.test_client()
        big = io.BytesIO(b"x" * 5000)
        assert (
            await c.post("/upload", files={"file": FileStorage(big, filename="a.txt")}, headers=ADMIN)
        ).status_code == 413
        r = await c.post("/backend/api/session", json={"token": "x"}, headers={"Content-Length": "70000"})
        assert r.status_code == 413 and (await r.get_json())["error"] == "payload_too_large"


async def test_uploaded_chunks_are_encrypted_at_rest(prod_settings, make_app):
    from tests.conftest import fake_gemini

    qc = QdrantClient(":memory:")
    app = make_app(prod_settings, qdrant_client=qc, gemini_client=fake_gemini())
    async with app.test_app():
        c = app.test_client()
        doc = await ingest(c, "Highly confidential password policy.", "policy.txt")
        points, _ = qc.scroll("documents", limit=10)
        assert points[0].payload["text"].startswith("enc:v1:") and points[0].payload["doc_id"] == doc["id"]
        text = (await (await c.get(f"/document/{doc['id']}", headers=ADMIN)).get_json())["text"]
        assert text == "Highly confidential password policy."


async def test_legacy_points_without_doc_id_keep_working(app, client):
    qc = app.extensions["helios"].store.client
    legacy = [
        qm.PointStruct(
            id=str(uuid.uuid4()),
            vector=[1.0] + [0.0] * (DIM - 1),
            payload={
                "filename": "old.pdf",
                "chunk_index": i,
                "text": f"legacy chunk {i}",
                "uploaded_at": "2025-01-01 10:00",
                "type": "PDF",
            },
        )
        for i in range(3)
    ]
    qc.upsert("documents", points=legacy)
    new = await ingest(client, "A newer document that happens to be called old.pdf.", "old.pdf")

    docs = {d["id"]: d for d in await (await client.get("/documents", headers=ADMIN)).get_json()}
    assert docs["old.pdf"]["chunks"] == 3 and docs["old.pdf"]["type"] == "PDF"
    assert docs[new["id"]]["chunks"] == 1

    got = await (await client.get("/document/old.pdf", headers=ADMIN)).get_json()
    assert got["filename"] == "old.pdf" and got["chunks"] == 3
    assert got["text"] == "legacy chunk 0\nlegacy chunk 1\nlegacy chunk 2"

    # deleting the legacy id must not touch the newer document with the same filename
    assert (await client.delete("/document/old.pdf", headers=ADMIN)).status_code == 200
    remaining = await (await client.get("/documents", headers=ADMIN)).get_json()
    assert [d["id"] for d in remaining] == [new["id"]]


async def test_search_results_use_filename_as_legacy_doc_id(app, client):
    qc = app.extensions["helios"].store.client
    qc.upsert(
        "documents",
        points=[
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=[1.0] + [0.0] * (DIM - 1),
                payload={"filename": "legacy.txt", "text": "hello"},
            )
        ],
    )
    from backend.app.services.embeddings import hash_embedding

    vec = hash_embedding("anything", DIM)
    hits = await app.extensions["helios"].store.search(vec, 5)
    assert hits[0].doc_id == "legacy.txt" and hits[0].text == "hello"


async def test_listing_and_reading_paginate_beyond_one_scroll_page(client):
    vec = [1.0] + [0.0] * (DIM - 1)
    chunks = [{"text": f"chunk {i}", "vector": vec, "chunk_index": i} for i in range(700)]
    r = await client.post(
        "/admin/upload_vectors",
        json={"document": {"id": "big", "filename": "big.txt", "chunks": chunks}},
        headers=ADMIN,
    )
    assert (await r.get_json())["results"][0]["upserted"] == 700
    docs = await (await client.get("/documents", headers=ADMIN)).get_json()
    assert docs[0]["chunks"] == 700
    got = await (await client.get("/document/big", headers=ADMIN)).get_json()
    assert got["chunks"] == 700 and got["text"].splitlines()[699] == "chunk 699"
    assert (await (await client.delete("/document/big", headers=ADMIN)).get_json())["deleted"] == 700


async def test_upload_vectors_is_idempotent_for_same_document_id(client):
    vec = [1.0] + [0.0] * (DIM - 1)
    body = {"documents": [{"id": "d1", "filename": "d.txt", "chunks": [{"text": "a", "vector": vec}]}]}
    for _ in range(2):
        await client.post("/admin/upload_vectors", json=body, headers=ADMIN)
    assert (await (await client.get("/documents", headers=ADMIN)).get_json())[0]["chunks"] == 1


async def test_upload_vectors_validation(client, prod_settings, make_app):
    async def post(payload):
        return await client.post("/admin/upload_vectors", json=payload, headers=ADMIN)

    assert (await post([1])).status_code == 400
    assert (await post({"documents": []})).status_code == 400
    assert (await post({"documents": [{}] * 101})).status_code == 400
    vec = [1.0] * DIM
    results = (
        await (
            await post(
                {
                    "documents": [
                        {"filename": "a.txt", "chunks": []},
                        {"filename": "b.txt", "chunks": [{"text": "x", "vector": [1.0, 2.0]}]},
                        {"filename": "c.txt", "chunks": [{"text": 5, "vector": vec}]},
                        {"filename": "d.txt", "chunks": [{"text": "no vector given"}]},
                        {"filename": "e.txt", "chunks": [{"text": "ok", "vector": vec, "chunk_index": "bad"}]},
                    ]
                }
            )
        ).get_json()
    )["results"]
    assert [r["status"] for r in results] == ["skipped", "error", "skipped", "upserted", "upserted"]
    assert results[1]["detail"] == "embedding_dim_mismatch"


async def test_upload_vectors_requires_vectors_outside_demo(prod_settings, make_app):
    from tests.conftest import fake_gemini

    app = make_app(prod_settings, gemini_client=fake_gemini())
    async with app.test_app():
        r = await app.test_client().post(
            "/admin/upload_vectors", json={"document": {"filename": "a.txt", "chunks": [{"text": "x"}]}}, headers=ADMIN
        )
        assert (await r.get_json())["results"][0]["reason"] == "missing vector for chunk 0"


async def test_dimension_mismatch_on_upload_is_409(app, client, monkeypatch):
    embedder = app.extensions["helios"].embedder

    async def wrong_size(texts, *, is_query=False):
        return [[0.5] * (DIM + 3) for _ in texts]

    monkeypatch.setattr(embedder, "embed", wrong_size)
    r = await client.post("/upload", json={"text": "hello world.", "filename": "a.txt"}, headers=ADMIN)
    body = await r.get_json()
    assert r.status_code == 409 and body["error"] == "embedding_dim_mismatch" and str(DIM + 3) in body["details"]


async def test_embedding_failure_during_upload_is_502_and_nothing_is_indexed(prod_settings, make_app):
    from tests.conftest import fake_gemini

    gem = fake_gemini()
    gem.aio.models.fail = True
    app = make_app(prod_settings, gemini_client=gem)
    async with app.test_app():
        c = app.test_client()
        r = await c.post("/upload", json={"text": "hello.", "filename": "a.txt"}, headers=ADMIN)
        assert r.status_code == 502 and "quota" not in await r.get_data(as_text=True)
        assert await (await c.get("/documents", headers=ADMIN)).get_json() == []


async def test_reset_and_optimize(client):
    for i in range(3):
        await ingest(client, f"doc {i}.", f"d{i}.txt")
    assert (await client.post("/optimize", headers=ADMIN)).status_code == 200
    body = await (await client.post("/reset", headers=ADMIN)).get_json()
    assert body == {"status": "cleared", "deleted": 3}
    assert await (await client.get("/documents", headers=ADMIN)).get_json() == []


async def test_demo_lock_returns_403_on_mutations(settings, make_app):
    app = make_app(replace(settings, demo_locked=True))
    async with app.test_app():
        c = app.test_client()
        for method, path in [
            ("POST", "/upload"),
            ("DELETE", "/document/x"),
            ("POST", "/reset"),
            ("POST", "/optimize"),
            ("POST", "/admin/upload_vectors"),
            ("DELETE", "/backend/api/conversations"),
            ("DELETE", "/backend/api/conversations/1"),
        ]:
            r = await c.open(path, method=method, headers=ADMIN)
            assert r.status_code == 403, path
            assert (await r.get_json())["error"] == "read_only_demo"
        assert (await c.get("/documents", headers=ADMIN)).status_code == 200  # reads still work
        assert (await c.get("/")).status_code == 200


async def test_conversation_history_is_pii_redacted_and_deletable(client):
    await client.post(
        "/search",
        json={"query": "contact anna@example.dk about 010203-1234"},
        headers={"Authorization": "Bearer test-search"},
    )
    page = await (await client.get("/backend/api/conversations", headers=ADMIN)).get_json()
    assert page["total"] == 1 and "anna@example.dk" not in page["items"][0]["query"]
    assert "[email-redacted]" in page["items"][0]["query"]
    found = await (await client.get("/backend/api/conversations?q=redacted&limit=5", headers=ADMIN)).get_json()
    assert found["total"] == 1
    assert (await client.get("/backend/api/conversations?before_id=abc", headers=ADMIN)).status_code == 400
    assert (await client.get("/backend/api/conversations?limit=zzz", headers=ADMIN)).status_code == 200
    cid = page["items"][0]["id"]
    assert (await (await client.delete(f"/backend/api/conversations/{cid}", headers=ADMIN)).get_json())["deleted"] == 1
    assert (await (await client.delete("/backend/api/conversations", headers=ADMIN)).get_json())["deleted"] == 0
