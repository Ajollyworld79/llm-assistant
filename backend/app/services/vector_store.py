"""Qdrant access: collection lifecycle, paginated scrolls, search and document operations.

Document identity: every point written by this version carries a ``doc_id`` payload
(a UUID shared by all chunks of one upload). Points written by older versions have no
``doc_id``; they are addressed by their ``filename``, which doubles as the document id.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from backend.app.core.config import Settings
from backend.app.infra.crypto import PayloadCipher

logger = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("6f1c2c52-6a0e-4c33-9b0f-3a0c7d1c9a11")
_SCROLL_PAGE = 256
_UPSERT_BATCH = 256
_LIST_FIELDS = ["doc_id", "filename", "uploaded_at", "type"]


class StoreUnavailable(RuntimeError):
    """No usable vector store connection."""


class DimensionMismatch(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"collection vectors have size {expected}, embeddings have size {actual}")
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class Hit:
    doc_id: str
    filename: str
    text: str
    score: float


@dataclass(frozen=True)
class DocumentInfo:
    id: str
    filename: str
    chunks: int
    uploaded_at: str
    type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "chunks": self.chunks,
            "uploaded_at": self.uploaded_at,
            "type": self.type,
        }


def _doc_filter(doc_id: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])


def _legacy_filter(filename: str) -> qm.Filter:
    """Points from before ``doc_id`` existed: addressed by filename, doc_id absent."""
    return qm.Filter(
        must=[
            qm.FieldCondition(key="filename", match=qm.MatchValue(value=filename)),
            qm.IsEmptyCondition(is_empty=qm.PayloadField(key="doc_id")),
        ]
    )


def point_id(doc_id: str, index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{index}"))


class VectorStore:
    def __init__(self, settings: Settings, cipher: PayloadCipher, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self.cipher = cipher
        self.client = client
        self.collection = settings.qdrant_collection
        self.vector_size: int | None = None
        self.collection_ready = False
        self._lock = asyncio.Lock()  # local Qdrant is not safe for concurrent use

    # -- lifecycle ----------------------------------------------------------

    def _open_client(self) -> QdrantClient:
        s = self.settings
        if s.qdrant_url:
            return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        if s.qdrant_path == ":memory:":
            return QdrantClient(":memory:")
        return QdrantClient(path=s.qdrant_path)

    def connect(self) -> None:
        """Open the client and make sure the collection exists. Never deletes data."""
        if self.client is None:
            try:
                self.client = self._open_client()
            except Exception as exc:
                logger.error("Could not open Qdrant: %s", type(exc).__name__)
                if not self.settings.demo:
                    return
                self.client = QdrantClient(":memory:")
        try:
            self._ensure_collection(self.client)
        except Exception as exc:
            logger.error("Qdrant collection check failed: %s", type(exc).__name__)

    def _ensure_collection(self, client: QdrantClient) -> None:
        dim = self.settings.embedding_dim
        if not client.collection_exists(self.collection):
            logger.info("Creating collection %r (size=%d)", self.collection, dim)
            client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
        params = client.get_collection(self.collection).config.params
        vectors = params.vectors
        self.vector_size = vectors.size if isinstance(vectors, qm.VectorParams) else None
        if self.vector_size is not None and self.vector_size != dim:
            logger.error(
                "Collection %r has vector size %d but EMBEDDING_DIM=%d; refusing to write until fixed",
                self.collection,
                self.vector_size,
                dim,
            )
            return
        self.collection_ready = True
        logger.info("Collection %r ready (%d points)", self.collection, client.count(self.collection).count)

    @property
    def available(self) -> bool:
        return self.client is not None

    def _require(self) -> QdrantClient:
        if self.client is None or not self.collection_ready:
            raise StoreUnavailable("vector store is not ready")
        return self.client

    async def _run[**P, T](self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        async with self._lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    async def status(self) -> dict[str, Any]:
        info: dict[str, Any] = {"collection_ready": self.collection_ready}
        if self.client is None:
            return {"status": "unavailable", **info}
        if not self.collection_ready:
            return {"status": "error", **info}
        try:
            count = await self._run(self.client.count, self.collection)
        except Exception as exc:
            logger.warning("Qdrant count failed: %s", type(exc).__name__)
            return {"status": "error", **info}
        return {"status": "healthy", **info, "points_count": count.count}

    async def close(self) -> None:
        if self.client is not None:
            await asyncio.to_thread(self.client.close)

    # -- reads --------------------------------------------------------------

    def _scroll_all(self, flt: qm.Filter | None, payload: bool | list[str]) -> list[qm.Record]:
        client = self._require()
        records: list[qm.Record] = []
        offset: Any = None
        while True:
            page, offset = client.scroll(
                self.collection,
                scroll_filter=flt,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=payload,
                with_vectors=False,
            )
            records.extend(page)
            if offset is None:
                return records

    async def list_documents(self) -> list[DocumentInfo]:
        records = await self._run(self._scroll_all, None, _LIST_FIELDS)
        groups: dict[str, dict[str, Any]] = {}
        for rec in records:
            p = rec.payload or {}
            filename = p.get("filename") or "unknown"
            doc_id = p.get("doc_id") or filename
            group = groups.setdefault(
                doc_id,
                {
                    "filename": filename,
                    "chunks": 0,
                    "uploaded_at": p.get("uploaded_at", "N/A"),
                    "type": p.get("type", "FILE"),
                },
            )
            group["chunks"] += 1
        return [DocumentInfo(id=k, **v) for k, v in groups.items()]

    def _read_document(self, doc_id: str) -> list[qm.Record]:
        records = self._scroll_all(_doc_filter(doc_id), True)
        return records or self._scroll_all(_legacy_filter(doc_id), True)

    async def get_document(self, doc_id: str) -> tuple[str, str, int] | None:
        """Return (filename, full text, chunk count) or None when no such document exists."""
        records = await self._run(self._read_document, doc_id)
        if not records:
            return None
        records.sort(key=lambda r: int((r.payload or {}).get("chunk_index", 0)))
        filename = str((records[0].payload or {}).get("filename") or doc_id)
        text = "\n".join(self.cipher.decrypt(str((r.payload or {}).get("text", ""))) for r in records)
        return filename, text, len(records)

    def _search(self, vector: Sequence[float], limit: int, file_type: str | None) -> list[Hit]:
        client = self._require()
        flt = None
        if file_type:
            flt = qm.Filter(must=[qm.FieldCondition(key="type", match=qm.MatchValue(value=file_type.upper()))])
        res = client.query_points(self.collection, query=list(vector), query_filter=flt, limit=limit, with_payload=True)
        hits = []
        for point in res.points:
            p = point.payload or {}
            filename = str(p.get("filename") or "unknown")
            hits.append(
                Hit(
                    doc_id=str(p.get("doc_id") or filename),
                    filename=filename,
                    text=self.cipher.decrypt(str(p.get("text", ""))),
                    score=float(point.score or 0.0),
                )
            )
        return hits

    async def search(self, vector: Sequence[float], limit: int, file_type: str | None = None) -> list[Hit]:
        return await self._run(self._search, vector, limit, file_type)

    # -- writes -------------------------------------------------------------

    def _upsert(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        uploaded_at: str,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        indices: Sequence[int],
    ) -> int:
        client = self._require()
        points = [
            qm.PointStruct(
                id=point_id(doc_id, idx),
                vector=list(vec),
                payload={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "text": self.cipher.encrypt(text),
                    "uploaded_at": uploaded_at,
                    "type": file_type,
                },
            )
            for text, vec, idx in zip(texts, vectors, indices, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            client.upsert(self.collection, points=points[start : start + _UPSERT_BATCH])
        return len(points)

    async def upsert_chunks(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        uploaded_at: str,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        indices: Sequence[int] | None = None,
    ) -> int:
        self._require()
        if self.vector_size is not None:
            for vec in vectors:
                if len(vec) != self.vector_size:
                    raise DimensionMismatch(self.vector_size, len(vec))
        idx = list(indices) if indices is not None else list(range(len(texts)))
        return await self._run(self._upsert, doc_id, filename, file_type, uploaded_at, texts, vectors, idx)

    def _delete_document(self, doc_id: str) -> int:
        client = self._require()
        for flt in (_doc_filter(doc_id), _legacy_filter(doc_id)):
            count = client.count(self.collection, count_filter=flt, exact=True).count
            if count:
                client.delete(self.collection, points_selector=qm.FilterSelector(filter=flt))
                return count
        return 0

    async def delete_document(self, doc_id: str) -> int:
        return await self._run(self._delete_document, doc_id)

    def _clear(self) -> int:
        client = self._require()
        total = 0
        while True:
            page, _ = client.scroll(self.collection, limit=_SCROLL_PAGE, with_payload=False, with_vectors=False)
            if not page:
                return total
            client.delete(self.collection, points_selector=[p.id for p in page])
            total += len(page)

    async def clear(self) -> int:
        """Delete every point (the collection itself is kept)."""
        return await self._run(self._clear)

    def _optimize(self) -> None:
        client = self._require()
        client.update_collection(self.collection, optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=0))

    async def optimize(self) -> None:
        await self._run(self._optimize)

    def _encrypt_legacy(self) -> int:
        client = self._require()
        migrated = 0
        for rec in self._scroll_all(None, ["text"]):
            text = (rec.payload or {}).get("text")
            if isinstance(text, str) and text and not self.cipher.is_encrypted(text):
                client.set_payload(self.collection, payload={"text": self.cipher.encrypt(text)}, points=[rec.id])
                migrated += 1
        return migrated

    async def encrypt_legacy_payloads(self) -> int:
        """Encrypt plaintext ``text`` payloads written before encryption was enabled."""
        if not self.cipher.enabled or not self.collection_ready:
            return 0
        migrated = await self._run(self._encrypt_legacy)
        if migrated:
            logger.info("Encrypted %d legacy plaintext points", migrated)
        return migrated
