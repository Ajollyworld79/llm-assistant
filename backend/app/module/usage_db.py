"""Postgres-backed usage history for the /backend graphs.

Schema:
    usage_events(id BIGSERIAL, ts TIMESTAMPTZ, session_id TEXT, endpoint TEXT,
                 model TEXT, tokens_input INT, tokens_output INT, tokens_cached INT,
                 latency_ms INT, status_code INT, error_kind TEXT)

If DATABASE_URL is not set or asyncpg is unavailable, the module falls back to
an in-memory deque (capped at 5000 events) so local dev works without postgres.
Inserts are fire-and-forget — never block a request on a logging failure.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import os
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import asyncpg  # type: ignore
    HAS_ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore
    HAS_ASYNCPG = False


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS usage_events (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id    TEXT,
    endpoint      TEXT NOT NULL,
    model         TEXT,
    tokens_input  INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    tokens_cached INTEGER DEFAULT 0,
    latency_ms    INTEGER DEFAULT 0,
    status_code   INTEGER DEFAULT 0,
    error_kind    TEXT
);
CREATE INDEX IF NOT EXISTS usage_events_ts_idx ON usage_events (ts);

CREATE TABLE IF NOT EXISTS chat_events (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    thread_id     TEXT,
    query         TEXT,
    answer        TEXT,
    embed_ms      INTEGER DEFAULT 0,
    ai_ms         INTEGER DEFAULT 0,
    total_ms      INTEGER DEFAULT 0,
    model         TEXT,
    tokens_input  INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    doc_count     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS chat_events_ts_idx ON chat_events (ts);
CREATE INDEX IF NOT EXISTS chat_events_thread_idx ON chat_events (thread_id);
"""


class UsageDB:
    def __init__(self) -> None:
        self._pool: Optional[Any] = None
        self._database_url: Optional[str] = os.getenv('DATABASE_URL')
        self._mem_events: deque = deque(maxlen=5000)
        self._mem_chats: deque = deque(maxlen=2000)
        self._mem_chat_seq: int = 0
        self._initialized = False

    @property
    def backend_kind(self) -> str:
        if self._pool is not None:
            return 'postgres'
        return 'memory'

    @property
    def enabled(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self._database_url or not HAS_ASYNCPG:
            logger.info("UsageDB: using in-memory backend (DATABASE_URL=%s, asyncpg=%s)",
                        bool(self._database_url), HAS_ASYNCPG)
            return
        try:
            self._pool = await asyncpg.create_pool(self._database_url, min_size=1, max_size=4)
            async with self._pool.acquire() as conn:
                await conn.execute(_SCHEMA_SQL)
            logger.info("UsageDB: postgres pool ready and schema ensured")
        except Exception as e:
            logger.exception("UsageDB: failed to init postgres, falling back to memory: %s", e)
            self._pool = None

    async def shutdown(self) -> None:
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                pass
            self._pool = None

    def log_event(self, *, session_id: Optional[str], endpoint: str,
                  model: Optional[str], tokens_input: int = 0, tokens_output: int = 0,
                  tokens_cached: int = 0, latency_ms: int = 0,
                  status_code: int = 200, error_kind: Optional[str] = None) -> None:
        """Fire-and-forget insert. Never raises to caller."""
        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc),
            "session_id": session_id or "",
            "endpoint": endpoint,
            "model": model or "",
            "tokens_input": int(tokens_input or 0),
            "tokens_output": int(tokens_output or 0),
            "tokens_cached": int(tokens_cached or 0),
            "latency_ms": int(latency_ms or 0),
            "status_code": int(status_code or 0),
            "error_kind": error_kind,
        }
        if self._pool is not None:
            asyncio.create_task(self._insert_pg(record))
        else:
            self._mem_events.append(record)

    async def _insert_pg(self, record: Dict[str, Any]) -> None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore
                await conn.execute(
                    """INSERT INTO usage_events
                       (ts, session_id, endpoint, model, tokens_input, tokens_output,
                        tokens_cached, latency_ms, status_code, error_kind)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)""",
                    record["ts"], record["session_id"], record["endpoint"], record["model"],
                    record["tokens_input"], record["tokens_output"], record["tokens_cached"],
                    record["latency_ms"], record["status_code"], record["error_kind"],
                )
        except Exception as e:
            logger.warning("usage_events insert failed: %s", e, exc_info=True)

    def log_chat(self, *, thread_id: Optional[str], query: str, answer: str,
                 embed_ms: int = 0, ai_ms: int = 0, total_ms: int = 0,
                 model: Optional[str] = None, tokens_input: int = 0,
                 tokens_output: int = 0, doc_count: int = 0) -> None:
        """Fire-and-forget insert of a chat exchange. Never raises."""
        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc),
            "thread_id": thread_id or "",
            "query": query or "",
            "answer": answer or "",
            "embed_ms": int(embed_ms or 0),
            "ai_ms": int(ai_ms or 0),
            "total_ms": int(total_ms or 0),
            "model": model or "",
            "tokens_input": int(tokens_input or 0),
            "tokens_output": int(tokens_output or 0),
            "doc_count": int(doc_count or 0),
        }
        if self._pool is not None:
            asyncio.create_task(self._insert_chat_pg(record))
        else:
            self._mem_chat_seq += 1
            record["id"] = self._mem_chat_seq
            self._mem_chats.append(record)

    async def _insert_chat_pg(self, record: Dict[str, Any]) -> None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore
                await conn.execute(
                    """INSERT INTO chat_events
                       (ts, thread_id, query, answer, embed_ms, ai_ms, total_ms,
                        model, tokens_input, tokens_output, doc_count)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                    record["ts"], record["thread_id"], record["query"], record["answer"],
                    record["embed_ms"], record["ai_ms"], record["total_ms"],
                    record["model"], record["tokens_input"], record["tokens_output"],
                    record["doc_count"],
                )
        except Exception as e:
            logger.warning("chat_events insert failed: %s", e, exc_info=True)

    async def list_chats(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Return chat events newest-first as JSON-serializable dicts."""
        limit = max(1, min(int(limit or 200), 1000))
        if self._pool is not None:
            try:
                async with self._pool.acquire() as conn:  # type: ignore
                    rows = await conn.fetch(
                        """SELECT id, ts, thread_id, query, answer, embed_ms, ai_ms,
                                  total_ms, model, tokens_input, tokens_output, doc_count
                           FROM chat_events
                           ORDER BY id DESC
                           LIMIT $1""",
                        limit,
                    )
                    return [{
                        "id": r["id"],
                        "ts": r["ts"].isoformat() if r["ts"] else None,
                        "thread_id": r["thread_id"],
                        "query": r["query"],
                        "answer": r["answer"],
                        "embed_ms": r["embed_ms"],
                        "ai_ms": r["ai_ms"],
                        "total_ms": r["total_ms"],
                        "model": r["model"],
                        "tokens_input": r["tokens_input"],
                        "tokens_output": r["tokens_output"],
                        "doc_count": r["doc_count"],
                    } for r in rows]
            except Exception as e:
                logger.warning("chat_events list query failed: %s", e, exc_info=True)
                return []
        # Memory fallback — newest first
        out = []
        for r in list(self._mem_chats)[-limit:][::-1]:
            out.append({
                "id": r.get("id"),
                "ts": r["ts"].isoformat() if isinstance(r.get("ts"), datetime.datetime) else r.get("ts"),
                "thread_id": r.get("thread_id", ""),
                "query": r.get("query", ""),
                "answer": r.get("answer", ""),
                "embed_ms": r.get("embed_ms", 0),
                "ai_ms": r.get("ai_ms", 0),
                "total_ms": r.get("total_ms", 0),
                "model": r.get("model", ""),
                "tokens_input": r.get("tokens_input", 0),
                "tokens_output": r.get("tokens_output", 0),
                "doc_count": r.get("doc_count", 0),
            })
        return out

    async def delete_chat(self, chat_id: int) -> bool:
        if self._pool is not None:
            try:
                async with self._pool.acquire() as conn:  # type: ignore
                    res = await conn.execute("DELETE FROM chat_events WHERE id = $1", int(chat_id))
                    # res is e.g. "DELETE 1"
                    parts = res.split()
                    return len(parts) >= 2 and parts[-1].isdigit() and int(parts[-1]) > 0
            except Exception as e:
                logger.debug("chat_events delete failed: %s", e)
                return False
        before = len(self._mem_chats)
        self._mem_chats = deque(
            (r for r in self._mem_chats if r.get("id") != int(chat_id)),
            maxlen=self._mem_chats.maxlen,
        )
        return len(self._mem_chats) < before

    async def delete_all_chats(self) -> int:
        if self._pool is not None:
            try:
                async with self._pool.acquire() as conn:  # type: ignore
                    res = await conn.execute("DELETE FROM chat_events")
                    parts = res.split()
                    return int(parts[-1]) if parts and parts[-1].isdigit() else 0
            except Exception as e:
                logger.debug("chat_events bulk delete failed: %s", e)
                return 0
        n = len(self._mem_chats)
        self._mem_chats.clear()
        return n

    async def query_summary(self) -> Dict[str, Any]:
        if self._pool is not None:
            try:
                async with self._pool.acquire() as conn:  # type: ignore
                    row = await conn.fetchrow("""
                        SELECT
                            COUNT(*) FILTER (WHERE ts >= now() - interval '24 hours') AS req_24h,
                            COUNT(*) FILTER (WHERE ts >= now() - interval '7 days')  AS req_7d,
                            COALESCE(SUM(tokens_input)  FILTER (WHERE ts >= now() - interval '24 hours'), 0) AS tok_in_24h,
                            COALESCE(SUM(tokens_output) FILTER (WHERE ts >= now() - interval '24 hours'), 0) AS tok_out_24h,
                            COALESCE(SUM(tokens_cached) FILTER (WHERE ts >= now() - interval '24 hours'), 0) AS tok_cached_24h,
                            COUNT(*) FILTER (WHERE error_kind IS NOT NULL AND ts >= now() - interval '24 hours') AS errors_24h
                        FROM usage_events
                    """)
                    if row:
                        return dict(row)
            except Exception as e:
                logger.debug("usage_events query_summary failed: %s", e)
        # In-memory aggregate
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff_24h = now - datetime.timedelta(hours=24)
        cutoff_7d = now - datetime.timedelta(days=7)
        e24 = [e for e in self._mem_events if e["ts"] >= cutoff_24h]
        e7d = [e for e in self._mem_events if e["ts"] >= cutoff_7d]
        return {
            "req_24h": len(e24),
            "req_7d": len(e7d),
            "tok_in_24h": sum(e["tokens_input"] for e in e24),
            "tok_out_24h": sum(e["tokens_output"] for e in e24),
            "tok_cached_24h": sum(e["tokens_cached"] for e in e24),
            "errors_24h": sum(1 for e in e24 if e.get("error_kind")),
        }

    async def query_timeseries(self, metric: str, range_label: str = "24h") -> List[Dict[str, Any]]:
        """Return [{ts, value}] bucketed for the requested metric and range."""
        if range_label == "7d":
            interval = "7 days"
            bucket = "1 hour"
        elif range_label == "30d":
            interval = "30 days"
            bucket = "1 day"
        else:
            interval = "24 hours"
            bucket = "1 hour"

        metric_expr = {
            "requests": "COUNT(*)",
            "tokens": "COALESCE(SUM(tokens_input + tokens_output), 0)",
            "cache_hit_ratio": "CASE WHEN SUM(tokens_input) > 0 THEN ROUND(100.0 * SUM(tokens_cached)::numeric / SUM(tokens_input), 1) ELSE 0 END",
            "fallback_rate": "CASE WHEN COUNT(*) > 0 THEN ROUND(100.0 * COUNT(*) FILTER (WHERE error_kind = 'fallback') / COUNT(*), 1) ELSE 0 END",
        }.get(metric, "COUNT(*)")

        if self._pool is not None:
            try:
                async with self._pool.acquire() as conn:  # type: ignore
                    rows = await conn.fetch(f"""
                        SELECT date_trunc('{bucket.split()[1]}', ts) AS bucket,
                               {metric_expr} AS value
                        FROM usage_events
                        WHERE ts >= now() - interval '{interval}'
                        GROUP BY bucket
                        ORDER BY bucket
                    """)
                    return [{"ts": r["bucket"].isoformat(), "value": float(r["value"])} for r in rows]
            except Exception as e:
                logger.debug("usage_events query_timeseries failed: %s", e)

        # In-memory bucketed fallback
        return self._mem_timeseries(metric, range_label)

    def _mem_timeseries(self, metric: str, range_label: str) -> List[Dict[str, Any]]:
        now = datetime.datetime.now(datetime.timezone.utc)
        if range_label == "7d":
            cutoff = now - datetime.timedelta(days=7)
            bucket_seconds = 3600
        elif range_label == "30d":
            cutoff = now - datetime.timedelta(days=30)
            bucket_seconds = 86400
        else:
            cutoff = now - datetime.timedelta(hours=24)
            bucket_seconds = 3600

        relevant = [e for e in self._mem_events if e["ts"] >= cutoff]
        if not relevant:
            return []

        buckets: Dict[float, List[Dict[str, Any]]] = {}
        for e in relevant:
            bts = int(e["ts"].timestamp() // bucket_seconds) * bucket_seconds
            buckets.setdefault(bts, []).append(e)

        out: List[Dict[str, Any]] = []
        for bts in sorted(buckets):
            es = buckets[bts]
            if metric == "requests":
                value = len(es)
            elif metric == "tokens":
                value = sum(e["tokens_input"] + e["tokens_output"] for e in es)
            elif metric == "cache_hit_ratio":
                ti = sum(e["tokens_input"] for e in es)
                tc = sum(e["tokens_cached"] for e in es)
                value = round(100.0 * tc / ti, 1) if ti > 0 else 0.0
            elif metric == "fallback_rate":
                total = len(es)
                fb = sum(1 for e in es if e.get("error_kind") == "fallback")
                value = round(100.0 * fb / total, 1) if total > 0 else 0.0
            else:
                value = len(es)
            out.append({
                "ts": datetime.datetime.fromtimestamp(bts, datetime.timezone.utc).isoformat(),
                "value": float(value),
            })
        return out


usage_db = UsageDB()
