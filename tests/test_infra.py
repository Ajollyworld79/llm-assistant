import asyncio

from backend.app.infra.model_catalog import catalog_for_viewer
from backend.app.infra.monitor import Monitor, rss_mb
from backend.app.infra.privacy import redact_pii
from backend.app.infra.tasks import drain, spawn
from backend.app.infra.usage_db import UsageDB


def test_redact_pii():
    out = redact_pii("mail me at anna.k@example.dk, cpr 010203-1234, phone +45 12 34 56 78")
    assert "anna.k@example.dk" not in out and "010203" not in out and "12 34 56 78" not in out
    assert "[email-redacted]" in out and "[cpr-redacted]" in out and "[phone-redacted]" in out


def test_redact_leaves_ordinary_text_and_short_numbers():
    text = "Error 0x80070005 on Windows 11 after update 2026"
    assert redact_pii(text) == text and redact_pii("") == ""


def test_model_catalog_roles():
    roles = {m["id"]: m["role"] for m in catalog_for_viewer("gemini-3.6-flash", "gemini-3.5-flash-lite")}
    assert roles["gemini-3.6-flash"] == "primary" and roles["gemini-3.5-flash-lite"] == "fallback"
    assert roles["gemini-2.5-pro"] == "alternative"


def test_monitor_counts_and_health():
    m = Monitor()
    m.record_request("/a", 10, 200)
    m.record_request("/a", 30, 500, "boom")
    m.record_llm(100, 25, True)
    stats = m.stats()
    assert stats["requests"] == {"total": 2, "success": 1, "failed": 1}
    assert stats["endpoints"]["/a"]["avg_time"] == 20 and stats["recent_errors"][0]["error"] == "boom"
    assert m.cache_hit_ratio == 0.25 and m.fallback_invocations == 1
    assert "Recent restart detected" in m.health_summary()["issues"]
    assert rss_mb() > 0


async def test_spawn_keeps_reference_and_drains():
    done = []

    async def work():
        await asyncio.sleep(0)
        done.append(1)

    spawn(work())
    await drain()
    assert done == [1]


async def test_usage_db_memory_backend_paging_and_filter():
    db = UsageDB(None)
    await db.initialize()
    assert db.backend_kind == "memory" and not db.configured and not await db.ping()
    for i in range(5):
        db.log_chat(thread_id=f"t{i}", query=f"question {i}", answer="Alpha" if i % 2 else "Beta")
    db.log_event(session_id=None, endpoint="/search", model="m", tokens_input=10, tokens_output=5, tokens_cached=2)
    db.log_event(session_id=None, endpoint="/search", model="m", error_kind="fallback")

    page = await db.list_chats(limit=2)
    assert [r["thread_id"] for r in page["items"]] == ["t4", "t3"] and page["total"] == 5
    nxt = await db.list_chats(limit=2, before_id=page["next_before_id"])
    assert [r["thread_id"] for r in nxt["items"]] == ["t2", "t1"]
    assert (await db.list_chats(q="alpha"))["total"] == 2
    assert (await db.list_chats(q="t0"))["total"] == 1
    assert (await db.list_chats(limit="bad"))["total"] == 5  # type: ignore[arg-type]

    assert await db.count_chats() == 5
    assert await db.delete_chat(page["items"][0]["id"]) is True and await db.delete_chat(9999) is False
    assert await db.delete_all_chats() == 4

    summary = await db.query_summary()
    assert summary["req_24h"] == 2 and summary["tok_in_24h"] == 10 and summary["errors_24h"] == 1
    assert (await db.query_timeseries("requests", "24h"))[0]["value"] == 2.0
    assert (await db.query_timeseries("tokens", "7d"))[0]["value"] == 15.0
    assert (await db.query_timeseries("cache_hit_ratio", "30d"))[0]["value"] == 20.0
    assert (await db.query_timeseries("fallback_rate", "24h"))[0]["value"] == 50.0
    await db.shutdown()


async def test_usage_db_postgres_failure_falls_back_to_memory():
    db = UsageDB("postgresql://u:p@127.0.0.1:1/none")
    await asyncio.wait_for(db.initialize(), 30)
    assert db.backend_kind == "memory" and db.init_error and db.configured
    await db.initialize()  # idempotent


def test_make_client_requires_key():
    from dataclasses import replace

    from backend.app.core.config import Settings
    from backend.app.services.gemini import make_client

    assert make_client(Settings(gemini_api_key=None)) is None
    assert make_client(replace(Settings(), gemini_api_key="test-key")) is not None


def test_setup_logging_writes_file(tmp_path):
    import logging

    from backend.app.core.config import Settings
    from backend.app.core.logging import setup_logging

    setup_logging(Settings(log_to_file=True, log_path=str(tmp_path / "logs")))
    logging.getLogger("x").info("hello file")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello file" in next((tmp_path / "logs").glob("logger_*.log")).read_text()
    setup_logging(Settings(log_to_file=False))


class FakeConn:
    def __init__(self, log):
        self.log = log

    async def execute(self, sql, *args):
        self.log.append(("execute", sql.split()[0], args))
        return "DELETE 3" if sql.startswith("DELETE") else "INSERT 0 1"

    async def fetchval(self, sql, *args):
        self.log.append(("fetchval", sql, args))
        return 7

    async def fetch(self, sql, *args):
        self.log.append(("fetch", sql, args))
        if "chat_events" in sql:
            row = {
                "id": 5,
                "ts": None,
                "thread_id": "t",
                "query": "q",
                "answer": "a",
                "embed_ms": 1,
                "ai_ms": 2,
                "total_ms": 3,
                "model": "m",
                "tokens_input": 1,
                "tokens_output": 2,
                "doc_count": 3,
            }
            return [row, {**row, "id": 4}]
        import datetime

        return [{"bucket": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC), "value": 2}]

    async def fetchrow(self, sql, *args):
        return {"req_24h": 1, "req_7d": 2}


class FakePool:
    def __init__(self):
        self.log = []
        self.closed = False

    def acquire(self):
        pool = self

        class Ctx:
            async def __aenter__(self):
                return FakeConn(pool.log)

            async def __aexit__(self, *exc):
                return False

        return Ctx()

    async def close(self):
        self.closed = True


async def test_usage_db_postgres_paths_with_fake_pool():
    db = UsageDB("postgresql://x")
    db._pool = pool = FakePool()
    db._initialized = True
    assert db.backend_kind == "postgres" and await db.ping() and await db.count_chats() == 7
    db.log_chat(thread_id="t", query="q", answer="a")
    db.log_event(session_id=None, endpoint="/search", model="m")
    await drain()
    assert [e[1] for e in pool.log if e[0] == "execute"] == ["INSERT", "INSERT"]

    page = await db.list_chats(limit=1, before_id=10, q="50%_off")
    assert page["total"] == 7 and page["next_before_id"] == 5 and len(page["items"]) == 1
    sql_args = next(e for e in pool.log if e[0] == "fetch")[2]
    assert sql_args[0] == "%50\\%\\_off%" and 10 in sql_args
    assert await db.delete_chat(1) is True
    assert await db.delete_all_chats() == 3
    assert (await db.query_summary())["req_7d"] == 2
    series = await db.query_timeseries("requests", "7d")
    assert series == [{"ts": "2026-01-01T00:00:00+00:00", "value": 2.0}]
    await db.shutdown()
    assert pool.closed and db.backend_kind == "memory"
