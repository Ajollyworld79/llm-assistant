import asyncio
import json
from datetime import UTC, datetime, timedelta

from backend.app.infra.rate_limit import FailureThrottle, RateLimiter


def limiter(tmp_path, **kw):
    args = {"per_min": 3, "per_day": 5, "per_session": 2, "daily_token_budget": 1000}
    args.update(kw)
    return RateLimiter(budget_file=tmp_path / "budget.json", **args)


async def test_ip_minute_and_day_limits(tmp_path):
    lim = limiter(tmp_path)
    assert [await lim.check_ip("1.1.1.1") for _ in range(3)] == [None] * 3
    assert "minute" in await lim.check_ip("1.1.1.1")
    assert await lim.check_ip("2.2.2.2") is None  # other IPs unaffected
    lim = limiter(tmp_path, per_min=100)
    for _ in range(5):
        assert await lim.check_ip("3.3.3.3") is None
    assert "daily" in await lim.check_ip("3.3.3.3")


async def test_session_limit_and_rotation(tmp_path):
    lim = limiter(tmp_path)
    assert await lim.check_session(None) is None
    assert [await lim.check_session("s") for _ in range(2)] == [None, None]
    assert "session" in await lim.check_session("s")
    lim._sessions_date = "2000-01-01"
    assert await lim.check_session("s") is None


async def test_prune_drops_idle_ips(tmp_path):
    lim = limiter(tmp_path)
    await lim.check_ip("9.9.9.9")
    lim._ip_day["9.9.9.9"].clear()
    lim._ip_minute["9.9.9.9"].clear()
    lim._prune(10**12)
    assert "9.9.9.9" not in lim._ip_day and lim.snapshot()["active_ips"] == 0


async def test_budget_persists_across_restart(tmp_path):
    first = limiter(tmp_path)
    assert await first.reserve(300) is None
    await first.settle(300, 450)
    second = limiter(tmp_path)  # simulated restart
    assert second.snapshot()["tokens_used_today"] == 450
    await second.reserve(600)
    assert "budget" in await second.check_budget()
    assert "budget" in await second.reserve(1)


async def test_stale_budget_file_is_ignored(tmp_path):
    yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    (tmp_path / "budget.json").write_text(json.dumps({"date": yesterday, "tokens_used": 999}))
    assert limiter(tmp_path).snapshot()["tokens_used_today"] == 0


async def test_corrupt_budget_file_starts_at_zero(tmp_path):
    (tmp_path / "budget.json").write_text("{not json")
    assert limiter(tmp_path).snapshot()["tokens_used_today"] == 0


async def test_day_rollover_resets_budget(tmp_path):
    lim = limiter(tmp_path)
    await lim.reserve(900)
    lim._budget["date"] = "2000-01-01"
    assert await lim.check_budget() is None
    assert lim.snapshot()["tokens_used_today"] == 0


async def test_concurrent_reservations_cannot_overshoot_far(tmp_path):
    lim = limiter(tmp_path, daily_token_budget=1000)
    results = await asyncio.gather(*(lim.reserve(400) for _ in range(10)))
    assert results.count(None) == 3  # 0, 400, 800 used -> the 4th is refused
    assert lim.snapshot()["tokens_used_today"] == 1200


async def test_settle_never_goes_negative(tmp_path):
    lim = limiter(tmp_path)
    await lim.settle(500, 0)
    assert lim.snapshot()["tokens_used_today"] == 0


def test_failure_throttle():
    t = FailureThrottle(max_failures=2, window_seconds=300)
    assert not t.blocked("ip")
    t.record_failure("ip")
    t.record_failure("ip")
    assert t.blocked("ip") and not t.blocked("other")
    t.reset("ip")
    assert not t.blocked("ip")


def test_failure_throttle_window_expires():
    t = FailureThrottle(max_failures=1, window_seconds=0)
    t.record_failure("ip")
    assert not t.blocked("ip")
