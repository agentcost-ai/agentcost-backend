"""
Redis-backed rate limiter.

The fallback path is always exercised. The shared-counter path needs a live
Redis and skips without one, so it runs in any environment that has Redis
(CI, staging, production) rather than silently never running:

    docker run -d -p 6379:6379 redis:7-alpine
    pytest tests/test_rate_limiter_redis.py -v
"""

import asyncio

import pytest

from app.utils.rate_limiter import RateLimiter, RedisRateLimiter, check_rate_limit

REDIS_URL = "redis://127.0.0.1:6379/0"
NAMESPACE = "agentcost:pytest:"


async def _redis_available() -> bool:
    probe = RedisRateLimiter(REDIS_URL, 5, 60, namespace=NAMESPACE)
    try:
        return await probe.is_allowed("availability-probe") is not None
    finally:
        await probe.close()


@pytest.fixture
def redis_url():
    if not asyncio.run(_redis_available()):
        pytest.skip("no Redis server on 127.0.0.1:6379")
    return REDIS_URL


class TestRedisFallback:
    """An outage must degrade to local limiting, never reject traffic."""

    def test_unreachable_redis_returns_none_not_denied(self):
        """None means 'fall back'; False would mean 'rate limited'."""
        limiter = RedisRateLimiter("redis://127.0.0.1:6399/0", 5, 60)

        async def run():
            try:
                return await limiter.is_allowed("k")
            finally:
                await limiter.close()

        assert asyncio.run(run()) is None

    def test_requests_still_served_while_redis_is_down(self):
        allowed, remaining, _ = asyncio.run(check_rate_limit("probe-key"))
        assert allowed is True
        assert remaining >= 0


class TestRedisSharedCounter:
    """Two limiter instances stand in for two uvicorn workers."""

    def test_limit_is_shared_across_instances(self, redis_url):
        async def run():
            a = RedisRateLimiter(redis_url, 3, 60, namespace=NAMESPACE)
            b = RedisRateLimiter(redis_url, 3, 60, namespace=NAMESPACE)
            key = "shared-budget"
            try:
                client = await a._ensure_client()
                await client.delete(f"{NAMESPACE}{key}")
                verdicts = [
                    await a.is_allowed(key),
                    await b.is_allowed(key),
                    await a.is_allowed(key),
                    await b.is_allowed(key),
                ]
                await client.delete(f"{NAMESPACE}{key}")
                return verdicts
            finally:
                await a.close()
                await b.close()

        verdicts = asyncio.run(run())
        allowed = [v[0] for v in verdicts]
        # Per-process limiters would have allowed all four.
        assert allowed == [True, True, True, False]
        assert verdicts[-1][1] == 0
        assert verdicts[-1][2] > 0, "a blocked request must report a reset window"

    def test_concurrent_requests_cannot_exceed_the_limit(self, redis_url):
        """The check must be atomic — separate round trips race."""
        async def run():
            key = "atomic-budget"
            limiters = [RedisRateLimiter(redis_url, 5, 60, namespace=NAMESPACE)
                        for _ in range(10)]
            try:
                client = await limiters[0]._ensure_client()
                await client.delete(f"{NAMESPACE}{key}")
                verdicts = await asyncio.gather(*[l.is_allowed(key) for l in limiters])
                await client.delete(f"{NAMESPACE}{key}")
                return verdicts
            finally:
                for l in limiters:
                    await l.close()

        verdicts = asyncio.run(run())
        assert sum(1 for v in verdicts if v[0]) == 5, "limit exceeded under concurrency"
