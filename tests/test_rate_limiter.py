"""
Tests for the in-memory rate limiter.

Focus: the counter dict must not grow without bound (it previously kept one
entry per distinct IP / API key forever), and ingest traffic must not consume
the dashboard's budget.
"""

import pytest
from starlette.requests import Request

from app.utils.rate_limiter import RateLimiter


class _Clock:
    """Controllable stand-in for time.monotonic()."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limiter(requests_per_window=3, window_seconds=60, max_keys=1000):
    limiter = RateLimiter(requests_per_window, window_seconds, max_keys=max_keys)
    clock = _Clock()
    limiter._now = clock  # instance attribute shadows the static method
    limiter._last_sweep = clock.now
    return limiter, clock


def _request(path: str, *, auth: str | None = None, ip: str = "10.0.0.1") -> Request:
    headers = [(b"authorization", auth.encode())] if auth else []
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("test", 80),
            "path": path,
            "query_string": b"",
            "headers": headers,
            "client": (ip, 1234),
        }
    )


def test_limit_is_enforced_within_the_window():
    limiter, _ = _limiter(requests_per_window=3)

    assert [limiter.is_allowed("k")[0] for _ in range(3)] == [True, True, True]
    allowed, remaining, reset_in = limiter.is_allowed("k")
    assert allowed is False
    assert remaining == 0
    assert 0 < reset_in <= 60


def test_window_expiry_restores_budget():
    limiter, clock = _limiter(requests_per_window=2)

    limiter.is_allowed("k")
    limiter.is_allowed("k")
    assert limiter.is_allowed("k")[0] is False

    clock.advance(61)
    assert limiter.is_allowed("k")[0] is True


def test_inspecting_an_unknown_key_does_not_create_an_entry():
    """defaultdict(list) used to mint a permanent entry on every lookup."""
    limiter, clock = _limiter()

    assert limiter._cleanup_old_requests("never-seen", clock.now) == []
    assert limiter.tracked_keys == 0


def test_expired_keys_are_forgotten():
    """Idle clients must not accumulate forever -- this was the memory leak."""
    limiter, clock = _limiter()

    for i in range(500):
        limiter.is_allowed(f"ip:10.0.0.{i}")
    assert limiter.tracked_keys == 500

    # Past the window, the next touch of a key drops it...
    clock.advance(61)
    limiter._cleanup_old_requests("ip:10.0.0.0", clock.now)
    assert "ip:10.0.0.0" not in limiter._requests

    # ...and the periodic sweep clears the rest without needing a touch.
    limiter.is_allowed("ip:10.0.0.999")
    assert limiter.tracked_keys == 1


def test_key_count_is_capped_for_active_keys():
    """A rotating key-space (new bearer token per request) stays bounded."""
    limiter, clock = _limiter(requests_per_window=10, max_keys=10)

    for i in range(50):
        limiter.is_allowed(f"api_key:{i}")
        clock.advance(0.1)

    # Force the periodic sweep: every key is still inside the window, so the
    # cap -- not expiry -- has to do the evicting.
    limiter._last_sweep = clock.now - 61
    limiter.is_allowed("api_key:fresh")

    assert limiter.tracked_keys <= limiter.max_keys + 1
    # The most recent callers are the ones kept.
    assert "api_key:49" in limiter._requests
    assert "api_key:0" not in limiter._requests


def test_ingest_and_dashboard_use_separate_buckets():
    """A busy SDK must not spend the dashboard's allowance on the same key."""
    limiter, _ = _limiter()

    auth = "Bearer sk_test_12345"
    assert limiter.get_key_from_request(
        _request("/v1/events/batch", auth=auth)
    ) != limiter.get_key_from_request(
        _request("/v1/analytics/overview", auth=auth)
    )


def test_anonymous_requests_are_keyed_by_ip():
    limiter, _ = _limiter()

    key = limiter.get_key_from_request(_request("/v1/analytics/overview", ip="203.0.113.7"))
    assert key == "api:ip:203.0.113.7"


@pytest.mark.parametrize(
    "path,bucket",
    [
        ("/v1/events/batch", "ingest"),
        ("/v1/events", "ingest"),
        ("/v1/analytics/report", "api"),
        ("/v1/optimizations", "api"),
    ],
)
def test_bucket_classification(path, bucket):
    assert RateLimiter.get_bucket(path) == bucket
