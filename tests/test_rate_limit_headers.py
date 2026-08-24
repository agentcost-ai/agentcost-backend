"""Rate limit headers let an agent pace itself instead of discovering the
limit by tripping it. Emitted in three forms because clients are split between
the current IETF draft, the earlier unprefixed triple, and the X- prefixed one.

Probed against a non-exempt path: RateLimitMiddleware.EXEMPT_PATHS skips /v1/health
and the docs routes entirely, so those carry no headers by design.
"""

# Non-exempt, and cheap -- the 404 comes from the router, after the middleware
# has already counted the request and attached its headers.
PROBE = "/v1/rate-limit-probe"

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ratelimit_headers_on_a_normal_response(client: AsyncClient):
    # Headers ride on every response the limiter sees, whatever the status --
    # a client that only learns its quota from a 200 cannot pace itself through
    # a run of errors.
    response = await client.get(PROBE)

    # Current IETF draft form: RateLimit-Policy: "default";q=<limit>;w=<window>
    policy = response.headers["ratelimit-policy"]
    assert policy.startswith('"default";q=')
    assert ";w=" in policy

    # RateLimit: "default";r=<remaining>;t=<reset>
    current = response.headers["ratelimit"]
    assert current.startswith('"default";r=')
    assert ";t=" in current

    # Earlier triple, still what most scanners and clients read.
    assert int(response.headers["ratelimit-limit"]) > 0
    assert int(response.headers["ratelimit-remaining"]) >= 0
    assert int(response.headers["ratelimit-reset"]) >= 0

    # The original X- prefixed form is kept so existing clients don't break.
    assert response.headers["x-ratelimit-limit"] == response.headers["ratelimit-limit"]


@pytest.mark.asyncio
async def test_remaining_decreases_across_requests(client: AsyncClient):
    first = await client.get(PROBE)
    second = await client.get(PROBE)
    assert int(second.headers["ratelimit-remaining"]) < int(
        first.headers["ratelimit-remaining"]
    )


@pytest.mark.asyncio
async def test_policy_and_limit_agree(client: AsyncClient):
    response = await client.get(PROBE)
    limit = int(response.headers["ratelimit-limit"])
    assert f"q={limit}" in response.headers["ratelimit-policy"]


@pytest.mark.asyncio
async def test_429_carries_retry_after_and_the_error_envelope(client: AsyncClient):
    """Exhaust the window, then check the response an agent actually has to parse."""
    from app.config import get_settings

    limit = get_settings().rate_limit_requests

    response = None
    for _ in range(limit + 5):
        response = await client.get(PROBE)
        if response.status_code == 429:
            break

    assert response.status_code == 429, "rate limiter never tripped"
    assert int(response.headers["retry-after"]) >= 0
    assert response.headers["ratelimit-remaining"] == "0"

    body = response.json()
    assert body["error"]["code"] == "rate_limited"
    assert body["error"]["hint"]
    assert isinstance(body["detail"], str)
    # Fields the previous shape carried, kept so existing clients keep working.
    assert body["limit"] == limit
    assert "retry_after" in body
