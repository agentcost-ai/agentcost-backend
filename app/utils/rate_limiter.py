"""
AgentCost Backend - Rate Limiting Middleware

Sliding-window rate limiting, in two backends: RateLimiter keeps counters in
process memory, RedisRateLimiter shares them across workers. Selected by
RATE_LIMIT_BACKEND; see check_rate_limit for the fallback rules.
"""

import logging
import time
import hashlib
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import get_settings

logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    """The caller's address, taken from the part of X-Forwarded-For we control.

    Proxies *append* the address they observed, so the rightmost entries are
    written by our own infrastructure and everything to their left is whatever
    the caller sent. Reading the leftmost entry -- the obvious choice, and the
    one this used to make -- lets any caller mint a fresh rate-limit bucket per
    request just by varying the header, which removes IP limiting from the
    unauthenticated routes (/v1/auth/login) that rely on it.

    settings.trusted_proxy_hops says how many entries are ours. If the header
    is missing, shorter than expected, or we are deployed with no proxy at all,
    fall back to the peer address, which cannot be forged over TCP.
    """
    hops = get_settings().trusted_proxy_hops
    peer = request.client.host if request.client else "unknown"

    if hops <= 0:
        return peer

    forwarded = request.headers.get("X-Forwarded-For")
    if not forwarded:
        return peer

    chain = [part.strip() for part in forwarded.split(",") if part.strip()]
    if len(chain) < hops:
        return peer
    return chain[-hops]


class RateLimiter:
    """
    In-memory rate limiter using sliding window.

    There is no lock: is_allowed is synchronous and never awaits, so the event
    loop cannot interleave two calls to it. That holds only while the caller
    stays on one thread -- running it from a thread pool would need real
    locking.

    KNOWN LIMITATION (multi-process): counters live in this process only, so
    running N uvicorn workers gives each client an effective N x limit, and a
    client is only limited on the worker that happens to serve it. Set
    RATE_LIMIT_BACKEND=redis with REDIS_URL to share counters across workers
    (see RedisRateLimiter); this class stays the fallback when Redis is not
    configured or is unreachable.
    """

    # Upper bound on tracked keys. Each key holds at most requests_per_window
    # floats, so this bounds memory even under a distributed key-space attack
    # (an attacker rotating IPs or bearer tokens mints a new key every request).
    DEFAULT_MAX_KEYS = 50_000

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        max_keys: int = DEFAULT_MAX_KEYS,
    ):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        # Plain dict, never a defaultdict: reading a missing key must not create
        # an entry. The previous defaultdict(list) grew one permanent entry per
        # distinct IP / API key ever seen, including keys whose requests had all
        # expired, and nothing ever removed them.
        self._requests: dict[str, list[float]] = {}
        self._last_sweep = self._now()

    @staticmethod
    def _now() -> float:
        # Monotonic: a wall-clock adjustment (NTP step, DST on a naive host)
        # must not make windows expire early or hang around forever.
        return time.monotonic()

    def _cleanup_old_requests(self, key: str, now: float) -> list[float]:
        """Drop timestamps outside the window; forget the key when it empties."""
        cutoff = now - self.window_seconds
        timestamps = self._requests.get(key)
        if timestamps is None:
            return []

        fresh = [ts for ts in timestamps if ts > cutoff]
        if fresh:
            self._requests[key] = fresh
        else:
            self._requests.pop(key, None)
        return fresh

    def _sweep(self, now: float) -> None:
        """Evict keys with no live requests; enforce the key cap if still over."""
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now

        cutoff = now - self.window_seconds
        for key in [k for k, ts in self._requests.items() if not ts or max(ts) <= cutoff]:
            self._requests.pop(key, None)

        if len(self._requests) > self.max_keys:
            # Still over budget: drop the least recently active keys. Forgetting
            # a counter can only let a request through, never wrongly reject one.
            overflow = len(self._requests) - self.max_keys
            oldest = sorted(self._requests.items(), key=lambda item: max(item[1], default=0.0))
            for key, _ in oldest[:overflow]:
                self._requests.pop(key, None)
            logger.warning(
                "Rate limiter key cap (%d) exceeded; evicted %d idle keys",
                self.max_keys,
                overflow,
            )

    def is_allowed(self, key: str) -> tuple[bool, int, int]:
        """
        Check if request is allowed for the given key.

        Returns:
            Tuple of (is_allowed, remaining_requests, reset_in_seconds)
        """
        now = self._now()
        self._sweep(now)
        timestamps = self._cleanup_old_requests(key, now)

        current_count = len(timestamps)
        remaining = max(0, self.requests_per_window - current_count)

        # Calculate when the oldest request will expire
        if timestamps:
            oldest = min(timestamps)
            reset_in = max(0, int(oldest + self.window_seconds - now))
        else:
            reset_in = self.window_seconds

        if current_count >= self.requests_per_window:
            return False, 0, reset_in

        # Record this request
        if timestamps:
            timestamps.append(now)
        else:
            self._requests[key] = [now]
        return True, remaining - 1, reset_in

    @property
    def tracked_keys(self) -> int:
        """Number of keys currently held (exposed for tests / diagnostics)."""
        return len(self._requests)

    @staticmethod
    def get_bucket(path: str) -> str:
        """
        Traffic class for a request path.

        Ingest and dashboard reads share one Authorization header, so a busy SDK
        used to spend the dashboard's budget and lock the UI out of its own
        project. Separate buckets keep a burst of events from blanking charts.
        """
        return "ingest" if path.startswith("/v1/events") else "api"

    def get_key_from_request(self, request: Request) -> str:
        """
        Extract rate limit key from request.

        Kept for callers that want a single identifying key; the middleware uses
        get_keys_from_request so a rotating token cannot escape the IP budget.
        """
        return self.get_keys_from_request(request)[0]

    def get_keys_from_request(self, request: Request) -> list[str]:
        """Every bucket this request must fit in; the IP bucket always among them.

        The IP must be counted unconditionally: any key derived from the
        Authorization header -- alone or combined with the IP -- is minted fresh
        by a caller rotating `Bearer <random>` per request, which is how login
        throttling was bypassed entirely. The token bucket is kept as a second,
        narrower limit so one busy SDK key cannot exhaust a shared NAT's budget.
        """
        bucket = self.get_bucket(request.url.path)
        keys = [f"{bucket}:ip:{_client_ip(request)}"]

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            key_hash = hashlib.sha256(auth_header.encode()).hexdigest()
            keys.append(f"{bucket}:api_key:{key_hash}")

        return keys


class RedisRateLimiter:
    """
    Sliding-window limiter backed by Redis, shared across worker processes.

    Runs as a Lua script so trim-count-insert is atomic; as separate round
    trips, concurrent workers all read the same count and each admits a
    request over the limit. Scores are wall clock, not monotonic — monotonic
    is per process and meaningless once the counter is shared.

    If Redis is slow or down the caller falls back to the in-memory limiter
    rather than failing the request: rate limiting is a guard rail, not a
    correctness requirement.
    """

    # KEYS[1] = bucket key; ARGV = now, window, limit, unique member.
    # Returns {allowed, remaining, oldest_score}.
    _SCRIPT = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
    local count = redis.call('ZCARD', key)
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')[2] or tostring(now)
    if count >= limit then
        return {0, 0, oldest}
    end
    redis.call('ZADD', key, now, ARGV[4])
    redis.call('EXPIRE', key, window + 1)
    return {1, limit - count - 1, oldest}
    """

    def __init__(self, url: str, requests_per_window: int, window_seconds: int,
                 namespace: str = "agentcost:rl:"):
        self.url = url
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.namespace = namespace
        self._client = None
        self._script = None
        self._unavailable_logged = False

    async def _ensure_client(self):
        """Connect lazily — import time has no running event loop."""
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self.url,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            self._script = self._client.register_script(self._SCRIPT)
        return self._client

    async def is_allowed(self, key: str):
        """
        (allowed, remaining, reset_in_seconds), or None if Redis is unusable.

        None is deliberately distinct from (False, ...) so an outage never
        reads as "rate limited".
        """
        try:
            await self._ensure_client()
            now = time.time()
            allowed, remaining, oldest = await self._script(
                keys=[f"{self.namespace}{key}"],
                args=[now, self.window_seconds, self.requests_per_window, uuid4().hex],
            )
            self._unavailable_logged = False
            reset_in = max(0, int(float(oldest) + self.window_seconds - now))
            return bool(int(allowed)), int(remaining), reset_in
        except Exception as exc:  # noqa: BLE001 — any Redis failure degrades
            if not self._unavailable_logged:
                self._unavailable_logged = True
                logger.warning(
                    "Redis rate limiter unavailable (%s: %s); falling back to the "
                    "in-memory limiter. Limits are per worker until Redis returns.",
                    type(exc).__name__, exc,
                )
            return None

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                pass
            self._client = None


# Global rate limiter instances
settings = get_settings()

rate_limiter = RateLimiter(
    requests_per_window=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_period,
)

redis_rate_limiter = None
_backend = (settings.rate_limit_backend or "memory").lower()
if _backend == "redis":
    try:
        if not settings.redis_url:
            raise RuntimeError("REDIS_URL is unset")
        import redis.asyncio  # noqa: F401 — availability probe only

        redis_rate_limiter = RedisRateLimiter(
            url=settings.redis_url,
            requests_per_window=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_period,
        )
        logger.info("Rate limiting shared via Redis")
    except (ImportError, RuntimeError) as exc:
        # Say why, so a deployment that asked for shared limits does not
        # quietly get per-worker ones.
        logger.warning("Redis rate limiting unavailable (%s); using in-memory limiter", exc)
elif _backend != "memory":
    logger.warning("Unknown RATE_LIMIT_BACKEND=%s; using in-memory limiter", _backend)


async def check_rate_limit(key: str) -> tuple[bool, int, int]:
    """Consult Redis when configured, otherwise the per-process limiter."""
    if redis_rate_limiter is not None:
        verdict = await redis_rate_limiter.is_allowed(key)
        if verdict is not None:
            return verdict
    return rate_limiter.is_allowed(key)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Applies to all /v1/ endpoints.
    Adds standard rate limit headers to responses.
    """

    # Paths that are exempt from rate limiting
    EXEMPT_PATHS = {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/v1/health",
    }

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Skip rate limiting for non-API paths
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        # Every bucket must admit the request; headers report the tightest
        # remaining count and the longest reset.
        is_allowed, remaining, reset_in = True, None, 0
        for key in rate_limiter.get_keys_from_request(request):
            allowed, key_remaining, key_reset = await check_rate_limit(key)
            is_allowed = is_allowed and allowed
            remaining = key_remaining if remaining is None else min(remaining, key_remaining)
            reset_in = max(reset_in, key_reset)
        remaining = remaining or 0

        # Add rate limit headers to all responses
        headers = {
            "X-RateLimit-Limit": str(settings.rate_limit_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_in),
        }

        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "retry_after": reset_in,
                    "limit": settings.rate_limit_requests,
                    "period": f"{settings.rate_limit_period} seconds",
                },
                headers={
                    **headers,
                    "Retry-After": str(reset_in),
                },
            )

        # Process the request
        response = await call_next(request)

        # Add rate limit headers to successful responses
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value

        return response
