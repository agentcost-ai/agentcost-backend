"""
Currency service.

Fetches and caches USD-based exchange rates from frankfurter.app (ECB-sourced,
no API key required) so the rest of the backend can convert event costs
(reported in USD) into a user's preferred display currency.

Designed to fail open: if the FX endpoint is unreachable, falls back to a
sensible static rate so budget evaluation never blocks the ingestion hot path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Final

import httpx


logger = logging.getLogger(__name__)


SUPPORTED_CURRENCIES: Final[tuple[str, ...]] = ("USD", "INR")
CURRENCY_SYMBOLS: Final[dict[str, str]] = {"USD": "$", "INR": "₹"}

# Fall-back rates (USD -> X). ONLY used when the live FX endpoint is
# unreachable. As soon as the network call to frankfurter.app succeeds the
# cache is populated with the real ECB rate, so steady-state operation never
# touches these constants. Reviewed periodically — bump if the live rate
# diverges materially from the fallback.
_FALLBACK_RATES: Final[dict[str, float]] = {"USD": 1.0, "INR": 95.0}

# Cache live for 6 hours. ECB rates only refresh once per business day so this
# is plenty fresh while keeping the network out of the hot path.
_CACHE_TTL_SECONDS: Final[int] = 6 * 60 * 60

# Minimum gap between background refresh attempts for the same currency.
_REFRESH_COOLDOWN_SECONDS: Final[int] = 60

# frankfurter.app moved to frankfurter.dev/v1/ in 2025; we hit the new URL
# directly and also enable follow_redirects defensively for future moves.
_FRANKFURTER_URL: Final[str] = "https://api.frankfurter.dev/v1/latest"


class CurrencyService:
    """Process-wide cache of USD->target FX rates."""

    # currency -> (rate, fetched_unix_ts)
    _cache: dict[str, tuple[float, float]] = {}
    _lock = asyncio.Lock()
    # In-flight background refreshes, kept referenced so they aren't GC'd.
    _refresh_tasks: dict[str, asyncio.Task] = {}
    # currency -> unix ts before which we won't schedule another refresh
    _refresh_cooldown: dict[str, float] = {}

    @staticmethod
    def normalize(currency: str | None) -> str:
        if not currency:
            return "USD"
        upper = currency.upper().strip()
        return upper if upper in SUPPORTED_CURRENCIES else "USD"

    @staticmethod
    def symbol(currency: str | None) -> str:
        return CURRENCY_SYMBOLS.get(CurrencyService.normalize(currency), "$")

    @classmethod
    def _cached_rate(cls, target: str) -> tuple[float | None, bool]:
        """Return (rate or None, is_fresh) for an already-known rate."""
        cached = cls._cache.get(target)
        if not cached:
            return None, False
        return cached[0], (time.time() - cached[1]) < _CACHE_TTL_SECONDS

    @classmethod
    def cached_usd_to(cls, currency: str | None) -> float:
        """Best known rate for *currency* without ever touching the network.

        For the ingestion hot path: a batch must not wait on api.frankfurter.dev
        (5s timeout, behind a process-wide lock, with the ingest transaction
        open) just to compare spend against a budget. A stale or fallback rate
        moves a budget threshold by a few percent; a stalled request loses
        telemetry. Refreshing happens in the background.
        """
        target = cls.normalize(currency)
        if target == "USD":
            return 1.0

        rate, fresh = cls._cached_rate(target)
        if rate is not None and fresh:
            return rate

        cls._schedule_refresh(target)
        return rate if rate is not None else _FALLBACK_RATES.get(target, 1.0)

    @classmethod
    def _schedule_refresh(cls, target: str) -> None:
        """Kick off a background refresh, at most one per currency.

        Cooldown included: a failed fetch caches the fallback with a short TTL,
        so without it every subsequent request would queue another attempt and
        we would hammer the FX provider for as long as it stayed down.
        """
        now = time.time()
        if now < cls._refresh_cooldown.get(target, 0.0):
            return
        existing = cls._refresh_tasks.get(target)
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no event loop (sync context) — nothing to do
            return

        cls._refresh_cooldown[target] = now + _REFRESH_COOLDOWN_SECONDS
        task = loop.create_task(cls._refresh(target))
        cls._refresh_tasks[target] = task
        task.add_done_callback(lambda t: cls._refresh_tasks.pop(target, None))

    @classmethod
    async def _refresh(cls, target: str) -> float:
        """Fetch and cache the live rate. Never raises."""
        # Serialize concurrent refreshes so a burst of events doesn't fan out
        # to the FX API a dozen times.
        async with cls._lock:
            rate, fresh = cls._cached_rate(target)
            if rate is not None and fresh:
                return rate

            try:
                async with httpx.AsyncClient(
                    timeout=5.0,
                    follow_redirects=True,
                ) as client:
                    response = await client.get(
                        _FRANKFURTER_URL,
                        params={"base": "USD", "symbols": target},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    live_rate = float(payload["rates"][target])
                    cls._cache[target] = (live_rate, time.time())
                    return live_rate
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Currency FX fetch failed for USD->%s (%s); using fallback rate",
                    target,
                    exc,
                )
                fallback = _FALLBACK_RATES.get(target, 1.0)
                # Cache fallback briefly so we don't hammer the network on
                # repeated failures, but with a short TTL so we retry soon.
                cls._cache[target] = (fallback, time.time() - _CACHE_TTL_SECONDS + 300)
                return fallback

    @classmethod
    async def usd_to(cls, currency: str | None) -> float:
        """Return how many units of *currency* equal 1 USD.

        Blocks on the FX provider only when nothing is cached yet; once any
        rate is known, callers get it immediately and the refresh runs in the
        background. Use :meth:`cached_usd_to` where blocking is never
        acceptable.
        """
        target = cls.normalize(currency)
        if target == "USD":
            return 1.0

        rate, fresh = cls._cached_rate(target)
        if rate is not None:
            if not fresh:
                cls._schedule_refresh(target)
            return rate

        # Cold cache: wait for the in-flight fetch rather than short-cutting to
        # _FALLBACK_RATES. _refresh re-checks the cache inside the lock, so a
        # waiter gets the rate the winner just fetched instead of a hardcoded
        # one. Ingest no longer reaches here (routes/events.py passes
        # hot_path=True, which uses cached_usd_to), so every remaining caller is
        # a dashboard or report read, where a stale-by-8% figure is worse than
        # waiting for the real one.
        return await cls._refresh(target)

    @classmethod
    async def convert_from_usd(cls, amount_usd: float, currency: str | None) -> float:
        """Convert *amount_usd* into the target currency."""
        rate = await cls.usd_to(currency)
        return amount_usd * rate

    @classmethod
    def format_amount(cls, amount: float, currency: str | None) -> str:
        """Render a currency-aware human-readable amount (used in emails)."""
        cur = cls.normalize(currency)
        sym = cls.symbol(cur)
        try:
            return f"{sym}{amount:,.2f}"
        except (TypeError, ValueError):
            return f"{sym}0.00"
