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

# frankfurter.app moved to frankfurter.dev/v1/ in 2025; we hit the new URL
# directly and also enable follow_redirects defensively for future moves.
_FRANKFURTER_URL: Final[str] = "https://api.frankfurter.dev/v1/latest"


class CurrencyService:
    """Process-wide cache of USD->target FX rates."""

    # currency -> (rate, fetched_unix_ts)
    _cache: dict[str, tuple[float, float]] = {}
    _lock = asyncio.Lock()

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
    async def usd_to(cls, currency: str | None) -> float:
        """Return how many units of *currency* equal 1 USD."""
        target = cls.normalize(currency)
        if target == "USD":
            return 1.0

        now = time.time()
        cached = cls._cache.get(target)
        if cached and (now - cached[1]) < _CACHE_TTL_SECONDS:
            return cached[0]

        # Serialize concurrent refreshes so a burst of events doesn't fan out
        # to the FX API a dozen times.
        async with cls._lock:
            cached = cls._cache.get(target)
            if cached and (now - cached[1]) < _CACHE_TTL_SECONDS:
                return cached[0]

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
                    rate = float(payload["rates"][target])
                    cls._cache[target] = (rate, time.time())
                    return rate
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
