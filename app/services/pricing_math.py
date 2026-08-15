"""
AgentCost Backend - Pricing Arithmetic

The one implementation of the cache-aware cost formula. Ingest and the
pricing service both price through here so the two can never drift and
report different costs for the same call.
"""

from typing import Optional

# Prices are quoted per 1000 tokens, so a single call's cost runs to small
# fractions of a cent. Every caller rounds at this precision.
COST_PRECISION = 8


def price_event(
    pricing: Optional[dict],
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Cost in USD for one call, accounting for prompt-cache rates.

    Cached input is a subset of input_tokens billed at a lower rate, so the
    uncached remainder and the cached portion are priced separately. Cache
    writes are additional tokens billed at a premium (Anthropic) and are
    added on top. Where a provider publishes no cache rate the tokens fall
    back to the standard input rate — a discount is never assumed.
    """
    if pricing is None:
        return 0.0

    input_rate = pricing.get("input") or 0.0
    output_rate = pricing.get("output") or 0.0
    # `or input_rate` and not `pricing.get(..., input_rate)`: the column is
    # nullable, so an explicit NULL must fall back too, not just a missing key.
    cached_rate = pricing.get("cached_input") or input_rate
    write_rate = pricing.get("cache_write") or input_rate

    cached = max(0, min(cached_tokens, input_tokens))
    uncached = input_tokens - cached

    total = (
        (uncached / 1000) * input_rate
        + (cached / 1000) * cached_rate
        + (max(0, cache_write_tokens) / 1000) * write_rate
        + (output_tokens / 1000) * output_rate
    )
    return round(total, COST_PRECISION)
