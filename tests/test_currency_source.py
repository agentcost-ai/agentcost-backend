"""The FX endpoint says whether its rate is live or the static fallback."""

import pytest

from app.services import currency_service as cs


@pytest.mark.asyncio
async def test_fallback_rate_is_labelled(monkeypatch):
    cs.CurrencyService._cache.clear()
    cs.CurrencyService._source.clear()

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            raise RuntimeError("provider down")

    monkeypatch.setattr(cs.httpx, "AsyncClient", Boom)
    rate = await cs.CurrencyService.usd_to("INR")
    assert rate == cs._FALLBACK_RATES["INR"]
    assert cs.CurrencyService.source_for("INR") == "fallback"
    assert cs.CurrencyService.source_for("USD") == "exact"


@pytest.mark.asyncio
async def test_live_rate_is_labelled(monkeypatch):
    cs.CurrencyService._cache.clear()
    cs.CurrencyService._source.clear()

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"rates": {"INR": 88.4}}

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return Resp()

    monkeypatch.setattr(cs.httpx, "AsyncClient", Client)
    assert await cs.CurrencyService.usd_to("INR") == 88.4
    assert cs.CurrencyService.source_for("INR") == "frankfurter.dev"
