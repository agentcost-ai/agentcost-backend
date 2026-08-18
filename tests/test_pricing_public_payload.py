"""Public /v1/pricing surface: cache rates for the SDK, deprecations feed.

The SDK's cost calculator reads `cached_input` / `cache_write` from
GET /v1/pricing; the route used to omit them, so every client-side estimate
billed cached tokens at the full input rate.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import ModelPricing


@pytest.fixture
async def catalogue(test_session: AsyncSession):
    test_session.add_all([
        ModelPricing(
            model_name="cache-model", input_price_per_1k=0.003,
            output_price_per_1k=0.015, cached_input_price_per_1k=0.0003,
            cache_write_price_per_1k=0.00375, provider="anthropic",
            is_active=True, mode="chat", deprecation_date="2026-12-31",
        ),
        ModelPricing(
            model_name="plain-model", input_price_per_1k=0.001,
            output_price_per_1k=0.002, provider="openai", is_active=True,
        ),
        ModelPricing(
            model_name="dead-model", input_price_per_1k=0.001,
            output_price_per_1k=0.002, provider="openai", is_active=False,
            deprecation_date="2026-01-01",
        ),
    ])
    await test_session.commit()


@pytest.mark.asyncio
async def test_public_pricing_includes_cache_rates(client: AsyncClient, catalogue):
    resp = await client.get("/v1/pricing")
    assert resp.status_code == 200
    pricing = resp.json()["pricing"]

    assert pricing["cache-model"]["cached_input"] == pytest.approx(0.0003)
    assert pricing["cache-model"]["cache_write"] == pytest.approx(0.00375)
    # No published rate stays None -- the SDK bills full input rate on None,
    # while 0.0 would mean "cached tokens are free".
    assert pricing["plain-model"]["cached_input"] is None
    assert pricing["plain-model"]["cache_write"] is None


@pytest.mark.asyncio
async def test_deprecations_lists_active_models_soonest_first(
    client: AsyncClient, catalogue
):
    resp = await client.get("/v1/pricing/deprecations")
    assert resp.status_code == 200
    body = resp.json()

    models = [d["model"] for d in body["deprecations"]]
    assert models == ["cache-model"]      # inactive rows are excluded
    assert body["deprecations"][0]["deprecation_date"] == "2026-12-31"
    assert body["count"] == 1


@pytest.mark.asyncio
async def test_deprecations_does_not_shadow_model_lookup(
    client: AsyncClient, catalogue
):
    """/deprecations is registered before /{model_name}; both must resolve."""
    resp = await client.get("/v1/pricing/cache-model")
    assert resp.status_code == 200
    assert resp.json()["matched_to"] == "cache-model"
