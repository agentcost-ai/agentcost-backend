"""Catalogue integrity rules added after the Aug 2026 sync audit.

Covers: first-party listings beating reseller medians, admin price overrides
outlasting syncs, mode/deprecation_date ingestion, retired models remaining
billable at their last-known rate, and output-cap field precedence.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import ModelPricing
from app.services.pricing_service import (
    RETIRED_ABSENT_NOTE,
    RETIRED_REJECTED_NOTE,
    RETIRED_UNPRICED_NOTE,
    PricingService,
)


def _payload(models: dict) -> dict:
    """Build a LiteLLM-shaped response from {name: (in_per_token, out_per_token)}."""
    return {
        name: {
            "input_cost_per_token": rates[0],
            "output_cost_per_token": rates[1],
            "litellm_provider": "openai",
            "max_tokens": 4096,
        }
        for name, rates in models.items()
    }


def _service(db: AsyncSession, payload: dict) -> PricingService:
    """A PricingService whose HTTP client returns `payload`."""
    service = PricingService(db)
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload)
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    service._get_client = AsyncMock(return_value=client)
    return service


@pytest.mark.asyncio
async def test_first_party_listing_wins_over_reseller_median(
    test_session: AsyncSession,
):
    """The production bug this guards: claude-sonnet-4-5 was attributed to
    provider 'snowflake' because with two same-priced listings the median
    landed on the reseller's key. The maker's own listing must win outright,
    even when a reseller undercuts it."""
    payload = {
        "claude-test": {"input_cost_per_token": 0.000003,
                        "output_cost_per_token": 0.000015,
                        "litellm_provider": "anthropic"},
        "snowflake/claude-test": {"input_cost_per_token": 0.000003,
                                  "output_cost_per_token": 0.000015,
                                  "litellm_provider": "snowflake"},
        "hyperbolic/claude-test": {"input_cost_per_token": 0.000001,
                                   "output_cost_per_token": 0.000005,
                                   "litellm_provider": "hyperbolic"},
    }

    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "claude-test")
    )).scalar_one()
    assert row.provider == "anthropic"
    assert row.input_price_per_1k == pytest.approx(0.003)
    assert row.output_price_per_1k == pytest.approx(0.015)


@pytest.mark.asyncio
async def test_admin_override_prices_survive_sync(test_session: AsyncSession):
    """An admin price correction must outlast the next sync; before this fix
    the sync silently reverted it within 24h. Capabilities still refresh."""
    payload = {
        "overridden-model": {"input_cost_per_token": 0.000001,
                             "output_cost_per_token": 0.000002,
                             "litellm_provider": "openai",
                             "max_output_tokens": 4096},
    }
    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "overridden-model")
    )).scalar_one()
    row.input_price_per_1k = 0.123
    row.pricing_source = "admin_override"
    await test_session.commit()

    moved = {
        "overridden-model": {"input_cost_per_token": 0.000009,
                             "output_cost_per_token": 0.000002,
                             "litellm_provider": "openai",
                             "max_output_tokens": 8192,
                             "supports_vision": True},
    }
    result = await _service(test_session, moved).sync_from_litellm(track_changes=True)
    await test_session.commit()
    await test_session.refresh(row)

    assert row.input_price_per_1k == pytest.approx(0.123)
    assert row.pricing_source == "admin_override"
    # Non-price facts still come from upstream.
    assert row.max_tokens == 8192
    assert row.supports_vision is True
    # The unapplied upstream move must not be reported as a price change.
    assert result["changes"]["price_changes"] == []
    # And the row must not be retired as unlisted.
    assert result["models_deactivated"] == 0


@pytest.mark.asyncio
async def test_mode_and_deprecation_date_are_stored(test_session: AsyncSession):
    payload = {
        "chat-model": {"input_cost_per_token": 0.000001,
                       "output_cost_per_token": 0.000002,
                       "litellm_provider": "openai",
                       "mode": "chat",
                       "deprecation_date": "2026-09-30"},
        "embed-model": {"input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0,
                        "litellm_provider": "openai",
                        "mode": "embedding",
                        # sample_spec-style junk must not be stored.
                        "deprecation_date": "date when the model is retired"},
    }
    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    rows = {
        r.model_name: r
        for r in (await test_session.execute(select(ModelPricing))).scalars()
    }
    assert rows["chat-model"].mode == "chat"
    assert rows["chat-model"].deprecation_date == "2026-09-30"
    assert rows["embed-model"].mode == "embedding"
    assert rows["embed-model"].deprecation_date is None


@pytest.mark.asyncio
async def test_retired_model_still_prices_exact_lookups(
    test_session: AsyncSession,
):
    """A model dropped upstream is still billable at its last-known rate; the
    fuzzy path would otherwise silently bill a sibling's rate."""
    both = _payload({"stays": (0.000001, 0.000002), "vanishes": (0.000003, 0.000004)})
    only_stays = _payload({"stays": (0.000001, 0.000002)})

    await _service(test_session, both).sync_from_litellm()
    await test_session.commit()
    await _service(test_session, only_stays).sync_from_litellm()
    await test_session.commit()

    gone = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "vanishes")
    )).scalar_one()
    assert gone.is_active is False
    assert gone.notes == RETIRED_ABSENT_NOTE

    pricing = await PricingService(test_session).get_model_pricing("vanishes")
    assert pricing is not None
    assert pricing["match"] == "exact"
    assert pricing["input"] == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_unit_error_retirement_never_prices_lookups(
    test_session: AsyncSession,
):
    """Rows retired because every listing was implausible hold wrong prices,
    not stale ones -- the exact-on-retired fallback must skip them."""
    test_session.add(ModelPricing(
        model_name="only-bad", input_price_per_1k=8.0, output_price_per_1k=35.0,
        provider="wandb", pricing_source="litellm", is_active=True,
    ))
    await test_session.commit()

    payload = {
        "wandb/only-bad": {"input_cost_per_token": 0.008,
                           "output_cost_per_token": 0.035,
                           "litellm_provider": "wandb"},
    }
    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "only-bad")
    )).scalar_one()
    assert row.is_active is False
    assert row.notes == RETIRED_REJECTED_NOTE

    assert await PricingService(test_session).get_model_pricing("only-bad") is None


@pytest.mark.asyncio
async def test_model_that_goes_unpriced_is_noted_and_not_priced(
    test_session: AsyncSession,
):
    """A model still listed upstream but with no token price (went free, or
    priced per image/second) must not keep billing its stale paid rate."""
    priced = _payload({"stays": (0.000001, 0.000002),
                       "goes-free": (0.000003, 0.000004)})
    await _service(test_session, priced).sync_from_litellm()
    await test_session.commit()

    now_free = _payload({"stays": (0.000001, 0.000002)})
    now_free["goes-free"] = {"input_cost_per_token": 0,
                             "output_cost_per_token": 0,
                             "litellm_provider": "openai"}
    result = await _service(test_session, now_free).sync_from_litellm()
    await test_session.commit()
    assert result["models_deactivated"] == 1

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "goes-free")
    )).scalar_one()
    assert row.is_active is False
    assert row.notes == RETIRED_UNPRICED_NOTE

    assert await PricingService(test_session).get_model_pricing("goes-free") is None


@pytest.mark.asyncio
async def test_output_cap_prefers_max_output_tokens_over_legacy_field(
    test_session: AsyncSession,
):
    """LiteLLM's max_tokens is legacy: on entries without max_output_tokens it
    holds the context window. When both exist, max_output_tokens is the cap."""
    payload = {
        "capped-model": {"input_cost_per_token": 0.000001,
                         "output_cost_per_token": 0.000002,
                         "litellm_provider": "openai",
                         "max_tokens": 128000,
                         "max_output_tokens": 8192},
    }
    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "capped-model")
    )).scalar_one()
    assert row.max_tokens == 8192
