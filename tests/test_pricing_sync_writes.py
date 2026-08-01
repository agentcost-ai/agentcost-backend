"""LiteLLM sync writes only what changed.

The sync used to assign every field on every row each run, so all ~3,500 rows
were UPDATEd whether or not anything moved upstream. That inflated the reported
update count into meaninglessness and made a sync take minutes.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import ModelPricing
from app.services.pricing_service import PricingService


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
async def test_first_sync_creates_then_second_reports_no_changes(test_session: AsyncSession):
    payload = _payload({"gpt-test-a": (0.000001, 0.000002),
                        "gpt-test-b": (0.000003, 0.000004)})

    first = await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()
    assert first["models_created"] == 2
    assert first["models_updated"] == 0

    # Same upstream data: nothing should be rewritten.
    second = await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()
    assert second["models_created"] == 0
    assert second["models_updated"] == 0
    assert second["models_unchanged"] == 2


@pytest.mark.asyncio
async def test_only_the_moved_price_counts_as_updated(test_session: AsyncSession):
    payload = _payload({"gpt-test-a": (0.000001, 0.000002),
                        "gpt-test-b": (0.000003, 0.000004)})
    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    moved = _payload({"gpt-test-a": (0.000009, 0.000002),   # input changed
                      "gpt-test-b": (0.000003, 0.000004)})  # unchanged
    result = await _service(test_session, moved).sync_from_litellm()
    await test_session.commit()

    assert result["models_updated"] == 1
    assert result["models_unchanged"] == 1

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "gpt-test-a")
    )).scalar_one()
    assert row.input_price_per_1k == pytest.approx(0.000009 * 1000)


@pytest.mark.asyncio
async def test_unchanged_rows_keep_their_original_updated_at(test_session: AsyncSession):
    """updated_at is what the public catalogue reports, so an unchanged row
    must not be touched."""
    payload = _payload({"gpt-test-a": (0.000001, 0.000002)})
    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "gpt-test-a")
    )).scalar_one()
    before = row.updated_at

    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()
    await test_session.refresh(row)

    assert row.updated_at == before


@pytest.mark.asyncio
async def test_two_keys_canonicalizing_to_one_name_do_not_violate_uniqueness(
    test_session: AsyncSession,
):
    """model_name is UNIQUE, and LiteLLM ships prefixed and bare keys for the
    same model. The bulk lookup must see a row created earlier in the same run."""
    payload = {
        "gpt-dup-test": {"input_cost_per_token": 0.000001,
                         "output_cost_per_token": 0.000002,
                         "litellm_provider": "openai"},
        "openai/gpt-dup-test": {"input_cost_per_token": 0.000005,
                                "output_cost_per_token": 0.000006,
                                "litellm_provider": "openai"},
    }

    result = await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()  # must not raise IntegrityError

    rows = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "gpt-dup-test")
    )).scalars().all()
    assert len(rows) == 1
    assert result["models_created"] == 1


@pytest.mark.asyncio
async def test_price_change_tracking_still_reports_movement(test_session: AsyncSession):
    """track_changes must keep working now that writes are conditional."""
    await _service(test_session, _payload({"gpt-test-a": (0.000001, 0.000002)})).sync_from_litellm()
    await test_session.commit()

    result = await _service(
        test_session, _payload({"gpt-test-a": (0.000002, 0.000002)})
    ).sync_from_litellm(track_changes=True)
    await test_session.commit()

    assert result["has_changes"] is True
    assert any(c["model"] == "gpt-test-a" for c in result["changes"]["price_changes"])
