"""LiteLLM sync writes only what changed.

The sync used to assign every field on every row each run, so all ~3,500 rows
were UPDATEd whether or not anything moved upstream. That inflated the reported
update count into meaninglessness and made a sync take minutes.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common import MAX_PRICE_PER_1K
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
async def test_multi_host_listings_are_resolved_once_not_overwritten(
    test_session: AsyncSession,
):
    """LiteLLM ships the same model under several hosting providers at different
    rates. Writing them in sequence rewrote the row inside a single run, which
    was then reported as a price change on every sync."""
    payload = {
        "anyscale/shared-model": {"input_cost_per_token": 0.000001,
                                  "output_cost_per_token": 0.000001,
                                  "litellm_provider": "anyscale"},
        "hyperbolic/shared-model": {"input_cost_per_token": 0.00000012,
                                    "output_cost_per_token": 0.0000003,
                                    "litellm_provider": "hyperbolic"},
    }

    first = await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()
    assert first["models_created"] == 1
    assert first["models_updated"] == 0      # no intra-run overwrite
    assert first["models_deduplicated"] == 1

    # The same payload again must report nothing at all.
    second = await _service(test_session, payload).sync_from_litellm(track_changes=True)
    await test_session.commit()
    assert second["models_updated"] == 0
    assert second["models_unchanged"] == 1
    assert second["changes"]["price_changes"] == []


@pytest.mark.asyncio
async def test_collision_winner_is_the_median_not_the_last_key(
    test_session: AsyncSession,
):
    """The bug this guards: the winner used to be whichever key sorted last in
    the upstream JSON, so wandb's per-million listing priced
    deepseek-ai/DeepSeek-R1-0528 at $135/1k against deepinfra's $0.0005."""
    payload = {
        "deepinfra/shared": {"input_cost_per_token": 0.0000005,
                             "output_cost_per_token": 0.00000215,
                             "litellm_provider": "deepinfra"},
        "hyperbolic/shared": {"input_cost_per_token": 0.00000025,
                              "output_cost_per_token": 0.00000025,
                              "litellm_provider": "hyperbolic"},
        # Implausible, and deliberately LAST -- the old rule would pick it.
        "wandb/shared": {"input_cost_per_token": 0.135,
                         "output_cost_per_token": 0.540,
                         "litellm_provider": "wandb"},
    }

    result = await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "shared")
    )).scalar_one()
    # Median of the two plausible listings, never the $135 outlier.
    assert row.input_price_per_1k == pytest.approx(0.0005)
    assert row.output_price_per_1k == pytest.approx(0.00215)
    assert result["models_deduplicated"] == 2


@pytest.mark.asyncio
async def test_selection_is_independent_of_upstream_key_order(
    test_session: AsyncSession,
):
    """A reshuffle upstream must not rewrite prices or fabricate price changes."""
    rates = {
        "a/shuffled": (0.000003, 0.000003, "deepinfra"),
        "b/shuffled": (0.000001, 0.000001, "nscale"),
        "c/shuffled": (0.000002, 0.000002, "together_ai"),
    }
    def build(keys):
        return {k: {"input_cost_per_token": rates[k][0],
                    "output_cost_per_token": rates[k][1],
                    "litellm_provider": rates[k][2]} for k in keys}

    await _service(test_session, build(["a/shuffled", "b/shuffled", "c/shuffled"])).sync_from_litellm()
    await test_session.commit()

    # Same data, reversed order: nothing may change.
    second = await _service(
        test_session, build(["c/shuffled", "b/shuffled", "a/shuffled"])
    ).sync_from_litellm(track_changes=True)
    await test_session.commit()

    assert second["models_updated"] == 0
    assert second["models_unchanged"] == 1
    assert second["changes"]["price_changes"] == []


@pytest.mark.asyncio
async def test_model_with_only_implausible_listings_is_not_written(
    test_session: AsyncSession,
):
    """No price at all beats a unit-error price: event_service falls back to the
    SDK's own cost instead of billing $8/1k as an exact match."""
    payload = {
        "wandb/only-bad": {"input_cost_per_token": 0.008,
                           "output_cost_per_token": 0.035,
                           "litellm_provider": "wandb"},
    }

    result = await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    rows = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "only-bad")
    )).scalars().all()
    assert rows == []
    assert result["models_rejected"] == 1
    assert result["models_created"] == 0


@pytest.mark.asyncio
async def test_sync_never_writes_a_row_the_admin_ui_would_reject(
    test_session: AsyncSession,
):
    """MAX_PRICE_PER_1K bounds the admin PATCH route, so a row above it could be
    displayed but never re-saved. The sync must honour the same ceiling."""
    payload = {
        # Plausible: 0.000001/token -> $0.001 per 1k.
        "sane-a": {"input_cost_per_token": 0.000001,
                   "output_cost_per_token": 0.000002,
                   "litellm_provider": "openai"},
        "sane-b": {"input_cost_per_token": 0.00001,
                   "output_cost_per_token": 0.00003,
                   "litellm_provider": "openai"},
        # Unit errors: 0.05/token -> $50 per 1k, five times the ceiling.
        "unit-error-in": {"input_cost_per_token": 0.05,
                          "output_cost_per_token": 0.000002,
                          "litellm_provider": "wandb"},
        "unit-error-out": {"input_cost_per_token": 0.000001,
                           "output_cost_per_token": 0.05,
                           "litellm_provider": "wandb"},
    }
    result = await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    rows = (await test_session.execute(select(ModelPricing))).scalars().all()
    assert {r.model_name for r in rows} == {"sane-a", "sane-b"}
    assert result["models_rejected"] == 2
    for row in rows:
        assert row.input_price_per_1k <= MAX_PRICE_PER_1K
        assert row.output_price_per_1k <= MAX_PRICE_PER_1K


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


@pytest.mark.asyncio
async def test_vanished_models_are_retired_and_returning_ones_restored(
    test_session: AsyncSession,
):
    """A model dropped upstream must stop being recommended; one that returns
    is reactivated only if the sync itself retired it."""
    both = _payload({"stays": (0.000001, 0.000002), "vanishes": (0.000003, 0.000004)})
    only_stays = _payload({"stays": (0.000001, 0.000002)})

    await _service(test_session, both).sync_from_litellm()
    await test_session.commit()

    second = await _service(test_session, only_stays).sync_from_litellm()
    await test_session.commit()
    assert second["models_deactivated"] == 1

    gone = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "vanishes")
    )).scalar_one()
    assert gone.is_active is False

    third = await _service(test_session, both).sync_from_litellm()
    await test_session.commit()
    assert third["models_deactivated"] == 0
    await test_session.refresh(gone)
    assert gone.is_active is True


@pytest.mark.asyncio
async def test_admin_disabled_row_is_not_reactivated_by_the_sync(
    test_session: AsyncSession,
):
    payload = _payload({"admin-off": (0.000001, 0.000002)})
    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "admin-off")
    )).scalar_one()
    row.is_active = False
    row.notes = "disabled by admin: deprecated for our customers"
    await test_session.commit()

    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()
    await test_session.refresh(row)
    assert row.is_active is False, "sync must not override a deliberate admin disable"


@pytest.mark.asyncio
async def test_stale_unit_error_row_is_retired_when_all_listings_are_rejected(
    test_session: AsyncSession,
):
    """The production gap this guards: microsoft/Phi-4-mini-instruct was written
    at $8/$35 by the old code; the new sync rejected its only (wandb) listing,
    which also meant never touching -- and never retiring -- the stale bad row."""
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
    result = await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()

    row = (await test_session.execute(
        select(ModelPricing).where(ModelPricing.model_name == "only-bad")
    )).scalar_one()
    assert row.is_active is False
    assert result["models_rejected"] == 1
    assert result["models_deactivated"] == 1

    # A second identical sync must not flap it back on.
    await _service(test_session, payload).sync_from_litellm()
    await test_session.commit()
    await test_session.refresh(row)
    assert row.is_active is False

    # Upstream ships a sane price -> row returns with the corrected rate.
    sane = {"wandb/only-bad": {"input_cost_per_token": 0.000001,
                               "output_cost_per_token": 0.000002,
                               "litellm_provider": "wandb"}}
    await _service(test_session, sane).sync_from_litellm()
    await test_session.commit()
    await test_session.refresh(row)
    assert row.is_active is True
    assert row.input_price_per_1k == pytest.approx(0.001)
