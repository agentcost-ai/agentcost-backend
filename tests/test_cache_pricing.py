"""
Prompt-cache accounting.

Regression cover for a bug that overstated cost on every cache-heavy workload:
the SDK read cached_tokens off the provider response, but EventCreate did not
declare the field, so Pydantic's default extra="ignore" dropped it and the
server repriced the whole prompt at the full input rate.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.models.db_models import Event, ModelPricing
from app.services.event_service import price_event


CACHED_MODEL = "cache-test-model"


@pytest.fixture
async def priced_model(test_session):
    """A model with a published cache-read rate one tenth of the input rate."""
    row = ModelPricing(
        model_name=CACHED_MODEL,
        input_price_per_1k=10.0,
        output_price_per_1k=30.0,
        cached_input_price_per_1k=1.0,
        cache_write_price_per_1k=12.5,
        provider="test",
    )
    test_session.add(row)
    await test_session.commit()
    return row


@pytest.fixture
async def uncached_model(test_session):
    """A model with no published cache rate at all."""
    row = ModelPricing(
        model_name="no-cache-rate-model",
        input_price_per_1k=10.0,
        output_price_per_1k=30.0,
        provider="test",
    )
    test_session.add(row)
    await test_session.commit()
    return row


def _event(**overrides):
    base = {
        "agent_name": "coding-agent",
        "model": CACHED_MODEL,
        "input_tokens": 1000,
        "output_tokens": 100,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    base.update(overrides)
    return base


class TestPriceEventFormula:
    """The pure function, independent of ingest."""

    PRICING = {"input": 10.0, "output": 30.0, "cached_input": 1.0, "cache_write": 12.5}

    def test_no_cache_prices_everything_at_full_rate(self):
        # 1000/1000*10 + 100/1000*30
        assert price_event(self.PRICING, 1000, 100) == pytest.approx(13.0)

    def test_cached_portion_is_discounted(self):
        # 100 uncached at 10, 900 cached at 1, plus output
        assert price_event(self.PRICING, 1000, 100, cached_tokens=900) == pytest.approx(4.9)

    def test_fully_cached_prompt(self):
        assert price_event(self.PRICING, 1000, 100, cached_tokens=1000) == pytest.approx(4.0)

    def test_cache_writes_are_a_premium_added_on_top(self):
        # Writes are extra tokens, not a subset of input.
        assert price_event(
            self.PRICING, 1000, 100, cache_write_tokens=200
        ) == pytest.approx(13.0 + 2.5)

    def test_missing_cache_rate_falls_back_to_input_rate(self):
        """Never assume a discount the provider has not published."""
        pricing = {"input": 10.0, "output": 30.0}
        assert price_event(pricing, 1000, 100, cached_tokens=900) == pytest.approx(13.0)

    def test_explicit_null_cache_rate_also_falls_back(self):
        """The column is nullable, so None must behave like absent, not like 0."""
        pricing = {"input": 10.0, "output": 30.0, "cached_input": None}
        assert price_event(pricing, 1000, 100, cached_tokens=900) == pytest.approx(13.0)

    def test_cached_cannot_exceed_input(self):
        """A miscounting provider must not be able to produce a credit."""
        assert price_event(self.PRICING, 100, 0, cached_tokens=10_000) == pytest.approx(0.1)

    def test_no_pricing_row_is_free_not_an_error(self):
        assert price_event(None, 1000, 100, cached_tokens=500) == 0.0


class TestCachedTokensSurviveIngest:
    """The regression itself: the field must reach the database."""

    async def test_cached_tokens_are_persisted(self, client, test_project, priced_model, test_session):
        response = await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(cached_tokens=900, cache_write_tokens=50, streaming=True)],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["events_stored"] == 1

        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.cached_tokens == 900
        assert row.cache_write_tokens == 50
        assert row.streaming is True

    async def test_cost_reflects_the_cache_discount(self, client, test_project, priced_model, test_session):
        """Same tokens, different cache hit rate, materially different cost."""
        await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [_event()]},
        )
        await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [_event(cached_tokens=900)]},
        )

        rows = (await test_session.execute(select(Event).order_by(Event.cost.desc()))).scalars().all()
        assert len(rows) == 2
        full, cached = rows
        assert full.cost == pytest.approx(13.0)
        assert cached.cost == pytest.approx(4.9)
        # The bug: both used to be 13.0.
        assert cached.cost < full.cost

    async def test_cached_tokens_clamped_to_input(self, client, test_project, priced_model, test_session):
        response = await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(input_tokens=100, cached_tokens=999_999)],
            },
        )
        assert response.status_code == 200
        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.cached_tokens == 100
        assert row.cost >= 0

    async def test_model_without_cache_rate_is_not_discounted(
        self, client, test_project, uncached_model, test_session
    ):
        await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(model="no-cache-rate-model", cached_tokens=900)],
            },
        )
        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.cost == pytest.approx(13.0)

    async def test_older_clients_omitting_the_fields_still_ingest(
        self, client, test_project, priced_model, test_session
    ):
        response = await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [_event()]},
        )
        assert response.status_code == 200
        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.cached_tokens is None
        assert row.cost == pytest.approx(13.0)
