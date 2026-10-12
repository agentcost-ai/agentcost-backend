"""
Two things a coding agent's events get wrong unless the server knows better:
one-hour cache writes cost more than the catalogue's cache-write rate, and a
call read from a session log has no latency, which is not the same as zero.
"""

from datetime import datetime, timezone

from sqlalchemy import select

from app.models.db_models import Event, ModelPricing
from app.services.pricing_math import price_event

# claude-haiku-5-5, per 1,000 tokens, as the catalogue carries it.
HAIKU = {"input": 0.0001, "output": 0.0005, "cached_input": 0.00001, "cache_write": 0.000125}

# One real Claude Code call: what it reported, and what it says it cost.
CALL = dict(input_tokens=14052, output_tokens=77, cached_tokens=14050, cache_write_tokens=12091)
CLAUDE_CODE_COST = 0.0025974


async def _seed_haiku(session):
    session.add(
        ModelPricing(
            model_name="claude-haiku-5-5",
            input_price_per_1k=HAIKU["input"],
            output_price_per_1k=HAIKU["output"],
            cached_input_price_per_1k=HAIKU["cached_input"],
            cache_write_price_per_1k=HAIKU["cache_write"],
            provider="anthropic",
            is_active=True,
        )
    )
    await session.commit()


def _event(**overrides):
    base = {
        "agent_name": "claude-code",
        "model": "claude-haiku-5-5",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **CALL,
    }
    base.update(overrides)
    return base


async def _ingest(client, project, *events):
    response = await client.post(
        "/v1/events/batch", json={"project_id": project.id, "events": list(events)}
    )
    assert response.status_code == 200, response.text


def test_one_hour_writes_reproduce_claude_codes_own_cost():
    five_minute = price_event(HAIKU, **CALL)
    one_hour = price_event(HAIKU, **CALL, cache_write_1h_tokens=CALL["cache_write_tokens"])

    assert round(one_hour, 7) == CLAUDE_CODE_COST
    assert five_minute < one_hour * 0.7  # what pricing them as five-minute writes loses


def test_one_hour_share_is_bounded_by_what_was_written():
    everything = price_event(HAIKU, **CALL, cache_write_1h_tokens=CALL["cache_write_tokens"])

    assert price_event(HAIKU, **CALL, cache_write_1h_tokens=10**9) == everything
    assert price_event(HAIKU, **CALL, cache_write_1h_tokens=-5) == price_event(HAIKU, **CALL)


async def test_stated_one_hour_writes_are_priced_as_such(client, test_project, test_session):
    await _seed_haiku(test_session)

    await _ingest(
        client, test_project, _event(metadata={"cache_write_1h_tokens": 12091}, event_id="stated")
    )

    cost = (await test_session.execute(select(Event.cost))).scalar_one()
    assert round(cost, 7) == CLAUDE_CODE_COST


async def test_lifetime_is_read_off_the_clients_cost_when_not_stated(
    client, test_project, test_session
):
    """Claude Code's OpenTelemetry export has one cache-write count and a cost."""
    await _seed_haiku(test_session)

    await _ingest(
        client,
        test_project,
        _event(cost=CLAUDE_CODE_COST, event_id="otel"),
        # The SDK's own estimate uses the five-minute rate; that stays as it was.
        _event(cost=price_event(HAIKU, **CALL), event_id="sdk"),
        _event(event_id="silent"),
    )

    rows = dict((await test_session.execute(select(Event.event_id, Event.cost))).all())
    assert round(rows["otel"], 7) == CLAUDE_CODE_COST
    assert rows["sdk"] == rows["silent"] == price_event(HAIKU, **CALL)


async def test_a_wild_client_cost_cannot_set_the_price(client, test_project, test_session):
    await _seed_haiku(test_session)

    await _ingest(client, test_project, _event(cost=9999.0, event_id="inflated"))

    cost = (await test_session.execute(select(Event.cost))).scalar_one()
    # The most it can do is select the one-hour reading, which is still ours.
    assert round(cost, 7) == CLAUDE_CODE_COST


async def test_unknown_latency_does_not_drag_the_average_down(client, test_project):
    await _ingest(
        client,
        test_project,
        _event(model="unpriced", latency_ms=1200, event_id="measured-1"),
        _event(model="unpriced", latency_ms=800, event_id="measured-2"),
        # Read from a session log: no latency was ever measured.
        _event(model="unpriced", event_id="unknown-1"),
        _event(model="unpriced", event_id="unknown-2"),
    )

    overview = (await client.get("/v1/analytics/overview?range=24h")).json()
    assert overview["total_calls"] == 4
    assert overview["avg_latency_ms"] == 1000.0

    agents = (await client.get("/v1/analytics/agents?range=24h")).json()
    assert agents[0]["avg_latency_ms"] == 1000.0


async def test_all_unknown_latency_reads_as_zero_not_an_error(client, test_project):
    await _ingest(client, test_project, _event(model="unpriced", event_id="only"))

    overview = (await client.get("/v1/analytics/overview?range=24h")).json()
    assert overview["avg_latency_ms"] == 0.0
    assert (await client.get("/v1/analytics/agents?range=24h")).status_code == 200
    assert (await client.get("/v1/analytics/models?range=24h")).status_code == 200
    assert (await client.get("/v1/analytics/report?range=24h")).status_code == 200
    assert (await client.get("/v1/metrics")).status_code == 200
