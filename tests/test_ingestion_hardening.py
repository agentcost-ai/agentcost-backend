"""
Regression tests for the event-ingestion path.

Each test here pins down a failure mode that made ingestion lose data:
all-or-nothing batch validation, budget enforcement reading the wrong cost,
alerting rolling back ingested events, FX calls in the hot path, and
non-deterministic pricing matches.
"""

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import event as sa_event, select

from app.models.db_models import (
    BudgetThresholdAlert,
    Event,
    InputPatternCache,
    ModelPricing,
)
from app.models.schemas import EventBatchRequest
from app.services.budget_service import BudgetService
from app.services.currency_service import CurrencyService
from app.services.pricing_service import PricingService


@pytest.fixture(autouse=True)
def isolate_currency_service(monkeypatch):
    """No FX network calls and no cache/cooldown bleed between tests."""

    async def _no_network(cls, target):
        return 1.0

    def _reset():
        CurrencyService._cache.clear()
        CurrencyService._refresh_cooldown.clear()
        CurrencyService._refresh_tasks.clear()

    _reset()
    monkeypatch.setattr(CurrencyService, "_refresh", classmethod(_no_network))
    yield
    _reset()


def _event(**overrides) -> dict:
    """A minimal valid event payload; override individual fields per test."""
    payload = {
        "agent_name": "a",
        "model": "gpt-4",
        "input_tokens": 100,
        "output_tokens": 100,
        "total_tokens": 200,
        "cost": 0.01,
        "latency_ms": 100,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    payload.update(overrides)
    return payload


async def _seed_pricing(session, model_name: str, input_price: float, output_price: float):
    session.add(
        ModelPricing(
            model_name=model_name,
            input_price_per_1k=input_price,
            output_price_per_1k=output_price,
            provider="test",
            is_active=True,
        )
    )
    await session.commit()


# ───────────────── optional fields / partial acceptance ─────────────────


async def test_batch_accepts_events_without_derived_fields(client, test_project):
    """total_tokens / cost / latency_ms are server-derived, not client contracts."""
    response = await client.post(
        "/v1/events/batch",
        json={
            "project_id": test_project.id,
            "events": [
                {
                    "agent_name": "a",
                    "model": "gpt-4",
                    "input_tokens": 120,
                    "output_tokens": 80,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["events_stored"] == 1

    listed = (await client.get("/v1/events")).json()
    assert listed[0]["total_tokens"] == 200  # derived from input + output
    assert listed[0]["cost"] == 0.0
    assert listed[0]["latency_ms"] == 0


async def test_one_malformed_event_does_not_reject_the_batch(client, test_project):
    """The old behaviour 422'd all 3 events; the SDK then retried forever."""
    response = await client.post(
        "/v1/events/batch",
        json={
            "project_id": test_project.id,
            "events": [
                _event(agent_name="good-1"),
                _event(agent_name="bad", model=None),
                _event(agent_name="good-2"),
            ],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"  # the SDK's only success signal
    assert body["events_stored"] == 2
    assert body["events_received"] == 3
    assert body["events_rejected"] == 1
    assert body["rejected"][0]["index"] == 1

    stored = (await client.get("/v1/events")).json()
    assert {e["agent_name"] for e in stored} == {"good-1", "good-2"}


async def test_batch_of_only_malformed_events_is_not_retryable(client, test_project):
    """A payload that can never be accepted must not come back 4xx/5xx."""
    response = await client.post(
        "/v1/events/batch",
        json={
            "project_id": test_project.id,
            "events": [_event(input_tokens=-5), _event(timestamp="not-a-date")],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["events_stored"] == 0
    assert body["events_rejected"] == 2


async def test_missing_events_key_still_rejected(client, test_project):
    response = await client.post(
        "/v1/events/batch", json={"project_id": test_project.id}
    )
    assert response.status_code == 422


async def test_oversized_batch_is_measured_by_what_was_sent(client, test_project):
    """Dropping invalid rows must not sneak an over-limit batch past the cap."""
    events = [_event() for _ in range(100)] + [_event(model=None)]
    response = await client.post(
        "/v1/events/batch",
        json={"project_id": test_project.id, "events": events},
    )
    assert response.status_code == 422
    assert "101" in response.json()["detail"]


def test_batch_request_reports_received_count():
    request = EventBatchRequest.model_validate(
        {"project_id": "p", "events": [_event(), _event(model=None)]}
    )
    assert request.received_count == 2
    assert len(request.events) == 1
    assert len(request.rejected) == 1


# ───────────────── budget enforcement uses the stored cost ─────────────────


async def test_hard_cap_blocks_on_server_cost_when_client_reports_zero(
    client, test_session, test_project
):
    """A client reporting cost 0 used to walk straight through a hard cap."""
    await _seed_pricing(test_session, "priced-model", input_price=10.0, output_price=10.0)

    test_project.monthly_budget_usd = 1.0
    test_project.budget_enforcement_mode = "hard_cap"
    test_project.budget_alert_thresholds = [100.0]
    await test_session.commit()

    # 1000 + 1000 tokens at $10/1k = $20 server-side, but the client says $0.
    response = await client.post(
        "/v1/events/batch",
        json={
            "project_id": test_project.id,
            "events": [
                _event(model="priced-model", input_tokens=1000, output_tokens=1000, cost=0.0)
            ],
        },
    )

    assert response.status_code == 429, response.text
    stored = (await test_session.execute(select(Event))).scalars().all()
    assert stored == []


async def test_hard_cap_does_not_block_on_inflated_client_cost(
    client, test_session, test_project
):
    """An inflated client cost must not 429 a project that is under budget."""
    await _seed_pricing(test_session, "cheap-model", input_price=0.0001, output_price=0.0001)

    test_project.monthly_budget_usd = 1.0
    test_project.budget_enforcement_mode = "hard_cap"
    test_project.budget_alert_thresholds = [100.0]
    await test_session.commit()

    response = await client.post(
        "/v1/events/batch",
        json={
            "project_id": test_project.id,
            "events": [
                _event(model="cheap-model", input_tokens=100, output_tokens=100, cost=99999.0)
            ],
        },
    )

    assert response.status_code == 200, response.text
    stored = (await test_session.execute(select(Event))).scalars().all()
    assert len(stored) == 1
    assert stored[0].cost < 0.01  # server price, not the client's claim


# ───────────────── alerting must never destroy ingested events ─────────────────


async def test_threshold_insert_race_does_not_poison_the_transaction(
    test_session, test_project, monkeypatch
):
    """Simulate the losing side of the check-then-insert race."""
    service = BudgetService(test_session)
    await service.record_threshold_crossings(
        project_id=test_project.id,
        period_key="2026-03",
        crossed_thresholds=[80.0],
        spent_amount=80.0,
        budget_amount=100.0,
        utilization_percent=80.0,
        dispatch_notifications=False,
    )

    # Pretend the dedup SELECT ran before the other writer committed.
    async def _no_existing(self, project_id, period_key):
        return set()

    monkeypatch.setattr(BudgetService, "_existing_thresholds", _no_existing)

    inserted = await service.record_threshold_crossings(
        project_id=test_project.id,
        period_key="2026-03",
        crossed_thresholds=[80.0],
        spent_amount=80.0,
        budget_amount=100.0,
        utilization_percent=80.0,
        dispatch_notifications=False,
    )
    assert inserted == []  # the other writer owns the alert

    # The session must still be usable — this is the bit that used to 500 the
    # ingest and roll back the events flushed moments earlier.
    test_session.add(
        Event(
            id=str(uuid.uuid4()),
            project_id=test_project.id,
            agent_name="after-race",
            model="gpt-4",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cost=0.01,
            latency_ms=1,
            timestamp=datetime.now(timezone.utc),
            success=True,
        )
    )
    await test_session.commit()

    rows = (
        await test_session.execute(
            select(BudgetThresholdAlert).where(
                BudgetThresholdAlert.project_id == test_project.id
            )
        )
    ).scalars().all()
    assert len(rows) == 1
    events = (await test_session.execute(select(Event))).scalars().all()
    assert [e.agent_name for e in events] == ["after-race"]


async def test_ingest_survives_duplicate_threshold_alert(
    client, test_session, test_project, monkeypatch
):
    """End to end: a racing alert insert must not cost the batch."""
    test_project.monthly_budget_usd = 100.0
    test_project.budget_enforcement_mode = "warn"
    test_project.budget_alert_thresholds = [50.0]
    test_session.add(
        BudgetThresholdAlert(
            id=str(uuid.uuid4()),
            project_id=test_project.id,
            period_key=BudgetService._period_key(datetime.now(timezone.utc)),
            threshold_percent=50.0,
            spent_amount=50.0,
            budget_amount=100.0,
            utilization_percent=50.0,
        )
    )
    await test_session.commit()

    async def _no_existing(self, project_id, period_key):
        return set()

    monkeypatch.setattr(BudgetService, "_existing_thresholds", _no_existing)

    response = await client.post(
        "/v1/events/batch",
        json={
            "project_id": test_project.id,
            "events": [_event(model="unpriced-model", cost=60.0)],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["events_stored"] == 1
    stored = (await test_session.execute(select(Event))).scalars().all()
    assert len(stored) == 1


# ───────────────── no FX / no spend query in the hot path ─────────────────


async def test_ingest_skips_budget_evaluation_without_a_budget(
    client, test_project, monkeypatch
):
    calls: list[str] = []

    async def _boom(self, project, additional_cost=0.0, **kwargs):
        calls.append(project.id)
        raise AssertionError("evaluate() must not run for a budget-less project")

    monkeypatch.setattr(BudgetService, "evaluate", _boom)

    response = await client.post(
        "/v1/events/batch",
        json={"project_id": test_project.id, "events": [_event()]},
    )

    assert response.status_code == 200, response.text
    assert calls == []


async def test_hot_path_evaluation_never_awaits_the_fx_provider(
    test_session, test_project, monkeypatch
):
    async def _network(cls, currency):
        raise AssertionError("hot path must not call the FX provider")

    monkeypatch.setattr(CurrencyService, "usd_to", classmethod(_network))

    test_project.monthly_budget_usd = 100.0
    test_project.budget_currency = "INR"
    test_project.budget_enforcement_mode = "warn"
    await test_session.commit()

    result = await BudgetService(test_session).evaluate(test_project, hot_path=True)
    assert result["currency"] == "INR"
    assert result["fx_rate"] > 0  # fallback / cached rate, not a network call


def test_cached_usd_to_never_blocks_and_prefers_a_stale_rate():
    CurrencyService._cache.pop("INR", None)
    assert CurrencyService.cached_usd_to("USD") == 1.0
    # No event loop here, so the background refresh is a no-op and the caller
    # still gets a usable rate instead of blocking.
    assert CurrencyService.cached_usd_to("INR") > 1.0

    # Once a rate has been seen, even a long-stale one beats the fallback.
    CurrencyService._cache["INR"] = (77.0, 0.0)  # fetched at epoch = long stale
    assert CurrencyService.cached_usd_to("INR") == 77.0


async def test_usd_to_serves_a_stale_rate_and_refreshes_in_the_background(monkeypatch):
    """Every caller used to queue on one 5s request behind a process-wide lock."""
    CurrencyService._cache["INR"] = (77.0, 0.0)  # long stale
    refreshed: list[str] = []

    async def _fake_refresh(cls, target):
        refreshed.append(target)
        return 88.0

    monkeypatch.setattr(CurrencyService, "_refresh", classmethod(_fake_refresh))

    assert await CurrencyService.usd_to("INR") == 77.0  # served immediately
    assert refreshed == []  # nothing awaited on the caller's path
    await asyncio.sleep(0)  # let the scheduled task run
    assert refreshed == ["INR"]


# ───────────────── deterministic pricing match + honest cost_source ─────────────────


async def test_exact_match_wins_and_is_labelled_exact(test_session):
    await _seed_pricing(test_session, "gpt-4o-mini", 0.00015, 0.0006)
    await _seed_pricing(test_session, "gpt-4", 0.03, 0.06)

    pricing = await PricingService(test_session).get_model_pricing("gpt-4o-mini")
    assert pricing["match"] == "exact"
    assert pricing["input"] == 0.00015


async def test_fuzzy_match_prefers_the_most_specific_family(test_session):
    """'gpt-4' must never win a 'gpt-4o-mini-…' lookup — that is a 200x error."""
    await _seed_pricing(test_session, "gpt-4", 0.03, 0.06)
    await _seed_pricing(test_session, "gpt-4o-mini", 0.00015, 0.0006)

    pricing = await PricingService(test_session).get_model_pricing(
        "gpt-4o-mini-2024-07-18"
    )
    assert pricing["matched_model"] == "gpt-4o-mini"
    assert pricing["match"] == "fuzzy"


async def test_fuzzy_superset_match_is_deterministic(test_session):
    """Several dated variants match; the same one must win every time."""
    await _seed_pricing(test_session, "claude-3-5-haiku-20241022", 0.001, 0.005)
    await _seed_pricing(test_session, "claude-3-5-haiku-latest-experimental", 9.0, 9.0)

    service = PricingService(test_session)
    first = await service.get_model_pricing("claude-3-5-haiku")
    second = await service.get_model_pricing("claude-3-5-haiku")

    assert first["matched_model"] == second["matched_model"] == "claude-3-5-haiku-20241022"
    assert first["match"] == "fuzzy"


async def test_unknown_model_has_no_pricing(test_session):
    await _seed_pricing(test_session, "gpt-4", 0.03, 0.06)
    assert await PricingService(test_session).get_model_pricing("llama-9000") is None


async def test_like_metacharacters_do_not_match_a_different_model(test_session):
    """'_' is a LIKE wildcard; unescaped it silently prices the wrong model."""
    await _seed_pricing(test_session, "gpt-4x", 0.03, 0.06)
    assert await PricingService(test_session).get_model_pricing("gpt_4x") is None


async def test_cost_source_distinguishes_exact_fuzzy_and_client(
    client, test_session, test_project
):
    await _seed_pricing(test_session, "gpt-4o-mini", 0.00015, 0.0006)

    response = await client.post(
        "/v1/events/batch",
        json={
            "project_id": test_project.id,
            "events": [
                _event(agent_name="exact", model="gpt-4o-mini"),
                _event(agent_name="fuzzy", model="gpt-4o-mini-2024-07-18"),
                _event(agent_name="client", model="totally-unknown-model", cost=0.5),
            ],
        },
    )
    assert response.status_code == 200, response.text

    rows = (await test_session.execute(select(Event))).scalars().all()
    sources = {row.agent_name: row.cost_source for row in rows}
    assert sources == {
        "exact": "database-exact",
        "fuzzy": "database-fuzzy",
        "client": "client-sdk",
    }


# ───────────────── API key must not answer for another project ─────────────────


async def test_api_key_serves_its_own_project_without_a_project_id(client):
    """The SDK sends no project_id; the key is the only identifier it has."""
    assert (await client.get("/v1/events")).status_code == 200


async def test_api_key_rejects_a_mismatched_project_id(client):
    response = await client.get("/v1/events", params={"project_id": str(uuid.uuid4())})
    assert response.status_code == 403
    assert "does not match" in response.json()["detail"]


async def test_api_key_rejects_an_empty_project_id(client):
    """``?project_id=`` is a request for a project, not an omission."""
    response = await client.get("/v1/events?project_id=")
    assert response.status_code == 403


async def test_api_key_accepts_its_own_project_id(client, test_project):
    response = await client.get("/v1/events", params={"project_id": test_project.id})
    assert response.status_code == 200


# ───────────────── pattern recording is genuinely batched ─────────────────


async def test_patterns_are_aggregated_per_batch(client, test_session, test_project):
    events = [_event(input_hash="h1") for _ in range(5)]
    events += [_event(input_hash="h2") for _ in range(3)]

    response = await client.post(
        "/v1/events/batch", json={"project_id": test_project.id, "events": events}
    )
    assert response.status_code == 200, response.text

    rows = (
        await test_session.execute(select(InputPatternCache))
    ).scalars().all()
    counts = {row.input_hash: row.occurrence_count for row in rows}
    assert counts == {"h1": 5, "h2": 3}

    # A second batch accumulates onto the same rows rather than duplicating.
    await client.post(
        "/v1/events/batch",
        json={"project_id": test_project.id, "events": [_event(input_hash="h1")]},
    )
    test_session.expire_all()  # the request wrote through a different session
    rows = (
        await test_session.execute(select(InputPatternCache))
    ).scalars().all()
    assert len(rows) == 2
    assert {r.input_hash: r.occurrence_count for r in rows} == {"h1": 6, "h2": 3}
    h1 = next(r for r in rows if r.input_hash == "h1")
    assert h1.avg_cost_per_occurrence == pytest.approx(
        h1.total_cost_for_pattern / h1.occurrence_count
    )


async def test_pattern_recording_does_not_scale_with_batch_size(
    client, test_engine, test_project
):
    """The per-event SELECT+flush loop was ~2 round trips per event."""
    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        if "input_pattern_cache" in statement.lower():
            statements.append(statement)

    sa_event.listen(test_engine.sync_engine, "before_cursor_execute", _record)
    try:
        response = await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(input_hash=f"h{i % 4}") for i in range(20)],
            },
        )
    finally:
        sa_event.remove(test_engine.sync_engine, "before_cursor_execute", _record)

    assert response.status_code == 200, response.text
    # One SELECT for the whole batch, then writes bounded by the number of
    # *distinct* patterns (4) — not by the 20 events that referenced them.
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1, statements
    assert len(statements) <= 5, statements
