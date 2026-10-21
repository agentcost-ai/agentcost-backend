"""
Subscription usage versus metered spend.

A coding agent on a flat plan reports real token counts but owes nothing per
token. Its cost is what the usage is worth at API rates: worth showing, wrong
to count against a budget.
"""

from datetime import datetime, timezone

from sqlalchemy import select

from app.models.db_models import Event


def _event(**overrides):
    base = {
        "agent_name": "claude-code",
        "model": "unpriced-model",  # not in the test catalogue: client cost is kept
        "input_tokens": 1000,
        "output_tokens": 100,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    base.update(overrides)
    return base


async def _ingest(client, project, *events):
    return await client.post(
        "/v1/events/batch", json={"project_id": project.id, "events": list(events)}
    )


async def test_billing_is_promoted_only_in_a_known_form(client, test_project, test_session):
    await _ingest(
        client,
        test_project,
        _event(metadata={"billing": "subscription"}),
        _event(metadata={"billing": "api"}),
        _event(metadata={"billing": "unknown"}),
        _event(metadata={"billing": {"nested": True}}),
        _event(),
    )

    rows = (await test_session.execute(select(Event.billing))).scalars().all()
    assert sorted(rows, key=str) == [None, None, None, "api", "subscription"]


async def test_overview_separates_subscription_value_from_spend(client, test_project):
    await _ingest(
        client,
        test_project,
        _event(metadata={"billing": "subscription"}, cost=8.0),
        _event(metadata={"billing": "api"}, cost=1.5),
        _event(cost=0.5),
    )

    overview = (await client.get("/v1/analytics/overview?range=24h")).json()
    assert overview["total_cost"] == 10.0
    assert overview["subscription_cost"] == 8.0

    groups = (await client.get("/v1/analytics/by/billing?range=24h")).json()
    assert {group["key"]: group["total_cost"] for group in groups} == {
        "subscription": 8.0,
        "api": 1.5,
    }


async def test_subscription_usage_never_touches_the_budget(client, test_project, test_session):
    test_project.monthly_budget_usd = 1.0
    test_project.budget_enforcement_mode = "hard_cap"
    test_project.budget_alert_thresholds = [100.0]
    await test_session.commit()

    plan = {"billing": "subscription"}
    first = await _ingest(client, test_project, _event(metadata=plan, cost=50.0))
    second = await _ingest(client, test_project, _event(metadata=plan, cost=50.0))
    assert first.status_code == 200 and second.status_code == 200

    state = (await client.get(f"/v1/projects/{test_project.id}/budget-state")).json()
    assert state["spend_mtd"] == 0.0
    assert state["exhausted"] is False

    # Metered spend in the same project is still capped.
    over = await _ingest(client, test_project, _event(metadata={"billing": "api"}, cost=5.0))
    assert over.status_code == 429


async def test_mixed_batch_is_judged_on_its_metered_part(client, test_project, test_session):
    test_project.monthly_budget_usd = 1.0
    test_project.budget_enforcement_mode = "hard_cap"
    test_project.budget_alert_thresholds = [100.0]
    await test_session.commit()

    response = await _ingest(
        client,
        test_project,
        _event(metadata={"billing": "subscription"}, cost=50.0),
        _event(metadata={"billing": "api"}, cost=0.25),
    )

    assert response.status_code == 200
    assert response.json()["events_stored"] == 2


async def test_timeseries_carries_the_subscription_share_of_each_bucket(client, test_project):
    await _ingest(
        client,
        test_project,
        _event(metadata={"billing": "subscription"}, cost=8.0),
        _event(cost=0.5),
    )

    series = (await client.get("/v1/analytics/timeseries?range=24h&granularity=hour")).json()
    assert round(sum(p["cost"] for p in series), 6) == 8.5
    assert round(sum(p["subscription_cost"] for p in series), 6) == 8.0


async def test_savings_percentage_is_measured_against_metered_spend(client, test_project):
    await _ingest(client, test_project, _event(metadata={"billing": "subscription"}, cost=90.0))

    summary = (await client.get("/v1/optimizations/summary")).json()
    assert summary["current_monthly_spend"] == 0.0
    assert summary["window_days"] == 30
