"""Tests for run outcomes and cost per completed outcome."""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from app.models.db_models import TraceOutcome


def _event(**overrides):
    base = {
        "agent_name": "worker",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.01,
        "latency_ms": 200,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    base.update(overrides)
    return base


def _batch(project_id, events, outcomes=None):
    payload = {"project_id": project_id, "events": events}
    if outcomes is not None:
        payload["outcomes"] = outcomes
    return payload


@pytest.mark.asyncio
async def test_batch_without_outcomes_still_ingests(client: AsyncClient, test_project):
    """Older SDKs send no outcomes field at all."""
    response = await client.post(
        "/v1/events/batch", json=_batch(test_project.id, [_event(trace_id="t1")])
    )
    assert response.status_code == 200
    assert response.json()["events_stored"] == 1


@pytest.mark.asyncio
async def test_outcome_is_persisted(client: AsyncClient, test_session, test_project):
    await client.post(
        "/v1/events/batch",
        json=_batch(
            test_project.id,
            [_event(trace_id="t1", workflow="w", step_name="s")],
            [{"trace_id": "t1", "workflow": "w", "success": True, "label": "resolved"}],
        ),
    )

    rows = (await test_session.execute(TraceOutcome.__table__.select())).fetchall()
    assert len(rows) == 1
    assert rows[0].trace_id == "t1"
    assert rows[0].success is True
    assert rows[0].label == "resolved"


@pytest.mark.asyncio
async def test_resent_outcome_updates_rather_than_duplicates(
    client: AsyncClient, test_session, test_project
):
    """A run may report success optimistically and fail later."""
    for success in (True, False):
        await client.post(
            "/v1/events/batch",
            json=_batch(
                test_project.id,
                [_event(trace_id="t1", workflow="w")],
                [{"trace_id": "t1", "workflow": "w", "success": success}],
            ),
        )

    rows = (await test_session.execute(TraceOutcome.__table__.select())).fetchall()
    assert len(rows) == 1
    assert rows[0].success is False


@pytest.mark.asyncio
async def test_duplicate_outcomes_in_one_batch_collapse(
    client: AsyncClient, test_session, test_project
):
    """Two records for one trace in a single batch must not trip the constraint."""
    response = await client.post(
        "/v1/events/batch",
        json=_batch(
            test_project.id,
            [_event(trace_id="t1", workflow="w")],
            [
                {"trace_id": "t1", "workflow": "w", "success": True},
                {"trace_id": "t1", "workflow": "w", "success": False},
            ],
        ),
    )
    assert response.status_code == 200

    rows = (await test_session.execute(TraceOutcome.__table__.select())).fetchall()
    assert len(rows) == 1
    assert rows[0].success is False


@pytest.mark.asyncio
async def test_cost_per_success_charges_failures_to_the_successes(
    client: AsyncClient, test_project
):
    """
    Two runs at $0.10 each, one succeeded. A result cost $0.20, not $0.10 --
    the failed run was paid for too.
    """
    events = [
        _event(trace_id="ok", workflow="w", step_name="s", cost=0.10),
        _event(trace_id="bad", workflow="w", step_name="s", cost=0.10),
    ]
    outcomes = [
        {"trace_id": "ok", "workflow": "w", "success": True},
        {"trace_id": "bad", "workflow": "w", "success": False},
    ]
    await client.post("/v1/events/batch", json=_batch(test_project.id, events, outcomes))

    rows = (await client.get("/v1/analytics/workflows/outcomes")).json()
    assert len(rows) == 1

    row = rows[0]
    assert row["succeeded"] == 1
    assert row["failed"] == 1
    assert row["cost_per_success"] == pytest.approx(0.20, rel=1e-3)
    assert row["success_rate"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_runs_without_an_outcome_are_unknown_not_failed(
    client: AsyncClient, test_project
):
    """A partly-instrumented project must not read as a failing one."""
    events = [
        _event(trace_id="ok", workflow="w", step_name="s", cost=0.10),
        _event(trace_id="silent", workflow="w", step_name="s", cost=0.10),
    ]
    outcomes = [{"trace_id": "ok", "workflow": "w", "success": True}]
    await client.post("/v1/events/batch", json=_batch(test_project.id, events, outcomes))

    row = (await client.get("/v1/analytics/workflows/outcomes")).json()[0]
    assert row["succeeded"] == 1
    assert row["failed"] == 0
    assert row["unknown"] == 1
    assert row["success_rate"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_cost_per_success_is_null_without_a_success(
    client: AsyncClient, test_project
):
    await client.post(
        "/v1/events/batch",
        json=_batch(
            test_project.id,
            [_event(trace_id="bad", workflow="w", step_name="s", cost=0.10)],
            [{"trace_id": "bad", "workflow": "w", "success": False}],
        ),
    )

    row = (await client.get("/v1/analytics/workflows/outcomes")).json()[0]
    assert row["cost_per_success"] is None
    assert row["cost_on_failure"] == pytest.approx(0.10, rel=1e-3)


@pytest.mark.asyncio
async def test_outcomes_do_not_leak_across_projects(
    client: AsyncClient, test_session, test_project
):
    """An outcome row for another project must not join onto these traces."""
    import uuid

    from app.models.db_models import Project
    from app.utils.auth import hash_api_key

    other = Project(
        id=str(uuid.uuid4()),
        name="other",
        api_key=hash_api_key("sk_other"),
        created_at=datetime.now(timezone.utc),
    )
    test_session.add(other)
    await test_session.flush()
    test_session.add(
        TraceOutcome(
            project_id=other.id,
            trace_id="shared",
            workflow="w",
            success=False,
            recorded_at=datetime.now(timezone.utc),
        )
    )
    await test_session.commit()

    await client.post(
        "/v1/events/batch",
        json=_batch(
            test_project.id,
            [_event(trace_id="shared", workflow="w", step_name="s", cost=0.10)],
            [{"trace_id": "shared", "workflow": "w", "success": True}],
        ),
    )

    row = (await client.get("/v1/analytics/workflows/outcomes")).json()[0]
    assert row["succeeded"] == 1
    assert row["failed"] == 0
