"""
Tests for the per-agent insight endpoints behind the Agents pages.
"""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient


def _event(agent: str, **overrides):
    base = {
        "agent_name": agent,
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cost": 0.001,
        "latency_ms": 400,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
        "error": None,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_agent_summaries_empty(client: AsyncClient):
    response = await client.get("/v1/analytics/agents/summary")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_agent_summaries(client: AsyncClient, sample_events):
    await client.post("/v1/events/batch", json=sample_events)

    response = await client.get("/v1/analytics/agents/summary")
    assert response.status_code == 200
    rows = response.json()

    assert [r["agent_name"] for r in rows] == ["research-agent", "writer-agent"]
    top = rows[0]
    assert top["total_cost"] == pytest.approx(0.015, rel=1e-3)
    assert top["share_percent"] == pytest.approx(83.33, abs=0.1)
    assert top["previous_cost"] == 0
    assert top["cost_change_percent"] is None
    assert top["runs"] == 0
    assert top["cost_per_run"] is None
    assert top["models"][0]["model"] == "gpt-4"
    assert len(top["daily"]) == 1
    assert top["daily"][0]["cost"] == pytest.approx(0.015, rel=1e-3)
    # No trace ids anywhere, so the most useful thing to say is "instrument it".
    assert top["signal"]["kind"] == "untraced"


@pytest.mark.asyncio
async def test_agent_summary_failed_spend_signal(client: AsyncClient, test_project):
    events = [_event("flaky", cost=0.01) for _ in range(5)]
    events += [_event("flaky", cost=0.01, success=False, error="context length exceeded") for _ in range(2)]
    await client.post("/v1/events/batch", json={"project_id": test_project.id, "events": events})

    rows = (await client.get("/v1/analytics/agents/summary")).json()
    flaky = rows[0]
    assert flaky["failed_calls"] == 2
    assert flaky["failed_cost"] == pytest.approx(0.02, rel=1e-3)
    assert flaky["signal"]["kind"] == "failed_spend"
    assert flaky["signal"]["amount"] == pytest.approx(0.02, rel=1e-3)


@pytest.mark.asyncio
async def test_agent_summary_classification_signal(client: AsyncClient, test_project):
    for _ in range(3):
        events = [_event("labeler", output_tokens=8, total_tokens=108) for _ in range(40)]
        response = await client.post(
            "/v1/events/batch", json={"project_id": test_project.id, "events": events}
        )
        assert response.status_code == 200, response.text

    rows = (await client.get("/v1/analytics/agents/summary")).json()
    assert rows[0]["signal"]["kind"] == "classification"


@pytest.mark.asyncio
async def test_agent_detail(client: AsyncClient, sample_events):
    await client.post("/v1/events/batch", json=sample_events)

    response = await client.get("/v1/analytics/agents/research-agent")
    assert response.status_code == 200
    detail = response.json()

    assert detail["summary"]["agent_name"] == "research-agent"
    assert detail["summary"]["total_calls"] == 1
    assert detail["by_model"][0]["key"] == "gpt-4"
    assert detail["by_user"] == []
    assert detail["steps"] == []
    assert detail["distribution"] is None
    assert detail["traces"] == []
    assert detail["outcomes"] is None
    assert detail["latency"]["sample_size"] == 1
    assert detail["latency"]["p50"] == 1200


@pytest.mark.asyncio
async def test_agent_detail_unknown(client: AsyncClient):
    response = await client.get("/v1/analytics/agents/nobody")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_agent_filter_on_shared_endpoints(client: AsyncClient, sample_events):
    await client.post("/v1/events/batch", json=sample_events)

    series = (
        await client.get(
            "/v1/analytics/timeseries", params={"agent_name": "writer-agent", "granularity": "day"}
        )
    ).json()
    assert sum(p["cost"] for p in series) == pytest.approx(0.003, rel=1e-3)

    by_model = (
        await client.get("/v1/analytics/by/model", params={"agent_name": "writer-agent"})
    ).json()
    assert [row["key"] for row in by_model] == ["gpt-3.5-turbo"]

    for path in (
        "/v1/analytics/workflows",
        "/v1/analytics/workflows/steps",
        "/v1/analytics/workflows/tools",
        "/v1/analytics/workflows/repeated-work",
        "/v1/analytics/workflows/outcomes",
        "/v1/analytics/traces",
    ):
        response = await client.get(path, params={"agent_name": "writer-agent"})
        assert response.status_code == 200, path
        assert response.json() == []

    response = await client.get("/v1/analytics/workflows/distribution", params={"agent_name": "writer-agent"})
    assert response.status_code == 200
    assert response.json() is None
