"""
Tests for analytics API endpoints.
"""

import pytest
from httpx import AsyncClient
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_analytics_overview_empty(client: AsyncClient):
    """Test analytics overview with no data"""
    response = await client.get("/v1/analytics/overview")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_cost"] == 0
    assert data["total_calls"] == 0
    assert data["total_tokens"] == 0


@pytest.mark.asyncio
async def test_analytics_overview_with_data(client: AsyncClient, sample_events):
    """Test analytics overview after ingesting data"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/analytics/overview")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_calls"] == 2
    assert data["total_tokens"] == 1600  # 300 + 1300
    assert data["total_cost"] == pytest.approx(0.018, rel=0.01)  # 0.015 + 0.003


@pytest.mark.asyncio
async def test_analytics_overview_time_ranges(client: AsyncClient, sample_events):
    """Test analytics overview with different time ranges"""
    await client.post("/v1/events/batch", json=sample_events)
    
    for range_val in ["1h", "24h", "7d", "30d", "90d"]:
        response = await client.get("/v1/analytics/overview", params={"range": range_val})
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_agent_stats(client: AsyncClient, sample_events):
    """Test per-agent statistics"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/analytics/agents")
    
    assert response.status_code == 200
    agents = response.json()
    
    assert len(agents) == 2
    
    # Find research agent
    research = next((a for a in agents if a["agent_name"] == "research-agent"), None)
    assert research is not None
    assert research["total_calls"] == 1
    assert research["total_tokens"] == 300


@pytest.mark.asyncio
async def test_model_stats(client: AsyncClient, sample_events):
    """Test per-model statistics"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/analytics/models")
    
    assert response.status_code == 200
    models = response.json()
    
    assert len(models) == 2
    
    # Find GPT-4
    gpt4 = next((m for m in models if m["model"] == "gpt-4"), None)
    assert gpt4 is not None
    assert gpt4["total_calls"] == 1


@pytest.mark.asyncio
async def test_timeseries(client: AsyncClient, sample_events):
    """Test timeseries data"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/analytics/timeseries")
    
    assert response.status_code == 200
    timeseries = response.json()
    
    # Should have at least one data point
    assert isinstance(timeseries, list)


@pytest.mark.asyncio
async def test_agent_stats_limit(client: AsyncClient, sample_events):
    """Test agent stats respects limit parameter"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/analytics/agents", params={"limit": 1})
    
    assert response.status_code == 200
    agents = response.json()
    
    assert len(agents) == 1


@pytest.mark.asyncio
async def test_model_stats_limit(client: AsyncClient, sample_events):
    """Test model stats respects limit parameter"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/analytics/models", params={"limit": 1})
    
    assert response.status_code == 200
    models = response.json()
    
    assert len(models) == 1


@pytest.mark.asyncio
async def test_overview_counts_every_agent_and_model_and_the_previous_window(client: AsyncClient, test_project):
    """The overview's counts are not the length of a top-10 list, and it carries
    the previous window's cost so a change needs no second query."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)

    def event(agent, model, cost, when):
        return {
            "agent_name": agent, "model": model, "input_tokens": 10, "output_tokens": 1,
            "timestamp": when.isoformat(), "success": True, "cost": cost,
        }

    current = [event(f"agent-{i}", f"unpriced-model-{i % 12}", 1.0, now - timedelta(hours=1)) for i in range(14)]
    earlier = [event("agent-0", "unpriced-model-0", 3.0, now - timedelta(days=10))]
    response = await client.post(
        "/v1/events/batch", json={"project_id": test_project.id, "events": current + earlier}
    )
    assert response.status_code == 200

    overview = (await client.get("/v1/analytics/overview?range=7d")).json()
    assert overview["distinct_agents"] == 14
    assert overview["distinct_models"] == 12
    assert overview["previous_total_cost"] == 3.0
    assert overview["previous_subscription_cost"] == 0.0
