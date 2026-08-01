"""
Tests for events API endpoints.
"""

import pytest
from httpx import AsyncClient
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_ingest_events_batch(client: AsyncClient, sample_events):
    """Test ingesting a batch of events"""
    response = await client.post("/v1/events/batch", json=sample_events)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ok"
    assert data["events_stored"] == 2


@pytest.mark.asyncio
async def test_ingest_empty_batch(client: AsyncClient, test_project):
    """Test ingesting empty batch returns validation error"""
    response = await client.post("/v1/events/batch", json={
        "project_id": test_project.id,
        "events": [],
    })
    
    # Empty events list triggers validation error (min 1 event required)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ingest_events_wrong_project(client: AsyncClient):
    """Test ingesting events with wrong project ID"""
    response = await client.post("/v1/events/batch", json={
        "project_id": "wrong-project-id",
        "events": [{
            "agent_name": "test",
            "model": "gpt-4",
            "input_tokens": 100,
            "output_tokens": 100,
            "total_tokens": 200,
            "cost": 0.01,
            "latency_ms": 500,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": True,
        }],
    })
    
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_events(client: AsyncClient, sample_events):
    """Test listing events after ingestion"""
    # First ingest
    await client.post("/v1/events/batch", json=sample_events)
    
    # Then list
    response = await client.get("/v1/events")
    
    assert response.status_code == 200
    events = response.json()
    
    assert len(events) == 2


@pytest.mark.asyncio
async def test_list_events_with_agent_filter(client: AsyncClient, sample_events):
    """Test filtering events by agent name"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/events", params={"agent_name": "research-agent"})
    
    assert response.status_code == 200
    events = response.json()
    
    assert len(events) == 1
    assert events[0]["agent_name"] == "research-agent"


@pytest.mark.asyncio
async def test_list_events_with_model_filter(client: AsyncClient, sample_events):
    """Test filtering events by model"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/events", params={"model": "gpt-3.5-turbo"})
    
    assert response.status_code == 200
    events = response.json()
    
    assert len(events) == 1
    assert events[0]["model"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_list_events_pagination(client: AsyncClient, sample_events, test_project):
    """Test events pagination"""
    # Ingest multiple batches
    for _ in range(3):
        await client.post("/v1/events/batch", json=sample_events)
    
    # Get first page
    response = await client.get("/v1/events", params={"limit": 2, "offset": 0})
    assert len(response.json()) == 2
    
    # Get second page
    response = await client.get("/v1/events", params={"limit": 2, "offset": 2})
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_event_count(client: AsyncClient, sample_events):
    """Test event count endpoint"""
    await client.post("/v1/events/batch", json=sample_events)
    
    response = await client.get("/v1/events/count")
    
    assert response.status_code == 200
    assert response.json()["count"] == 2


@pytest.mark.asyncio
async def test_unauthorized_request(client: AsyncClient, sample_events):
    """Test request without API key is rejected"""
    # Create a new client without auth header
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as unauth_client:
        response = await unauth_client.post("/v1/events/batch", json=sample_events)
        assert response.status_code == 401
