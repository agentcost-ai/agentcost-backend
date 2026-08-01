"""
Tests for health and root endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test root endpoint returns API info"""
    response = await client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "name" in data
    assert "version" in data
    assert "docs" in data
    assert data["docs"] == "/docs"


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Test health endpoint returns status"""
    response = await client.get("/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_timestamp_format(client: AsyncClient):
    """Test health endpoint timestamp is ISO format"""
    response = await client.get("/v1/health")
    data = response.json()
    
    from datetime import datetime
    
    # Should parse without error
    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    assert timestamp is not None
