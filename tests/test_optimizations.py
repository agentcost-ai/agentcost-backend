"""
Tests for the refactored optimization service.
"""

import pytest
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_optimizations_empty(client: AsyncClient, test_project):
    """Test optimizations endpoint with no data."""
    response = await client.get(
        "/v1/optimizations",
        headers={"X-API-Key": test_project.api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_optimization_summary(client: AsyncClient, test_project):
    """Test optimization summary endpoint."""
    response = await client.get(
        "/v1/optimizations/summary",
        headers={"X-API-Key": test_project.api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "total_potential_savings_monthly" in data
    assert "suggestion_count" in data
    assert "effectiveness" in data


@pytest.mark.asyncio
async def test_get_pending_recommendations(client: AsyncClient, test_project):
    """Test pending recommendations endpoint."""
    response = await client.get(
        "/v1/optimizations/recommendations",
        headers={"X-API-Key": test_project.api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_recommendation_effectiveness(client: AsyncClient, test_project):
    """Test recommendation effectiveness endpoint."""
    response = await client.get(
        "/v1/optimizations/recommendations/effectiveness",
        headers={"X-API-Key": test_project.api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "total_recommendations" in data
    assert "implementation_rate" in data
    assert "accuracy_percent" in data


@pytest.mark.asyncio
async def test_get_baselines_empty(client: AsyncClient, test_project):
    """Test get baselines endpoint with no data."""
    response = await client.get(
        "/v1/optimizations/baselines",
        headers={"X-API-Key": test_project.api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_caching_opportunities_empty(client: AsyncClient, test_project):
    """Test caching opportunities endpoint with no data."""
    response = await client.get(
        "/v1/optimizations/caching-opportunities",
        headers={"X-API-Key": test_project.api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "opportunities" in data
    assert "total_potential_monthly_savings" in data
