"""
Tests for projects API endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_project(client: AsyncClient, auth_headers):
    """Test creating a new project (requires an authenticated user)"""
    response = await client.post(
        "/v1/projects",
        json={"name": "New Test Project"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "New Test Project"
    assert "id" in data
    assert "api_key" in data
    assert data["api_key"].startswith("sk_")  # Uses sk_ prefix


@pytest.mark.asyncio
async def test_get_project(client: AsyncClient, test_project):
    """Test getting a specific project"""
    response = await client.get(f"/v1/projects/{test_project.id}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["id"] == test_project.id
    assert data["name"] == test_project.name


@pytest.mark.asyncio
async def test_get_nonexistent_project(client: AsyncClient, test_project):
    """Test getting a project that doesn't exist returns forbidden (not accessible)"""
    response = await client.get("/v1/projects/nonexistent-id")
    
    # Returns 403 since project ID doesn't match auth context
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_project_name_required(client: AsyncClient, auth_headers):
    """Test that project name is required"""
    response = await client.post("/v1/projects", json={}, headers=auth_headers)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_create_project_generates_unique_api_key(client: AsyncClient, auth_headers):
    """Test that each project gets a unique API key"""
    response1 = await client.post(
        "/v1/projects", json={"name": "Project 1"}, headers=auth_headers
    )
    response2 = await client.post(
        "/v1/projects", json={"name": "Project 2"}, headers=auth_headers
    )

    assert response1.json()["api_key"] != response2.json()["api_key"]
