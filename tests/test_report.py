"""
Tests for the Executive Report endpoint (/v1/analytics/report).
"""

import pytest
from datetime import datetime, timedelta, timezone

from httpx import AsyncClient


@pytest.mark.asyncio
async def test_report_empty(client: AsyncClient):
    """Report renders with no data — zeros, not errors."""
    response = await client.get("/v1/analytics/report")

    assert response.status_code == 200
    data = response.json()

    assert data["overview"]["total_calls"] == 0
    assert data["summary"]["cost"]["current"] == 0
    assert data["latency"]["sample_size"] == 0
    assert data["model_pareto"]["top_count"] == 0
    assert data["range_label"] == "30d"
    assert data["is_custom_range"] is False


@pytest.mark.asyncio
async def test_report_with_data(client: AsyncClient, sample_events):
    """Report aggregates ingested events and exposes deep sections."""
    await client.post("/v1/events/batch", json=sample_events)

    response = await client.get("/v1/analytics/report", params={"range": "30d"})

    assert response.status_code == 200
    data = response.json()

    assert data["overview"]["total_calls"] == 2
    assert data["summary"]["cost"]["current"] == pytest.approx(0.018, rel=0.05)
    # Two models → both appear in breakdown and efficiency
    assert len(data["models"]) == 2
    assert len(data["efficiency"]["by_model"]) == 2
    # Latency percentiles computed from the two events
    assert data["latency"]["sample_size"] == 2
    assert data["latency"]["p50"] > 0
    # Cadence has full 7-day and 24-hour buckets
    assert len(data["cadence"]["by_dow"]) == 7
    assert len(data["cadence"]["by_hour"]) == 24
    assert data["cadence"]["busiest_day"] is not None
    # Run-rate projects forward
    assert data["run_rate"]["projected_monthly_cost"] >= 0
    # Agent cost shares present for each returned agent
    for agent in data["agents"]:
        assert agent["agent_name"] in data["agent_cost_share"]


@pytest.mark.asyncio
async def test_report_custom_range(client: AsyncClient, sample_events):
    """Explicit start/end overrides the preset and flags a custom range."""
    await client.post("/v1/events/batch", json=sample_events)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=3)

    response = await client.get(
        "/v1/analytics/report",
        params={"start": start.isoformat(), "end": end.isoformat()},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["is_custom_range"] is True
    assert data["range_label"] == "Custom range"
    # Previous window is the equal-length window immediately before start.
    prev_start = datetime.fromisoformat(data["previous_period_start"])
    prev_end = datetime.fromisoformat(data["previous_period_end"])
    assert (prev_end - prev_start) == timedelta(days=3)


@pytest.mark.asyncio
async def test_report_mtd(client: AsyncClient, sample_events):
    """Month-to-date resolves a current-month window."""
    await client.post("/v1/events/batch", json=sample_events)

    response = await client.get("/v1/analytics/report", params={"range": "mtd"})

    assert response.status_code == 200
    data = response.json()
    assert data["range_label"] == "Month to date"
    start = datetime.fromisoformat(data["period_start"])
    assert start.day == 1
