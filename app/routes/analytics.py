"""
AgentCost Backend - Analytics API Routes

Endpoints for analytics queries.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..database import get_db
from ..models.schemas import (
    AnalyticsOverview,
    AnalyticsResponse,
    AgentStats,
    ModelStats,
    TimeSeriesPoint,
)
from ..models.db_models import Project
from ..services.analytics_service import AnalyticsService
from ..utils.auth import validate_api_key

router = APIRouter(prefix="/v1/analytics", tags=["Analytics"])


def parse_time_range(range_str: str) -> tuple[datetime, datetime]:
    """Parse time range string into start/end datetimes"""
    end_time = datetime.now(timezone.utc)
    
    if range_str == "1h":
        start_time = end_time - timedelta(hours=1)
    elif range_str == "24h":
        start_time = end_time - timedelta(hours=24)
    elif range_str == "7d":
        start_time = end_time - timedelta(days=7)
    elif range_str == "30d":
        start_time = end_time - timedelta(days=30)
    elif range_str == "90d":
        start_time = end_time - timedelta(days=90)
    else:
        # Default to 7 days
        start_time = end_time - timedelta(days=7)
    
    return start_time, end_time


@router.get("/overview", response_model=AnalyticsOverview)
async def get_overview(
    range: str = Query("7d", description="Time range: 1h, 24h, 7d, 30d, 90d"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get overview metrics for the project.
    
    Returns total cost, calls, tokens, and averages.
    """
    start_time, end_time = parse_time_range(range)
    
    analytics = AnalyticsService(db)
    return await analytics.get_overview(project.id, start_time, end_time)


@router.get("/agents", response_model=list[AgentStats])
async def get_agent_stats(
    range: str = Query("7d", description="Time range: 1h, 24h, 7d, 30d, 90d"),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get per-agent statistics.
    
    Returns cost, calls, and performance metrics per agent.
    """
    start_time, end_time = parse_time_range(range)
    
    analytics = AnalyticsService(db)
    return await analytics.get_agent_stats(project.id, start_time, end_time, limit)


@router.get("/models", response_model=list[ModelStats])
async def get_model_stats(
    range: str = Query("7d", description="Time range: 1h, 24h, 7d, 30d, 90d"),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get per-model statistics.
    
    Returns cost, calls, and performance metrics per model.
    """
    start_time, end_time = parse_time_range(range)
    
    analytics = AnalyticsService(db)
    return await analytics.get_model_stats(project.id, start_time, end_time, limit)


@router.get("/timeseries", response_model=list[TimeSeriesPoint])
async def get_timeseries(
    range: str = Query("7d", description="Time range: 1h, 24h, 7d, 30d, 90d"),
    granularity: str = Query("day", description="Granularity: hour, day"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get time series data.
    
    Returns cost, calls, and tokens over time.
    """
    start_time, end_time = parse_time_range(range)
    
    analytics = AnalyticsService(db)
    return await analytics.get_timeseries(project.id, start_time, end_time, granularity)


@router.get("/full", response_model=AnalyticsResponse)
async def get_full_analytics(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get complete analytics response.
    
    Includes overview, agent stats, model stats, and time series.
    """
    analytics = AnalyticsService(db)
    return await analytics.get_full_analytics(project.id, days)
