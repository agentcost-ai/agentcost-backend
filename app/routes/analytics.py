"""
AgentCost Backend - Analytics API Routes

Endpoints for analytics queries.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal

from ..database import get_db
from ..models.schemas import (
    AnalyticsOverview,
    AnalyticsResponse,
    AgentStats,
    ModelStats,
    TimeSeriesPoint,
    ExecutiveReport,
)
from ..models.db_models import Project
from ..services.analytics_service import AnalyticsService
from ..services.report_service import ReportService
from ..utils.auth import validate_project_access

router = APIRouter(prefix="/v1/analytics", tags=["Analytics"])


def parse_time_range(range_str: str) -> tuple[datetime, datetime]:
    """Parse a time range string into start/end datetimes, snapping the end up
    to the next minute so a dashboard's parallel requests share one window."""
    now = datetime.now(timezone.utc)
    end_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)

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
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d", description="Time range: 1h, 24h, 7d, 30d, 90d"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
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
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d", description="Time range: 1h, 24h, 7d, 30d, 90d"),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
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
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d", description="Time range: 1h, 24h, 7d, 30d, 90d"),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
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
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d", description="Time range: 1h, 24h, 7d, 30d, 90d"),
    granularity: Literal["hour", "day"] = Query("day", description="Granularity: hour, day"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """
    Get time series data.
    
    Returns cost, calls, and tokens over time.
    """
    start_time, end_time = parse_time_range(range)
    
    analytics = AnalyticsService(db)
    return await analytics.get_timeseries(project.id, start_time, end_time, granularity)


def _first_of_month(now: datetime) -> datetime:
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc)


@router.get("/report", response_model=ExecutiveReport)
async def get_executive_report(
    range: Literal["1h", "24h", "7d", "30d", "90d", "mtd"] = Query(
        "30d", description="Preset window. Ignored when start+end are both provided."
    ),
    start: Optional[datetime] = Query(
        None, description="Custom window start (ISO 8601). Requires end."
    ),
    end: Optional[datetime] = Query(
        None, description="Custom window end (ISO 8601). Requires start."
    ),
    top_n: int = Query(10, ge=3, le=50, description="Rows for model/agent breakdown tables."),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """
    Build the Executive Cost & Usage Report.

    One composite payload: executive summary (KPIs + period-over-period deltas)
    plus deep breakdowns — latency percentiles, cost concentration / Pareto,
    token efficiency, error analysis, usage cadence, run-rate projection,
    budget status, and an optimization-savings rollup.

    Supports preset ranges, ``mtd`` (month-to-date), and a custom
    ``start``/``end`` window. Deltas compare against the immediately preceding
    window of equal length.
    """
    now = datetime.now(timezone.utc)

    if start is not None and end is not None:
        # Custom range. Normalize naive datetimes to UTC.
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        start_time, end_time = start, end
        range_label = "Custom range"
        is_custom = True
    elif range == "mtd":
        start_time, end_time = _first_of_month(now), now
        range_label = "Month to date"
        is_custom = False
    else:
        start_time, end_time = parse_time_range(range)
        range_label = range
        is_custom = False

    # Previous equal-length window for period-over-period deltas.
    span = end_time - start_time
    prev_start = start_time - span
    prev_end = start_time

    report = ReportService(db)
    return await report.build_report(
        project=project,
        start=start_time,
        end=end_time,
        prev_start=prev_start,
        prev_end=prev_end,
        top_n=top_n,
        range_label=range_label,
        is_custom_range=is_custom,
    )


@router.get("/full", response_model=AnalyticsResponse)
async def get_full_analytics(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """
    Get complete analytics response.
    
    Includes overview, agent stats, model stats, and time series.
    """
    analytics = AnalyticsService(db)
    return await analytics.get_full_analytics(project.id, days)
