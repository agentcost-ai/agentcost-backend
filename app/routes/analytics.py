"""
AgentCost Backend - Analytics API Routes

Endpoints for analytics queries.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
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
from ..services.trace_service import TraceService
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


# --- Trace analytics ------------------------------------------------------
# Cost attributed to the shape of a run rather than to the model that served
# it. Every endpoint below reads only events carrying trace structure; runs
# recorded by an SDK predating workflow()/step() are absent by design rather
# than silently blended into these totals.


@router.get("/workflows")
async def get_workflow_stats(
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Cost per workflow, including the average cost of a single run."""
    start_time, end_time = parse_time_range(range)
    return await TraceService(db).get_workflow_stats(
        project.id, start_time, end_time, limit=limit
    )


@router.get("/workflows/steps")
async def get_step_stats(
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d"),
    workflow: Optional[str] = Query(None, description="Restrict to one workflow"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Cost per step. calls_per_run above 1 indicates retries or a loop."""
    start_time, end_time = parse_time_range(range)
    return await TraceService(db).get_step_stats(
        project.id, start_time, end_time, workflow=workflow, limit=limit
    )


@router.get("/workflows/tools")
async def get_tool_stats(
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """LLM spend incurred underneath each named tool."""
    start_time, end_time = parse_time_range(range)
    return await TraceService(db).get_tool_stats(
        project.id, start_time, end_time, limit=limit
    )


@router.get("/workflows/repeated-work")
async def get_repeated_work(
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d"),
    limit: int = Query(25, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Identical calls repeated within a single run, and what they cost."""
    start_time, end_time = parse_time_range(range)
    return await TraceService(db).detect_repeated_work(
        project.id, start_time, end_time, limit=limit
    )


@router.get("/workflows/outcomes")
async def get_outcome_stats(
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Cost per completed outcome. Requires track_costs.outcome() in the run."""
    start_time, end_time = parse_time_range(range)
    return await TraceService(db).get_outcome_stats(
        project.id, start_time, end_time, limit=limit
    )


@router.get("/workflows/distribution")
async def get_run_cost_distribution(
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d"),
    workflow: Optional[str] = Query(None, description="Defaults to the highest-spend workflow"),
    buckets: int = Query(24, ge=6, le=60),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """
    Distribution of cost per run, with percentiles and the tail's share of spend.

    Computed over every run in the window rather than a top-N slice, because a
    distribution drawn from the most expensive runs is not a distribution.
    """
    start_time, end_time = parse_time_range(range)
    service = TraceService(db)

    target = workflow
    if target is None:
        ranked = await service.get_workflow_stats(project.id, start_time, end_time, limit=1)
        if not ranked:
            return None
        target = ranked[0]["workflow"]

    return await service.get_run_cost_distribution(
        project.id, start_time, end_time, workflow=target, buckets=buckets
    )


@router.get("/traces")
async def list_traces(
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query("7d"),
    workflow: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Individual runs, most expensive first."""
    start_time, end_time = parse_time_range(range)
    return await TraceService(db).list_traces(
        project.id, start_time, end_time, workflow=workflow, limit=limit
    )


@router.get("/traces/{trace_id}")
async def get_trace_detail(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Every span of one run, ordered as it executed."""
    detail = await TraceService(db).get_trace_detail(project.id, trace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return detail
