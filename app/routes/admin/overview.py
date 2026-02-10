"""
Admin routes -- platform overview.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta, timezone

from ...database import get_db
from ...models.user_models import User
from ...models.db_models import Project, Event
from ._deps import require_superuser

router = APIRouter()


@router.get("/overview/stats")
async def get_platform_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Aggregated platform-wide statistics."""
    user_count = (await db.execute(select(func.count(User.id)))).scalar() or 0
    active_user_count = (await db.execute(
        select(func.count(User.id)).where(User.is_active == True)
    )).scalar() or 0

    project_count = (await db.execute(select(func.count(Project.id)))).scalar() or 0
    active_project_count = (await db.execute(
        select(func.count(Project.id)).where(Project.is_active == True)
    )).scalar() or 0

    event_agg = (await db.execute(
        select(
            func.count(Event.id).label("total_events"),
            func.coalesce(func.sum(Event.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(Event.cost), 0).label("total_cost"),
        )
    )).one()

    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    active_sdk = (await db.execute(
        select(func.count(func.distinct(Event.project_id))).where(
            Event.timestamp >= week_ago
        )
    )).scalar() or 0

    return {
        "total_users": user_count,
        "active_users": active_user_count,
        "total_projects": project_count,
        "active_projects": active_project_count,
        "total_events": int(event_agg.total_events),
        "total_tokens": int(event_agg.total_tokens),
        "total_cost": float(event_agg.total_cost),
        "active_sdk_installations": active_sdk,
    }


@router.get("/overview/timeseries")
async def get_platform_timeseries(
    range: str = Query("30d", description="Time range: 7d, 30d, 90d"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Platform-wide daily timeseries: events, cost, tokens."""
    days = {"7d": 7, "30d": 30, "90d": 90}.get(range, 30)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    rows = (await db.execute(
        select(
            func.date(Event.timestamp).label("date"),
            func.count(Event.id).label("events"),
            func.coalesce(func.sum(Event.cost), 0).label("cost"),
            func.coalesce(func.sum(Event.total_tokens), 0).label("tokens"),
        )
        .where(Event.timestamp >= start)
        .group_by(func.date(Event.timestamp))
        .order_by(func.date(Event.timestamp))
    )).all()

    return [
        {
            "date": str(r.date),
            "events": int(r.events),
            "cost": float(r.cost),
            "tokens": int(r.tokens),
        }
        for r in rows
    ]
