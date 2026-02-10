"""
Admin routes -- system monitoring and health.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, desc

from ...database import get_db
from ...models.user_models import User, UserSession
from ...models.db_models import Project, Event, DailyAggregate, ModelPricing, Feedback
from ...config import get_settings
from ._deps import require_superuser

router = APIRouter()
settings = get_settings()


@router.get("/system/health")
async def system_health(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """System health: db connectivity, table counts, uptime indicators."""
    tables = {}
    for model_cls, name in [
        (User, "users"),
        (Project, "projects"),
        (Event, "events"),
        (ModelPricing, "model_pricing"),
        (Feedback, "feedback"),
        (UserSession, "user_sessions"),
        (DailyAggregate, "daily_aggregates"),
    ]:
        count = (await db.execute(select(func.count(model_cls.id)))).scalar() or 0
        tables[name] = int(count)

    day_ago = datetime.now(timezone.utc) - timedelta(days=1)
    error_stats = (await db.execute(
        select(
            func.count(Event.id).label("total"),
            func.sum(case((Event.success == False, 1), else_=0)).label("errors"),
        ).where(Event.timestamp >= day_ago)
    )).one()

    total_24h = int(error_stats.total) if error_stats.total else 0
    errors_24h = int(error_stats.errors) if error_stats.errors else 0
    error_rate = (errors_24h / total_24h * 100) if total_24h > 0 else 0.0

    last_event = (await db.execute(
        select(Event.timestamp).order_by(desc(Event.timestamp)).limit(1)
    )).scalar()

    return {
        "status": "operational",
        "database": {
            "connected": True,
            "tables": tables,
        },
        "ingestion": {
            "events_24h": total_24h,
            "errors_24h": errors_24h,
            "error_rate_24h": round(error_rate, 2),
            "last_event_at": last_event.isoformat() if last_event else None,
        },
        "pricing": {
            "total_models": tables.get("model_pricing", 0),
        },
        "version": settings.app_version,
        "environment": settings.environment,
    }


@router.get("/system/ingestion-stats")
async def ingestion_stats(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Ingestion throughput stats over time."""
    hours = {"1h": 1, "24h": 24, "7d": 168}.get(range, 24)
    start = datetime.now(timezone.utc) - timedelta(hours=hours)

    rows = (await db.execute(
        select(
            func.date(Event.timestamp).label("date"),
            func.count(Event.id).label("count"),
            func.sum(case((Event.success == True, 1), else_=0)).label("success"),
            func.sum(case((Event.success == False, 1), else_=0)).label("failed"),
        )
        .where(Event.timestamp >= start)
        .group_by(func.date(Event.timestamp))
        .order_by(func.date(Event.timestamp))
    )).all()

    return [
        {
            "date": str(r.date),
            "total": int(r.count),
            "success": int(r.success) if r.success else 0,
            "failed": int(r.failed) if r.failed else 0,
        }
        for r in rows
    ]
