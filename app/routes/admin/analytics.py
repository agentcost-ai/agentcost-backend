"""
Admin routes -- cross-tenant platform analytics.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ...database import get_db
from ...models.user_models import User
from ...models.db_models import Project, Event, ModelPricing
from ._deps import require_superuser

router = APIRouter()


@router.get("/analytics/top-models")
async def top_models(
    range: str = Query("30d"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Most used models across all tenants."""
    days = {"7d": 7, "30d": 30, "90d": 90}.get(range, 30)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    rows = (await db.execute(
        select(
            Event.model,
            func.count(Event.id).label("calls"),
            func.coalesce(func.sum(Event.total_tokens), 0).label("tokens"),
            func.coalesce(func.sum(Event.cost), 0).label("cost"),
            func.count(func.distinct(Event.project_id)).label("projects"),
        )
        .where(Event.timestamp >= start)
        .group_by(Event.model)
        .order_by(desc(func.count(Event.id)))
        .limit(limit)
    )).all()

    return [
        {
            "model": r.model,
            "calls": int(r.calls),
            "tokens": int(r.tokens),
            "cost": float(r.cost),
            "project_count": int(r.projects),
        }
        for r in rows
    ]


@router.get("/analytics/top-spenders")
async def top_spenders(
    range: str = Query("30d"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Highest spend tenants (by project)."""
    days = {"7d": 7, "30d": 30, "90d": 90}.get(range, 30)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    rows = (await db.execute(
        select(
            Event.project_id,
            Project.name.label("project_name"),
            User.email.label("owner_email"),
            func.coalesce(func.sum(Event.cost), 0).label("cost"),
            func.count(Event.id).label("calls"),
            func.coalesce(func.sum(Event.total_tokens), 0).label("tokens"),
        )
        .join(Project, Event.project_id == Project.id)
        .outerjoin(User, Project.owner_id == User.id)
        .where(Event.timestamp >= start)
        .group_by(Event.project_id, Project.name, User.email)
        .order_by(desc(func.sum(Event.cost)))
        .limit(limit)
    )).all()

    return [
        {
            "project_id": r.project_id,
            "project_name": r.project_name,
            "owner_email": r.owner_email,
            "cost": float(r.cost),
            "calls": int(r.calls),
            "tokens": int(r.tokens),
        }
        for r in rows
    ]


@router.get("/analytics/provider-growth")
async def provider_growth(
    range: str = Query("30d"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Provider usage growth over time."""
    days = {"7d": 7, "30d": 30, "90d": 90}.get(range, 30)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    rows = (await db.execute(
        select(
            func.date(Event.timestamp).label("date"),
            ModelPricing.provider.label("provider"),
            func.count(Event.id).label("calls"),
            func.coalesce(func.sum(Event.cost), 0).label("cost"),
        )
        .outerjoin(ModelPricing, Event.model == ModelPricing.model_name)
        .where(Event.timestamp >= start)
        .group_by(func.date(Event.timestamp), ModelPricing.provider)
        .order_by(func.date(Event.timestamp))
    )).all()

    return [
        {
            "date": str(r.date),
            "provider": r.provider or "unknown",
            "calls": int(r.calls),
            "cost": float(r.cost),
        }
        for r in rows
    ]


@router.get("/analytics/cost-per-user")
async def avg_cost_per_user(
    range: str = Query("30d"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Average cost per user across the platform."""
    days = {"7d": 7, "30d": 30, "90d": 90}.get(range, 30)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    row = (await db.execute(
        select(
            func.coalesce(func.sum(Event.cost), 0).label("total_cost"),
            func.count(func.distinct(Project.owner_id)).label("unique_users"),
        )
        .join(Project, Event.project_id == Project.id)
        .where(Event.timestamp >= start, Project.owner_id.isnot(None))
    )).one()

    total_cost = float(row.total_cost)
    unique_users = int(row.unique_users) if row.unique_users else 0
    avg = total_cost / unique_users if unique_users > 0 else 0.0

    return {
        "total_cost": total_cost,
        "unique_users": unique_users,
        "avg_cost_per_user": round(avg, 4),
    }
