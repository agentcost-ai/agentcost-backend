"""
Admin routes -- demo mode analytics.

How many visitors used the no-signup demo, what they explored, and how many
converted to real accounts.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...models.db_models import DemoSession
from ...models.user_models import User
from ._deps import require_superuser

router = APIRouter()


@router.get("/demo/stats")
async def get_demo_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Aggregated demo usage and conversion statistics."""
    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    total = (await db.execute(select(func.count(DemoSession.id)))).scalar() or 0
    last_24h = (await db.execute(
        select(func.count(DemoSession.id)).where(DemoSession.started_at >= day_ago)
    )).scalar() or 0
    last_7d = (await db.execute(
        select(func.count(DemoSession.id)).where(DemoSession.started_at >= week_ago)
    )).scalar() or 0
    last_30d = (await db.execute(
        select(func.count(DemoSession.id)).where(DemoSession.started_at >= month_ago)
    )).scalar() or 0

    signup_clicks = (await db.execute(
        select(func.count(DemoSession.id)).where(DemoSession.signup_clicked == True)
    )).scalar() or 0
    conversions = (await db.execute(
        select(func.count(DemoSession.id)).where(DemoSession.converted == True)
    )).scalar() or 0

    avg_page_views = (await db.execute(
        select(func.avg(DemoSession.page_views))
    )).scalar() or 0

    # Source and page breakdowns (pages is JSON; aggregate in Python — demo
    # session volume is small enough that this stays cheap).
    sessions = (await db.execute(
        select(DemoSession.source, DemoSession.pages).where(
            DemoSession.started_at >= month_ago
        )
    )).all()
    source_counts = Counter((s.source or "direct") for s in sessions)
    page_counts: Counter = Counter()
    for s in sessions:
        for page in s.pages or []:
            page_counts[page] += 1

    return {
        "total_sessions": total,
        "sessions_24h": last_24h,
        "sessions_7d": last_7d,
        "sessions_30d": last_30d,
        "signup_clicks": signup_clicks,
        "conversions": conversions,
        "click_through_rate": round(100 * signup_clicks / total, 1) if total else 0.0,
        "conversion_rate": round(100 * conversions / total, 1) if total else 0.0,
        "avg_page_views": round(float(avg_page_views), 1),
        "top_sources": [
            {"source": k, "sessions": v} for k, v in source_counts.most_common(8)
        ],
        "top_pages": [
            {"page": k, "views": v} for k, v in page_counts.most_common(8)
        ],
    }


@router.get("/demo/timeseries")
async def get_demo_timeseries(
    range: str = Query("30d", description="Time range: 7d, 30d, 90d"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Daily demo sessions and conversions for the admin chart."""
    days = {"7d": 7, "30d": 30, "90d": 90}.get(range, 30)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    rows = (await db.execute(
        select(
            func.date(DemoSession.started_at).label("date"),
            func.count(DemoSession.id).label("sessions"),
            func.sum(
                case((DemoSession.converted == True, 1), else_=0)
            ).label("conversions"),
        )
        .where(DemoSession.started_at >= start)
        .group_by(func.date(DemoSession.started_at))
        .order_by(func.date(DemoSession.started_at))
    )).all()

    return [
        {
            "date": str(r.date),
            "sessions": int(r.sessions),
            "conversions": int(r.conversions or 0),
        }
        for r in rows
    ]
