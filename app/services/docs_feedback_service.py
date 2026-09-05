"""
AgentCost Backend - Docs Feedback Service

Stores anonymous "Was this page helpful?" votes from the public docs and
aggregates them per page for the admin console.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import DocsFeedback
from ..models.schemas import DocsFeedbackPageSummary, DocsFeedbackSummary


async def record_vote(db: AsyncSession, page: str, helpful: bool) -> DocsFeedback:
    vote = DocsFeedback(page=page.rstrip("/") or "/docs", helpful=helpful)
    db.add(vote)
    await db.flush()
    return vote


async def summarize(db: AsyncSession) -> DocsFeedbackSummary:
    helpful_sum = func.sum(case((DocsFeedback.helpful.is_(True), 1), else_=0))
    rows = await db.execute(
        select(
            DocsFeedback.page,
            helpful_sum.label("helpful"),
            func.count().label("total"),
            func.max(DocsFeedback.created_at).label("last_vote_at"),
        )
        .group_by(DocsFeedback.page)
        .order_by(func.count().desc(), DocsFeedback.page)
    )
    pages = []
    total_votes = 0
    for row in rows:
        helpful = int(row.helpful or 0)
        total = int(row.total)
        total_votes += total
        pages.append(
            DocsFeedbackPageSummary(
                page=row.page,
                helpful=helpful,
                not_helpful=total - helpful,
                total=total,
                score=round(100 * helpful / total, 1) if total else None,
                last_vote_at=row.last_vote_at,
            )
        )
    since = datetime.now(timezone.utc) - timedelta(days=30)
    recent = await db.execute(select(func.count()).where(DocsFeedback.created_at >= since))
    return DocsFeedbackSummary(
        pages=pages, total_votes=total_votes, votes_last_30d=int(recent.scalar() or 0)
    )
