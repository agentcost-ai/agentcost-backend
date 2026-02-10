"""
Admin routes -- error and incident logs.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ...database import get_db
from ...models.user_models import User
from ...models.db_models import Project, Event, Feedback
from ._deps import require_superuser

router = APIRouter()


@router.get("/incidents/events")
async def failed_events(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    project_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Failed event ingestions (success=False)."""
    query = select(Event, Project.name.label("project_name")).join(
        Project, Event.project_id == Project.id
    ).where(Event.success == False)

    if project_id:
        query = query.where(Event.project_id == project_id)

    count_q = select(func.count(Event.id)).where(Event.success == False)
    if project_id:
        count_q = count_q.where(Event.project_id == project_id)
    total = (await db.execute(count_q)).scalar() or 0

    query = query.order_by(desc(Event.timestamp)).limit(limit).offset(offset)
    rows = (await db.execute(query)).all()

    return {
        "items": [
            {
                "id": ev.id,
                "project_id": ev.project_id,
                "project_name": pname,
                "model": ev.model,
                "agent_name": ev.agent_name,
                "error": ev.error,
                "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
            }
            for ev, pname in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/incidents/feedback")
async def feedback_incidents(
    status_filter: Optional[str] = Query(None, alias="status"),
    priority_filter: Optional[str] = Query(None, alias="priority"),
    type_filter: Optional[str] = Query(None, alias="type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Bug reports and incident-type feedback."""
    query = select(Feedback)

    if type_filter:
        query = query.where(Feedback.type == type_filter)
    else:
        query = query.where(
            Feedback.type.in_(["bug_report", "security_report", "performance_issue"])
        )
    if status_filter:
        query = query.where(Feedback.status == status_filter)
    if priority_filter:
        query = query.where(Feedback.priority == priority_filter)

    count_sub = select(func.count(Feedback.id)).where(
        Feedback.type.in_(["bug_report", "security_report", "performance_issue"])
    )
    total = (await db.execute(count_sub)).scalar() or 0

    query = query.order_by(desc(Feedback.created_at)).limit(limit).offset(offset)
    items = (await db.execute(query)).scalars().all()

    return {
        "items": [
            {
                "id": f.id,
                "type": f.type,
                "title": f.title,
                "description": f.description[:200] + "..." if f.description and len(f.description) > 200 else f.description,
                "status": f.status,
                "priority": f.priority,
                "user_email": f.user_email,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "admin_response": f.admin_response,
            }
            for f in items
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
