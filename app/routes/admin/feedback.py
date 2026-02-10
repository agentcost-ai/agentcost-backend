"""
Admin routes -- feedback management.
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, or_

from ...database import get_db
from ...models.user_models import User
from ...models.db_models import Feedback, FeedbackComment, FeedbackEvent
from ...services.admin_service import update_feedback as svc_update_feedback
from ...services.email_service import send_feedback_update_email
from ._deps import require_superuser

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/feedback")
async def list_all_feedback(
    status_filter: Optional[str] = Query(None, alias="status"),
    priority_filter: Optional[str] = Query(None, alias="priority"),
    type_filter: Optional[str] = Query(None, alias="type"),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """
    List all feedback with filtering by status, priority, type, and search.
    Unlike /incidents/feedback this returns ALL feedback types.
    """
    query = select(Feedback)
    count_q = select(func.count(Feedback.id))

    if type_filter:
        query = query.where(Feedback.type == type_filter)
        count_q = count_q.where(Feedback.type == type_filter)
    if status_filter:
        query = query.where(Feedback.status == status_filter)
        count_q = count_q.where(Feedback.status == status_filter)
    if priority_filter:
        query = query.where(Feedback.priority == priority_filter)
        count_q = count_q.where(Feedback.priority == priority_filter)
    if search:
        pattern = f"%{search}%"
        query = query.where(
            or_(Feedback.title.ilike(pattern), Feedback.description.ilike(pattern))
        )
        count_q = count_q.where(
            or_(Feedback.title.ilike(pattern), Feedback.description.ilike(pattern))
        )

    total = (await db.execute(count_q)).scalar() or 0
    query = query.order_by(desc(Feedback.created_at)).limit(limit).offset(offset)
    items = (await db.execute(query)).scalars().all()

    return {
        "items": [
            {
                "id": f.id,
                "type": f.type,
                "title": f.title,
                "description": f.description,
                "status": f.status,
                "priority": f.priority,
                "upvotes": f.upvotes,
                "user_id": f.user_id,
                "user_email": f.user_email,
                "user_name": f.user_name,
                "model_name": f.model_name,
                "model_provider": f.model_provider,
                "admin_response": f.admin_response,
                "admin_responded_at": f.admin_responded_at.isoformat() if f.admin_responded_at else None,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "updated_at": f.updated_at.isoformat() if f.updated_at else None,
            }
            for f in items
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/feedback/{feedback_id}")
async def get_feedback_detail(
    feedback_id: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Full feedback detail including comments and event history."""
    feedback = (
        await db.execute(select(Feedback).where(Feedback.id == feedback_id))
    ).scalar_one_or_none()

    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")

    comments = (await db.execute(
        select(FeedbackComment)
        .where(FeedbackComment.feedback_id == feedback_id)
        .order_by(FeedbackComment.created_at)
    )).scalars().all()

    events = (await db.execute(
        select(FeedbackEvent)
        .where(FeedbackEvent.feedback_id == feedback_id)
        .order_by(desc(FeedbackEvent.created_at))
    )).scalars().all()

    return {
        "id": feedback.id,
        "type": feedback.type,
        "title": feedback.title,
        "description": feedback.description,
        "status": feedback.status,
        "priority": feedback.priority,
        "upvotes": feedback.upvotes,
        "user_id": feedback.user_id,
        "user_email": feedback.user_email,
        "user_name": feedback.user_name,
        "model_name": feedback.model_name,
        "model_provider": feedback.model_provider,
        "environment": feedback.environment,
        "is_confidential": feedback.is_confidential,
        "admin_response": feedback.admin_response,
        "admin_responded_at": feedback.admin_responded_at.isoformat() if feedback.admin_responded_at else None,
        "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
        "updated_at": feedback.updated_at.isoformat() if feedback.updated_at else None,
        "comments": [
            {
                "id": c.id,
                "user_name": c.user_name,
                "comment": c.comment,
                "is_admin": c.is_admin,
                "is_internal": c.is_internal,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in comments
        ],
        "events": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "actor_id": e.actor_id,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    }


@router.patch("/feedback/{feedback_id}")
async def patch_feedback(
    feedback_id: str,
    body: Dict[str, Any] = Body(...),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """
    Update feedback status, priority, and/or admin response.
    Sends an email notification to the user when a response is provided.
    """
    try:
        feedback = await svc_update_feedback(
            db,
            feedback_id=feedback_id,
            admin=admin,
            status=body.get("status"),
            priority=body.get("priority"),
            admin_response=body.get("admin_response"),
            ip_address=request.client.host if request and request.client else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if body.get("admin_response") and feedback.user_email:
        try:
            await send_feedback_update_email(
                email=feedback.user_email,
                title=feedback.title,
                status=feedback.status,
                admin_response=body["admin_response"],
                name=feedback.user_name,
                feedback_id=feedback.id,
            )
        except Exception as e:
            logger.warning("Failed to send feedback notification email: %s", e)

    await db.commit()

    return {
        "id": feedback.id,
        "status": feedback.status,
        "priority": feedback.priority,
        "admin_response": feedback.admin_response,
        "message": "Feedback updated",
    }
