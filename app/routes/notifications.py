"""
AgentCost Backend - Notifications API

Endpoints for the per-user in-app notification feed.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.db_models import Notification
from ..models.schemas import (
    NotificationCountResponse,
    NotificationListResponse,
    NotificationResponse,
)
from ..models.user_models import User
from ..services.notification_service import NotificationService
from ..utils.auth import get_required_user


router = APIRouter(prefix="/v1/notifications", tags=["Notifications"])


@router.get("", response_model=NotificationListResponse)
async def list_notifications(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    unread_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_required_user),
):
    """List notifications for the current user."""
    service = NotificationService(db)
    items = await service.list_for_user(
        user.id, limit=limit, offset=offset, unread_only=unread_only
    )

    total_stmt = select(func.count(Notification.id)).where(Notification.user_id == user.id)
    if unread_only:
        total_stmt = total_stmt.where(Notification.is_read.is_(False))
    total = int((await db.execute(total_stmt)).scalar() or 0)

    unread = await service.unread_count(user.id)

    return NotificationListResponse(
        items=[NotificationResponse.model_validate(n) for n in items],
        total=total,
        unread_count=unread,
    )


@router.get("/unread-count", response_model=NotificationCountResponse)
async def get_unread_count(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_required_user),
):
    """Return the unread notification count for the current user."""
    service = NotificationService(db)
    return NotificationCountResponse(unread_count=await service.unread_count(user.id))


@router.post("/{notification_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_notification_read(
    notification_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_required_user),
):
    """Mark a single notification as read."""
    service = NotificationService(db)
    updated = await service.mark_read(user_id=user.id, notification_id=notification_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Notification not found")
    return None


@router.post("/read-all", response_model=NotificationCountResponse)
async def mark_all_notifications_read(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_required_user),
):
    """Mark every notification for the current user as read."""
    service = NotificationService(db)
    await service.mark_all_read(user_id=user.id)
    return NotificationCountResponse(unread_count=0)
