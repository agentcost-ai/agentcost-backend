"""
In-app notification service.

Persists per-user notifications used by the dashboard's notification bell.
Email delivery is handled separately by ``email_service``.
"""

from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import Notification


VALID_SEVERITIES = {"info", "warning", "critical"}


class NotificationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        *,
        user_id: str,
        type: str,
        title: str,
        body: Optional[str] = None,
        severity: str = "info",
        link: Optional[str] = None,
        project_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> Notification:
        if severity not in VALID_SEVERITIES:
            severity = "info"

        notif = Notification(
            user_id=user_id,
            type=type,
            severity=severity,
            title=title[:255],
            body=body,
            link=link,
            project_id=project_id,
            payload=payload,
        )
        self.db.add(notif)
        await self.db.flush()
        return notif

    async def create_bulk(
        self,
        *,
        user_ids: Sequence[str],
        type: str,
        title: str,
        body: Optional[str] = None,
        severity: str = "info",
        link: Optional[str] = None,
        project_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> list[Notification]:
        created: list[Notification] = []
        for uid in set(uid for uid in user_ids if uid):
            created.append(
                await self.create(
                    user_id=uid,
                    type=type,
                    title=title,
                    body=body,
                    severity=severity,
                    link=link,
                    project_id=project_id,
                    payload=payload,
                )
            )
        return created

    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        unread_only: bool = False,
    ) -> list[Notification]:
        stmt = select(Notification).where(Notification.user_id == user_id)
        if unread_only:
            stmt = stmt.where(Notification.is_read.is_(False))
        stmt = stmt.order_by(Notification.created_at.desc()).limit(limit).offset(offset)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def unread_count(self, user_id: str) -> int:
        result = await self.db.execute(
            select(func.count(Notification.id)).where(
                Notification.user_id == user_id,
                Notification.is_read.is_(False),
            )
        )
        return int(result.scalar() or 0)

    async def mark_read(self, *, user_id: str, notification_id: str) -> bool:
        result = await self.db.execute(
            update(Notification)
            .where(Notification.id == notification_id, Notification.user_id == user_id)
            .values(is_read=True, read_at=datetime.now(timezone.utc))
            .execution_options(synchronize_session=False)
        )
        await self.db.flush()
        return (result.rowcount or 0) > 0

    async def mark_all_read(self, *, user_id: str) -> int:
        result = await self.db.execute(
            update(Notification)
            .where(Notification.user_id == user_id, Notification.is_read.is_(False))
            .values(is_read=True, read_at=datetime.now(timezone.utc))
            .execution_options(synchronize_session=False)
        )
        await self.db.flush()
        return int(result.rowcount or 0)
