"""
Admin routes -- audit log.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...models.user_models import User
from ...services.admin_service import get_admin_activity_log as svc_get_audit_log
from ._deps import require_superuser

router = APIRouter()


@router.get("/audit-log")
async def get_audit_log(
    action_type: Optional[str] = Query(None),
    target_type: Optional[str] = Query(None),
    admin_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """
    Retrieve the admin activity audit trail.
    Supports filtering by action_type, target_type, and actor.
    """
    return await svc_get_audit_log(
        db,
        action_type=action_type,
        target_type=target_type,
        admin_id=admin_id,
        limit=limit,
        offset=offset,
    )
