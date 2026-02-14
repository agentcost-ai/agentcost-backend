"""
Admin routes -- user and tenant management.
"""

from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, desc, or_
from datetime import datetime, timezone

from ...database import get_db
from ...models.user_models import User, UserSession, ProjectMember
from ...models.db_models import Project, Event, UserMilestone
from ...services.admin_service import (
    log_admin_action,
    soft_delete_user as svc_soft_delete_user,
    delete_user_permanently as svc_delete_user,
    update_admin_notes as svc_update_admin_notes,
)
from ...services.email_service import send_admin_email
from ._deps import require_superuser

router = APIRouter()


def _escape_like(value: str) -> str:
    """Escape special LIKE/ILIKE characters to prevent wildcard injection."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class AdminUserUpdate(BaseModel):
    """Typed body for admin user update."""
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


class AdminNotesUpdate(BaseModel):
    """Typed body for admin notes update."""
    notes: str = Field(default="", max_length=10000)


class AdminEmailBody(BaseModel):
    """Typed body for admin direct email."""
    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=50000)


@router.get("/users")
async def list_users(
    search: Optional[str] = Query(None, description="Search by email or name"),
    is_active: Optional[bool] = Query(None),
    is_superuser: Optional[bool] = Query(None),
    include_deleted: bool = Query(False, description="Include soft-deleted users"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort: str = Query("created_at", description="Sort field"),
    order: str = Query("desc", description="asc or desc"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """List all users with search, filter, and pagination."""
    query = select(User)

    # Exclude soft-deleted users by default
    if not include_deleted:
        query = query.where(User.is_deleted == False)

    if search:
        escaped = _escape_like(search)
        pattern = f"%{escaped}%"
        query = query.where(
            or_(User.email.ilike(pattern), User.name.ilike(pattern))
        )
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    if is_superuser is not None:
        query = query.where(User.is_superuser == is_superuser)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    sort_col = getattr(User, sort, User.created_at)
    query = query.order_by(desc(sort_col) if order == "desc" else sort_col)
    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    users = result.scalars().all()

    return {
        "items": [
            {
                "id": u.id,
                "email": u.email,
                "name": u.name,
                "is_active": u.is_active,
                "is_superuser": u.is_superuser,
                "email_verified": u.email_verified,
                "auth_provider": getattr(u, 'auth_provider', 'email'),
                "user_number": u.user_number,
                "milestone_badge": u.milestone_badge,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
                "last_active_at": u.last_active_at.isoformat() if u.last_active_at else None,
                "is_deleted": u.is_deleted,
                "deleted_at": u.deleted_at.isoformat() if u.deleted_at else None,
            }
            for u in users
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Full user profile with project memberships and usage footprint."""
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    owned = (await db.execute(
        select(Project).where(Project.owner_id == user_id)
    )).scalars().all()

    memberships = (await db.execute(
        select(ProjectMember, Project)
        .join(Project, ProjectMember.project_id == Project.id)
        .where(ProjectMember.user_id == user_id)
    )).all()

    project_ids = [p.id for p in owned]
    usage = None
    if project_ids:
        usage_row = (await db.execute(
            select(
                func.count(Event.id).label("total_events"),
                func.coalesce(func.sum(Event.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(Event.cost), 0).label("total_cost"),
            ).where(Event.project_id.in_(project_ids))
        )).one()
        usage = {
            "total_events": int(usage_row.total_events),
            "total_tokens": int(usage_row.total_tokens),
            "total_cost": float(usage_row.total_cost),
        }

    session_count = (await db.execute(
        select(func.count(UserSession.id)).where(
            UserSession.user_id == user_id,
            UserSession.is_revoked == False,
            UserSession.expires_at > datetime.now(timezone.utc),
        )
    )).scalar() or 0

    # Fetch milestones
    milestones = (await db.execute(
        select(UserMilestone)
        .where(UserMilestone.user_id == user_id)
        .order_by(desc(UserMilestone.achieved_at))
    )).scalars().all()

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "email_verified": user.email_verified,
        "auth_provider": getattr(user, 'auth_provider', 'email'),
        "admin_notes": user.admin_notes,
        "user_number": user.user_number,
        "milestone_badge": user.milestone_badge,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        "last_active_at": user.last_active_at.isoformat() if user.last_active_at else None,
        "email_verification_sent_at": user.email_verification_sent_at.isoformat() if user.email_verification_sent_at else None,
        "is_deleted": user.is_deleted,
        "deleted_at": user.deleted_at.isoformat() if user.deleted_at else None,
        "active_sessions": session_count,
        "owned_projects": [
            {
                "id": p.id,
                "name": p.name,
                "is_active": p.is_active,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in owned
        ],
        "memberships": [
            {
                "project_id": pm.project_id,
                "project_name": proj.name,
                "role": pm.role,
            }
            for pm, proj in memberships
        ],
        "milestones": [
            {
                "id": m.id,
                "milestone_type": m.milestone_type,
                "milestone_name": m.milestone_name,
                "milestone_description": m.milestone_description,
                "metadata_json": m.metadata_json,
                "achieved_at": m.achieved_at.isoformat() if m.achieved_at else None,
            }
            for m in milestones
        ],
        "usage": usage or {"total_events": 0, "total_tokens": 0, "total_cost": 0},
    }


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    body: AdminUserUpdate,
    request: Request = None,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Admin user update: toggle is_active, is_superuser."""
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user_id == admin.id and body.is_superuser is not None and not body.is_superuser:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove your own superuser status",
        )

    changes = {}
    if body.is_active is not None:
        old_val = user.is_active
        user.is_active = body.is_active
        if old_val != body.is_active:
            changes["is_active"] = {"old": old_val, "new": body.is_active}

    if body.is_superuser is not None:
        old_val = user.is_superuser
        user.is_superuser = body.is_superuser
        if old_val != body.is_superuser:
            changes["is_superuser"] = {"old": old_val, "new": body.is_superuser}

    if body.is_active is not None and not body.is_active:
        await db.execute(
            update(UserSession)
            .where(UserSession.user_id == user_id, UserSession.is_revoked == False)
            .values(is_revoked=True)
        )

    if changes:
        ip = request.client.host if request and request.client else None
        await log_admin_action(
            db,
            admin_id=admin.id,
            action_type="user_updated",
            target_type="user",
            target_id=user_id,
            details=changes,
            ip_address=ip,
            user_agent=request.headers.get("user-agent") if request else None,
        )

    await db.commit()
    await db.refresh(user)

    return {
        "id": user.id,
        "email": user.email,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "message": "User updated",
    }


@router.post("/users/{user_id}/revoke-sessions")
async def revoke_user_sessions(
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Revoke all active sessions for a user."""
    result = await db.execute(
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.is_revoked == False)
        .values(is_revoked=True)
    )
    await log_admin_action(
        db,
        admin_id=admin.id,
        action_type="sessions_revoked",
        target_type="user",
        target_id=user_id,
        details={"revoked_count": result.rowcount},
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    await db.commit()
    return {"revoked": result.rowcount}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    permanent: bool = Query(False, description="Permanently delete instead of soft-delete"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """
    Delete a user. Performs soft-delete by default (marks as deleted).
    Pass ?permanent=true to permanently remove the user and all associated data.
    Cannot delete superuser accounts or your own account.
    """
    try:
        if permanent:
            result = await svc_delete_user(
                db,
                user_id=user_id,
                admin=admin,
                ip_address=request.client.host if request.client else None,
            )
            await db.commit()
            return {"message": "User deleted permanently", **result}
        else:
            result = await svc_soft_delete_user(
                db,
                user_id=user_id,
                admin=admin,
                ip_address=request.client.host if request.client else None,
            )
            await db.commit()
            return {"message": "User soft-deleted", **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/users/{user_id}/notes")
async def set_admin_notes(
    user_id: str,
    body: AdminNotesUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """
    Update the internal admin notes for a user.
    """
    try:
        user = await svc_update_admin_notes(
            db, user_id=user_id, notes=body.notes, admin=admin
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    await db.commit()
    return {"message": "Notes updated", "user_id": user.id}


@router.post("/users/{user_id}/send-email")
async def send_email_to_user(
    user_id: str,
    body: AdminEmailBody,
    request: Request = None,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """
    Send a direct email to a user from the admin panel.
    """
    subject = body.subject.strip()
    email_body = body.body.strip()

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    success = send_admin_email(user.email, subject, email_body)

    await log_admin_action(
        db,
        admin_id=admin.id,
        action_type="email_sent",
        target_type="user",
        target_id=user_id,
        details={"subject": subject, "success": success},
        ip_address=request.client.host if request and request.client else None,
        user_agent=request.headers.get("user-agent") if request else None,
    )
    await db.commit()

    if not success:
        raise HTTPException(
            status_code=502,
            detail="Email delivery failed. Check Resend configuration.",
        )

    return {"message": "Email sent", "recipient": user.email}
