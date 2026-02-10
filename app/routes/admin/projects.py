"""
Admin routes -- project and API key governance.
"""

import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, or_

from ...database import get_db
from ...models.user_models import User, ProjectMember
from ...models.db_models import Project, Event
from ...services.admin_service import log_admin_action
from ._deps import require_superuser

router = APIRouter()


@router.get("/projects")
async def list_projects(
    search: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """List all projects with owner info and usage stats."""
    query = select(Project, User).outerjoin(User, Project.owner_id == User.id)

    if search:
        pattern = f"%{search}%"
        query = query.where(
            or_(Project.name.ilike(pattern), User.email.ilike(pattern))
        )
    if is_active is not None:
        query = query.where(Project.is_active == is_active)

    count_q = select(func.count(Project.id))
    if search:
        count_q = count_q.outerjoin(User, Project.owner_id == User.id).where(
            or_(Project.name.ilike(f"%{search}%"), User.email.ilike(f"%{search}%"))
        )
    if is_active is not None:
        count_q = count_q.where(Project.is_active == is_active)
    total = (await db.execute(count_q)).scalar() or 0

    sort_col = getattr(Project, sort, Project.created_at)
    query = query.order_by(desc(sort_col) if order == "desc" else sort_col)
    query = query.limit(limit).offset(offset)

    rows = (await db.execute(query)).all()

    items = []
    for proj, owner in rows:
        event_stats = (await db.execute(
            select(
                func.count(Event.id).label("event_count"),
                func.max(Event.timestamp).label("last_event"),
            ).where(Event.project_id == proj.id)
        )).one()

        items.append({
            "id": proj.id,
            "name": proj.name,
            "description": proj.description,
            "is_active": proj.is_active,
            "key_prefix": proj.api_key[:7] + "..." if proj.api_key else None,
            "owner_email": owner.email if owner else None,
            "owner_name": owner.name if owner else None,
            "created_at": proj.created_at.isoformat() if proj.created_at else None,
            "event_count": int(event_stats.event_count),
            "last_event_at": event_stats.last_event.isoformat() if event_stats.last_event else None,
        })

    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/projects/{project_id}")
async def get_project_detail(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Detailed project view with members, usage, and key info."""
    proj = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    owner = None
    if proj.owner_id:
        owner = (await db.execute(select(User).where(User.id == proj.owner_id))).scalar_one_or_none()

    members_rows = (await db.execute(
        select(ProjectMember, User)
        .join(User, ProjectMember.user_id == User.id)
        .where(ProjectMember.project_id == project_id)
    )).all()

    event_stats = (await db.execute(
        select(
            func.count(Event.id).label("event_count"),
            func.coalesce(func.sum(Event.total_tokens), 0).label("tokens"),
            func.coalesce(func.sum(Event.cost), 0).label("cost"),
            func.max(Event.timestamp).label("last_event"),
            func.min(Event.timestamp).label("first_event"),
        ).where(Event.project_id == project_id)
    )).one()

    return {
        "id": proj.id,
        "name": proj.name,
        "description": proj.description,
        "is_active": proj.is_active,
        "key_prefix": proj.api_key[:7] + "..." if proj.api_key else None,
        "key_created_at": proj.created_at.isoformat() if proj.created_at else None,
        "owner": {
            "id": owner.id,
            "email": owner.email,
            "name": owner.name,
        } if owner else None,
        "members": [
            {
                "user_id": pm.user_id,
                "email": u.email,
                "name": u.name,
                "role": pm.role,
            }
            for pm, u in members_rows
        ],
        "usage": {
            "total_events": int(event_stats.event_count),
            "total_tokens": int(event_stats.tokens),
            "total_cost": float(event_stats.cost),
            "first_event_at": event_stats.first_event.isoformat() if event_stats.first_event else None,
            "last_event_at": event_stats.last_event.isoformat() if event_stats.last_event else None,
        },
    }


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: str,
    body: Dict[str, Any] = Body(...),
    request: Request = None,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Admin update: toggle is_active (freeze/unfreeze ingestion)."""
    proj = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    changes = {}
    if "is_active" in body:
        old_val = proj.is_active
        proj.is_active = body["is_active"]
        if old_val != body["is_active"]:
            changes["is_active"] = {"old": old_val, "new": body["is_active"]}

    if changes:
        action = "project_resumed" if body.get("is_active") else "project_frozen"
        await log_admin_action(
            db,
            admin_id=admin.id,
            action_type=action,
            target_type="project",
            target_id=project_id,
            details=changes,
            ip_address=request.client.host if request and request.client else None,
        )

    await db.commit()
    await db.refresh(proj)
    return {"id": proj.id, "is_active": proj.is_active, "message": "Project updated"}


@router.post("/projects/{project_id}/rotate-key")
async def rotate_project_key(
    project_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Rotate (regenerate) a project's API key."""
    proj = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    new_key = f"sk_{uuid.uuid4().hex}"
    proj.api_key = new_key

    await log_admin_action(
        db,
        admin_id=admin.id,
        action_type="project_key_rotated",
        target_type="project",
        target_id=project_id,
        details={"new_key_prefix": new_key[:7] + "..."},
        ip_address=request.client.host if request.client else None,
    )
    await db.commit()

    return {
        "id": proj.id,
        "key_prefix": new_key[:7] + "...",
        "message": "API key rotated. Users must update their SDK configuration.",
    }


@router.post("/projects/{project_id}/revoke-key")
async def revoke_project_key(
    project_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Revoke a project key and deactivate the project."""
    proj = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    proj.is_active = False

    await log_admin_action(
        db,
        admin_id=admin.id,
        action_type="project_key_revoked",
        target_type="project",
        target_id=project_id,
        details={},
        ip_address=request.client.host if request.client else None,
    )
    await db.commit()

    return {"id": proj.id, "is_active": False, "message": "Project key revoked and project deactivated"}
