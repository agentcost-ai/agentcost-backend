"""
AgentCost Backend - Admin Service

Centralised business logic for all admin operations.
Handles audit logging, user management helpers, and feedback
management so that route handlers stay thin.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import select, update, delete, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import (
    AdminActivityLog,
    DailyAggregate,
    Event,
    Feedback,
    FeedbackEvent,
    InputPatternCache,
    OptimizationRecommendation,
    Project,
    ProjectBaseline,
)
from ..models.user_models import PendingEmailInvitation, User, UserSession
from ..services.email_service import send_admin_email, send_account_deletion_email

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

async def log_admin_action(
    db: AsyncSession,
    *,
    admin_id: str,
    action_type: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """
    Record an immutable audit entry for an admin action.

    Parameters
    ----------
    db : AsyncSession
        Active database session (caller is responsible for committing).
    admin_id : str
        ID of the admin performing the action.
    action_type : str
        Machine-readable label, e.g. ``user_disabled``, ``feedback_status_changed``.
    target_type : str, optional
        Entity category: ``user``, ``project``, ``feedback``, ``system``.
    target_id : str, optional
        Primary key of the affected entity.
    details : dict, optional
        Arbitrary metadata serialised as JSON.
    ip_address : str, optional
        Client IP address.
    user_agent : str, optional
        Client user-agent header.
    """
    entry = AdminActivityLog(
        id=str(uuid4()),
        admin_id=admin_id,
        action_type=action_type,
        target_type=target_type,
        target_id=target_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(entry)
    # Caller commits with the rest of its transaction


# ---------------------------------------------------------------------------
# User management helpers
# ---------------------------------------------------------------------------

async def suspend_user(
    db: AsyncSession,
    *,
    user_id: str,
    admin: User,
    ip_address: Optional[str] = None,
) -> User:
    """
    Disable a user and revoke all their sessions.

    Returns the updated User object.
    """
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise ValueError("User not found")

    if user.id == admin.id:
        raise ValueError("Cannot disable your own account")

    previous_state = user.is_active
    user.is_active = False

    # Revoke every active session
    await db.execute(
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.is_revoked == False)
        .values(is_revoked=True)
    )

    await log_admin_action(
        db,
        admin_id=admin.id,
        action_type="user_disabled",
        target_type="user",
        target_id=user_id,
        details={"previous_is_active": previous_state},
        ip_address=ip_address,
    )

    return user


async def soft_delete_user(
    db: AsyncSession,
    *,
    user_id: str,
    admin: User,
    ip_address: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Soft-delete a user: mark as deleted, revoke all sessions.

    The user record is preserved but hidden from normal queries.
    Returns a summary dict.
    """
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise ValueError("User not found")

    if admin and user.id == admin.id:
        raise ValueError("Cannot delete your own account")

    if user.is_superuser:
        raise ValueError("Cannot delete a superuser account")

    if user.is_deleted:
        raise ValueError("User is already deleted")

    user.is_deleted = True
    user.deleted_at = datetime.now(timezone.utc)
    user.is_active = False

    # Revoke all active sessions so tokens fail immediately
    await db.execute(
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.is_revoked == False)
        .values(is_revoked=True)
    )

    # Send notification email
    from ..config import get_settings
    settings = get_settings()
    grace_days = settings.deletion_grace_days
    expiry_date = (datetime.now(timezone.utc) + timedelta(days=grace_days)).strftime("%B %d, %Y")
    
    # We fire and forget the email (async)
    await send_account_deletion_email(
        email=user.email,
        name=user.name,
        grace_expiry_date=expiry_date
    )

    await log_admin_action(
        db,
        admin_id=admin.id,
        action_type="user_soft_deleted",
        target_type="user",
        target_id=user_id,
        details={"email": user.email},
        ip_address=ip_address,
    )

    return {"user_id": user_id, "email": user.email}


async def delete_user_permanently(
    db: AsyncSession,
    *,
    user_id: str,
    admin: Optional[User] = None,
    ip_address: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Permanently remove a user and all associated data: sessions,
    memberships, owned projects (with events, aggregates,
    baselines, patterns, recommendations), and any pending email
    invitations matching the user's address.
    """
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise ValueError("User not found")

    if admin and user.id == admin.id:
        raise ValueError("Cannot delete your own account")

    if user.is_superuser and (admin and not admin.is_superuser):
        # Allow system (admin=None) to delete superusers? 
        # Maybe safest to say system CAN delete anyone if expired.
        # But let's prevent non-super admin from deleting super.
        raise ValueError("Cannot delete a superuser account")
        
    if user.is_superuser and admin is None:
        # Prevent auto-purge of superusers just in case?
        # Actually superusers shouldn't expire if they are active?
        # If a superuser was soft-deleted, they should be purgeable.
        pass

    user_email = user.email

    # Revoke all active sessions so in-flight refresh tokens fail immediately.
    await db.execute(
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.is_revoked == False)
        .values(is_revoked=True)
    )

    # Delete owned projects and all project-scoped data
    owned_project_ids_q = select(Project.id).where(Project.owner_id == user_id)
    owned_ids = (await db.execute(owned_project_ids_q)).scalars().all()

    if owned_ids:
        # Child tables of projects (no CASCADE FK defined, so delete explicitly)
        for model in (
            Event,
            DailyAggregate,
            OptimizationRecommendation,
            ProjectBaseline,
            InputPatternCache,
        ):
            await db.execute(
                delete(model).where(model.project_id.in_(owned_ids))
            )

        # PendingEmailInvitation has CASCADE on project_id, but be explicit
        await db.execute(
            delete(PendingEmailInvitation)
            .where(PendingEmailInvitation.project_id.in_(owned_ids))
        )

        # Now remove the projects themselves
        await db.execute(
            delete(Project).where(Project.id.in_(owned_ids))
        )

    # Clean up pending invitations for this email address
    # These are invitations to other users' projects — if the user
    # re-registers with the same email they should NOT inherit access.
    await db.execute(
        delete(PendingEmailInvitation)
        .where(PendingEmailInvitation.email == user_email.lower())
    )

    # Audit log (before the user row disappears)
    await log_admin_action(
        db,
        admin_id=admin.id if admin else "SYSTEM",
        action_type="user_deleted",
        target_type="user",
        target_id=user_id,
        details={
            "email": user_email,
            "permanent": True,
            "projects_deleted": len(owned_ids) if owned_ids else 0,
        },
        ip_address=ip_address,
    )

    # Finally delete the user (cascades sessions, memberships, etc.)
    await db.delete(user)

    return {"user_id": user_id, "email": user_email}


async def update_admin_notes(
    db: AsyncSession,
    *,
    user_id: str,
    notes: str,
    admin: User,
) -> User:
    """Update the internal admin notes on a user record."""
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise ValueError("User not found")

    user.admin_notes = notes

    await log_admin_action(
        db,
        admin_id=admin.id,
        action_type="admin_notes_updated",
        target_type="user",
        target_id=user_id,
        details={"notes_length": len(notes) if notes else 0},
    )

    return user


# ---------------------------------------------------------------------------
# Feedback management helpers
# ---------------------------------------------------------------------------

async def update_feedback(
    db: AsyncSession,
    *,
    feedback_id: str,
    admin: User,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    admin_response: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> Feedback:
    """
    Update feedback status, priority, and/or admin response.
    Creates FeedbackEvent audit entries for each changed field.
    """
    feedback = (
        await db.execute(select(Feedback).where(Feedback.id == feedback_id))
    ).scalar_one_or_none()

    if not feedback:
        raise ValueError("Feedback not found")

    changes: Dict[str, Any] = {}

    if status is not None and status != feedback.status:
        old_status = feedback.status
        feedback.status = status
        changes["status"] = {"old": old_status, "new": status}

        # Record feedback event
        db.add(FeedbackEvent(
            id=str(uuid4()),
            feedback_id=feedback_id,
            event_type="status_change",
            old_value={"status": old_status},
            new_value={"status": status},
            actor_id=admin.id,
        ))

    if priority is not None and priority != feedback.priority:
        old_priority = feedback.priority
        feedback.priority = priority
        changes["priority"] = {"old": old_priority, "new": priority}

        db.add(FeedbackEvent(
            id=str(uuid4()),
            feedback_id=feedback_id,
            event_type="priority_change",
            old_value={"priority": old_priority},
            new_value={"priority": priority},
            actor_id=admin.id,
        ))

    if admin_response is not None:
        feedback.admin_response = admin_response
        feedback.admin_responded_at = datetime.now(timezone.utc)
        changes["admin_response"] = True

        db.add(FeedbackEvent(
            id=str(uuid4()),
            feedback_id=feedback_id,
            event_type="admin_response",
            old_value=None,
            new_value={"response_length": len(admin_response)},
            actor_id=admin.id,
        ))

    if not changes:
        return feedback

    await log_admin_action(
        db,
        admin_id=admin.id,
        action_type="feedback_updated",
        target_type="feedback",
        target_id=feedback_id,
        details=changes,
        ip_address=ip_address,
    )

    return feedback


# ---------------------------------------------------------------------------
# Audit log queries
# ---------------------------------------------------------------------------

async def get_admin_activity_log(
    db: AsyncSession,
    *,
    action_type: Optional[str] = None,
    target_type: Optional[str] = None,
    admin_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Query the admin activity log with optional filters.
    Returns paginated results with actor information.
    """
    query = select(AdminActivityLog, User.email.label("admin_email"), User.name.label("admin_name")).outerjoin(
        User, AdminActivityLog.admin_id == User.id
    )

    filters = []
    if action_type:
        filters.append(AdminActivityLog.action_type == action_type)
    if target_type:
        filters.append(AdminActivityLog.target_type == target_type)
    if admin_id:
        filters.append(AdminActivityLog.admin_id == admin_id)

    if filters:
        query = query.where(*filters)

    # Count
    count_q = select(func.count(AdminActivityLog.id))
    if filters:
        count_q = count_q.where(*filters)
    total = (await db.execute(count_q)).scalar() or 0

    query = query.order_by(desc(AdminActivityLog.created_at)).limit(limit).offset(offset)
    rows = (await db.execute(query)).all()

    items = []
    for log_entry, admin_email, admin_name in rows:
        items.append({
            "id": log_entry.id,
            "admin_id": log_entry.admin_id,
            "admin_email": admin_email,
            "admin_name": admin_name,
            "action_type": log_entry.action_type,
            "target_type": log_entry.target_type,
            "target_id": log_entry.target_id,
            "details": log_entry.details,
            "ip_address": log_entry.ip_address,
            "user_agent": log_entry.user_agent,
            "created_at": log_entry.created_at.isoformat() if log_entry.created_at else None,
        })

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
