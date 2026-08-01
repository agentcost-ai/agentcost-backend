"""Feedback API -- submit, list, vote, comment, and admin triage."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select, func, exists, or_, literal
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.db_models import Feedback, FeedbackUpvote, FeedbackComment, FeedbackEvent
from ..models.schemas import (
    FeedbackCreate,
    FeedbackUpdate,
    FeedbackResponse,
    FeedbackListResponse,
    FeedbackCommentCreate,
    FeedbackCommentListResponse,
    FeedbackCreatedResponse,
    FeedbackSummaryResponse,
    FeedbackEventResponse,
)
from ..models.user_models import User
from ..services.auth_service import get_current_user
from ..services.email_service import (
    send_feedback_admin_notification,
    send_feedback_update_email,
)
from ..utils.auth import get_required_user as _shared_get_required_user

router = APIRouter(prefix="/v1/feedback", tags=["Feedback"])
security = HTTPBearer(auto_error=False)

ALLOWED_TYPES = {
    "feature_request", "bug_report", "model_request",
    "general", "security_report", "performance_issue",
}
ALLOWED_STATUSES = {
    "open", "under_review", "needs_info", "in_progress",
    "completed", "shipped", "rejected", "duplicate",
}
ALLOWED_PRIORITIES = {"low", "medium", "high", "critical"}


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    if not credentials:
        return None

    user = await get_current_user(db, credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_admin_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_current_user(db, credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return user


def build_display_name(
    user: Optional[User],
    provided_name: Optional[str],
    provided_email: Optional[str],
) -> str:
    if provided_name:
        return provided_name
    if user and user.name:
        return user.name
    return "Anonymous"


def build_user_email(
    user: Optional[User],
    provided_email: Optional[str],
) -> Optional[str]:
    if provided_email:
        return provided_email
    if user and user.email:
        return user.email
    return None


def normalize_search_term(term: Optional[str]) -> Optional[str]:
    if not term:
        return None
    stripped = term.strip()
    if not stripped:
        return None
    return stripped.lower()


def escape_like(value: str) -> str:
    """Escape special LIKE characters to prevent wildcard injection."""
    return value.replace("%", "\\%").replace("_", "\\_")


def serialize_feedback(
    feedback: Feedback,
    comment_count: int,
    user_has_upvoted: bool,
) -> FeedbackResponse:
    return FeedbackResponse(
        id=feedback.id,
        type=feedback.type,
        title=feedback.title,
        description=feedback.description,
        status=feedback.status,
        priority=feedback.priority,
        upvotes=feedback.upvotes,
        user_has_upvoted=user_has_upvoted,
        model_name=feedback.model_name,
        model_provider=feedback.model_provider,
        admin_response=feedback.admin_response,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
        comment_count=comment_count,
        user_name=feedback.user_name,
        metadata=feedback.type_metadata,
        attachments=feedback.attachments,
        environment=feedback.environment,
        is_confidential=feedback.is_confidential,
    )


@router.post("", response_model=FeedbackCreatedResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback: FeedbackCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Submit feedback or a request.
    Accepts authenticated and anonymous submissions.
    """
    user_email = build_user_email(user, feedback.user_email)
    display_name = build_display_name(user, feedback.user_name, user_email)

    model_name = feedback.model_name if feedback.type == "model_request" else None
    model_provider = feedback.model_provider if feedback.type == "model_request" else None

    # Pick up client fingerprint for audit (available on every request)
    ip_address = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.headers.get("X-Real-IP")
        or (request.client.host if request.client else None)
    )
    user_agent_str = request.headers.get("User-Agent")

    new_feedback = Feedback(
        user_id=user.id if user else None,
        user_email=user_email,
        user_name=display_name,
        type=feedback.type,
        title=feedback.title.strip(),
        description=feedback.description.strip(),
        model_name=model_name,
        model_provider=model_provider,
        type_metadata=feedback.metadata,
        attachments=feedback.attachments,
        environment=feedback.environment,
        client_metadata=feedback.client_metadata,
        is_confidential=feedback.type == "security_report",
        ip_address=ip_address,
        user_agent=user_agent_str,
    )

    db.add(new_feedback)
    await db.commit()
    await db.refresh(new_feedback)

    await send_feedback_admin_notification(
        feedback_id=new_feedback.id,
        feedback_type=new_feedback.type,
        title=new_feedback.title,
        description=new_feedback.description,
        submitted_by=display_name,
    )

    return FeedbackCreatedResponse(
        id=new_feedback.id,
        message="Feedback submitted successfully.",
    )


@router.get("", response_model=FeedbackListResponse)
async def list_feedback(
    type: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    sort_by: str = Query("recent", pattern="^(recent|popular|oldest)$"),
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """List feedback with filtering, search, and sorting."""
    if type and type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid feedback type")
    if status and status not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    if priority and priority not in ALLOWED_PRIORITIES:
        raise HTTPException(status_code=400, detail="Invalid priority")

    search_term = normalize_search_term(search)

    comment_count_subquery = (
        select(func.count(FeedbackComment.id))
        .where(FeedbackComment.feedback_id == Feedback.id)
        .correlate(Feedback)
        .scalar_subquery()
    )

    if user:
        user_has_upvoted_expr = exists().where(
            FeedbackUpvote.feedback_id == Feedback.id,
            FeedbackUpvote.user_id == user.id,
        )
    else:
        user_has_upvoted_expr = literal(False)

    query = select(
        Feedback,
        comment_count_subquery.label("comment_count"),
        user_has_upvoted_expr.label("user_has_upvoted"),
    )

    if type:
        query = query.where(Feedback.type == type)
    if status:
        query = query.where(Feedback.status == status)
    if priority:
        query = query.where(Feedback.priority == priority)
    if search_term:
        like_term = f"%{escape_like(search_term)}%"
        query = query.where(
            or_(
                func.lower(Feedback.title).like(like_term),
                func.lower(Feedback.description).like(like_term),
                func.lower(Feedback.model_name).like(like_term),
                func.lower(Feedback.model_provider).like(like_term),
            )
        )

    # Non-admins must not see confidential items (e.g. security reports)
    if not (user and user.is_superuser):
        query = query.where(Feedback.is_confidential == False)

    if sort_by == "popular":
        query = query.order_by(Feedback.upvotes.desc(), Feedback.created_at.desc())
    elif sort_by == "oldest":
        query = query.order_by(Feedback.created_at.asc())
    else:
        query = query.order_by(Feedback.created_at.desc())

    query = query.limit(limit).offset(offset)

    total_query = select(func.count(Feedback.id))
    if type:
        total_query = total_query.where(Feedback.type == type)
    if status:
        total_query = total_query.where(Feedback.status == status)
    if priority:
        total_query = total_query.where(Feedback.priority == priority)
    if search_term:
        like_term = f"%{escape_like(search_term)}%"
        total_query = total_query.where(
            or_(
                func.lower(Feedback.title).like(like_term),
                func.lower(Feedback.description).like(like_term),
                func.lower(Feedback.model_name).like(like_term),
                func.lower(Feedback.model_provider).like(like_term),
            )
        )

    # Same confidential filter on the count query
    if not (user and user.is_superuser):
        total_query = total_query.where(Feedback.is_confidential == False)

    result = await db.execute(query)
    rows = result.all()

    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    items = [
        serialize_feedback(feedback, comment_count, user_has_upvoted)
        for feedback, comment_count, user_has_upvoted in rows
    ]

    return FeedbackListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/summary", response_model=FeedbackSummaryResponse)
async def get_feedback_summary(
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Get summary counts for feedback."""
    # Exclude confidential items from counts for non-admin users
    is_admin = user and getattr(user, 'is_superuser', False)
    base_filter = True if is_admin else Feedback.is_confidential == False  # noqa: E712

    total_result = await db.execute(
        select(func.count(Feedback.id)).where(base_filter)
    )
    total = total_result.scalar() or 0

    type_rows = await db.execute(
        select(Feedback.type, func.count(Feedback.id)).where(base_filter).group_by(Feedback.type)
    )
    status_rows = await db.execute(
        select(Feedback.status, func.count(Feedback.id)).where(base_filter).group_by(Feedback.status)
    )

    by_type = {row[0]: row[1] for row in type_rows.all()}
    by_status = {row[0]: row[1] for row in status_rows.all()}

    return FeedbackSummaryResponse(total=total, by_type=by_type, by_status=by_status)


@router.get("/{feedback_id}", response_model=FeedbackResponse)
async def get_feedback(
    feedback_id: str,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Get a single feedback item."""
    comment_count_subquery = (
        select(func.count(FeedbackComment.id))
        .where(FeedbackComment.feedback_id == Feedback.id)
        .correlate(Feedback)
        .scalar_subquery()
    )

    if user:
        user_has_upvoted_expr = exists().where(
            FeedbackUpvote.feedback_id == Feedback.id,
            FeedbackUpvote.user_id == user.id,
        )
    else:
        user_has_upvoted_expr = literal(False)

    query = select(
        Feedback,
        comment_count_subquery.label("comment_count"),
        user_has_upvoted_expr.label("user_has_upvoted"),
    ).where(Feedback.id == feedback_id)

    result = await db.execute(query)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Feedback not found")

    feedback, comment_count, user_has_upvoted = row

    # Block access to confidential items for non-admin users
    is_admin = user and getattr(user, 'is_superuser', False)
    if feedback.is_confidential and not is_admin:
        raise HTTPException(status_code=404, detail="Feedback not found")

    return serialize_feedback(feedback, comment_count, user_has_upvoted)


@router.post("/{feedback_id}/upvote")
async def toggle_upvote(
    feedback_id: str,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Toggle an upvote on a feedback item (requires authentication)."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    feedback_result = await db.execute(select(Feedback).where(Feedback.id == feedback_id))
    feedback = feedback_result.scalar_one_or_none()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")

    upvote_result = await db.execute(
        select(FeedbackUpvote).where(
            FeedbackUpvote.feedback_id == feedback_id,
            FeedbackUpvote.user_id == user.id,
        )
    )
    existing = upvote_result.scalar_one_or_none()

    if existing:
        await db.delete(existing)
        feedback.upvotes = max(0, feedback.upvotes - 1)
        action = "removed"
    else:
        db.add(FeedbackUpvote(feedback_id=feedback_id, user_id=user.id))
        feedback.upvotes += 1
        action = "added"

    feedback.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "action": action,
        "upvotes": feedback.upvotes,
    }


@router.post("/{feedback_id}/comments", status_code=status.HTTP_201_CREATED)
async def add_comment(
    feedback_id: str,
    comment: FeedbackCommentCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(_shared_get_required_user),
):
    """Add a comment to feedback. Requires authentication (L4 fix)."""
    feedback_result = await db.execute(select(Feedback).where(Feedback.id == feedback_id))
    feedback = feedback_result.scalar_one_or_none()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")

    display_name = build_display_name(user, comment.user_name, None)

    new_comment = FeedbackComment(
        feedback_id=feedback_id,
        user_id=user.id,
        user_name=display_name,
        comment=comment.comment.strip(),
        is_admin=user.is_superuser if user else False,
    )

    db.add(new_comment)
    feedback.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {"message": "Comment added"}


@router.get("/{feedback_id}/comments", response_model=FeedbackCommentListResponse)
async def list_comments(
    feedback_id: str,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Get all comments for a feedback item."""
    feedback_result = await db.execute(select(Feedback.id).where(Feedback.id == feedback_id))
    if not feedback_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Feedback not found")

    query = (
        select(FeedbackComment)
        .where(FeedbackComment.feedback_id == feedback_id)
        .order_by(FeedbackComment.created_at.asc())
    )

    # Strip internal notes from non-admin views
    if not (user and user.is_superuser):
        query = query.where(FeedbackComment.is_internal == False)

    result = await db.execute(query)
    comments = result.scalars().all()

    items = [
        {
            "id": c.id,
            "user_name": c.user_name,
            "comment": c.comment,
            "is_admin": c.is_admin,
            "created_at": c.created_at,
        }
        for c in comments
    ]

    return FeedbackCommentListResponse(items=items, total=len(items))


@router.patch("/admin/{feedback_id}", response_model=FeedbackResponse)
async def update_feedback(
    feedback_id: str,
    payload: FeedbackUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    """Update feedback status and admin response (admin only)."""
    feedback_result = await db.execute(select(Feedback).where(Feedback.id == feedback_id))
    feedback = feedback_result.scalar_one_or_none()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")

    # Snapshot before mutation for the audit event
    old_status = feedback.status
    old_priority = feedback.priority

    feedback.status = payload.status
    if payload.priority:
        feedback.priority = payload.priority
    if payload.admin_response is not None:
        feedback.admin_response = payload.admin_response
        feedback.admin_responded_at = datetime.now(timezone.utc)

    feedback.updated_at = datetime.now(timezone.utc)

    # Record an audit event on every admin change
    changes = {}
    if old_status != payload.status:
        changes["status"] = {"old": old_status, "new": payload.status}
    if payload.priority and old_priority != payload.priority:
        changes["priority"] = {"old": old_priority, "new": payload.priority}
    if payload.admin_response is not None:
        changes["admin_response"] = True

    if changes:
        audit_event = FeedbackEvent(
            feedback_id=feedback.id,
            event_type="admin_update",
            old_value={"status": old_status, "priority": old_priority},
            new_value={"status": payload.status, "priority": payload.priority or feedback.priority},
            actor_id=_admin.id,
        )
        db.add(audit_event)

    await db.commit()
    await db.refresh(feedback)

    if feedback.user_email:
        await send_feedback_update_email(
            email=feedback.user_email,
            title=feedback.title,
            status=feedback.status,
            admin_response=feedback.admin_response,
            name=feedback.user_name,
            feedback_id=feedback.id,
        )

    comment_count_result = await db.execute(
        select(func.count(FeedbackComment.id)).where(FeedbackComment.feedback_id == feedback.id)
    )
    comment_count = comment_count_result.scalar() or 0

    return serialize_feedback(feedback, comment_count, False)


@router.post("/admin/{feedback_id}/internal-note", status_code=status.HTTP_201_CREATED)
async def add_internal_note(
    feedback_id: str,
    comment: FeedbackCommentCreate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """
    Add an internal-only note to a feedback item (admin only).

    Internal notes are hidden from non-admin users.
    """
    feedback_result = await db.execute(select(Feedback).where(Feedback.id == feedback_id))
    feedback_obj = feedback_result.scalar_one_or_none()
    if not feedback_obj:
        raise HTTPException(status_code=404, detail="Feedback not found")

    display_name = admin.name or admin.email.split("@", 1)[0]

    new_comment = FeedbackComment(
        feedback_id=feedback_id,
        user_id=admin.id,
        user_name=display_name,
        comment=comment.comment.strip(),
        is_admin=True,
        is_internal=True,
    )

    db.add(new_comment)
    feedback_obj.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {"message": "Internal note added"}


@router.get("/admin/{feedback_id}/events", response_model=list[FeedbackEventResponse])
async def get_feedback_events(
    feedback_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    """
    Get the audit trail for a feedback item (admin only).

    Returns all lifecycle events: status changes, priority changes, admin actions.
    """
    feedback_result = await db.execute(select(Feedback.id).where(Feedback.id == feedback_id))
    if not feedback_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Feedback not found")

    query = (
        select(FeedbackEvent)
        .where(FeedbackEvent.feedback_id == feedback_id)
        .order_by(FeedbackEvent.created_at.asc())
    )
    result = await db.execute(query)
    events = result.scalars().all()

    return [
        FeedbackEventResponse(
            id=e.id,
            feedback_id=e.feedback_id,
            event_type=e.event_type,
            old_value=e.old_value,
            new_value=e.new_value,
            actor_id=e.actor_id,
            created_at=e.created_at,
        )
        for e in events
    ]
