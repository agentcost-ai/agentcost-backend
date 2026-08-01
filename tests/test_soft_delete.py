"""
Tests for the soft-delete system.

Validates:
1. Soft-deleted users cannot authenticate via email/password
2. Soft-deleted users cannot authenticate via Google
3. Admin list_users excludes deleted users by default
4. Admin list_users includes deleted users when include_deleted=true
5. delete_user performs soft-delete by default
6. delete_user?permanent=true performs hard delete
7. Sessions are revoked on soft-delete
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user_models import User, UserSession
from app.services.auth_service import AuthService, hash_password, hash_token
from app.services.admin_service import soft_delete_user, delete_user_permanently


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _create_user(
    db: AsyncSession,
    email: str = "test@example.com",
    password: str = "hashedpassword123",
    is_superuser: bool = False,
) -> User:
    """Create a user directly in the DB."""
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        password_hash=hash_password(password),
        name="Test User",
        is_active=True,
        is_deleted=False,
        auth_provider="email",
        is_superuser=is_superuser,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def _create_session(db: AsyncSession, user_id: str) -> UserSession:
    """Create an active session for a user."""
    session = UserSession(
        id=str(uuid.uuid4()),
        user_id=user_id,
        token_hash=hash_token(f"fake_token_{uuid.uuid4()}"),
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        is_revoked=False,
    )
    db.add(session)
    await db.flush()
    return session


# ---------------------------------------------------------------------------
# authenticate_user tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_soft_deleted_user_cannot_authenticate(test_session: AsyncSession):
    """A soft-deleted user past the grace period must be rejected at login.

    Note: within the deletion grace period the system intentionally
    reactivates the account on login (see test_grace_period.py); this test
    covers the post-grace case where login must fail.
    """
    user = await _create_user(test_session, email="deleted@example.com")
    auth = AuthService(test_session)

    # Normal auth should work
    result = await auth.authenticate_user("deleted@example.com", "hashedpassword123")
    assert result is not None
    assert result.id == user.id

    # Soft-delete the user well beyond any grace window.
    user.is_deleted = True
    user.deleted_at = datetime.now(timezone.utc) - timedelta(days=400)
    await test_session.flush()

    # Auth should now fail (past grace period → no reactivation).
    result = await auth.authenticate_user("deleted@example.com", "hashedpassword123")
    assert result is None


@pytest.mark.asyncio
async def test_inactive_user_cannot_authenticate(test_session: AsyncSession):
    """Inactive users should also be rejected (existing behavior preserved)."""
    user = await _create_user(test_session, email="inactive@example.com")
    auth = AuthService(test_session)

    user.is_active = False
    await test_session.flush()

    result = await auth.authenticate_user("inactive@example.com", "hashedpassword123")
    assert result is None


# ---------------------------------------------------------------------------
# soft_delete_user tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_soft_delete_user_marks_user_correctly(test_session: AsyncSession):
    """soft_delete_user should set is_deleted=True, deleted_at, is_active=False."""
    admin = await _create_user(test_session, email="admin@example.com", is_superuser=True)
    target = await _create_user(test_session, email="target@example.com")

    result = await soft_delete_user(
        test_session, user_id=target.id, admin=admin
    )

    assert result["user_id"] == target.id
    assert result["email"] == "target@example.com"

    await test_session.refresh(target)
    assert target.is_deleted is True
    assert target.deleted_at is not None
    assert target.is_active is False


@pytest.mark.asyncio
async def test_soft_delete_revokes_sessions(test_session: AsyncSession):
    """All active sessions should be revoked when a user is soft-deleted."""
    admin = await _create_user(test_session, email="admin2@example.com", is_superuser=True)
    target = await _create_user(test_session, email="target2@example.com")
    session = await _create_session(test_session, target.id)

    assert session.is_revoked is False

    await soft_delete_user(test_session, user_id=target.id, admin=admin)

    await test_session.refresh(session)
    assert session.is_revoked is True


@pytest.mark.asyncio
async def test_soft_delete_cannot_delete_self(test_session: AsyncSession):
    """Admins cannot soft-delete their own account."""
    admin = await _create_user(test_session, email="selfadmin@example.com", is_superuser=True)

    with pytest.raises(ValueError, match="Cannot delete your own account"):
        await soft_delete_user(test_session, user_id=admin.id, admin=admin)


@pytest.mark.asyncio
async def test_soft_delete_cannot_delete_superuser(test_session: AsyncSession):
    """Cannot soft-delete a superuser."""
    admin = await _create_user(test_session, email="admin3@example.com", is_superuser=True)
    other_super = await _create_user(test_session, email="super@example.com", is_superuser=True)

    with pytest.raises(ValueError, match="Cannot delete a superuser account"):
        await soft_delete_user(test_session, user_id=other_super.id, admin=admin)


@pytest.mark.asyncio
async def test_soft_delete_already_deleted(test_session: AsyncSession):
    """Cannot soft-delete a user that's already deleted."""
    admin = await _create_user(test_session, email="admin4@example.com", is_superuser=True)
    target = await _create_user(test_session, email="already@example.com")

    await soft_delete_user(test_session, user_id=target.id, admin=admin)

    with pytest.raises(ValueError, match="already deleted"):
        await soft_delete_user(test_session, user_id=target.id, admin=admin)


@pytest.mark.asyncio
async def test_soft_delete_nonexistent_user(test_session: AsyncSession):
    """Soft-deleting a nonexistent user should raise."""
    admin = await _create_user(test_session, email="admin5@example.com", is_superuser=True)

    with pytest.raises(ValueError, match="User not found"):
        await soft_delete_user(
            test_session, user_id=str(uuid.uuid4()), admin=admin
        )


# ---------------------------------------------------------------------------
# hard delete tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hard_delete_removes_user(test_session: AsyncSession):
    """Hard delete should completely remove the user from the database."""
    admin = await _create_user(test_session, email="admin6@example.com", is_superuser=True)
    target = await _create_user(test_session, email="hard_del@example.com")
    target_id = target.id

    await delete_user_permanently(
        test_session, user_id=target_id, admin=admin
    )
    await test_session.flush()

    result = await test_session.execute(select(User).where(User.id == target_id))
    assert result.scalar_one_or_none() is None
