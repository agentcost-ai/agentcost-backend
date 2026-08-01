"""
Tests for the grace period deletion flow.

Validates:
1. Reactivation: Soft-deleted user logging in within 7 days -> access granted + restored.
2. Expiry: Soft-deleted user logging in after 7 days -> access denied.
3. Purge: Background job correctly removes only expired users.
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user_models import User
from app.services.auth_service import AuthService, hash_password
from app.services.admin_service import soft_delete_user
from app.services.cron import purge_expired_soft_deletes
from app.config import get_settings


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


# ---------------------------------------------------------------------------
# Auth Reactivation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reactivation_within_grace_period(test_session: AsyncSession):
    """User should be automatically reactivated if logging in within grace period."""
    # 1. Setup
    admin = await _create_user(test_session, email="admin@example.com", is_superuser=True)
    user = await _create_user(test_session, email="grace@example.com")
    auth = AuthService(test_session)
    
    # 2. Soft-delete the user
    await soft_delete_user(test_session, user_id=user.id, admin=admin)
    await test_session.commit() # Commit to ensure state is saved
    
    # 3. Authenticate (should succeed because < 7 days)
    # Note: soft_delete_user sets deleted_at = now, so it's definitely within 7 days
    result = await auth.authenticate_user("grace@example.com", "hashedpassword123")
    
    assert result is not None, "Login should succeed within grace period"
    assert result.id == user.id
    
    # 4. Verify user is restored
    await test_session.refresh(user)
    assert user.is_deleted is False
    assert user.deleted_at is None
    assert user.is_active is True


@pytest.mark.asyncio
async def test_login_blocked_after_grace_period(test_session: AsyncSession):
    """User should be blocked if logging in after grace period expired."""
    # 1. Setup
    admin = await _create_user(test_session, email="admin2@example.com", is_superuser=True)
    user = await _create_user(test_session, email="expired@example.com")
    auth = AuthService(test_session)
    
    # 2. Soft-delete normally
    await soft_delete_user(test_session, user_id=user.id, admin=admin)
    
    # 3. Manually age the deleted_at timestamp to 8 days ago
    settings = get_settings()
    past_date = datetime.now(timezone.utc) - timedelta(days=settings.deletion_grace_days + 1)
    user.deleted_at = past_date
    await test_session.commit()
    
    # 4. Authenticate (should fail)
    result = await auth.authenticate_user("expired@example.com", "hashedpassword123")
    
    assert result is None, "Login should fail after grace period"
    
    # 5. Verify user remains deleted
    await test_session.refresh(user)
    assert user.is_deleted is True
    # 5. Verify user remains deleted
    await test_session.refresh(user)
    assert user.is_deleted is True
    # Force naive datetime from DB to be treated as UTC
    deleted_at_utc = user.deleted_at.replace(tzinfo=timezone.utc)
    assert abs(deleted_at_utc.timestamp() - past_date.timestamp()) < 1.0


# ---------------------------------------------------------------------------
# Cron Job Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_purge_expired_soft_deletes_logic(test_session: AsyncSession):
    """Purge function should only remove users past the grace period."""
    # 1. Setup
    settings = get_settings()
    now = datetime.now(timezone.utc)
    
    # User A: Deleted just now (Safe)
    user_safe = await _create_user(test_session, email="safe@example.com")
    user_safe.is_deleted = True
    user_safe.deleted_at = now
    user_safe.is_active = False
    
    # User B: Deleted 8 days ago (Expired)
    user_expired = await _create_user(test_session, email="gone@example.com")
    user_expired.is_deleted = True
    user_expired.deleted_at = now - timedelta(days=settings.deletion_grace_days + 1)
    user_expired.is_active = False
    
    # Safe user C: Not deleted (Control)
    user_active = await _create_user(test_session, email="active@example.com")
    
    await test_session.commit()
    
    # 2. Run purge
    deleted_count = await purge_expired_soft_deletes(test_session)
    
    # 3. Verify
    assert deleted_count == 1, "Should have deleted exactly 1 user"
    
    # Check User B is gone
    result_expired = await test_session.execute(select(User).where(User.email == "gone@example.com"))
    assert result_expired.scalar_one_or_none() is None
    
    # Check User A still exists
    result_safe = await test_session.execute(select(User).where(User.email == "safe@example.com"))
    u_safe = result_safe.scalar_one_or_none()
    assert u_safe is not None
    assert u_safe.is_deleted is True
    
    # Check User C still exists
    result_active = await test_session.execute(select(User).where(User.email == "active@example.com"))
    assert result_active.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_unattended_purge_writes_a_null_admin_id(test_session: AsyncSession):
    """An unattended purge must leave a usable audit row.

    admin_id is a FK to users.id. Writing a sentinel string satisfied SQLite
    but violated the constraint on PostgreSQL, so every scheduled purge aborted
    while the logs reported success. conftest now enables PRAGMA foreign_keys,
    so a regression here fails instead of passing silently.
    """
    from app.models.db_models import AdminActivityLog

    settings = get_settings()
    user = await _create_user(test_session, email="purge-audit@example.com")
    user.is_deleted = True
    user.deleted_at = datetime.now(timezone.utc) - timedelta(
        days=settings.deletion_grace_days + 1
    )
    user.is_active = False
    await test_session.commit()

    assert await purge_expired_soft_deletes(test_session) == 1

    entry = (await test_session.execute(
        select(AdminActivityLog).where(AdminActivityLog.action_type == "user_deleted")
    )).scalars().first()

    assert entry is not None, "purge left no audit trail"
    assert entry.admin_id is None
    assert entry.details.get("actor") == "system"
