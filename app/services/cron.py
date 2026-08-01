"""
Background Cron Jobs

Handles periodic tasks like purging expired soft-deleted users.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_session
from ..config import get_settings
from ..models.db_models import PricingSyncLog
from ..models.user_models import User
from .admin_service import delete_user_permanently

logger = logging.getLogger(__name__)

# How often the cron loop wakes. Shorter than the pricing interval so a host
# that sleeps and restarts gets several chances to notice a sync is due.
_CRON_TICK_SECONDS = 3600


async def purge_expired_soft_deletes(db: AsyncSession) -> int:
    """
    Hard-delete users whose grace period has expired.
    
    Returns:
        Number of users permanently deleted.
    """
    settings = get_settings()
    grace_days = settings.deletion_grace_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=grace_days)
    
    # Find all soft-deleted users past the cutoff
    query = select(User).where(
        User.is_deleted == True,
        User.deleted_at <= cutoff
    )
    result = await db.execute(query)
    expired_users = result.scalars().all()
    
    count = 0
    for user in expired_users:
        # Capture before the delete: after a rollback the instance is expired
        # and touching user.id would raise inside the error handler.
        user_id = str(user.id)
        try:
            # Skip superusers — they should never be purged automatically
            if user.is_superuser:
                logger.warning(f"Skipping purge of superuser {user_id} — superusers cannot be auto-deleted")
                continue

            await delete_user_permanently(db, user_id=user_id, admin=None)
            # Commit per user so one failure cannot discard the deletions that
            # already succeeded, and so `count` always matches what was durably
            # removed rather than what was merely attempted.
            await db.commit()
            count += 1
        except Exception as e:
            logger.error(f"Failed to purge user {user_id}: {e}")
            await db.rollback()

    if count > 0:
        logger.info(f"Purged {count} expired soft-deleted users")

    return count


async def _hours_since_last_pricing_sync(db: AsyncSession, source: str) -> Optional[float]:
    """Hours since the last successful sync, or None if there has never been one.

    Read from the database rather than process memory so the schedule survives
    restarts. A host that sleeps between requests restarts constantly; without
    durable state it would either re-sync ~3,500 models on every wake or never
    sync at all.
    """
    last = (await db.execute(
        select(PricingSyncLog.created_at)
        .where(PricingSyncLog.source == source, PricingSyncLog.status == "ok")
        .order_by(PricingSyncLog.created_at.desc())
        .limit(1)
    )).scalar_one_or_none()

    if last is None:
        return None
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds() / 3600


async def sync_pricing_if_due(db: AsyncSession) -> bool:
    """Refresh model pricing when the configured interval has elapsed.

    Returns True if a sync ran. Records the outcome in PricingSyncLog, which is
    also what schedules the next run.
    """
    settings = get_settings()
    interval = settings.pricing_sync_interval_hours
    if interval <= 0:
        return False

    elapsed = await _hours_since_last_pricing_sync(db, "litellm")
    if elapsed is not None and elapsed < interval:
        return False

    # Imported here so a pricing-service import error cannot stop the purge job.
    from .pricing_service import PricingService

    logger.info(
        "Pricing sync due (last run %s); syncing from LiteLLM",
        "never" if elapsed is None else f"{elapsed:.1f}h ago",
    )
    pricing_service = PricingService(db)
    started = datetime.now(timezone.utc)
    try:
        result = await pricing_service.sync_from_litellm(track_changes=False)
        duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        failed = result.get("status") == "error"

        db.add(PricingSyncLog(
            source="litellm",
            status="error" if failed else "ok",
            models_created=result.get("models_created", 0),
            models_updated=result.get("models_updated", 0),
            models_skipped=result.get("models_skipped", 0),
            error_message=result.get("error"),
            duration_ms=duration_ms,
        ))
        await db.commit()

        if failed:
            logger.warning("Pricing sync failed: %s", result.get("error"))
            return False

        logger.info(
            "Pricing sync complete: %d created, %d updated (%d ms)",
            result.get("models_created", 0),
            result.get("models_updated", 0),
            duration_ms,
        )
        return True
    finally:
        await pricing_service.close()


async def cron_loop():
    """
    Background task that runs periodic jobs.
    Wakes hourly; each job decides for itself whether it is due.
    """
    logger.info("Starting background cron loop")

    try:
        while True:
            # Each job gets its own session and its own error boundary, so one
            # failing job cannot stop the others from running.
            for job in (purge_expired_soft_deletes, sync_pricing_if_due):
                try:
                    async for db in get_db_session():
                        await job(db)
                        break  # get_db_session yields once
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Cron job %s failed: %s", job.__name__, e)

            await asyncio.sleep(_CRON_TICK_SECONDS)

    except asyncio.CancelledError:
        logger.info("Cron loop cancelled")
        raise
