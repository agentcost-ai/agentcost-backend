"""
Background Cron Jobs

Handles periodic tasks like purging expired soft-deleted users.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

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

# A "running" claim older than this is assumed to belong to a process that died
# mid-sync, so a crash cannot block pricing updates indefinitely. Comfortably
# above a normal full sync, which takes minutes.
_SYNC_STALE_AFTER_HOURS = 2.0


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


async def _last_sync(db: AsyncSession, source: str):
    """Most recent sync that should suppress a new one, with its age in hours.

    Read from the database rather than process memory so the schedule survives
    restarts. A host that sleeps between requests restarts constantly; without
    durable state it would either re-sync ~3,500 models on every wake or never
    sync at all.

    In-progress runs count. A full sync takes minutes, so considering only
    finished ones leaves that entire window open for a second run to start.
    """
    row = (await db.execute(
        select(PricingSyncLog)
        .where(
            PricingSyncLog.source == source,
            PricingSyncLog.status.in_(("ok", "running")),
        )
        .order_by(PricingSyncLog.created_at.desc())
        .limit(1)
    )).scalar_one_or_none()

    if row is None:
        return None, None
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return row, (datetime.now(timezone.utc) - created).total_seconds() / 3600


async def claim_pricing_sync(
    db: AsyncSession, source: str = "litellm", admin_id=None
) -> "PricingSyncLog | None":
    """Insert and commit a 'running' claim, or return None if a live one exists.

    Shared by cron and the admin sync routes so a manual sync and the scheduled
    one cannot run the same multi-minute job concurrently. A stale claim (older
    than _SYNC_STALE_AFTER_HOURS) is treated as a dead process and superseded.
    """
    last, elapsed = await _last_sync(db, source)
    if (
        last is not None
        and last.status == "running"
        and elapsed < _SYNC_STALE_AFTER_HOURS
    ):
        return None
    entry = PricingSyncLog(source=source, status="running", admin_id=admin_id)
    db.add(entry)
    await db.commit()
    return entry


async def sync_pricing_if_due(db: AsyncSession) -> bool:
    """Refresh model pricing when the configured interval has elapsed.

    Returns True if a sync ran. Records the outcome in PricingSyncLog, which is
    also what schedules the next run.
    """
    settings = get_settings()
    interval = settings.pricing_sync_interval_hours
    if interval <= 0:
        return False

    last, elapsed = await _last_sync(db, "litellm")
    if last is not None:
        if last.status == "running":
            # Someone else is mid-sync. Only step in once the run is old enough
            # that the process running it has certainly died, so a crash cannot
            # block syncing forever.
            if elapsed < _SYNC_STALE_AFTER_HOURS:
                return False
            logger.warning(
                "Previous pricing sync has been running %.1fh; treating it as dead",
                elapsed,
            )
        elif elapsed < interval:
            return False

    # Imported here so a pricing-service import error cannot stop the purge job.
    from .pricing_service import PricingService

    # Claim the slot BEFORE the work (see claim_pricing_sync); an admin-run
    # sync may have claimed it between our _last_sync read and now.
    entry = await claim_pricing_sync(db)
    if entry is None:
        return False

    logger.info(
        "Pricing sync due (last run %s); syncing from LiteLLM",
        "never" if last is None else f"{elapsed:.1f}h ago",
    )
    pricing_service = PricingService(db)
    started = datetime.now(timezone.utc)
    try:
        result = await pricing_service.sync_from_litellm(track_changes=False)
        failed = result.get("status") == "error"

        entry.status = "error" if failed else "ok"
        entry.models_created = result.get("models_created", 0)
        entry.models_updated = result.get("models_updated", 0)
        entry.models_skipped = result.get("models_skipped", 0)
        entry.error_message = result.get("error")
        entry.duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        await db.commit()

        if failed:
            logger.warning("Pricing sync failed: %s", result.get("error"))
            return False

        logger.info(
            "Pricing sync complete: %d created, %d updated (%d ms)",
            entry.models_created, entry.models_updated, entry.duration_ms,
        )
        return True
    except BaseException as exc:
        # Includes CancelledError from a shutdown mid-sync. Mark the claim
        # failed rather than leaving a "running" row to expire on its own.
        entry.status = "error"
        entry.error_message = f"{type(exc).__name__}: {exc}"[:500]
        entry.duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        try:
            await db.commit()
        except Exception:
            await db.rollback()
        raise
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
