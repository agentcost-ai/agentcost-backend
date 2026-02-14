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
from ..models.user_models import User
from .admin_service import delete_user_permanently

logger = logging.getLogger(__name__)


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
        try:
            # Skip superusers — they should never be purged automatically
            if user.is_superuser:
                logger.warning(f"Skipping purge of superuser {user.id} — superusers cannot be auto-deleted")
                continue
            
            await delete_user_permanently(db, user_id=str(user.id), admin=None)
            count += 1
        except Exception as e:
            logger.error(f"Failed to purge user {user.id}: {e}")
            # Rollback to prevent dirty session state from affecting subsequent deletes
            await db.rollback()
            
    if count > 0:
        await db.commit()
        logger.info(f"Purged {count} expired soft-deleted users")
        
    return count


async def cron_loop():
    """
    Background task that runs periodic jobs.
    Runs every 6 hours.
    """
    logger.info("Starting background cron loop")
    
    try:
        while True:
            # Run startup/periodic checks
            try:
                # Use a fresh session for each run
                async for db in get_db_session():
                    await purge_expired_soft_deletes(db)
                    break # get_db_session yields once
            except Exception as e:
                logger.error(f"Error in cron loop: {e}")
            
            # Sleep for 6 hours
            await asyncio.sleep(6 * 3600)
            
    except asyncio.CancelledError:
        logger.info("Cron loop cancelled")
        raise
