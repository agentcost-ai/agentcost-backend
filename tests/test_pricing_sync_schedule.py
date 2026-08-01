"""Background pricing sync scheduling.

The schedule lives in pricing_sync_log rather than process memory: a host that
sleeps between requests restarts constantly, and in-memory state would make it
either re-sync ~3,500 models on every wake or never sync at all.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.db_models import PricingSyncLog
from app.services import cron


@pytest.fixture
def fake_sync(monkeypatch):
    """Replace the network sync with a counter."""
    calls = {"n": 0}

    class _FakeService:
        def __init__(self, db):
            pass

        async def sync_from_litellm(self, track_changes=False):
            calls["n"] += 1
            return {"status": "ok", "models_created": 3, "models_updated": 7,
                    "models_skipped": 0}

        async def close(self):
            pass

    import app.services.pricing_service as pricing_module
    monkeypatch.setattr(pricing_module, "PricingService", _FakeService)
    return calls


@pytest.mark.asyncio
async def test_first_run_syncs_and_records_the_schedule(test_session: AsyncSession, fake_sync):
    """With no history the sync runs, and its log row is what schedules the next."""
    assert await cron.sync_pricing_if_due(test_session) is True
    assert fake_sync["n"] == 1

    entry = (await test_session.execute(
        select(PricingSyncLog).where(PricingSyncLog.source == "litellm")
    )).scalars().first()
    assert entry is not None
    assert entry.status == "ok"
    assert entry.models_updated == 7


@pytest.mark.asyncio
async def test_restart_does_not_resync_before_the_interval(test_session: AsyncSession, fake_sync):
    """The sleeping-host case: repeated wakes must not restart a full sync."""
    test_session.add(PricingSyncLog(
        source="litellm", status="ok",
        created_at=datetime.now(timezone.utc) - timedelta(hours=1),
    ))
    await test_session.commit()

    for _ in range(5):  # five wake-ups inside the interval
        assert await cron.sync_pricing_if_due(test_session) is False
    assert fake_sync["n"] == 0


@pytest.mark.asyncio
async def test_syncs_again_once_the_interval_has_elapsed(test_session: AsyncSession, fake_sync):
    settings = get_settings()
    test_session.add(PricingSyncLog(
        source="litellm", status="ok",
        created_at=datetime.now(timezone.utc)
        - timedelta(hours=settings.pricing_sync_interval_hours + 1),
    ))
    await test_session.commit()

    assert await cron.sync_pricing_if_due(test_session) is True
    assert fake_sync["n"] == 1


@pytest.mark.asyncio
async def test_a_failed_sync_does_not_count_as_scheduled(test_session: AsyncSession, fake_sync):
    """Only successful runs push the schedule forward, so failures retry."""
    test_session.add(PricingSyncLog(
        source="litellm", status="error",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    ))
    await test_session.commit()

    assert await cron.sync_pricing_if_due(test_session) is True
    assert fake_sync["n"] == 1


@pytest.mark.asyncio
async def test_zero_interval_disables_background_sync(test_session: AsyncSession, fake_sync, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "pricing_sync_interval_hours", 0)

    assert await cron.sync_pricing_if_due(test_session) is False
    assert fake_sync["n"] == 0
