"""`last_updated` reports the last sync, not the last price movement.

Now that the sync only writes rows whose values changed, max(updated_at) means
"when a price last moved". A quiet week upstream would make a freshly-checked
catalogue look a week stale, so the public field reads the sync log instead.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import ModelPricing, PricingSyncLog
from app.routes.pricing import _last_synced_at


@pytest.mark.asyncio
async def test_reports_the_latest_successful_sync(test_session: AsyncSession):
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    test_session.add_all([
        PricingSyncLog(source="litellm", status="ok",
                       created_at=datetime.now(timezone.utc) - timedelta(days=3)),
        PricingSyncLog(source="litellm", status="ok", created_at=recent),
    ])
    await test_session.commit()

    assert (await _last_synced_at(test_session)).replace(tzinfo=timezone.utc) == pytest.approx(
        recent, abs=timedelta(seconds=1)
    )


@pytest.mark.asyncio
async def test_failed_and_running_syncs_do_not_count(test_session: AsyncSession):
    """Only a completed sync proves the catalogue was refreshed."""
    good = datetime.now(timezone.utc) - timedelta(days=2)
    test_session.add_all([
        PricingSyncLog(source="litellm", status="ok", created_at=good),
        PricingSyncLog(source="litellm", status="error",
                       created_at=datetime.now(timezone.utc)),
        PricingSyncLog(source="litellm", status="running",
                       created_at=datetime.now(timezone.utc)),
    ])
    await test_session.commit()

    assert (await _last_synced_at(test_session)).replace(tzinfo=timezone.utc) == pytest.approx(
        good, abs=timedelta(seconds=1)
    )


@pytest.mark.asyncio
async def test_populated_catalogue_with_no_sync_log_reports_its_own_high_water_mark(
    test_session: AsyncSession,
):
    """A populated DB with no sync logged is stale-dated, not unknown -- None
    here rendered as "Never" on the public /docs/models page."""
    test_session.add(ModelPricing(model_name="m", input_price_per_1k=0.01,
                                  output_price_per_1k=0.02, provider="openai"))
    await test_session.commit()

    assert await _last_synced_at(test_session) is not None


@pytest.mark.asyncio
async def test_none_only_when_there_is_nothing_at_all(test_session: AsyncSession):
    """Empty catalogue and empty history genuinely has no date to report."""
    assert await _last_synced_at(test_session) is None
