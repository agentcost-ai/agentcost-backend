"""
Tests for BudgetService and budget enforcement on the ingestion route.

These tests exercise the service directly against the in-memory SQLite
fixture defined in ``conftest.py`` so they remain fast and deterministic.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import (
    BudgetThresholdAlert,
    Event,
    Notification,
    Project,
)
from app.models.user_models import ProjectMember, User, UserRole
from app.services.budget_service import BudgetService
from app.services.currency_service import CurrencyService


# ─────────────────────────── helpers ───────────────────────────


async def _seed_event(
    session: AsyncSession,
    project_id: str,
    *,
    cost: float,
    when: datetime | None = None,
) -> Event:
    event = Event(
        id=str(uuid.uuid4()),
        project_id=project_id,
        agent_name="t",
        model="gpt-4",
        input_tokens=10,
        output_tokens=10,
        total_tokens=20,
        cost=cost,
        latency_ms=100,
        timestamp=when or datetime.now(timezone.utc),
        success=True,
    )
    session.add(event)
    await session.flush()
    return event


async def _make_user(session: AsyncSession, *, email: str, name: str = "Owner") -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        name=name,
        is_active=True,
        email_verified=True,
    )
    session.add(user)
    await session.flush()
    return user


# ─────────────────────────── pure helpers ───────────────────────────


def test_normalize_thresholds_defaults_when_empty():
    assert BudgetService.normalize_thresholds(None) == [50.0, 80.0, 100.0]
    assert BudgetService.normalize_thresholds([]) == [50.0, 80.0, 100.0]
    assert BudgetService.normalize_thresholds("not-a-list") == [50.0, 80.0, 100.0]


def test_normalize_thresholds_dedup_and_sort():
    assert BudgetService.normalize_thresholds([80, 50, 100, 80, 50]) == [50.0, 80.0, 100.0]


def test_normalize_thresholds_filters_out_of_range():
    # 0 and 150 are out of range; rest survive
    assert BudgetService.normalize_thresholds([0, 25, 150, 99]) == [25.0, 99.0]


def test_period_key_format():
    assert BudgetService._period_key(datetime(2026, 3, 22, tzinfo=timezone.utc)) == "2026-03"
    assert BudgetService._period_key(datetime(2026, 12, 1, tzinfo=timezone.utc)) == "2026-12"


def test_month_window_handles_year_rollover():
    start, end = BudgetService._month_window(datetime(2026, 12, 15, tzinfo=timezone.utc))
    assert start == datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert end == datetime(2027, 1, 1, tzinfo=timezone.utc)


# ─────────────────────────── evaluate() ───────────────────────────


@pytest.mark.asyncio
async def test_evaluate_disabled_when_no_budget(test_session, test_project):
    test_project.monthly_budget_usd = None
    await test_session.commit()

    result = await BudgetService(test_session).evaluate(test_project)
    assert result["enabled"] is False
    assert result["mode"] == "off"
    assert result["should_block"] is False
    assert result["budget"] is None
    assert result["utilization_percent"] is None
    assert result["crossed_thresholds"] == []


@pytest.mark.asyncio
async def test_evaluate_reports_utilization_and_crossings(test_session, test_project):
    test_project.monthly_budget_usd = 100.0
    test_project.budget_enforcement_mode = "warn"
    test_project.budget_alert_thresholds = [50.0, 80.0, 100.0]
    await test_session.commit()

    await _seed_event(test_session, test_project.id, cost=60.0)

    result = await BudgetService(test_session).evaluate(test_project)

    assert result["enabled"] is True
    assert result["budget"] == 100.0
    assert result["current_spend"] == 60.0
    assert result["projected_spend"] == 60.0
    assert result["utilization_percent"] == 60.0
    assert result["crossed_thresholds"] == [50.0]
    assert result["should_block"] is False
    assert result["mode"] == "warn"


@pytest.mark.asyncio
async def test_evaluate_should_block_only_in_hard_cap(test_session, test_project):
    test_project.monthly_budget_usd = 10.0
    test_project.budget_enforcement_mode = "warn"
    test_project.budget_alert_thresholds = [50.0, 100.0]
    await test_session.commit()
    await _seed_event(test_session, test_project.id, cost=10.0)

    warn_eval = await BudgetService(test_session).evaluate(test_project)
    assert warn_eval["utilization_percent"] == 100.0
    assert warn_eval["should_block"] is False  # warn mode never blocks

    test_project.budget_enforcement_mode = "hard_cap"
    await test_session.commit()

    cap_eval = await BudgetService(test_session).evaluate(test_project)
    assert cap_eval["should_block"] is True
    assert 100.0 in cap_eval["crossed_thresholds"]


@pytest.mark.asyncio
async def test_evaluate_projects_with_additional_cost(test_session, test_project):
    test_project.monthly_budget_usd = 100.0
    test_project.budget_enforcement_mode = "hard_cap"
    test_project.budget_alert_thresholds = [80.0, 100.0]
    await test_session.commit()
    await _seed_event(test_session, test_project.id, cost=70.0)

    # An incoming batch that would push us over the budget must be blocked.
    eval_with_pending = await BudgetService(test_session).evaluate(
        test_project, additional_cost=40.0
    )
    assert eval_with_pending["projected_spend"] == 110.0
    assert eval_with_pending["utilization_percent"] == 110.0
    assert eval_with_pending["should_block"] is True


# ─────────────────────────── record_threshold_crossings ───────────────────────────


@pytest.mark.asyncio
async def test_record_threshold_crossings_is_deduplicated(test_session, test_project):
    service = BudgetService(test_session)

    first = await service.record_threshold_crossings(
        project_id=test_project.id,
        period_key="2026-03",
        crossed_thresholds=[50.0, 80.0],
        spent_amount=80.0,
        budget_amount=100.0,
        utilization_percent=80.0,
        dispatch_notifications=False,
    )
    assert sorted(first) == [50.0, 80.0]

    # Second call with overlap should only insert the new (100) threshold.
    second = await service.record_threshold_crossings(
        project_id=test_project.id,
        period_key="2026-03",
        crossed_thresholds=[50.0, 80.0, 100.0],
        spent_amount=100.0,
        budget_amount=100.0,
        utilization_percent=100.0,
        dispatch_notifications=False,
    )
    assert second == [100.0]

    rows = (
        await test_session.execute(
            select(BudgetThresholdAlert).where(
                BudgetThresholdAlert.project_id == test_project.id
            )
        )
    ).scalars().all()
    assert {r.threshold_percent for r in rows} == {50.0, 80.0, 100.0}


@pytest.mark.asyncio
async def test_record_threshold_crossings_notifies_owner_and_admin(
    test_session, test_project
):
    owner = await _make_user(test_session, email="owner@example.com", name="Owner")
    admin = await _make_user(test_session, email="admin@example.com", name="Admin")
    viewer = await _make_user(test_session, email="viewer@example.com", name="Viewer")

    test_project.owner_id = owner.id
    test_project.name = "Production"
    test_project.monthly_budget_usd = 100.0
    test_project.budget_enforcement_mode = "warn"
    test_session.add(
        ProjectMember(
            project_id=test_project.id,
            user_id=admin.id,
            role=UserRole.ADMIN.value,
            accepted_at=datetime.now(timezone.utc),
        )
    )
    test_session.add(
        ProjectMember(
            project_id=test_project.id,
            user_id=viewer.id,
            role=UserRole.VIEWER.value,
            accepted_at=datetime.now(timezone.utc),
        )
    )
    await test_session.commit()

    service = BudgetService(test_session)
    await service.record_threshold_crossings(
        project_id=test_project.id,
        period_key="2026-03",
        crossed_thresholds=[80.0],
        spent_amount=80.0,
        budget_amount=100.0,
        utilization_percent=80.0,
        project=test_project,
    )

    notifs = (
        await test_session.execute(select(Notification))
    ).scalars().all()

    recipients = {n.user_id for n in notifs}
    assert owner.id in recipients
    assert admin.id in recipients
    assert viewer.id not in recipients  # viewers are not alerted

    for n in notifs:
        assert n.type == "budget_threshold"
        assert n.severity == "warning"
        assert n.project_id == test_project.id
        assert n.payload["thresholds_crossed"] == [80.0]
        assert n.payload["period_key"] == "2026-03"


@pytest.mark.asyncio
async def test_record_threshold_crossings_critical_on_hard_cap(
    test_session, test_project
):
    owner = await _make_user(test_session, email="owner2@example.com", name="Owner2")
    test_project.owner_id = owner.id
    test_project.monthly_budget_usd = 100.0
    test_project.budget_enforcement_mode = "hard_cap"
    await test_session.commit()

    service = BudgetService(test_session)
    await service.record_threshold_crossings(
        project_id=test_project.id,
        period_key="2026-03",
        crossed_thresholds=[100.0],
        spent_amount=120.0,
        budget_amount=100.0,
        utilization_percent=120.0,
        project=test_project,
    )

    notif = (
        await test_session.execute(
            select(Notification).where(Notification.user_id == owner.id)
        )
    ).scalar_one()

    assert notif.type == "budget_hard_cap"
    assert notif.severity == "critical"
    assert "hard cap" in notif.title.lower()


@pytest.fixture(autouse=True)
def stub_currency_fx(monkeypatch):
    """
    Use a deterministic FX rate in tests so we don't hit the network.
    1 USD = 83 INR keeps the math easy to verify (independent of live FX).
    """

    async def _fake_usd_to(currency):
        cur = (currency or "USD").upper()
        if cur == "INR":
            return 83.0
        return 1.0

    # Reset the process-wide cache to avoid bleed-through across tests
    CurrencyService._cache.clear()
    monkeypatch.setattr(CurrencyService, "usd_to", classmethod(lambda cls, c: _fake_usd_to(c)))


def test_currency_service_normalize_and_symbol():
    assert CurrencyService.normalize(None) == "USD"
    assert CurrencyService.normalize("inr") == "INR"
    assert CurrencyService.normalize("eur") == "USD"  # unsupported -> USD
    assert CurrencyService.symbol("INR") == "₹"
    assert CurrencyService.symbol("USD") == "$"


def test_currency_service_format_amount():
    assert CurrencyService.format_amount(1234.5, "USD") == "$1,234.50"
    assert CurrencyService.format_amount(1234.5, "INR") == "₹1,234.50"


@pytest.mark.asyncio
async def test_evaluate_converts_usd_spend_to_inr(test_session, test_project):
    # ₹4,000 monthly budget in INR
    test_project.monthly_budget_usd = 4000.0
    test_project.budget_currency = "INR"
    test_project.budget_enforcement_mode = "warn"
    test_project.budget_alert_thresholds = [50.0, 100.0]
    await test_session.commit()

    # $24 USD of spend -> 24 * 83 = ₹1,992 (49.8% of ₹4,000)
    await _seed_event(test_session, test_project.id, cost=24.0)

    result = await BudgetService(test_session).evaluate(test_project)

    assert result["currency"] == "INR"
    assert result["fx_rate"] == 83.0
    assert result["current_spend_usd"] == 24.0
    assert result["current_spend"] == pytest.approx(1992.0)
    assert result["budget"] == 4000.0
    assert result["utilization_percent"] == pytest.approx(49.8, rel=0.01)
    assert result["crossed_thresholds"] == []  # 49.8% < 50%


@pytest.mark.asyncio
async def test_evaluate_hard_cap_triggers_in_user_currency(test_session, test_project):
    # ₹4,000 INR budget, hard-cap mode
    test_project.monthly_budget_usd = 4000.0
    test_project.budget_currency = "INR"
    test_project.budget_enforcement_mode = "hard_cap"
    test_project.budget_alert_thresholds = [100.0]
    await test_session.commit()

    # $50 USD spend = ₹4,150 -> 103.75% utilization, hard cap breached
    await _seed_event(test_session, test_project.id, cost=50.0)

    result = await BudgetService(test_session).evaluate(test_project)

    assert result["should_block"] is True
    assert result["utilization_percent"] > 100
    assert 100.0 in result["crossed_thresholds"]


@pytest.mark.asyncio
async def test_evaluate_defaults_to_usd_when_currency_unset(test_session, test_project):
    """Project with no explicit currency falls back to USD, no FX conversion."""
    test_project.monthly_budget_usd = 100.0
    test_project.budget_enforcement_mode = "warn"
    # budget_currency keeps its DB default ('USD') — exercise the normal path
    await test_session.commit()
    await _seed_event(test_session, test_project.id, cost=50.0)

    result = await BudgetService(test_session).evaluate(test_project)
    assert result["currency"] == "USD"
    assert result["fx_rate"] == 1.0
    assert result["current_spend"] == 50.0
    assert result["current_spend_usd"] == 50.0


@pytest.mark.asyncio
async def test_record_threshold_crossings_noop_when_empty(test_session, test_project):
    service = BudgetService(test_session)
    inserted = await service.record_threshold_crossings(
        project_id=test_project.id,
        period_key="2026-03",
        crossed_thresholds=[],
        spent_amount=10.0,
        budget_amount=100.0,
        utilization_percent=10.0,
        dispatch_notifications=False,
    )
    assert inserted == []
    rows = (
        await test_session.execute(select(BudgetThresholdAlert))
    ).scalars().all()
    assert rows == []
