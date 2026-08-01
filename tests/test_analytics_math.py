"""
Regression tests for analytics/report math that used to be quietly wrong:

* cost shares divided by a top-N subtotal instead of real spend
* Pareto quoting the top-N slice as the total model count
* p95/p99 taken from the fastest rows after an ascending LIMIT
* time buckets that depended on the DB session timezone
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import Event
from app.services.analytics_service import AnalyticsService
from app.services.report_service import ReportService


def _event(project_id: str, *, model="gpt-4", agent="agent-a", cost=0.01,
           latency_ms=100, timestamp=None, success=True) -> Event:
    return Event(
        id=str(uuid.uuid4()),
        project_id=project_id,
        agent_name=agent,
        model=model,
        input_tokens=100,
        output_tokens=100,
        total_tokens=200,
        cost=cost,
        latency_ms=latency_ms,
        success=success,
        timestamp=timestamp or datetime.now(timezone.utc),
    )


async def _add_all(session: AsyncSession, events) -> None:
    session.add_all(events)
    await session.commit()


# ── Cost share ────────────────────────────────────────────────────────────


async def test_model_cost_share_is_against_full_window_spend(
    test_session: AsyncSession, test_project
):
    """Top-N shares must not be renormalized to 100% over the truncated slice."""
    # 12 models: the three biggest hold 30 of 78 total cost units.
    events = [
        _event(test_project.id, model=f"model-{i:02d}", cost=float(i))
        for i in range(1, 13)
    ]
    await _add_all(test_session, events)

    analytics = AnalyticsService(test_session)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    start = end - timedelta(days=1)

    assert await analytics.get_distinct_model_count(test_project.id, start, end) == 12

    models = await analytics.get_model_stats(test_project.id, start, end, limit=3)
    assert [m.model for m in models] == ["model-12", "model-11", "model-10"]

    # 12 + 11 + 10 = 33 of 78 → shares must sum to ~42%, not 100%.
    assert sum(m.cost_share for m in models) == pytest.approx(42.3, abs=0.2)
    assert models[0].cost_share == pytest.approx(12 / 78 * 100, abs=0.1)


async def test_model_cost_share_totals_100_when_nothing_is_truncated(
    test_session: AsyncSession, test_project
):
    events = [
        _event(test_project.id, model="a", cost=3.0),
        _event(test_project.id, model="b", cost=1.0),
    ]
    await _add_all(test_session, events)

    analytics = AnalyticsService(test_session)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    models = await analytics.get_model_stats(test_project.id, end - timedelta(days=1), end)

    assert sum(m.cost_share for m in models) == pytest.approx(100.0, abs=0.1)


async def test_empty_window_reports_zero_success_rate(
    test_session: AsyncSession, test_project
):
    """No calls means no successes -- 100% made empty windows look healthy."""
    analytics = AnalyticsService(test_session)
    end = datetime.now(timezone.utc)
    overview = await analytics.get_overview(test_project.id, end - timedelta(days=1), end)

    assert overview.total_calls == 0
    assert overview.success_rate == 0.0


# ── Pareto ────────────────────────────────────────────────────────────────


async def test_pareto_counts_every_model_not_just_the_top_slice(
    test_session: AsyncSession, test_project
):
    events = [
        _event(test_project.id, model=f"model-{i:02d}", cost=float(i))
        for i in range(1, 13)
    ]
    await _add_all(test_session, events)

    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    start = end - timedelta(days=1)
    report = await ReportService(test_session).build_report(
        project=test_project,
        start=start,
        end=end,
        prev_start=start - timedelta(days=1),
        prev_end=start,
        top_n=3,
        range_label="24h",
        is_custom_range=False,
    )

    # 12 models exist; only 3 were listed.
    assert len(report.models) == 3
    assert report.model_pareto.total_models == 12
    # The 3 listed models cover 33/78 of spend, so they cannot claim 80%.
    assert report.model_pareto.top_share == pytest.approx(42.3, abs=0.2)
    assert report.model_pareto.top_count == 3


async def test_agent_and_model_shares_use_the_same_denominator(
    test_session: AsyncSession, test_project
):
    """The report used to show agent shares ~60% beside model shares at 100%."""
    events = []
    for i in range(1, 7):
        events.append(
            _event(test_project.id, model=f"model-{i}", agent=f"agent-{i}", cost=float(i))
        )
    await _add_all(test_session, events)

    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    start = end - timedelta(days=1)
    report = await ReportService(test_session).build_report(
        project=test_project,
        start=start,
        end=end,
        prev_start=start - timedelta(days=1),
        prev_end=start,
        top_n=2,
        range_label="24h",
        is_custom_range=False,
    )

    model_share = sum(m.cost_share for m in report.models)
    agent_share = sum(report.agent_cost_share.values())
    assert model_share == pytest.approx(agent_share, abs=0.2)
    assert model_share < 100.0


# ── Latency percentiles ───────────────────────────────────────────────────


async def test_percentiles_are_exact_over_the_whole_set(
    test_session: AsyncSession, test_project
):
    """p50/p95/p99 match percentile_cont's linear interpolation."""
    await _add_all(
        test_session,
        [_event(test_project.id, latency_ms=i) for i in range(1, 1001)],
    )

    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    start = end - timedelta(days=1)
    latency = await ReportService(test_session)._latency(test_project.id, start, end)

    # rank = p * (n - 1) over 1..1000
    assert latency.sample_size == 1000
    assert latency.p50 == pytest.approx(500.5, abs=0.01)
    assert latency.p95 == pytest.approx(950.05, abs=0.01)
    assert latency.p99 == pytest.approx(990.01, abs=0.01)
    assert latency.avg == pytest.approx(500.5, abs=0.01)


async def test_percentiles_see_the_slow_tail(test_session: AsyncSession, test_project):
    """The old ascending LIMIT discarded exactly these rows."""
    events = [_event(test_project.id, latency_ms=100) for _ in range(990)]
    events += [_event(test_project.id, latency_ms=5000) for _ in range(10)]
    await _add_all(test_session, events)

    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    start = end - timedelta(days=1)
    latency = await ReportService(test_session)._latency(test_project.id, start, end)

    assert latency.p50 == pytest.approx(100.0)
    assert latency.p95 == pytest.approx(100.0)
    assert latency.p99 > 100.0  # tail is visible
    assert latency.sample_size == 1000


# ── UTC bucketing ─────────────────────────────────────────────────────────


async def test_day_buckets_follow_utc_midnight(test_session: AsyncSession, test_project):
    """Buckets are cut on UTC day boundaries and labelled as UTC."""
    late = datetime(2026, 3, 10, 23, 30, tzinfo=timezone.utc)
    early = datetime(2026, 3, 11, 0, 30, tzinfo=timezone.utc)
    await _add_all(
        test_session,
        [
            _event(test_project.id, timestamp=late),
            _event(test_project.id, timestamp=early),
        ],
    )

    analytics = AnalyticsService(test_session)
    points = await analytics.get_timeseries(
        test_project.id,
        datetime(2026, 3, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 12, tzinfo=timezone.utc),
        granularity="day",
    )

    assert [p.timestamp.date().isoformat() for p in points] == ["2026-03-10", "2026-03-11"]
    assert all(p.timestamp.tzinfo is not None for p in points)
    assert all(p.calls == 1 for p in points)


async def test_hour_buckets_follow_utc_hours(test_session: AsyncSession, test_project):
    stamps = [
        datetime(2026, 3, 10, 22, 5, tzinfo=timezone.utc),
        datetime(2026, 3, 10, 22, 55, tzinfo=timezone.utc),
        datetime(2026, 3, 10, 23, 5, tzinfo=timezone.utc),
    ]
    await _add_all(
        test_session, [_event(test_project.id, timestamp=s) for s in stamps]
    )

    analytics = AnalyticsService(test_session)
    points = await analytics.get_timeseries(
        test_project.id,
        datetime(2026, 3, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 11, tzinfo=timezone.utc),
        granularity="hour",
    )

    assert [(p.timestamp.hour, p.calls) for p in points] == [(22, 2), (23, 1)]


async def test_cadence_buckets_are_utc(test_session: AsyncSession, test_project):
    """Busiest hour/day are reported in UTC, matching the UTC window bounds."""
    # 2026-03-10 is a Tuesday in UTC; 21:00 UTC is already Wednesday east of +03.
    stamps = [datetime(2026, 3, 10, 21, m, tzinfo=timezone.utc) for m in (0, 10, 20)]
    await _add_all(
        test_session, [_event(test_project.id, timestamp=s) for s in stamps]
    )

    cadence = await ReportService(test_session)._cadence(
        test_project.id,
        datetime(2026, 3, 9, tzinfo=timezone.utc),
        datetime(2026, 3, 12, tzinfo=timezone.utc),
    )

    assert cadence.busiest_hour == "21:00"
    assert cadence.busiest_day == "Tuesday"
