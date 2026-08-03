"""
Query-count regression tests for the optimization analyzers.

Each analyzer used to issue queries per (agent, model) group -- two for
capability inference, one per row for baselines -- so a project with 20 groups
spent 100+ round trips on a single /v1/optimizations request. These tests pin
the cost as flat in the number of groups rather than asserting a magic number.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import Event, ModelPricing, ProjectBaseline
from app.services.optimization_service import OptimizationService
from app.services.pricing_service import PricingService


CALLS_PER_GROUP = 12  # above the analyzers' 10-call cutoff


def _events_for_groups(project_id: str, groups: int, calls: int = CALLS_PER_GROUP):
    """`groups` distinct (agent, model) pairs, each above the analyzer cutoffs."""
    now = datetime.now(timezone.utc)
    rows = []
    for g in range(groups):
        for c in range(calls):
            rows.append(
                Event(
                    id=str(uuid.uuid4()),
                    project_id=project_id,
                    agent_name=f"agent-{g}",
                    model=f"model-{g}",
                    input_tokens=1000,
                    output_tokens=1000,
                    total_tokens=2000,
                    cost=0.05,
                    latency_ms=900,
                    # A third of the calls fail, so the error analyzer engages.
                    success=(c % 3 != 0),
                    error=None if c % 3 else "boom",
                    timestamp=now - timedelta(hours=1),
                    extra_data={"tools": [{"name": "search"}]},
                )
            )
    return rows


def _pricing_for_groups(groups: int, start: int = 0):
    """Priced models, as production has after a LiteLLM sync."""
    return [
        ModelPricing(
            model_name=f"model-{g}",
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
            provider="openai",
            is_active=True,
        )
        for g in range(start, groups)
    ]


def _baselines_for_groups(project_id: str, groups: int):
    return [
        ProjectBaseline(
            project_id=project_id,
            agent_name=f"agent-{g}",
            model=f"model-{g}",
            avg_error_rate=0.01,
            avg_latency_ms=100.0,
            stddev_latency_ms=10.0,
            p95_latency_ms=150.0,
            last_calculated_at=datetime.now(timezone.utc),
        )
        for g in range(groups)
    ]


async def _count_queries(engine, awaitable_factory) -> int:
    """Count cursor executions issued while running the given coroutine."""
    counter = {"n": 0}

    def _before(conn, cursor, statement, parameters, context, executemany):
        counter["n"] += 1

    event.listen(engine.sync_engine, "before_cursor_execute", _before)
    try:
        await awaitable_factory()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _before)
    return counter["n"]


async def _analyzer_queries(engine, session: AsyncSession, project_id: str, phases) -> int:
    start = datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)

    async def run():
        # A fresh service per run: the request-scoped caches must not leak
        # between measurements.
        service = OptimizationService(session)
        for phase in phases:
            if phase == "model_usage":
                await service._analyze_model_usage(project_id, start, end, days=1)
            elif phase == "errors":
                await service._analyze_error_patterns(project_id, start, end, days=1)
            else:
                await service._analyze_latency_issues(project_id, start, end)

    return await _count_queries(engine, run)


async def _seed(session: AsyncSession, project_id: str, groups: int, start: int = 0):
    events = _events_for_groups(project_id, groups=groups, calls=CALLS_PER_GROUP)
    session.add_all(events[start * CALLS_PER_GROUP :])
    session.add_all(_baselines_for_groups(project_id, groups=groups)[start:])
    session.add_all(_pricing_for_groups(groups=groups, start=start))
    await session.commit()


async def test_error_and_latency_analyzers_do_not_query_per_row(
    test_engine, test_session: AsyncSession, test_project
):
    """Both used to call get_baseline() once per (agent, model) row."""
    session = test_session

    await _seed(session, test_project.id, groups=2)
    small = await _analyzer_queries(
        test_engine, session, test_project.id, ["errors", "latency"]
    )

    await _seed(session, test_project.id, groups=12, start=2)
    large = await _analyzer_queries(
        test_engine, session, test_project.id, ["errors", "latency"]
    )

    # One aggregate each plus a single shared baseline prefetch, whatever the
    # number of groups.
    assert small == large == 3


async def test_model_usage_analyzer_query_count_barely_grows(
    test_engine, test_session: AsyncSession, test_project
):
    session = test_session

    await _seed(session, test_project.id, groups=2)
    small = await _analyzer_queries(test_engine, session, test_project.id, ["model_usage"])

    await _seed(session, test_project.id, groups=12, start=2)
    large = await _analyzer_queries(test_engine, session, test_project.id, ["model_usage"])

    # Fixed cost is 3 queries (usage aggregate, bulk capability sample, bulk
    # pricing prefetch). What is left per group belongs to
    # PricingService.discover_alternatives, which has to ask about each distinct
    # source model: the learned-alternatives lookup and the dynamic scan. That
    # is 2 per group -- it used to be ~7 (2 capability + 1 source pricing + 2
    # baseline + the alternatives pair).
    per_group = (large - small) / 10
    assert per_group <= 2, f"{small} -> {large} queries for 2 -> 12 groups"


async def test_capability_inference_reads_event_metadata(
    test_session: AsyncSession, test_project
):
    """Events carry a "tools" key, so function calling is detected, and nothing
    signals vision or JSON mode."""
    session = test_session
    session.add_all(_events_for_groups(test_project.id, groups=3))
    await session.commit()

    service = OptimizationService(session)
    bulk = await service._infer_capability_requirements_bulk(
        project_id=test_project.id,
        pairs={(f"agent-{g}", f"model-{g}"): 12 for g in range(3)},
        start_time=datetime.now(timezone.utc) - timedelta(days=1),
        end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert bulk[("agent-1", "model-1")] == {
        "requires_vision": "false",
        "requires_function_calling": "true",
        "requires_json_mode": "false",
    }


async def test_capability_inference_needs_enough_calls(
    test_session: AsyncSession, test_project
):
    """Below the 10-call floor the state stays unknown, as before."""
    session = test_session
    session.add_all(_events_for_groups(test_project.id, groups=1, calls=3))
    await session.commit()

    states = await OptimizationService(session)._infer_capability_requirements_bulk(
        project_id=test_project.id,
        pairs={("agent-0", "model-0"): 3},
        start_time=datetime.now(timezone.utc) - timedelta(days=1),
        end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert states[("agent-0", "model-0")] == {
        "requires_vision": "unknown",
        "requires_function_calling": "unknown",
        "requires_json_mode": "unknown",
    }


async def test_baselines_are_prefetched_once_per_project(
    test_session: AsyncSession, test_project
):
    session = test_session
    session.add_all(_baselines_for_groups(test_project.id, groups=5))
    await session.commit()

    service = OptimizationService(session)
    baselines = await service._get_baselines(test_project.id)

    assert len(baselines) == 5
    assert baselines[("agent-3", "model-3")].avg_error_rate == pytest.approx(0.01)
    # Cached: a second call reuses the same mapping object.
    assert await service._get_baselines(test_project.id) is baselines


async def test_model_pricing_lookups_are_memoized(test_session: AsyncSession, test_project):
    """OptimizationService opts its PricingService into the request-scoped memo."""
    pricing = OptimizationService(test_session).pricing_service

    first = await pricing.get_model_pricing("gpt-4")
    second = await pricing.get_model_pricing("gpt-4")

    assert first == second
    assert "gpt-4" in pricing._lookup_memo
    # A plain PricingService keeps looking every model up.
    assert PricingService(test_session)._lookup_memo is None


@pytest.mark.asyncio
async def test_output_cap_is_not_filtered_against_input_size(
    test_session: AsyncSession,
):
    """max_tokens is the output cap; max_input_tokens the context cap. Filtering
    max_tokens against input+output hid an 8k-output model from a 20k-input
    workload its 128k context handles fine."""
    from app.services.pricing_service import PricingService

    test_session.add_all([
        ModelPricing(model_name="expensive-src", input_price_per_1k=0.01,
                     output_price_per_1k=0.03, provider="openai", is_active=True),
        # Fits: large context, small output cap.
        ModelPricing(model_name="cheap-small-output", input_price_per_1k=0.001,
                     output_price_per_1k=0.002, provider="openai", is_active=True,
                     max_tokens=8192, max_input_tokens=128_000),
        # Does not fit: context smaller than the workload's input.
        ModelPricing(model_name="cheap-small-context", input_price_per_1k=0.001,
                     output_price_per_1k=0.002, provider="openai", is_active=True,
                     max_tokens=8192, max_input_tokens=4_000),
    ])
    await test_session.commit()

    service = PricingService(test_session)
    alts = await service._discover_dynamically(
        model="expensive-src",
        source_pricing={"input": 0.01, "output": 0.03},
        source_total_cost=0.04,
        source_provider="openai",
        avg_input_tokens=20_000,
        avg_output_tokens=1_000,
        requires_vision=False,
        requires_function_calling=False,
        same_provider_only=False,
        max_results=10,
    )
    names = {a["model"] for a in alts}
    assert "cheap-small-output" in names
    assert "cheap-small-context" not in names
