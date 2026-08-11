"""
Tests for classifier-shaped workload detection.

The value of this recommendation is precision: telling a team their
summarisation agent is "really a classifier" burns trust fast, so most of
these tests pin the cases it must stay quiet on.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import Event
from app.services.optimization_service import OptimizationService, OptimizationType


async def _seed(
    session: AsyncSession,
    project_id: str,
    *,
    count: int,
    output_tokens,
    distinct_inputs: int,
    agent: str = "intent-router",
    model: str = "gpt-4o",
    cost: float = 0.02,
    success: bool = True,
):
    """Write `count` events, cycling `distinct_inputs` different input hashes."""
    now = datetime.now(timezone.utc)
    for i in range(count):
        tokens = output_tokens(i) if callable(output_tokens) else output_tokens
        session.add(
            Event(
                id=f"{agent}-{model}-{i}",
                project_id=project_id,
                agent_name=agent,
                model=model,
                input_tokens=120,
                output_tokens=tokens,
                total_tokens=120 + tokens,
                cost=cost,
                latency_ms=180,
                timestamp=now - timedelta(minutes=i),
                success=success,
                input_hash=f"hash_{i % distinct_inputs}",
            )
        )
    await session.commit()


async def _suggestions(session: AsyncSession, project_id: str):
    service = OptimizationService(session)
    all_suggestions = await service._generate_suggestions(project_id, days=30)
    return [
        s for s in all_suggestions
        if s["type"] == OptimizationType.NON_LLM_CANDIDATE.value
    ]


@pytest.mark.asyncio
async def test_flags_a_classifier_shaped_workload(test_session, test_project):
    """Short outputs, bounded input space, real volume."""
    await _seed(
        test_session, test_project.id,
        count=200, output_tokens=4, distinct_inputs=12,
    )

    found = await _suggestions(test_session, test_project.id)
    assert len(found) == 1

    suggestion = found[0]
    assert suggestion["agent_name"] == "intent-router"
    assert suggestion["model"] == "gpt-4o"
    assert suggestion["metrics"]["max_output_tokens"] == 4
    assert suggestion["metrics"]["calls"] == 200
    assert suggestion["metrics"]["savings_is_ceiling"] is True
    assert suggestion["estimated_savings_monthly"] > 0


@pytest.mark.asyncio
async def test_ignores_generation_workloads(test_session, test_project):
    """Long responses are the whole reason to be paying for a model."""
    await _seed(
        test_session, test_project.id,
        count=200, output_tokens=800, distinct_inputs=12,
        agent="summariser",
    )

    assert await _suggestions(test_session, test_project.id) == []


@pytest.mark.asyncio
async def test_one_long_response_disqualifies_the_group(test_session, test_project):
    """
    The reason the detector uses max and not mean.

    199 labels and one paragraph averages out to a classifier, but that one
    paragraph is exactly what breaks when the workload is moved off an LLM.
    """
    await _seed(
        test_session, test_project.id,
        count=200,
        output_tokens=lambda i: 900 if i == 0 else 3,
        distinct_inputs=12,
    )

    assert await _suggestions(test_session, test_project.id) == []


@pytest.mark.asyncio
async def test_ignores_unbounded_input_spaces(test_session, test_project):
    """Every input unique means open-ended work that answers briefly."""
    await _seed(
        test_session, test_project.id,
        count=200, output_tokens=4, distinct_inputs=200,
    )

    assert await _suggestions(test_session, test_project.id) == []


@pytest.mark.asyncio
async def test_ignores_low_volume(test_session, test_project):
    """A handful of short answers is a coincidence, not a workload."""
    await _seed(
        test_session, test_project.id,
        count=10, output_tokens=4, distinct_inputs=3,
    )

    assert await _suggestions(test_session, test_project.id) == []


@pytest.mark.asyncio
async def test_ignores_trivial_spend(test_session, test_project):
    """Nobody should re-architect an agent to save cents a month."""
    await _seed(
        test_session, test_project.id,
        count=200, output_tokens=4, distinct_inputs=12,
        cost=0.0000001,
    )

    assert await _suggestions(test_session, test_project.id) == []


@pytest.mark.asyncio
async def test_failed_calls_do_not_qualify_a_group(test_session, test_project):
    """A failed call has no output; its zero tokens must not look like a label."""
    await _seed(
        test_session, test_project.id,
        count=200, output_tokens=0, distinct_inputs=12,
        success=False,
    )

    assert await _suggestions(test_session, test_project.id) == []


@pytest.mark.asyncio
async def test_action_items_lead_with_caching_when_repeats_are_high(
    test_session, test_project
):
    """A cache is the same-day win; a model swap is next week's project."""
    await _seed(
        test_session, test_project.id,
        count=200, output_tokens=4, distinct_inputs=4,  # 98% repeat rate
    )

    suggestion = (await _suggestions(test_session, test_project.id))[0]
    assert any("Cache first" in action for action in suggestion["action_items"])


@pytest.mark.asyncio
async def test_separates_agents_that_share_a_model(test_session, test_project):
    """Recommendations are per (agent, model), not per model."""
    await _seed(
        test_session, test_project.id,
        count=200, output_tokens=4, distinct_inputs=12, agent="router",
    )
    await _seed(
        test_session, test_project.id,
        count=200, output_tokens=700, distinct_inputs=12, agent="writer",
    )

    found = await _suggestions(test_session, test_project.id)
    assert [s["agent_name"] for s in found] == ["router"]
