"""
Two ways a model-downgrade suggestion used to promise money that was not there:
pricing a cache-served prompt at the list rate, and counting usage a flat
subscription already covers. Both surfaced with coding agents, whose prompts are
almost entirely cache reads and are usually on a plan.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models.db_models import Event, ModelPricing
from app.services.optimization_service import CAPABILITY_KEY, OptimizationService

CALLS = 12  # above the analyzer's 10-call cutoff


@pytest.fixture
async def catalogue(test_session):
    test_session.add_all(
        [
            ModelPricing(
                model_name="premium-model",
                input_price_per_1k=10.0,
                output_price_per_1k=30.0,
                cached_input_price_per_1k=1.0,
                provider="test",
                supports_vision=True,
                supports_function_calling=True,
            ),
            ModelPricing(
                model_name="cheap-model",
                input_price_per_1k=1.0,
                output_price_per_1k=3.0,
                provider="test",
                supports_vision=True,
                supports_function_calling=True,
            ),
        ]
    )
    await test_session.commit()


def _events(project_id, *, cost, billing=None):
    now = datetime.now(timezone.utc)
    return [
        Event(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent_name="claude-code",
            model="premium-model",
            # A coding agent's call: a huge prompt, nearly all of it cache reads.
            input_tokens=400_000,
            cached_tokens=399_000,
            output_tokens=500,
            total_tokens=400_500,
            cost=cost,
            latency_ms=0,
            success=True,
            timestamp=now - timedelta(hours=1),
            extra_data={CAPABILITY_KEY: {"tools": True}},
            billing=billing,
        )
        for _ in range(CALLS)
    ]


async def _downgrades(test_session, project_id):
    suggestions = await OptimizationService(test_session).get_suggestions(
        project_id, days=7, persist_recommendations=False
    )
    return [s for s in suggestions if s.get("type") == "model_downgrade"]


async def test_savings_never_exceed_what_was_actually_spent(
    test_session, test_project, catalogue
):
    # 399K cache reads at $1/1k + 1K at $10/1k + 500 out at $30/1k = $424 a call.
    test_session.add_all(_events(test_project.id, cost=424.0))
    await test_session.commit()

    downgrades = await _downgrades(test_session, test_project.id)

    assert downgrades, "a 10x cheaper capable model should still be suggested"
    for suggestion in downgrades:
        spent = suggestion["metrics"]["current_monthly_cost"]
        assert 0 < suggestion["estimated_savings_monthly"] <= spent
        assert suggestion["estimated_savings_percent"] <= 100.0


async def test_subscription_usage_has_no_money_to_recover(
    test_session, test_project, catalogue
):
    test_session.add_all(_events(test_project.id, cost=424.0, billing="subscription"))
    await test_session.commit()

    assert await _downgrades(test_session, test_project.id) == []


async def test_metered_usage_beside_a_subscription_is_still_analysed(
    test_session, test_project, catalogue
):
    test_session.add_all(_events(test_project.id, cost=424.0, billing="subscription"))
    test_session.add_all(_events(test_project.id, cost=424.0, billing="api"))
    await test_session.commit()

    downgrades = await _downgrades(test_session, test_project.id)

    assert downgrades
    # Judged on the metered half only.
    assert downgrades[0]["metrics"]["current_calls"] == CALLS
