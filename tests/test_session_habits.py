"""
Session habits: advice for coding agents, where the user controls neither the
prompt nor the model, only how they use sessions.
"""

import uuid
from datetime import datetime, timedelta, timezone

from app.models.db_models import Event, SessionName
from app.services.optimization_service import OptimizationService
from app.services.session_habit_service import SessionHabitService

NOW = datetime.now(timezone.utc)


def _call(project_id, session, minutes, *, input_tokens=30_000, cost=0.05, cache_write=0,
          workflow="claude-code", success=True, error=None):
    return Event(
        id=str(uuid.uuid4()),
        project_id=project_id,
        agent_name=workflow,
        model="claude-opus-5-5",
        input_tokens=input_tokens,
        cached_tokens=max(input_tokens - cache_write - 10, 0),
        cache_write_tokens=cache_write,
        output_tokens=300,
        total_tokens=input_tokens + 300,
        cost=cost,
        latency_ms=0,
        success=success,
        error=error,
        timestamp=NOW - timedelta(days=2) + timedelta(minutes=minutes),
        trace_id=session,
        workflow=workflow,
        billing="subscription",
    )


async def _habits(test_session, project_id):
    found = await SessionHabitService(test_session).analyze(
        project_id, NOW - timedelta(days=7), NOW
    )
    return {habit["type"]: habit for habit in found}


async def test_a_light_user_gets_no_advice(test_session, test_project):
    for session in ("s1", "s2", "s3", "s4"):
        test_session.add_all(_call(test_project.id, session, m) for m in range(8))
    await test_session.commit()

    assert await _habits(test_session, test_project.id) == {}


async def test_long_conversations_are_named_with_their_share(test_session, test_project):
    test_session.add_all(_call(test_project.id, "short", m) for m in range(20))  # $1.00
    test_session.add_all(
        _call(test_project.id, "long", m, input_tokens=450_000, cost=0.30) for m in range(10)
    )  # $3.00
    await test_session.commit()

    habit = (await _habits(test_session, test_project.id))["session_large_context"]

    assert habit["agent_name"] == "claude-code"
    assert habit["metrics"]["share_percent"] == 75.0
    assert habit["metrics"]["headline_value"] == "75%"
    assert habit["metrics"]["large_context_calls"] == 10
    assert "75%" in habit["title"] and "$3.00" in habit["description"]
    assert habit["priority"] == "high"
    # Usage on a plan is not money to recover.
    assert habit["estimated_savings_monthly"] == 0
    assert any("/compact" in step for step in habit["action_items"])


async def test_coming_back_to_an_old_session_is_a_cache_rebuild(test_session, test_project):
    # The first call of each session writes its cache: expected, not a rebuild.
    for session in ("a", "b", "c"):
        test_session.add(
            _call(test_project.id, session, 0, input_tokens=60_000, cache_write=59_000, cost=0.40)
        )
        test_session.add_all(_call(test_project.id, session, m) for m in range(1, 8))
    # Hours later, the whole context is written again, three times.
    for index, session in enumerate(("a", "a", "b")):
        test_session.add(
            _call(test_project.id, session, 300 + index, input_tokens=180_000,
                  cache_write=179_000, cost=1.20)
        )
    await test_session.commit()

    habits = await _habits(test_session, test_project.id)
    habit = habits["session_cache_rebuild"]

    assert habit["metrics"]["rebuilds"] == 3
    assert habit["metrics"]["sessions_affected"] == 2
    assert habit["metrics"]["window_cost"] == 3.6
    assert "3 times" in habit["title"]


async def test_one_runaway_session_is_named(test_session, test_project):
    test_session.add_all(_call(test_project.id, "runaway", m, cost=0.50) for m in range(20))
    for session in ("x", "y", "z"):
        test_session.add_all(_call(test_project.id, session, m, cost=0.05) for m in range(6))
    test_session.add(
        SessionName(project_id=test_project.id, session_id="runaway",
                    name="Rewrite the billing service", updated_at=NOW)
    )
    await test_session.commit()

    habit = (await _habits(test_session, test_project.id))["session_dominant"]

    assert habit["metrics"]["trace_id"] == "runaway"
    assert habit["metrics"]["session_name"] == "Rewrite the billing service"
    assert '"Rewrite the billing service"' in habit["description"]
    assert round(habit["metrics"]["share_percent"]) == 92


async def test_hitting_the_plan_limit_is_counted(test_session, test_project):
    test_session.add_all(_call(test_project.id, "s1", m) for m in range(25))
    test_session.add_all(
        _call(test_project.id, "s1", 30 + m, input_tokens=0, cost=0.0, success=False,
              error="rate_limit (429)")
        for m in range(4)
    )
    test_session.add(
        _call(test_project.id, "s1", 40, input_tokens=0, cost=0.0, success=False,
              error="server_error")
    )
    await test_session.commit()

    habit = (await _habits(test_session, test_project.id))["session_rate_limited"]

    assert habit["metrics"]["rate_limited"] == 4
    assert habit["metrics"]["headline_value"] == "4"
    assert "4 times" in habit["title"]


async def test_an_agent_barely_used_has_no_habits(test_session, test_project):
    # One session dominates, but all of it together is worth three cents.
    test_session.add_all(_call(test_project.id, "big", m, cost=0.001) for m in range(23))
    for session in ("x", "y", "z"):
        test_session.add_all(_call(test_project.id, session, m, cost=0.0005) for m in range(4))
    await test_session.commit()

    assert await _habits(test_session, test_project.id) == {}


async def test_apps_instrumented_with_the_sdk_are_left_alone(test_session, test_project):
    test_session.add_all(
        _call(test_project.id, "run", m, input_tokens=450_000, cost=0.30, workflow="support-triage")
        for m in range(30)
    )
    await test_session.commit()

    assert await _habits(test_session, test_project.id) == {}


async def test_habits_reach_the_optimizer_without_adding_to_recoverable_spend(
    test_session, test_project
):
    test_session.add_all(_call(test_project.id, "short", m) for m in range(20))
    test_session.add_all(
        _call(test_project.id, "long", m, input_tokens=450_000, cost=0.30, workflow="codex")
        for m in range(25)
    )
    await test_session.commit()

    service = OptimizationService(test_session)
    suggestions = await service.get_suggestions(
        test_project.id, days=7, persist_recommendations=True
    )
    summary = await service.get_summary(test_project.id, days=7)

    habits = [s for s in suggestions if s["type"].startswith("session_")]
    assert [h["agent_name"] for h in habits] == ["codex"]
    assert "Codex" in habits[0]["title"]
    assert summary["total_potential_savings_monthly"] == 0
