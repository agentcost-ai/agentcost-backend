"""Tests for agent guardrails: declared tool policy vs observed usage.

Compliance is a separate concept from success rate — these tests pin the
separation: breaches come only from tool boundaries, never from call failures.
"""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from app.models.schemas import GuardrailUpsert
from app.services.guardrail_service import GuardrailService


def _event(**overrides):
    base = {
        "agent_name": "worker",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.01,
        "latency_ms": 200,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    base.update(overrides)
    return base


async def _ingest(client, project_id, events):
    response = await client.post(
        "/v1/events/batch", json={"project_id": project_id, "events": events}
    )
    assert response.status_code == 200
    return response


@pytest.mark.asyncio
async def test_agent_without_guardrail_reports_no_guardrail(
    client: AsyncClient, test_project
):
    await _ingest(
        client,
        test_project.id,
        [_event(tool_name="web_search"), _event()],
    )

    response = await client.get("/v1/guardrails/compliance")
    assert response.status_code == 200
    data = response.json()

    assert data["total_calls"] == 2
    assert data["tool_tracked_calls"] == 1
    (agent,) = data["agents"]
    assert agent["agent_name"] == "worker"
    assert agent["status"] == "no_guardrail"
    assert agent["observed_tools"] == ["web_search"]
    assert agent["breaches"] == []


@pytest.mark.asyncio
async def test_undeclared_tool_is_a_breach(
    client: AsyncClient, test_session, test_project
):
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", allowed_tools=["web_search"]),
    )
    await test_session.commit()

    await _ingest(
        client,
        test_project.id,
        [
            _event(tool_name="web_search"),
            _event(tool_name="db_write"),
            _event(tool_name="db_write"),
        ],
    )

    response = await client.get("/v1/guardrails/compliance")
    data = response.json()
    (agent,) = data["agents"]
    assert agent["status"] == "breach"
    (breach,) = agent["breaches"]
    assert breach["subject"] == "db_write"
    assert breach["kind"] == "undeclared_tool"
    assert breach["count"] == 2


@pytest.mark.asyncio
async def test_failed_call_within_boundary_is_not_a_breach(
    client: AsyncClient, test_session, test_project
):
    """Success rate and compliance stay separate: a failed call on a
    permitted tool must not register as a guardrail breach."""
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", allowed_tools=["web_search"]),
    )
    await test_session.commit()

    await _ingest(
        client,
        test_project.id,
        [_event(tool_name="web_search", success=False, error="rate limited")],
    )

    response = await client.get("/v1/guardrails/compliance")
    (agent,) = response.json()["agents"]
    assert agent["status"] == "compliant"
    assert agent["breaches"] == []


@pytest.mark.asyncio
async def test_readonly_agent_write_tool_breach_and_unknown_tags(
    client: AsyncClient, test_session, test_project
):
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id, GuardrailUpsert(agent_name="worker", read_only=True)
    )
    await service.upsert_tool_tag(test_project.id, "db_write", "write")
    await service.upsert_tool_tag(test_project.id, "web_search", "read")
    await test_session.commit()

    await _ingest(
        client,
        test_project.id,
        [
            _event(tool_name="web_search"),
            _event(tool_name="db_write"),
            _event(tool_name="mystery_tool"),
        ],
    )

    response = await client.get("/v1/guardrails/compliance")
    (agent,) = response.json()["agents"]
    assert agent["status"] == "breach"
    (breach,) = agent["breaches"]
    assert breach["subject"] == "db_write"
    assert breach["kind"] == "write_in_readonly"
    # Untagged tools are reported, not silently judged.
    assert agent["unknown_access_tools"] == ["mystery_tool"]


@pytest.mark.asyncio
async def test_declared_guardrail_with_no_activity_is_listed(
    client: AsyncClient, test_session, test_project
):
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id, GuardrailUpsert(agent_name="idle-agent", allowed_tools=[])
    )
    await test_session.commit()

    response = await client.get("/v1/guardrails/compliance")
    (agent,) = response.json()["agents"]
    assert agent["agent_name"] == "idle-agent"
    assert agent["status"] == "compliant"
    assert agent["tracked_tool_calls"] == 0
    assert agent["total_calls"] == 0


@pytest.mark.asyncio
async def test_disabled_guardrail_reports_no_guardrail(
    client: AsyncClient, test_session, test_project
):
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", allowed_tools=[], enabled=False),
    )
    await test_session.commit()

    await _ingest(client, test_project.id, [_event(tool_name="anything")])

    response = await client.get("/v1/guardrails/compliance")
    (agent,) = response.json()["agents"]
    assert agent["status"] == "no_guardrail"


@pytest.mark.asyncio
async def test_upsert_replaces_rather_than_duplicates(test_session, test_project):
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id, GuardrailUpsert(agent_name="worker", allowed_tools=["a"])
    )
    await service.upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", allowed_tools=["b"], read_only=True),
    )
    await test_session.commit()

    rows = await service.list_guardrails(test_project.id)
    assert len(rows) == 1
    assert rows[0].allowed_tools == ["b"]
    assert rows[0].read_only is True


@pytest.mark.asyncio
async def test_guardrail_mutation_rejects_project_api_key(
    client: AsyncClient, test_project
):
    """Definitions are member-managed: the SDK key must not be able to
    rewrite the policy it is judged against."""
    response = await client.put(
        f"/v1/projects/{test_project.id}/guardrails",
        json={"agent_name": "worker", "read_only": True},
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_project_delete_removes_guardrail_rows(test_session, test_project):
    """Guardrail tables reference projects.id without CASCADE: the explicit
    cleanup in ProjectService.delete must include them or deletes fail on PG."""
    from sqlalchemy import select
    from app.models.db_models import AgentGuardrail, ToolAccessTag
    from app.services.event_service import ProjectService

    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id, GuardrailUpsert(agent_name="worker", read_only=True)
    )
    await service.upsert_tool_tag(test_project.id, "db_write", "write")
    await test_session.commit()

    assert await ProjectService(test_session).delete(test_project.id) is True
    await test_session.commit()

    for model in (AgentGuardrail, ToolAccessTag):
        rows = (await test_session.execute(select(model))).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_breach_at_ingest_notifies_owner_once_per_window(
    client: AsyncClient, test_session, test_project, test_user
):
    """A breach is an event, not a report: owners hear about it at ingest,
    and a breaching agent that keeps calling the tool does not page twice."""
    from sqlalchemy import select
    from app.models.db_models import Notification

    test_project.owner_id = test_user.id
    test_session.add(test_project)
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id, GuardrailUpsert(agent_name="worker", read_only=True)
    )
    await service.upsert_tool_tag(test_project.id, "send_email", "write")
    await test_session.commit()

    await _ingest(client, test_project.id, [_event(tool_name="send_email")])
    await _ingest(client, test_project.id, [_event(tool_name="send_email")] * 3)

    rows = (
        await test_session.execute(
            select(Notification).where(Notification.user_id == test_user.id)
        )
    ).scalars().all()
    assert len(rows) == 1
    notif = rows[0]
    assert notif.type == "guardrail_breach"
    assert notif.severity == "critical"
    assert notif.link == "/guardrails"
    assert notif.payload["agent_name"] == "worker"
    assert notif.payload["subject"] == "send_email"
    assert notif.payload["kind"] == "write_in_readonly"


@pytest.mark.asyncio
async def test_breach_fires_project_webhook(
    client: AsyncClient, test_session, test_project, monkeypatch
):
    from app.services import guardrail_service as module

    sent = []
    monkeypatch.setattr(
        module.webhook_service,
        "dispatch",
        lambda url, event_type, payload, secret=None: sent.append((url, event_type, payload)),
    )
    test_project.webhook_url = "https://hooks.example.com/agentcost"
    test_session.add(test_project)
    await GuardrailService(test_session).upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", allowed_tools=["web_search"]),
    )
    await test_session.commit()

    await _ingest(client, test_project.id, [_event(tool_name="db_write")])

    assert len(sent) == 1
    url, event_type, payload = sent[0]
    assert url == "https://hooks.example.com/agentcost"
    assert event_type == "guardrail.breach"
    assert payload["kind"] == "undeclared_tool"
    assert payload["subject"] == "db_write"


@pytest.mark.asyncio
async def test_compliant_batch_raises_no_alert(
    client: AsyncClient, test_session, test_project, test_user
):
    from sqlalchemy import select
    from app.models.db_models import Notification

    test_project.owner_id = test_user.id
    test_session.add(test_project)
    await GuardrailService(test_session).upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", allowed_tools=["web_search"]),
    )
    await test_session.commit()

    await _ingest(client, test_project.id, [_event(tool_name="web_search")])

    rows = (await test_session.execute(select(Notification))).scalars().all()
    assert rows == []


async def _member_session(client: AsyncClient):
    """Register a user through the API; the response carries a JWT and the
    default project (with its plaintext key) the way the dashboard sees it."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": "guardrails-owner@example.com",
            "password": "Str0ng-passw0rd!",
            "name": "Owner",
            "accept_terms": True,
            "accept_privacy": True,
            "terms_version": "1.0",
            "privacy_version": "1.0",
        },
    )
    assert response.status_code in (200, 201), response.text
    data = response.json()
    return (
        {"Authorization": f"Bearer {data['access_token']}"},
        data["default_project"]["id"],
    )


@pytest.mark.asyncio
async def test_member_can_upsert_and_read_back_guardrail_over_http(client: AsyncClient):
    """The full HTTP round trip, including response serialization of
    server-default timestamps (a MissingGreenlet the service tests could
    not see)."""
    jwt, project_id = await _member_session(client)

    response = await client.put(
        f"/v1/projects/{project_id}/guardrails",
        headers=jwt,
        json={"agent_name": "mailer", "read_only": True, "allowed_tools": None, "enabled": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent_name"] == "mailer"
    assert body["read_only"] is True
    assert body["allowed_tools"] is None
    assert body["created_at"]

    response = await client.put(
        f"/v1/projects/{project_id}/guardrails/tool-tags",
        headers=jwt,
        json={"tool_name": "send_email", "access": "write"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"tool_name": "send_email", "access": "write"}

    response = await client.put(
        f"/v1/projects/{project_id}/guardrails",
        headers=jwt,
        json={"agent_name": "mailer", "read_only": False, "allowed_tools": ["a"], "enabled": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["allowed_tools"] == ["a"]

    response = await client.delete(
        f"/v1/projects/{project_id}/guardrails/tool-tags/send_email", headers=jwt
    )
    assert response.status_code == 204
    response = await client.delete(
        f"/v1/projects/{project_id}/guardrails/mailer", headers=jwt
    )
    assert response.status_code == 204
    response = await client.delete(
        f"/v1/projects/{project_id}/guardrails/mailer", headers=jwt
    )
    assert response.status_code == 404


# ── v2 boundaries: models and per-run limits ──────────────────────────


@pytest.mark.asyncio
async def test_undeclared_model_is_a_breach(
    client: AsyncClient, test_session, test_project
):
    """A model allow-list is judged on every call, instrumented or not."""
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", allowed_models=["gpt-4o-mini"]),
    )
    await test_session.commit()

    await _ingest(
        client,
        test_project.id,
        [_event(model="gpt-4o-mini"), _event(model="gpt-4o"), _event(model="gpt-4o")],
    )

    (agent,) = (await client.get("/v1/guardrails/compliance")).json()["agents"]
    assert agent["status"] == "breach"
    assert sorted(agent["observed_models"]) == ["gpt-4o", "gpt-4o-mini"]
    (breach,) = agent["breaches"]
    assert breach["kind"] == "undeclared_model"
    assert breach["subject"] == "gpt-4o"
    assert breach["count"] == 2


@pytest.mark.asyncio
async def test_tool_calls_per_run_limit(
    client: AsyncClient, test_session, test_project
):
    """Per-run limits count tool calls sharing a trace_id; calls outside a
    workflow (no trace_id) are not a run and cannot breach."""
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", max_tool_calls_per_run=2),
    )
    await test_session.commit()

    run_a = "a" * 32
    run_b = "b" * 32
    await _ingest(
        client,
        test_project.id,
        [_event(tool_name="search", trace_id=run_a, span_id=f"{i:032x}") for i in range(3)]
        + [_event(tool_name="search", trace_id=run_b, span_id=f"{i:032x}") for i in range(2)]
        + [_event(tool_name="search")] * 5,
    )

    (agent,) = (await client.get("/v1/guardrails/compliance")).json()["agents"]
    assert agent["runs_seen"] == 2
    assert agent["status"] == "breach"
    (breach,) = agent["breaches"]
    assert breach["kind"] == "tool_calls_over_limit"
    assert breach["subject"] == run_a
    assert breach["count"] == 1
    assert breach["limit"] == 2
    assert breach["observed"] == 3


@pytest.mark.asyncio
async def test_run_cost_limit_reports_worst_run(
    client: AsyncClient, test_session, test_project
):
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", max_cost_per_run_usd=0.05),
    )
    await test_session.commit()

    cheap, mid, worst = "c" * 32, "d" * 32, "e" * 32
    await _ingest(
        client,
        test_project.id,
        [
            _event(trace_id=cheap, span_id="1" * 32, cost=0.01),
            _event(trace_id=mid, span_id="2" * 32, cost=0.04),
            _event(trace_id=mid, span_id="3" * 32, cost=0.04),
            _event(trace_id=worst, span_id="4" * 32, cost=0.30),
        ],
    )

    (agent,) = (await client.get("/v1/guardrails/compliance")).json()["agents"]
    (breach,) = agent["breaches"]
    assert breach["kind"] == "run_cost_over_limit"
    assert breach["count"] == 2
    assert breach["subject"] == worst
    assert breach["limit"] == 0.05
    assert breach["observed"] == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_run_limit_breach_alerts_across_batches(
    client: AsyncClient, test_session, test_project, test_user
):
    """A run that crosses its limit in a later batch still alerts, because
    the alert is judged on the run's stored totals, and only once."""
    from sqlalchemy import select
    from app.models.db_models import Notification

    test_project.owner_id = test_user.id
    test_session.add(test_project)
    await GuardrailService(test_session).upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", max_tool_calls_per_run=2),
    )
    await test_session.commit()

    run = "f" * 32
    await _ingest(client, test_project.id, [_event(tool_name="t", trace_id=run, span_id="1" * 32)] * 2)
    rows = (await test_session.execute(select(Notification))).scalars().all()
    assert rows == []

    await _ingest(client, test_project.id, [_event(tool_name="t", trace_id=run, span_id="2" * 32)])
    await _ingest(client, test_project.id, [_event(tool_name="t", trace_id=run, span_id="3" * 32)])
    rows = (await test_session.execute(select(Notification))).scalars().all()
    assert len(rows) == 1
    assert rows[0].payload["kind"] == "tool_calls_over_limit"
    assert rows[0].payload["subject"] == run
    assert rows[0].payload["limit"] == 2
    assert rows[0].payload["observed"] == 3


@pytest.mark.asyncio
async def test_guardrail_v2_fields_round_trip_and_validate(client: AsyncClient):
    jwt, project_id = await _member_session(client)

    response = await client.put(
        f"/v1/projects/{project_id}/guardrails",
        headers=jwt,
        json={
            "agent_name": "planner",
            "allowed_models": ["claude-sonnet-4", " claude-sonnet-4 ", ""],
            "max_tool_calls_per_run": 12,
            "max_cost_per_run_usd": 0.5,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["allowed_models"] == ["claude-sonnet-4"]
    assert body["max_tool_calls_per_run"] == 12
    assert body["max_cost_per_run_usd"] == 0.5
    assert body["allowed_tools"] is None

    response = await client.put(
        f"/v1/projects/{project_id}/guardrails",
        headers=jwt,
        json={"agent_name": "planner", "max_tool_calls_per_run": 0},
    )
    assert response.status_code == 422
    response = await client.put(
        f"/v1/projects/{project_id}/guardrails",
        headers=jwt,
        json={"agent_name": "planner", "max_cost_per_run_usd": -1},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_compliance_detail_reports_usage_and_run_stats(
    client: AsyncClient, test_session, test_project
):
    """The detail a user needs to set a boundary: per-tool and per-model usage
    with cost, run distribution percentiles, and a daily breach series."""
    service = GuardrailService(test_session)
    await service.upsert_guardrail(
        test_project.id,
        GuardrailUpsert(agent_name="worker", allowed_tools=["search"], allowed_models=["gpt-4o"]),
    )
    await test_session.commit()

    runs = ["1" * 32, "2" * 32, "3" * 32, "4" * 32]
    events = []
    for i, run in enumerate(runs):
        for j in range(i + 1):  # 1, 2, 3, 4 tool calls per run
            events.append(_event(tool_name="search", trace_id=run, span_id=f"{i}{j:031d}", cost=0.01))
    events.append(_event(tool_name="db_write", model="gpt-4o-mini", cost=0.5))

    await _ingest(client, test_project.id, events)
    (agent,) = (await client.get("/v1/guardrails/compliance")).json()["agents"]

    assert agent["total_calls"] == 11
    assert agent["total_cost"] == pytest.approx(0.6)
    usage = {u["tool_name"]: u for u in agent["tool_usage"]}
    assert usage["search"]["calls"] == 10 and usage["search"]["breach_kind"] is None
    assert usage["db_write"]["calls"] == 1 and usage["db_write"]["breach_kind"] == "undeclared_tool"
    models = {m["model"]: m for m in agent["model_usage"]}
    assert models["gpt-4o"]["permitted"] is True and models["gpt-4o"]["calls"] == 10
    assert models["gpt-4o-mini"]["permitted"] is False
    assert models["gpt-4o-mini"]["cost"] == pytest.approx(0.5)

    stats = agent["run_stats"]
    assert stats["runs"] == 4
    assert stats["max_tool_calls"] == 4
    assert stats["p50_tool_calls"] == 2
    assert stats["p95_tool_calls"] == 4
    assert stats["max_cost"] == pytest.approx(0.04)

    # Two breach kinds on the same day: the undeclared tool call and the
    # undeclared model call are the same event, counted once per boundary.
    assert len(agent["breach_series"]) == 1
    assert agent["breach_series"][0]["count"] == 2
    assert {b["kind"] for b in agent["breaches"]} == {"undeclared_tool", "undeclared_model"}
