"""
OTLP receiver.

The Claude Code fixture is a real export captured from one headless session
(two model calls, one Bash tool call), with the account identity replaced.
"""

import gzip
import json
from pathlib import Path

from sqlalchemy import select

from app.models.db_models import Event
from app.services.otlp_service import events_from_logs

CLAUDE_CODE = json.loads(
    (Path(__file__).parent / "fixtures" / "claude_code_otlp_logs.json").read_text(encoding="utf-8")
)
JSON = {"Content-Type": "application/json"}


def _attr(key, value):
    kind = "intValue" if isinstance(value, int) else "stringValue"
    return {"key": key, "value": {kind: str(value)}}


def _logs(*records, resource=()):
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": [_attr(k, v) for k, v in resource]},
                "scopeLogs": [{"logRecords": list(records)}],
            }
        ]
    }


def _codex_completed(**overrides):
    attrs = {
        "event.name": "codex.sse_event",
        "event.kind": "response.completed",
        "event.timestamp": "2026-10-01T10:00:00.000Z",
        "conversation.id": "01a0e65a-938b-70a1-96c0-bb5096c0fe02",
        "model": "gpt-6-luna",
        "auth_mode": "ChatGPT",
        "user.account_id": "acct_1",
        "input_token_count": 14565,
        "output_token_count": 103,
        "cached_token_count": 11008,
        "reasoning_token_count": 40,
        "tool_token_count": 14668,
    }
    attrs.update(overrides)
    return {"attributes": [_attr(k, v) for k, v in attrs.items()]}


# -- Mapping ---------------------------------------------------------------


def test_claude_code_export_yields_only_its_model_calls():
    events = events_from_logs(CLAUDE_CODE)

    assert [event["event_id"] for event in events] == [
        "cc-req_011Cfx6TtUm8F6k6qCda8TDs",
        "cc-req_011Cfx6U5KNnWAUm23HKyW4V",
    ]
    first, second = events
    assert first["model"] == "claude-haiku-5-5"
    # 2 uncached + 14,050 read from cache: the whole prompt is input.
    assert first["input_tokens"] == 14052
    assert first["cached_tokens"] == 14050
    assert first["cache_write_tokens"] == 12091
    assert first["output_tokens"] == 77
    assert first["latency_ms"] == 1670
    assert first["trace_id"] == "d5f007b2-e8b2-4961-87da-b38b1203d7d5"
    assert first["workflow"] == "claude-code" and first["agent_name"] == "claude-code"
    assert first["metadata"]["user_id"] == "dev@example.com"
    assert first["metadata"]["billing"] == "subscription"
    assert first["metadata"]["session_id"] == first["trace_id"]
    # The Bash decision logged just before the first call belongs to it.
    assert first["tool_name"] == "Bash"
    assert second["tool_name"] is None


def test_prompt_and_response_records_contribute_nothing():
    text = json.dumps(events_from_logs(CLAUDE_CODE))

    assert "REDACTED" not in text and "prompt" not in text


CODEX = json.loads(
    (Path(__file__).parent / "fixtures" / "codex_otlp_logs.json").read_text(encoding="utf-8")
)


def test_real_codex_export_matches_what_codex_logged_for_the_session():
    """A real export of one session: a warm-up, then two responses, one calling
    the shell. Codex's own session log recorded exactly the two responses."""
    events = events_from_logs(CODEX)

    assert [(e["input_tokens"], e["cached_tokens"], e["output_tokens"]) for e in events] == [
        (13677, 1792, 34),
        (13794, 13056, 16),
    ]
    first, second = events
    # The shell tool, under the name Codex's hooks give it.
    assert first["tool_name"] == "Bash" and second["tool_name"] is None
    assert first["model"] == "gpt-6-luna"
    assert first["trace_id"] == "01a12f9e-5f24-7f40-884e-be95cfff30ae"
    assert first["metadata"]["billing"] == "subscription"  # auth_mode: Chatgpt
    assert first["metadata"]["user_id"] == "dev@example.com"
    assert "<removed>" not in json.dumps(events)


def test_codex_id_is_the_one_the_sdk_collector_derives():
    import hashlib

    session = "01a12f9e-5f24-7f40-884e-be95cfff30ae"
    expected = "cx-" + hashlib.sha256(f"{session}|13677|1792|34".encode()).hexdigest()[:40]

    assert events_from_logs(CODEX)[0]["event_id"] == expected


def test_codex_completed_response_becomes_an_event():
    events = events_from_logs(_logs(_codex_completed(), _codex_completed(**{"event.kind": "response.created"})))

    assert len(events) == 1
    event = events[0]
    assert event["agent_name"] == "codex" and event["workflow"] == "codex"
    assert event["input_tokens"] == 14565 and event["cached_tokens"] == 11008
    assert event["output_tokens"] == 103
    assert event["trace_id"] == "01a0e65a-938b-70a1-96c0-bb5096c0fe02"
    assert event["metadata"]["billing"] == "subscription"
    assert event["metadata"]["user_id"] == "acct_1"
    assert event["event_id"].startswith("cx-")
    # Same call, same id: a retried export is a no-op.
    assert events_from_logs(_logs(_codex_completed()))[0]["event_id"] == event["event_id"]


def test_billing_can_be_stated_by_the_sender():
    api = events_from_logs(_logs(_codex_completed(auth_mode="ApiKey")))[0]
    assert api["metadata"]["billing"] == "api"

    stated = events_from_logs(
        _logs(_codex_completed(auth_mode="ApiKey"), resource=[("agentcost.billing", "subscription")])
    )[0]
    assert stated["metadata"]["billing"] == "subscription"


def test_a_rejected_tool_is_not_attributed():
    decision = {
        "attributes": [
            _attr("event.name", "tool_decision"),
            _attr("session.id", "s1"),
            _attr("tool_name", "Bash"),
            _attr("decision", "reject"),
        ]
    }
    request = {
        "attributes": [
            _attr("event.name", "api_request"),
            _attr("session.id", "s1"),
            _attr("event.timestamp", "2026-10-01T10:00:00.000Z"),
            _attr("model", "claude-opus-5-5"),
            _attr("input_tokens", 10),
            _attr("output_tokens", 5),
            _attr("agent.name", "Plan"),
        ]
    }

    event = events_from_logs(_logs(decision, request))[0]

    assert event["tool_name"] is None
    assert event["agent_name"] == "claude-code/Plan" and event["depth"] == 1
    assert event["event_id"] is None  # no request id, so no idempotency key


def test_junk_is_ignored_rather_than_raised():
    assert events_from_logs(None) == []
    assert events_from_logs({"resourceLogs": "nope"}) == []
    assert events_from_logs({"resourceLogs": [{"scopeLogs": [{"logRecords": [{"attributes": 7}]}]}]}) == []
    assert events_from_logs(_logs({"attributes": [_attr("event.name", "api_request")]})) == []


# -- Endpoint --------------------------------------------------------------


async def test_logs_endpoint_stores_calls_once(client, test_session):
    first = await client.post("/v1/otlp/v1/logs", json=CLAUDE_CODE)
    again = await client.post("/v1/otlp/v1/logs", json=CLAUDE_CODE)

    assert first.status_code == 200 and first.json() == {}
    assert again.status_code == 200
    rows = (await test_session.execute(select(Event).order_by(Event.timestamp))).scalars().all()
    assert len(rows) == 2
    assert rows[0].tool_name == "Bash"
    assert rows[0].billing == "subscription"
    assert rows[0].user_id == "dev@example.com"
    assert rows[0].session_id == "d5f007b2-e8b2-4961-87da-b38b1203d7d5"
    assert rows[0].cached_tokens == 14050


async def test_a_call_sent_both_ways_is_counted_once(client, test_project, test_session):
    """The SDK's collector and the OTLP export name the same call the same way."""
    collected = {
        "event_id": "cc-req_011Cfx6TtUm8F6k6qCda8TDs",
        "model": "claude-haiku-5-5",
        "input_tokens": 14052,
        "output_tokens": 77,
        "timestamp": "2026-10-12T09:19:46.930Z",
    }
    await client.post(
        "/v1/events/batch", json={"project_id": test_project.id, "events": [collected]}
    )

    await client.post("/v1/otlp/v1/logs", json=CLAUDE_CODE)

    ids = (await test_session.execute(select(Event.event_id))).scalars().all()
    assert sorted(ids) == ["cc-req_011Cfx6TtUm8F6k6qCda8TDs", "cc-req_011Cfx6U5KNnWAUm23HKyW4V"]


async def test_gzip_bodies_are_accepted(client, test_session):
    body = gzip.compress(json.dumps(_logs(_codex_completed())).encode())

    response = await client.post(
        "/v1/otlp/v1/logs", content=body, headers={**JSON, "Content-Encoding": "gzip"}
    )

    assert response.status_code == 200
    assert len((await test_session.execute(select(Event))).scalars().all()) == 1


async def test_protobuf_is_refused_with_the_setting_to_change(client):
    response = await client.post(
        "/v1/otlp/v1/logs", content=b"\x0a\x00", headers={"Content-Type": "application/x-protobuf"}
    )

    assert response.status_code == 415
    assert "http/json" in response.text


async def test_bad_bodies_are_400_not_500(client):
    bad_json = await client.post("/v1/otlp/v1/logs", content=b"{ nope", headers=JSON)
    bad_gzip = await client.post(
        "/v1/otlp/v1/logs", content=b"not gzip", headers={**JSON, "Content-Encoding": "gzip"}
    )

    assert bad_json.status_code == 400 and bad_gzip.status_code == 400


async def test_a_key_is_required(client):
    client.headers.pop("Authorization")

    assert (await client.post("/v1/otlp/v1/logs", json=CLAUDE_CODE)).status_code == 401
    assert (await client.post("/v1/otlp/v1/metrics", json={})).status_code == 401


async def test_metrics_and_traces_are_accepted_and_dropped(client, test_session):
    metrics = await client.post("/v1/otlp/v1/metrics", json={"resourceMetrics": []})
    traces = await client.post("/v1/otlp/v1/traces", json={"resourceSpans": []})

    assert metrics.status_code == 200 and traces.status_code == 200
    assert (await test_session.execute(select(Event))).scalars().all() == []


async def test_a_large_export_is_ingested_in_batches(client, test_session):
    records = [
        # The prompt grows with every call in a session, as it does in a real one.
        _codex_completed(input_token_count=14565 + i * 500)
        for i in range(230)
    ]

    response = await client.post("/v1/otlp/v1/logs", json=_logs(*records))

    assert response.status_code == 200
    assert len((await test_session.execute(select(Event.id))).scalars().all()) == 230
