"""
AgentCost Backend - OTLP log mapping

Claude Code and Codex can export their own telemetry over OpenTelemetry. Among
everything they send, one log record per model call carries the usage. This
module picks those records out of an OTLP/HTTP JSON logs payload and turns
each into an ordinary event; every other record is dropped unread.

Pure functions, no I/O: the route owns transport and persistence.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

# A sender states how the usage is paid for with a resource attribute, e.g.
# OTEL_RESOURCE_ATTRIBUTES="agentcost.billing=subscription".
BILLING_ATTRIBUTE = "agentcost.billing"

_ID_WIDTH = 64


def _value(wrapped: Any) -> Any:
    """Unwrap an OTLP AnyValue. Integers arrive as strings in the JSON form."""
    if not isinstance(wrapped, dict):
        return None
    if "stringValue" in wrapped:
        return wrapped["stringValue"]
    if "intValue" in wrapped:
        try:
            return int(wrapped["intValue"])
        except (TypeError, ValueError):
            return None
    if "doubleValue" in wrapped:
        return wrapped["doubleValue"]
    if "boolValue" in wrapped:
        return wrapped["boolValue"]
    return None


def _attributes(items: Any) -> Dict[str, Any]:
    if not isinstance(items, list):
        return {}
    return {
        item["key"]: _value(item.get("value"))
        for item in items
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }


def _count(attrs: Dict[str, Any], key: str) -> int:
    try:
        return max(int(attrs.get(key) or 0), 0)
    except (TypeError, ValueError):
        return 0


def _timestamp(attrs: Dict[str, Any], record: Dict[str, Any]) -> Optional[str]:
    stamp = attrs.get("event.timestamp")
    if isinstance(stamp, str) and stamp:
        return stamp
    try:
        nanos = int(record.get("timeUnixNano") or 0)
    except (TypeError, ValueError):
        nanos = 0
    if nanos <= 0:
        return None
    return datetime.fromtimestamp(nanos / 1e9, tz=timezone.utc).isoformat()


def _bounded(prefix: str, value: str) -> str:
    candidate = f"{prefix}{value}"
    if len(candidate) <= _ID_WIDTH:
        return candidate
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:40]


def _billing(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        if candidate in ("api", "subscription"):
            return candidate
    return None


def _event(
    *,
    source: str,
    event_id: Optional[str],
    session: Optional[str],
    subagent: Optional[str],
    model: str,
    timestamp: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int,
    tools: List[str],
    metadata: Dict[str, Any],
    cost: float = 0.0,
    latency_ms: int = 0,
) -> Dict[str, Any]:
    """The same shape the SDK's local collector sends for the same call."""
    metadata = {"source": source, "session_id": session, **metadata}
    event = {
        "event_id": event_id,
        "agent_name": f"{source}/{subagent}" if subagent else source,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens or None,
        "cache_write_tokens": cache_write_tokens or None,
        "cost": cost,
        "latency_ms": latency_ms,
        "timestamp": timestamp,
        "success": True,
        "trace_id": session[:_ID_WIDTH] if session else None,
        "workflow": source,
        "step_name": subagent or "main",
        "depth": 1 if subagent else 0,
        "tool_name": None,
        "metadata": {key: value for key, value in metadata.items() if value is not None},
    }
    _set_tools(event, tools)
    return event


def _set_tools(event: Dict[str, Any], tools: List[str]) -> None:
    if not tools:
        return
    event["tool_name"] = tools[0]
    event["metadata"]["_ac_caps"] = {"tools": True}
    if len(tools) > 1:
        event["metadata"]["tools"] = tools


# Codex's hooks call its shell tool "Bash", and that is the name a guardrail is
# written against; its telemetry uses the internal names. "exec" is the script
# wrapper around the real call in code mode, reported alongside it.
_CODEX_TOOL_NAMES = {"exec_command": "Bash", "shell": "Bash", "local_shell": "Bash"}
_CODEX_WRAPPER = "exec"


def _codex_tools(reported: List[str]) -> List[str]:
    real = [name for name in reported if name != _CODEX_WRAPPER] or reported
    return [_CODEX_TOOL_NAMES.get(name, name) for name in real]


def _claude_code(
    attrs: Dict[str, Any], record: Dict[str, Any], tools: List[str]
) -> Optional[Dict[str, Any]]:
    model, timestamp = attrs.get("model"), _timestamp(attrs, record)
    if not model or not timestamp:
        return None
    cache_read = _count(attrs, "cache_read_tokens")
    request_id = attrs.get("request_id")
    try:
        cost = max(float(attrs.get("cost_usd") or 0.0), 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return _event(
        source="claude-code",
        # The API request id, which is also what the SDK's collector sends for
        # the same call read from the transcript. A machine that does both is
        # therefore counted once.
        event_id=_bounded("cc-", str(request_id)) if request_id else None,
        session=attrs.get("session.id"),
        subagent=attrs.get("agent.name"),
        model=str(model),
        timestamp=timestamp,
        # Anthropic reports cache reads beside input_tokens, not inside it.
        input_tokens=_count(attrs, "input_tokens") + cache_read,
        output_tokens=_count(attrs, "output_tokens"),
        cached_tokens=cache_read,
        cache_write_tokens=_count(attrs, "cache_creation_tokens"),
        tools=tools,
        cost=cost,
        latency_ms=_count(attrs, "duration_ms"),
        metadata={
            "user_id": attrs.get("user.email") or attrs.get("user.account_uuid"),
            "billing": _billing(attrs.get(BILLING_ATTRIBUTE)),
            "surface": attrs.get("terminal.type"),
            "client_version": attrs.get("app.version"),
        },
    )


def _codex(attrs: Dict[str, Any], record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    model, timestamp = attrs.get("model"), _timestamp(attrs, record)
    if not model or not timestamp:
        return None
    session = attrs.get("conversation.id")
    input_tokens = _count(attrs, "input_token_count")
    output_tokens = _count(attrs, "output_token_count")
    # A completed response with no output is Codex warming its connection
    # before the turn. Codex leaves it out of its own usage totals and its
    # session log, so it is left out here.
    if not output_tokens:
        return None
    cached_tokens = _count(attrs, "cached_token_count")
    # Codex puts no response id on the record, so the id is derived from the
    # session and the counts. The SDK's collector derives the same id from
    # Codex's session log, so a machine reporting both ways is counted once,
    # and a retried export is a no-op.
    identity = f"{session}|{input_tokens}|{cached_tokens}|{output_tokens}"
    auth = str(attrs.get("auth_mode") or "").lower()
    return _event(
        source="codex",
        event_id="cx-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40],
        session=session,
        subagent=None,
        model=str(model),
        timestamp=timestamp,
        # OpenAI convention already: cached tokens are a subset of the input.
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=_count(attrs, "cache_write_token_count"),
        tools=[],
        metadata={
            "user_id": attrs.get("user.email") or attrs.get("user.account_id"),
            "billing": _billing(
                attrs.get(BILLING_ATTRIBUTE),
                "subscription" if "chatgpt" in auth else None,
                "api" if "api" in auth else None,
            ),
            "surface": attrs.get("originator") or attrs.get("terminal.type"),
            "client_version": attrs.get("app.version"),
            "reasoning_tokens": _count(attrs, "reasoning_token_count") or None,
        },
    )


def _records(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Every log record, with its resource's attributes folded underneath."""
    for resource_logs in payload.get("resourceLogs") or []:
        if not isinstance(resource_logs, dict):
            continue
        shared = _attributes((resource_logs.get("resource") or {}).get("attributes"))
        for scope_logs in resource_logs.get("scopeLogs") or []:
            if not isinstance(scope_logs, dict):
                continue
            for record in scope_logs.get("logRecords") or []:
                if isinstance(record, dict):
                    attrs = {**shared, **_attributes(record.get("attributes"))}
                    yield {"record": record, "attrs": attrs}


def events_from_logs(payload: Any) -> List[Dict[str, Any]]:
    """Model calls found in an OTLP logs payload, as event dicts.

    Anything that is not a recognised usage record -- prompts, tool results,
    settings, unknown services -- contributes nothing.
    """
    if not isinstance(payload, dict):
        return []
    events: List[Dict[str, Any]] = []
    # Claude Code logs a tool decision just before the request that asked for
    # the tool. Decisions seen since a session's previous request therefore
    # belong to its next one. A decision whose request lands in a later export
    # is simply not attributed; it is never given to the wrong call.
    pending_tools: Dict[Any, List[str]] = {}
    # Codex is the other way round: a tool's result is logged after the
    # response that called it, so results belong to the conversation's latest
    # response in this export.
    codex_calls: List[Dict[str, Any]] = []
    latest_codex: Dict[Any, Dict[str, Any]] = {}

    for item in _records(payload):
        record, attrs = item["record"], item["attrs"]
        name = str(attrs.get("event.name") or _value(record.get("body")) or "")
        session = attrs.get("session.id")

        if name in ("tool_decision", "claude_code.tool_decision"):
            if attrs.get("tool_name") and attrs.get("decision") != "reject":
                pending_tools.setdefault(session, []).append(str(attrs["tool_name"]))
        elif name in ("api_request", "claude_code.api_request"):
            event = _claude_code(attrs, record, pending_tools.pop(session, []))
            if event:
                events.append(event)
        elif name == "codex.sse_event" and attrs.get("event.kind") == "response.completed":
            event = _codex(attrs, record)
            if event:
                events.append(event)
                call = {"event": event, "tools": []}
                codex_calls.append(call)
                latest_codex[attrs.get("conversation.id")] = call
        elif name == "codex.tool_result" and attrs.get("tool_name"):
            call = latest_codex.get(attrs.get("conversation.id"))
            if call is not None:
                call["tools"].append(str(attrs["tool_name"]))

    for call in codex_calls:
        _set_tools(call["event"], _codex_tools(call["tools"]))
    return events
