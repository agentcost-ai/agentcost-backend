"""
AgentCost Backend - Guardrail Service

Declared agent policy evaluated against what the events table actually saw.
A guardrail has four optional boundaries: permitted tools, read-only (via
project tool tags), permitted models, and per-run limits (tool calls, cost).
Deliberately separate from success-rate math: success measures whether a call
raised; compliance measures whether an agent stayed inside its boundary.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select, func, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import (
    Event,
    AgentGuardrail,
    ToolAccessTag,
    Project,
    Notification,
)
from ..models.user_models import User, ProjectMember, UserRole
from . import webhook_service
from .notification_service import NotificationService
from ..models.schemas import (
    GuardrailUpsert,
    GuardrailBreach,
    AgentCompliance,
    GuardrailComplianceResponse,
    ToolAccessTagResponse,
    ToolUsage,
    ModelUsage,
    RunStats,
    DailyBreaches,
)


logger = logging.getLogger(__name__)

# One alert per (agent, subject, kind) per window: a breaching agent that keeps
# calling the tool must not page anyone on every batch.
ALERT_DEDUPE_WINDOW = timedelta(hours=1)
BREACH_NOTIFICATION_TYPE = "guardrail_breach"
BREACH_WEBHOOK_EVENT = "guardrail.breach"

# Kinds whose subject is a run (trace_id) rather than a tool or model.
RUN_KINDS = ("tool_calls_over_limit", "run_cost_over_limit")

_SEVERITY = {
    "write_in_readonly": "critical",
    "run_cost_over_limit": "critical",
    "undeclared_tool": "warning",
    "undeclared_model": "warning",
    "tool_calls_over_limit": "warning",
}


def breach_kind(
    guardrail: AgentGuardrail, tool_name: str, tags: Dict[str, str]
) -> Optional[str]:
    """Why a tool call breaches a guardrail, or None if it does not.

    Untagged tools under a read-only guardrail are not a breach: their access
    is unknown, and unknown is reported rather than silently judged.
    """
    if guardrail.allowed_tools is not None and tool_name not in guardrail.allowed_tools:
        return "undeclared_tool"
    if guardrail.read_only and tags.get(tool_name) == "write":
        return "write_in_readonly"
    return None


def model_breach(guardrail: AgentGuardrail, model: str) -> Optional[str]:
    """``undeclared_model`` when the agent has a model allow-list and this
    model is not on it."""
    if guardrail.allowed_models is not None and model not in guardrail.allowed_models:
        return "undeclared_model"
    return None


def _has_run_limits(guardrail: AgentGuardrail) -> bool:
    return (
        guardrail.max_tool_calls_per_run is not None
        or guardrail.max_cost_per_run_usd is not None
    )


def describe_breach(kind: str, subject: str, limit: Optional[float]) -> str:
    """Human sentence fragment for notifications."""
    if kind == "write_in_readonly":
        return f"called {subject}, a write tool, while declared read-only"
    if kind == "undeclared_tool":
        return f"called {subject}, which is outside its permitted tools"
    if kind == "undeclared_model":
        return f"used {subject}, which is outside its permitted models"
    if kind == "tool_calls_over_limit":
        return f"made more than {int(limit or 0)} tool calls in run {subject}"
    if kind == "run_cost_over_limit":
        return f"spent more than ${limit or 0:.2f} in run {subject}"
    return kind


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile over an ascending list; 0 for an empty list."""
    if not sorted_values:
        return 0.0
    rank = max(1, int(round(pct / 100 * len(sorted_values))))
    return float(sorted_values[min(rank, len(sorted_values)) - 1])


def _day(value) -> str:
    """ISO date for a DATE()/date_trunc result on SQLite or PostgreSQL."""
    return str(value)[:10]


class GuardrailService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Definitions ────────────────────────────────────────────────────

    async def list_guardrails(self, project_id: str) -> List[AgentGuardrail]:
        result = await self.db.execute(
            select(AgentGuardrail)
            .where(AgentGuardrail.project_id == project_id)
            .order_by(AgentGuardrail.agent_name)
        )
        return list(result.scalars().all())

    async def upsert_guardrail(
        self, project_id: str, payload: GuardrailUpsert
    ) -> AgentGuardrail:
        result = await self.db.execute(
            select(AgentGuardrail).where(
                AgentGuardrail.project_id == project_id,
                AgentGuardrail.agent_name == payload.agent_name,
            )
        )
        guardrail = result.scalar_one_or_none()
        if guardrail is None:
            guardrail = AgentGuardrail(
                project_id=project_id, agent_name=payload.agent_name
            )
            self.db.add(guardrail)
            try:
                async with self.db.begin_nested():
                    await self.db.flush()
            except IntegrityError:
                # Lost a concurrent insert race on the unique constraint:
                # the row now exists, so apply the update to it instead.
                result = await self.db.execute(
                    select(AgentGuardrail).where(
                        AgentGuardrail.project_id == project_id,
                        AgentGuardrail.agent_name == payload.agent_name,
                    )
                )
                guardrail = result.scalar_one()
        guardrail.allowed_tools = payload.allowed_tools
        guardrail.read_only = payload.read_only
        guardrail.allowed_models = payload.allowed_models
        guardrail.max_tool_calls_per_run = payload.max_tool_calls_per_run
        guardrail.max_cost_per_run_usd = payload.max_cost_per_run_usd
        guardrail.enabled = payload.enabled
        await self.db.flush()
        # Server-default timestamps are expired by the flush; load them now so
        # response serialization never lazy-loads outside the async context.
        await self.db.refresh(guardrail)
        return guardrail

    async def delete_guardrail(self, project_id: str, agent_name: str) -> bool:
        result = await self.db.execute(
            delete(AgentGuardrail).where(
                AgentGuardrail.project_id == project_id,
                AgentGuardrail.agent_name == agent_name,
            )
        )
        return result.rowcount > 0

    async def list_tool_tags(self, project_id: str) -> List[ToolAccessTag]:
        result = await self.db.execute(
            select(ToolAccessTag)
            .where(ToolAccessTag.project_id == project_id)
            .order_by(ToolAccessTag.tool_name)
        )
        return list(result.scalars().all())

    async def upsert_tool_tag(
        self, project_id: str, tool_name: str, access: str
    ) -> ToolAccessTag:
        result = await self.db.execute(
            select(ToolAccessTag).where(
                ToolAccessTag.project_id == project_id,
                ToolAccessTag.tool_name == tool_name,
            )
        )
        tag = result.scalar_one_or_none()
        if tag is None:
            tag = ToolAccessTag(project_id=project_id, tool_name=tool_name)
            self.db.add(tag)
            try:
                async with self.db.begin_nested():
                    await self.db.flush()
            except IntegrityError:
                result = await self.db.execute(
                    select(ToolAccessTag).where(
                        ToolAccessTag.project_id == project_id,
                        ToolAccessTag.tool_name == tool_name,
                    )
                )
                tag = result.scalar_one()
        tag.access = access
        await self.db.flush()
        await self.db.refresh(tag)
        return tag

    async def delete_tool_tag(self, project_id: str, tool_name: str) -> bool:
        result = await self.db.execute(
            delete(ToolAccessTag).where(
                ToolAccessTag.project_id == project_id,
                ToolAccessTag.tool_name == tool_name,
            )
        )
        return result.rowcount > 0

    # ── Compliance ─────────────────────────────────────────────────────

    async def compliance(
        self, project_id: str, start_time: datetime, end_time: datetime
    ) -> GuardrailComplianceResponse:
        guardrails = {g.agent_name: g for g in await self.list_guardrails(project_id)}
        tags = {t.tool_name: t.access for t in await self.list_tool_tags(project_id)}

        window = (
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        )
        day = func.date(Event.timestamp).label("day")

        # Tool usage per agent and per day (only calls wrapped in track_costs.tool()).
        tools_daily = (
            await self.db.execute(
                select(
                    Event.agent_name,
                    Event.tool_name,
                    day,
                    func.count().label("calls"),
                    func.max(Event.timestamp).label("last_seen"),
                )
                .where(*window, Event.tool_name.isnot(None))
                .group_by(Event.agent_name, Event.tool_name, day)
            )
        ).all()

        # Model usage per agent and per day: every call carries one.
        models_daily = (
            await self.db.execute(
                select(
                    Event.agent_name,
                    Event.model,
                    day,
                    func.count().label("calls"),
                    func.sum(Event.cost).label("cost"),
                    func.max(Event.timestamp).label("last_seen"),
                )
                .where(*window)
                .group_by(Event.agent_name, Event.model, day)
            )
        ).all()

        # All calls and spend per agent, for coverage and context.
        totals = (
            await self.db.execute(
                select(
                    Event.agent_name,
                    func.count().label("calls"),
                    func.sum(Event.cost).label("cost"),
                )
                .where(*window)
                .group_by(Event.agent_name)
            )
        ).all()
        total_by_agent = {row.agent_name: row.calls for row in totals}
        cost_by_agent = {row.agent_name: float(row.cost or 0) for row in totals}

        # Runs: every agent with trace_ids in the window, so run statistics
        # can suggest limits before any are declared.
        runs = (
            await self.db.execute(
                select(
                    Event.agent_name,
                    Event.trace_id,
                    func.count(Event.tool_name).label("tool_calls"),
                    func.sum(Event.cost).label("cost"),
                    func.max(Event.timestamp).label("last_seen"),
                )
                .where(*window, Event.trace_id.isnot(None))
                .group_by(Event.agent_name, Event.trace_id)
            )
        ).all()

        per_agent: Dict[str, AgentCompliance] = {}
        series: Dict[str, Dict[str, int]] = {}

        def active_guardrail(agent_name: str) -> Optional[AgentGuardrail]:
            guardrail = guardrails.get(agent_name)
            return guardrail if guardrail is not None and guardrail.enabled else None

        def entry(agent_name: str) -> AgentCompliance:
            if agent_name not in per_agent:
                guardrail = active_guardrail(agent_name)
                per_agent[agent_name] = AgentCompliance(
                    agent_name=agent_name,
                    status="compliant" if guardrail else "no_guardrail",
                    read_only=bool(guardrail.read_only) if guardrail else False,
                    allowed_tools=guardrail.allowed_tools if guardrail else None,
                    allowed_models=guardrail.allowed_models if guardrail else None,
                    max_tool_calls_per_run=(
                        guardrail.max_tool_calls_per_run if guardrail else None
                    ),
                    max_cost_per_run_usd=(
                        guardrail.max_cost_per_run_usd if guardrail else None
                    ),
                    total_calls=total_by_agent.get(agent_name, 0),
                    total_cost=round(cost_by_agent.get(agent_name, 0.0), 6),
                )
                series[agent_name] = {}
            return per_agent[agent_name]

        def add_breach(comp: AgentCompliance, breach: GuardrailBreach) -> None:
            comp.status = "breach"
            comp.breaches.append(breach)

        def bump_series(agent_name: str, day_value, count: int) -> None:
            bucket = series.setdefault(agent_name, {})
            key = _day(day_value)
            bucket[key] = bucket.get(key, 0) + count

        # ── Tools ──
        tool_totals: Dict[Tuple[str, str], dict] = {}
        for row in tools_daily:
            comp = entry(row.agent_name)
            comp.tracked_tool_calls += row.calls
            agg = tool_totals.setdefault(
                (row.agent_name, row.tool_name), {"calls": 0, "last_seen": None}
            )
            agg["calls"] += row.calls
            if row.last_seen and (agg["last_seen"] is None or row.last_seen > agg["last_seen"]):
                agg["last_seen"] = row.last_seen
            guardrail = active_guardrail(row.agent_name)
            if guardrail and breach_kind(guardrail, row.tool_name, tags):
                bump_series(row.agent_name, row.day, row.calls)

        for (agent_name, tool_name), agg in sorted(tool_totals.items()):
            comp = entry(agent_name)
            comp.observed_tools.append(tool_name)
            guardrail = active_guardrail(agent_name)
            kind = breach_kind(guardrail, tool_name, tags) if guardrail else None
            comp.tool_usage.append(
                ToolUsage(
                    tool_name=tool_name,
                    calls=agg["calls"],
                    last_seen=agg["last_seen"],
                    access=tags.get(tool_name),
                    breach_kind=kind,
                )
            )
            if guardrail is None:
                continue
            if kind is None and guardrail.read_only and tool_name not in tags:
                comp.unknown_access_tools.append(tool_name)
            if kind:
                add_breach(
                    comp,
                    GuardrailBreach(
                        kind=kind,
                        subject=tool_name,
                        count=agg["calls"],
                        last_seen=agg["last_seen"],
                    ),
                )

        # ── Models ──
        model_totals: Dict[Tuple[str, str], dict] = {}
        for row in models_daily:
            entry(row.agent_name)
            agg = model_totals.setdefault(
                (row.agent_name, row.model), {"calls": 0, "cost": 0.0, "last_seen": None}
            )
            agg["calls"] += row.calls
            agg["cost"] += float(row.cost or 0)
            if row.last_seen and (agg["last_seen"] is None or row.last_seen > agg["last_seen"]):
                agg["last_seen"] = row.last_seen
            guardrail = active_guardrail(row.agent_name)
            if guardrail and model_breach(guardrail, row.model):
                bump_series(row.agent_name, row.day, row.calls)

        for (agent_name, model), agg in sorted(model_totals.items()):
            comp = entry(agent_name)
            comp.observed_models.append(model)
            guardrail = active_guardrail(agent_name)
            kind = model_breach(guardrail, model) if guardrail else None
            comp.model_usage.append(
                ModelUsage(
                    model=model,
                    calls=agg["calls"],
                    cost=round(agg["cost"], 6),
                    permitted=kind is None,
                )
            )
            if kind:
                add_breach(
                    comp,
                    GuardrailBreach(
                        kind=kind,
                        subject=model,
                        count=agg["calls"],
                        last_seen=agg["last_seen"],
                    ),
                )

        # ── Runs ──
        run_rows: Dict[str, List[tuple]] = {}
        for row in runs:
            run_rows.setdefault(row.agent_name, []).append(row)

        for agent_name, rows in run_rows.items():
            comp = entry(agent_name)
            comp.runs_seen = len(rows)
            tool_counts = sorted(int(r.tool_calls or 0) for r in rows)
            costs = sorted(float(r.cost or 0) for r in rows)
            comp.run_stats = RunStats(
                runs=len(rows),
                p50_tool_calls=_percentile(tool_counts, 50),
                p95_tool_calls=_percentile(tool_counts, 95),
                max_tool_calls=int(tool_counts[-1]) if tool_counts else 0,
                p50_cost=round(_percentile(costs, 50), 6),
                p95_cost=round(_percentile(costs, 95), 6),
                max_cost=round(costs[-1], 6) if costs else 0.0,
            )

            guardrail = active_guardrail(agent_name)
            if guardrail is None or not _has_run_limits(guardrail):
                continue
            worst: Dict[str, dict] = {}
            for r in rows:
                checks = (
                    ("tool_calls_over_limit", guardrail.max_tool_calls_per_run, int(r.tool_calls or 0)),
                    ("run_cost_over_limit", guardrail.max_cost_per_run_usd, float(r.cost or 0)),
                )
                for kind, limit, observed_value in checks:
                    if limit is None or observed_value <= limit:
                        continue
                    agg = worst.setdefault(
                        kind,
                        {"count": 0, "observed": observed_value, "subject": r.trace_id,
                         "limit": limit, "last_seen": r.last_seen},
                    )
                    agg["count"] += 1
                    if observed_value > agg["observed"]:
                        agg["observed"], agg["subject"] = observed_value, r.trace_id
                    if r.last_seen and (agg["last_seen"] is None or r.last_seen > agg["last_seen"]):
                        agg["last_seen"] = r.last_seen
                    if r.last_seen:
                        bump_series(agent_name, r.last_seen.date(), 1)
            for kind, agg in worst.items():
                add_breach(
                    comp,
                    GuardrailBreach(
                        kind=kind,
                        subject=agg["subject"],
                        count=agg["count"],
                        limit=float(agg["limit"]),
                        observed=round(float(agg["observed"]), 6),
                        last_seen=agg["last_seen"],
                    ),
                )

        # Agents with a declared guardrail but no activity, and agents active
        # in the window with no tool instrumentation at all.
        for agent_name in guardrails:
            entry(agent_name)
        for agent_name in total_by_agent:
            entry(agent_name)

        for agent_name, comp in per_agent.items():
            comp.breach_series = [
                DailyBreaches(day=d, count=n) for d, n in sorted(series.get(agent_name, {}).items())
            ]

        agents = sorted(
            per_agent.values(),
            key=lambda c: (
                {"breach": 0, "compliant": 1, "no_guardrail": 2}[c.status],
                -c.total_cost,
                c.agent_name,
            ),
        )
        return GuardrailComplianceResponse(
            agents=agents,
            tool_tags=[
                ToolAccessTagResponse(tool_name=name, access=access)
                for name, access in sorted(tags.items())
            ],
            start_time=start_time,
            end_time=end_time,
            total_calls=sum(total_by_agent.values()),
            tool_tracked_calls=sum(c.tracked_tool_calls for c in per_agent.values()),
        )

    # ── Ingest-time alerts ────────────────────────────────────────────

    async def alert_breaches(self, project: Project, rows: Sequence[Event]) -> int:
        """Raise alerts for guardrail breaches in a freshly ingested batch.

        Owner and admin members get an in-app notification and the project
        webhook (if any) receives ``guardrail.breach``, once per
        (agent, subject, kind) per ALERT_DEDUPE_WINDOW. Per-run limits are
        judged on the run's stored totals so a limit crossed across several
        batches still alerts. Returns the number of distinct breaches alerted.
        Never raises: the events are already stored and alerting must not
        undo telemetry.
        """
        guardrails = {
            g.agent_name: g
            for g in await self.list_guardrails(project.id)
            if g.enabled
        }
        if not guardrails:
            return 0
        tags = {t.tool_name: t.access for t in await self.list_tool_tags(project.id)}

        # key -> [count, limit, observed]
        found: Dict[Tuple[str, str, str], list] = {}

        def note(key, limit=None, observed=None):
            agg = found.setdefault(key, [0, limit, observed])
            agg[0] += 1

        traces: Dict[Tuple[str, str], AgentGuardrail] = {}
        for row in rows:
            guardrail = guardrails.get(row.agent_name)
            if guardrail is None:
                continue
            if row.tool_name:
                kind = breach_kind(guardrail, row.tool_name, tags)
                if kind:
                    note((row.agent_name, row.tool_name, kind))
            kind = model_breach(guardrail, row.model)
            if kind:
                note((row.agent_name, row.model, kind))
            if row.trace_id and _has_run_limits(guardrail):
                traces[(row.agent_name, row.trace_id)] = guardrail

        if traces:
            totals = await self.db.execute(
                select(
                    Event.agent_name,
                    Event.trace_id,
                    func.count(Event.tool_name).label("tool_calls"),
                    func.sum(Event.cost).label("cost"),
                )
                .where(
                    Event.project_id == project.id,
                    Event.trace_id.in_({t for _, t in traces}),
                    Event.agent_name.in_({a for a, _ in traces}),
                )
                .group_by(Event.agent_name, Event.trace_id)
            )
            for row in totals:
                guardrail = traces.get((row.agent_name, row.trace_id))
                if guardrail is None:
                    continue
                limit = guardrail.max_tool_calls_per_run
                if limit is not None and row.tool_calls > limit:
                    found[(row.agent_name, row.trace_id, "tool_calls_over_limit")] = [
                        1, float(limit), float(row.tool_calls),
                    ]
                limit = guardrail.max_cost_per_run_usd
                cost = float(row.cost or 0)
                if limit is not None and cost > limit:
                    found[(row.agent_name, row.trace_id, "run_cost_over_limit")] = [
                        1, float(limit), round(cost, 6),
                    ]
        if not found:
            return 0

        since = datetime.now(timezone.utc) - ALERT_DEDUPE_WINDOW
        recent = await self.db.execute(
            select(Notification.payload).where(
                Notification.project_id == project.id,
                Notification.type == BREACH_NOTIFICATION_TYPE,
                Notification.created_at >= since,
            )
        )
        seen = {
            (p.get("agent_name"), p.get("subject"), p.get("kind"))
            for (p,) in recent
            if isinstance(p, dict)
        }
        fresh = {k: v for k, v in found.items() if k not in seen}
        if not fresh:
            return 0

        recipients = await self._resolve_recipients(project.id)
        notifications = NotificationService(self.db)
        for (agent_name, subject, kind), (count, limit, observed) in fresh.items():
            payload = {
                "project_id": project.id,
                "agent_name": agent_name,
                "kind": kind,
                "subject": subject,
                "count": count,
                "limit": limit,
                "observed": observed,
                "observed_at": datetime.now(timezone.utc).isoformat(),
            }
            if kind in RUN_KINDS:
                title = f"Guardrail breach: {agent_name} exceeded a per-run limit"
                body = f"{agent_name} {describe_breach(kind, subject, limit)}."
            else:
                title = f"Guardrail breach: {agent_name} used {subject}"
                body = (
                    f"{agent_name} {describe_breach(kind, subject, limit)} — "
                    f"{count} call{'s' if count != 1 else ''} in the latest batch."
                )
            for user in recipients:
                await notifications.create(
                    user_id=user.id,
                    type=BREACH_NOTIFICATION_TYPE,
                    title=title,
                    body=body,
                    severity=_SEVERITY.get(kind, "warning"),
                    link="/guardrails",
                    project_id=project.id,
                    payload=payload,
                )
            if getattr(project, "webhook_url", None):
                webhook_service.dispatch(
                    project.webhook_url,
                    BREACH_WEBHOOK_EVENT,
                    payload,
                    secret=getattr(project, "webhook_secret", None),
                )
        await self.db.flush()
        return len(fresh)

    async def _resolve_recipients(self, project_id: str) -> List[User]:
        """Owner plus accepted admin members, deduplicated, active."""
        owner_stmt = (
            select(User)
            .join(Project, Project.owner_id == User.id)
            .where(Project.id == project_id)
        )
        admins_stmt = (
            select(User)
            .join(ProjectMember, ProjectMember.user_id == User.id)
            .where(
                ProjectMember.project_id == project_id,
                ProjectMember.role == UserRole.ADMIN.value,
                ProjectMember.accepted_at.isnot(None),
            )
        )
        seen: Dict[str, User] = {}
        for stmt in (owner_stmt, admins_stmt):
            for user in (await self.db.execute(stmt)).scalars().all():
                if getattr(user, "is_active", True) and user.id not in seen:
                    seen[user.id] = user
        return list(seen.values())
