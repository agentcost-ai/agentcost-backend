"""
AgentCost Backend - Prometheus Metrics Export

Cost telemetry in the format the rest of an enterprise observability stack
already speaks. Without this, AgentCost is a place people have to go and look;
with it, cost sits on the same dashboards and alert rules as everything else.

Exposition format only — no client library dependency. The output is small and
entirely generated from aggregates, so rendering it by hand is less code than
wiring up a registry, and avoids the double-counting that a global registry
causes under multiple workers.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.db_models import Event, Project
from ..services.budget_service import BudgetService
from ..utils.auth import validate_project_access

router = APIRouter(prefix="/v1/metrics", tags=["Metrics"])

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Cardinality guard. Prometheus degrades badly with unbounded label values, and
# a project with thousands of agents or models would emit a series per one.
_MAX_SERIES_PER_DIMENSION = 50


def _escape(value: str) -> str:
    """Escape a Prometheus label value."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _line(name: str, labels: dict, value) -> str:
    if labels:
        rendered = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items() if v is not None)
        return f"{name}{{{rendered}}} {value}"
    return f"{name} {value}"


@router.get("", response_class=PlainTextResponse)
async def prometheus_metrics(
    window_hours: int = Query(24, ge=1, le=720, description="Aggregation window in hours"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """
    Prometheus exposition of this project's cost metrics.

    Authenticated with the project API key. Point a scrape job at it:

        scrape_configs:
          - job_name: agentcost
            metrics_path: /v1/metrics
            authorization:
              credentials: <project_api_key>
            static_configs:
              - targets: ['api.agentcost.tech']

    Values are windowed counters, not monotonic ones — `window_hours` sets the
    lookback. Use them with `max_over_time` style queries rather than `rate()`,
    which assumes a counter that only ever climbs.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=window_hours)

    scope = (
        Event.project_id == project.id,
        Event.timestamp >= start_time,
        Event.timestamp <= end_time,
    )

    totals = (
        await db.execute(
            select(
                func.count(Event.id),
                func.coalesce(func.sum(Event.cost), 0.0),
                func.coalesce(func.sum(Event.total_tokens), 0),
                func.coalesce(func.sum(Event.cached_tokens), 0),
                func.coalesce(func.avg(Event.latency_ms), 0.0),
                func.coalesce(
                    func.sum(case((Event.success == False, 1), else_=0)), 0  # noqa: E712
                ),
            ).where(*scope)
        )
    ).one()

    calls, cost, tokens, cached, latency, errors = totals
    labels = {"project": project.id}

    # Windowed gauges, so none carry the `_total` suffix Prometheus reserves
    # for monotonic counters — a name that invites rate() would mislead.
    lines = [
        "# HELP agentcost_calls LLM calls in the window.",
        "# TYPE agentcost_calls gauge",
        _line("agentcost_calls", labels, int(calls or 0)),
        "# HELP agentcost_cost_usd Total cost in USD in the window.",
        "# TYPE agentcost_cost_usd gauge",
        _line("agentcost_cost_usd", labels, round(float(cost or 0.0), 6)),
        "# HELP agentcost_tokens Total tokens in the window.",
        "# TYPE agentcost_tokens gauge",
        _line("agentcost_tokens", labels, int(tokens or 0)),
        "# HELP agentcost_cached_tokens Prompt-cache hits in the window.",
        "# TYPE agentcost_cached_tokens gauge",
        _line("agentcost_cached_tokens", labels, int(cached or 0)),
        "# HELP agentcost_latency_ms_avg Mean call latency in the window.",
        "# TYPE agentcost_latency_ms_avg gauge",
        _line("agentcost_latency_ms_avg", labels, round(float(latency or 0.0), 2)),
        "# HELP agentcost_errors Failed calls in the window.",
        "# TYPE agentcost_errors gauge",
        _line("agentcost_errors", labels, int(errors or 0)),
    ]

    # Per-model and per-agent cost, capped so one noisy project cannot blow up
    # a shared Prometheus instance's series count.
    for dimension, column in (("model", Event.model), ("agent", Event.agent_name)):
        rows = await db.execute(
            select(column, func.coalesce(func.sum(Event.cost), 0.0))
            .where(*scope)
            .group_by(column)
            .order_by(func.sum(Event.cost).desc())
            .limit(_MAX_SERIES_PER_DIMENSION)
        )
        metric = f"agentcost_cost_usd_by_{dimension}"
        lines.append(f"# HELP {metric} Cost in USD by {dimension}, top {_MAX_SERIES_PER_DIMENSION}.")
        lines.append(f"# TYPE {metric} gauge")
        for key, value in rows:
            lines.append(
                _line(metric, {**labels, dimension: key}, round(float(value or 0.0), 6))
            )

    # Budget position, so an alert rule can fire on utilization without the
    # alerting system needing to know the budget figure itself.
    if BudgetService.has_budget(project):
        state = await BudgetService(db).budget_state(project)
        lines += [
            "# HELP agentcost_budget_utilization_percent Month-to-date spend against budget.",
            "# TYPE agentcost_budget_utilization_percent gauge",
            _line(
                "agentcost_budget_utilization_percent",
                labels,
                state.get("utilization_percent") or 0.0,
            ),
            "# HELP agentcost_budget_remaining Budget remaining this period, in the project's currency.",
            "# TYPE agentcost_budget_remaining gauge",
            _line("agentcost_budget_remaining", labels, state.get("remaining") or 0.0),
        ]

    return PlainTextResponse("\n".join(lines) + "\n", media_type=CONTENT_TYPE)
