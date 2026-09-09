"""
AgentCost Backend - Agent insight service

Per-agent aggregates for the Agents list and the agent detail page. Composes
the analytics, trace, report and guardrail services under an agent filter and
adds what only makes sense per agent: previous-window delta, model mix, cache
economics, failed-call spend, repeated work and one computed signal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import Event
from ..models.schemas import (
    AgentDailyPoint,
    AgentDetail,
    AgentModelShare,
    AgentOutcomes,
    AgentSignal,
    AgentStepCost,
    AgentSummary,
    DimensionStat,
)
from ..utils.sql_dialect import dialect_name, utc_timestamp
from .analytics_service import AnalyticsService
from .guardrail_service import GuardrailService, describe_breach
from .pricing_service import PricingService
from .report_service import ReportService
from .trace_service import TraceService

# Signal thresholds. Repeated work and failed spend are shares of the agent's
# own spend in the window, so a large agent is not flagged for pocket change.
REPEATED_WORK_SHARE = 0.05
FAILED_SPEND_SHARE = 0.02
CLASSIFIER_MAX_OUTPUT_TOKENS = 16
CLASSIFIER_MIN_CALLS = 100
STEP_ROWS_CAP = 50_000


def _pct(part: float, whole: float) -> Optional[float]:
    return round(part / whole * 100, 2) if whole else None


def _nearest_rank(sorted_values: List[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


class AgentInsightService:
    """Everything the Agents pages need, computed per agent."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self._dialect = dialect_name(db)

    def _window(self, project_id: str, start: datetime, end: datetime, agent_name: Optional[str] = None):
        filters = [
            Event.project_id == project_id,
            Event.timestamp >= start,
            Event.timestamp <= end,
        ]
        if agent_name:
            filters.append(Event.agent_name == agent_name)
        return filters

    # ── Summaries (list page) ────────────────────────────────────────────

    async def summaries(
        self,
        project_id: str,
        start: datetime,
        end: datetime,
        limit: int = 50,
        agent_name: Optional[str] = None,
        compliance: Optional[Dict[str, Any]] = None,
    ) -> List[AgentSummary]:
        window = self._window(project_id, start, end, agent_name)

        base = (
            select(
                Event.agent_name,
                func.count(Event.id).label("calls"),
                func.sum(Event.total_tokens).label("tokens"),
                func.sum(Event.cost).label("cost"),
                func.avg(Event.latency_ms).label("latency"),
                func.sum(case((Event.success == True, 1), else_=0)).label("ok"),  # noqa: E712
                func.sum(case((Event.success == False, 1), else_=0)).label("failed"),  # noqa: E712
                func.sum(case((Event.success == False, Event.cost), else_=0.0)).label("failed_cost"),  # noqa: E712
                func.count(func.distinct(Event.trace_id)).label("runs"),
                func.count(func.distinct(Event.user_id)).label("developers"),
                func.count(func.distinct(Event.session_id)).label("sessions"),
                func.min(Event.timestamp).label("first_seen"),
                func.max(Event.timestamp).label("last_seen"),
                func.max(Event.output_tokens).label("max_output"),
            )
            .where(*window)
            .group_by(Event.agent_name)
            .order_by(desc("cost"))
            .limit(limit)
        )
        rows = (await self.db.execute(base)).all()
        if not rows:
            return []
        names = [row.agent_name for row in rows]

        # Share of the whole project, not of the top-N slice.
        project_cost = float(
            (await self.db.execute(
                select(func.sum(Event.cost)).where(*self._window(project_id, start, end))
            )).scalar() or 0.0
        )

        span = end - start
        previous = {
            row.agent_name: float(row.cost or 0.0)
            for row in await self.db.execute(
                select(Event.agent_name, func.sum(Event.cost).label("cost"))
                .where(
                    Event.project_id == project_id,
                    Event.timestamp >= start - span,
                    Event.timestamp < start,
                    Event.agent_name.in_(names),
                )
                .group_by(Event.agent_name)
            )
        }

        models_rows = (
            await self.db.execute(
                select(
                    Event.agent_name,
                    Event.model,
                    func.count(Event.id).label("calls"),
                    func.sum(Event.cost).label("cost"),
                    func.sum(Event.input_tokens).label("input_tokens"),
                    func.sum(Event.cached_tokens).label("cached"),
                    func.sum(Event.cache_write_tokens).label("written"),
                )
                .where(*window, Event.agent_name.in_(names))
                .group_by(Event.agent_name, Event.model)
            )
        ).all()

        day = func.date(utc_timestamp(self._dialect)).label("day")
        daily_rows = (
            await self.db.execute(
                select(
                    Event.agent_name,
                    day,
                    func.count(Event.id).label("calls"),
                    func.sum(Event.cost).label("cost"),
                    func.sum(case((Event.success == False, Event.cost), else_=0.0)).label("failed_cost"),  # noqa: E712
                )
                .where(*window, Event.agent_name.in_(names))
                .group_by(Event.agent_name, day)
                .order_by(day)
            )
        ).all()

        # Identical input repeated inside one run: the redundant occurrences
        # are waste. Same definition as TraceService.detect_repeated_work, but
        # folded per agent in SQL so a busy 90-day window never streams every
        # duplicate group into Python.
        groups = (
            select(
                Event.agent_name.label("agent_name"),
                Event.trace_id.label("trace_id"),
                func.count(Event.id).label("occurrences"),
                func.sum(Event.cost).label("spend"),
            )
            .where(
                *window,
                Event.agent_name.in_(names),
                Event.trace_id.isnot(None),
                Event.input_hash.isnot(None),
            )
            .group_by(Event.agent_name, Event.trace_id, Event.step_name, Event.input_hash, Event.model)
            .having(func.count(Event.id) > 1)
            .subquery()
        )
        repeated_rows = (
            await self.db.execute(
                select(
                    groups.c.agent_name,
                    # spend * (n - 1) / n per group: everything beyond the first call.
                    func.sum(groups.c.spend - groups.c.spend / groups.c.occurrences).label("wasted"),
                    func.count(func.distinct(groups.c.trace_id)).label("runs"),
                ).group_by(groups.c.agent_name)
            )
        ).all()

        repeated_cost: Dict[str, float] = {}
        repeated_runs: Dict[str, int] = {}
        for row in repeated_rows:
            repeated_cost[row.agent_name] = float(row.wasted or 0.0)
            repeated_runs[row.agent_name] = int(row.runs or 0)

        # Cache economics, priced per model the way ingest prices them.
        models: Dict[str, List[AgentModelShare]] = {}
        cached_tokens: Dict[str, int] = {}
        input_tokens: Dict[str, int] = {}
        cache_savings: Dict[str, float] = {}
        pricing_service = PricingService(self.db)
        pricing_cache: Dict[str, Optional[dict]] = {}
        try:
            for row in models_rows:
                cached = int(row.cached or 0)
                written = int(row.written or 0)
                inputs = int(row.input_tokens or 0)
                cached_tokens[row.agent_name] = cached_tokens.get(row.agent_name, 0) + cached
                input_tokens[row.agent_name] = input_tokens.get(row.agent_name, 0) + inputs
                models.setdefault(row.agent_name, []).append(
                    AgentModelShare(
                        model=row.model,
                        calls=int(row.calls or 0),
                        cost=round(float(row.cost or 0.0), 6),
                        cached_share=_pct(cached, inputs) if cached else None,
                    )
                )
                if not cached and not written:
                    continue
                if row.model not in pricing_cache:
                    pricing_cache[row.model] = await pricing_service.get_model_pricing(row.model)
                pricing = pricing_cache[row.model]
                if not pricing:
                    continue
                input_rate = pricing.get("input") or 0.0
                cached_rate = pricing.get("cached_input")
                write_rate = pricing.get("cache_write")
                saved = 0.0
                if cached and cached_rate is not None:
                    saved += (cached / 1000) * (input_rate - cached_rate)
                if written and write_rate is not None:
                    saved -= (written / 1000) * (write_rate - input_rate)
                cache_savings[row.agent_name] = cache_savings.get(row.agent_name, 0.0) + saved
        finally:
            await pricing_service.close()

        daily: Dict[str, List[AgentDailyPoint]] = {}
        for row in daily_rows:
            daily.setdefault(row.agent_name, []).append(
                AgentDailyPoint(
                    day=str(row.day)[:10],
                    cost=round(float(row.cost or 0.0), 6),
                    calls=int(row.calls or 0),
                    failed_cost=round(float(row.failed_cost or 0.0), 6),
                )
            )

        summaries: List[AgentSummary] = []
        for row in rows:
            name = row.agent_name
            calls = int(row.calls or 0)
            cost = float(row.cost or 0.0)
            ok = int(row.ok or 0)
            failed = int(row.failed or 0)
            failed_cost = float(row.failed_cost or 0.0)
            runs = int(row.runs or 0)
            prev = previous.get(name, 0.0)
            rep_cost = repeated_cost.get(name, 0.0)
            rep_runs = repeated_runs.get(name, 0)
            inputs = input_tokens.get(name, 0)
            cached = cached_tokens.get(name, 0)
            agent_models = sorted(models.get(name, []), key=lambda m: m.cost, reverse=True)
            agent_compliance = (compliance or {}).get(name)

            summaries.append(
                AgentSummary(
                    agent_name=name,
                    total_calls=calls,
                    total_tokens=int(row.tokens or 0),
                    total_cost=round(cost, 6),
                    avg_latency_ms=round(float(row.latency or 0.0), 2),
                    success_rate=round((ok / calls * 100) if calls else 100.0, 2),
                    share_percent=round((cost / project_cost * 100) if project_cost else 0.0, 2),
                    previous_cost=round(prev, 6),
                    cost_change_percent=round((cost - prev) / prev * 100, 1) if prev > 0 else None,
                    models=agent_models,
                    runs=runs,
                    cost_per_run=round(cost / runs, 6) if runs else None,
                    calls_per_run=round(calls / runs, 2) if runs else None,
                    cached_share=_pct(cached, inputs) if cached else None,
                    cache_savings=round(cache_savings.get(name, 0.0), 6),
                    failed_calls=failed,
                    failed_cost=round(failed_cost, 6),
                    repeated_cost=round(rep_cost, 6),
                    repeated_runs=rep_runs,
                    developers=int(row.developers or 0),
                    sessions=int(row.sessions or 0),
                    first_seen=row.first_seen,
                    last_seen=row.last_seen,
                    daily=daily.get(name, []),
                    signal=self._signal(
                        cost=cost,
                        calls=calls,
                        runs=runs,
                        failed=failed,
                        failed_cost=failed_cost,
                        repeated_cost=rep_cost,
                        repeated_runs=rep_runs,
                        max_output=int(row.max_output or 0),
                        compliance=agent_compliance,
                    ),
                )
            )
        return summaries

    @staticmethod
    def _signal(
        *,
        cost: float,
        calls: int,
        runs: int,
        failed: int,
        failed_cost: float,
        repeated_cost: float,
        repeated_runs: int,
        max_output: int,
        compliance: Optional[Any],
    ) -> AgentSignal:
        """The one most expensive thing we can prove about the agent."""
        if compliance is not None and getattr(compliance, "status", None) == "breach":
            breaches = list(getattr(compliance, "breaches", []) or [])
            count = sum(int(getattr(b, "count", 0) or 0) for b in breaches)
            first = breaches[0] if breaches else None
            detail = (
                describe_breach(first.kind, first.subject, first.limit) if first is not None else None
            )
            return AgentSignal(
                kind="breach",
                title=f"{count} guardrail breach{'es' if count != 1 else ''}",
                detail=detail,
                amount=None,
            )
        if cost > 0 and repeated_cost / cost >= REPEATED_WORK_SHARE:
            return AgentSignal(
                kind="repeated_work",
                title="Repeated work",
                detail=f"{repeated_runs} runs re-ran an identical call",
                amount=round(repeated_cost, 6),
            )
        if cost > 0 and failed_cost / cost >= FAILED_SPEND_SHARE:
            return AgentSignal(
                kind="failed_spend",
                title=f"{round(failed / calls * 100, 1) if calls else 0}% of calls fail and still bill",
                detail=f"{failed} failed calls",
                amount=round(failed_cost, 6),
            )
        if calls >= CLASSIFIER_MIN_CALLS and 0 < max_output <= CLASSIFIER_MAX_OUTPUT_TOKENS:
            return AgentSignal(
                kind="classification",
                title="Classification, not generation",
                detail=f"Output never above {max_output} tokens across {calls} calls",
            )
        if runs == 0 and calls > 0:
            return AgentSignal(
                kind="untraced",
                title="Untraced",
                detail="Wrap runs in workflow() to get cost per run and step costs",
            )
        return AgentSignal(kind="none", title="Nothing to flag")

    # ── Step costs (detail page) ─────────────────────────────────────────

    async def step_costs(
        self, project_id: str, start: datetime, end: datetime, agent_name: str
    ) -> List[AgentStepCost]:
        """Median and p95 cost of each step per run, across the agent's runs."""
        window = self._window(project_id, start, end, agent_name)
        per_run = (
            select(
                Event.step_name,
                Event.trace_id,
                func.count(Event.id).label("calls"),
                func.sum(Event.cost).label("cost"),
                func.sum(case((Event.success == False, 1), else_=0)).label("failures"),  # noqa: E712
                func.max(case((Event.tool_name.isnot(None), 1), else_=0)).label("is_tool"),
            )
            .where(*window, Event.trace_id.isnot(None), Event.step_name.isnot(None))
            .group_by(Event.step_name, Event.trace_id)
            .limit(STEP_ROWS_CAP)
        )
        rows = (await self.db.execute(per_run)).all()
        if not rows:
            return []

        models_rows = (
            await self.db.execute(
                select(Event.step_name, Event.model)
                .where(*window, Event.trace_id.isnot(None), Event.step_name.isnot(None))
                .group_by(Event.step_name, Event.model)
            )
        ).all()
        models: Dict[str, List[str]] = {}
        for row in models_rows:
            models.setdefault(row.step_name, []).append(row.model)

        groups: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            g = groups.setdefault(
                row.step_name,
                {"costs": [], "calls": 0, "max_calls": 0, "failures": 0, "tool": False},
            )
            calls = int(row.calls or 0)
            g["costs"].append(float(row.cost or 0.0))
            g["calls"] += calls
            g["max_calls"] = max(g["max_calls"], calls)
            g["failures"] += int(row.failures or 0)
            g["tool"] = g["tool"] or bool(row.is_tool)

        steps: List[AgentStepCost] = []
        for step_name, g in groups.items():
            costs = sorted(g["costs"])
            runs = len(costs)
            total = sum(costs)
            steps.append(
                AgentStepCost(
                    step_name=step_name,
                    tool=g["tool"],
                    models=sorted(models.get(step_name, [])),
                    runs=runs,
                    calls=g["calls"],
                    calls_per_run=round(g["calls"] / runs, 2) if runs else 0.0,
                    max_calls_per_run=g["max_calls"],
                    total_cost=round(total, 6),
                    median_cost_per_run=round(_nearest_rank(costs, 0.5), 6),
                    p95_cost_per_run=round(_nearest_rank(costs, 0.95), 6),
                    success_rate=round(
                        ((g["calls"] - g["failures"]) / g["calls"] * 100) if g["calls"] else 100.0, 2
                    ),
                )
            )
        steps.sort(key=lambda s: s.total_cost, reverse=True)
        return steps

    # ── Detail (agent page) ──────────────────────────────────────────────

    async def detail(
        self, project_id: str, start: datetime, end: datetime, agent_name: str
    ) -> Optional[AgentDetail]:
        compliance_response = await GuardrailService(self.db).compliance(project_id, start, end)
        compliance = {a.agent_name: a for a in compliance_response.agents}

        summaries = await self.summaries(
            project_id, start, end, limit=1, agent_name=agent_name, compliance=compliance
        )
        if not summaries:
            return None
        summary = summaries[0]

        analytics = AnalyticsService(self.db)
        trace = TraceService(self.db)

        async def by(dimension: str) -> List[DimensionStat]:
            rows = await analytics.get_dimension_stats(
                project_id, dimension, start, end, limit=8, agent_name=agent_name
            )
            return [DimensionStat(**row) for row in rows]

        latency = await ReportService(self.db).latency_percentiles(
            project_id, start, end, agent_name=agent_name
        )
        steps = await self.step_costs(project_id, start, end, agent_name)
        distribution = await trace.get_run_cost_distribution(
            project_id, start, end, workflow=None, buckets=24, agent_name=agent_name
        )
        repeated = await trace.detect_repeated_work(
            project_id, start, end, limit=8, agent_name=agent_name
        )
        traces = await trace.list_traces(project_id, start, end, limit=8, agent_name=agent_name)
        outcome_rows = await trace.get_outcome_stats(
            project_id, start, end, limit=100, agent_name=agent_name
        )

        outcomes: Optional[AgentOutcomes] = None
        if outcome_rows:
            runs = sum(r["runs"] for r in outcome_rows)
            succeeded = sum(r["succeeded"] for r in outcome_rows)
            failed = sum(r["failed"] for r in outcome_rows)
            unknown = sum(r["unknown"] for r in outcome_rows)
            on_success = sum(r["cost_on_success"] for r in outcome_rows)
            on_failure = sum(r["cost_on_failure"] for r in outcome_rows)
            declared = succeeded + failed
            outcomes = AgentOutcomes(
                runs=runs,
                succeeded=succeeded,
                failed=failed,
                unknown=unknown,
                cost_on_success=round(on_success, 6),
                cost_on_failure=round(on_failure, 6),
                cost_per_success=round((on_success + on_failure) / succeeded, 6) if succeeded else None,
                success_rate=round(succeeded / declared * 100, 2) if declared else None,
            )

        tail_cost = 0.0
        if distribution:
            tail_cost = distribution["total_cost"] * distribution["tail_share_percent"] / 100

        return AgentDetail(
            summary=summary,
            latency=latency,
            steps=steps,
            by_model=await by("model"),
            by_tool=await by("tool"),
            by_user=await by("user"),
            by_session=await by("session"),
            by_workflow=await by("workflow"),
            distribution=distribution,
            tail_cost=round(tail_cost, 6),
            repeated_work=repeated,
            traces=traces,
            outcomes=outcomes,
            compliance=compliance.get(agent_name),
        )
