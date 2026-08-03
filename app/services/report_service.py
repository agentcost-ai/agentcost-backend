"""
AgentCost Backend - Executive Report Service

Assembles a single board-ready "Executive Cost & Usage Report": an executive
summary (headline KPIs with period-over-period deltas) plus deep drill-down
sections (latency percentiles, cost concentration / Pareto, token efficiency,
error breakdown, usage cadence, run-rate projection, budget status, and an
optimization-savings rollup).

It composes the existing AnalyticsService / BudgetService / OptimizationService
rather than duplicating their aggregation logic; only the dimensions those
services don't already expose are computed here with new queries.
"""

import logging
import math
from datetime import datetime, timezone
from typing import List

from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import Event, Project
from ..models.schemas import (
    AnalyticsOverview,
    AgentStats,
    ModelStats,
    MetricDelta,
    ReportSummary,
    LatencyPercentiles,
    ModelEfficiency,
    TokenEfficiency,
    ParetoInfo,
    CadenceBucket,
    UsageCadence,
    ErrorBreakdownRow,
    TopError,
    RunRateProjection,
    BudgetStatus,
    SavingsRollup,
    ExecutiveReport,
)
from ..utils.sql_dialect import dialect_name, utc_timestamp
from .analytics_service import AnalyticsService
from .budget_service import BudgetService
from .optimization_service import OptimizationService

logger = logging.getLogger(__name__)

_DOW_LABELS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


class ReportService:
    """Builds the ExecutiveReport for a project over a time window."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.analytics = AnalyticsService(db)
        self._dialect = dialect_name(db)

    async def build_report(
        self,
        project: Project,
        start: datetime,
        end: datetime,
        prev_start: datetime,
        prev_end: datetime,
        top_n: int,
        range_label: str,
        is_custom_range: bool,
    ) -> ExecutiveReport:
        project_id = project.id
        window_days = max((end - start).total_seconds() / 86400.0, 1e-9)

        # Current + previous overviews drive the executive-summary deltas.
        overview = await self.analytics.get_overview(project_id, start, end)
        prev_overview = await self.analytics.get_overview(project_id, prev_start, prev_end)

        granularity = "day" if window_days > 2 else "hour"
        timeseries = await self.analytics.get_timeseries(project_id, start, end, granularity)

        # Shares and the Pareto denominator come from the whole window, not from
        # the top_n rows below. The overview already summed it, so pass that
        # figure down instead of re-running SUM(cost) over the same window.
        window_cost = overview.total_cost
        distinct_models = await self.analytics.get_distinct_model_count(project_id, start, end)

        models = await self.analytics.get_model_stats(
            project_id, start, end, top_n, window_cost=window_cost
        )
        agents = await self.analytics.get_agent_stats(project_id, start, end, top_n)

        summary = self._summary(overview, prev_overview)
        pareto = self._pareto(models, window_cost, distinct_models)
        agent_cost_share = self._agent_cost_share(agents, window_cost)
        latency = await self._latency(project_id, start, end)
        efficiency = self._efficiency(overview, models)
        errors, top_errors = await self._errors(project_id, start, end)
        cadence = await self._cadence(project_id, start, end)
        run_rate = self._run_rate(overview.total_cost, window_days)
        budget = await self._budget(project)
        savings = await self._savings(project_id, window_days)

        return ExecutiveReport(
            generated_at=datetime.now(timezone.utc),
            range_label=range_label,
            period_start=start,
            period_end=end,
            previous_period_start=prev_start,
            previous_period_end=prev_end,
            is_custom_range=is_custom_range,
            project_name=project.name,
            currency=budget.currency,
            summary=summary,
            overview=overview,
            timeseries=timeseries,
            models=models,
            model_pareto=pareto,
            agents=agents,
            agent_cost_share=agent_cost_share,
            latency=latency,
            efficiency=efficiency,
            errors=errors,
            top_errors=top_errors,
            cadence=cadence,
            run_rate=run_rate,
            budget=budget,
            savings=savings,
        )

    # ── Executive summary ─────────────────────────────────────────────────

    @staticmethod
    def _delta(current: float, previous: float, *, lower_is_better: bool = False) -> MetricDelta:
        if previous and previous != 0:
            change = (current - previous) / abs(previous) * 100.0
        else:
            change = 0.0
        # Direction reflects "good vs bad": for cost/latency, up is bad.
        if abs(change) < 0.05:
            direction = "neutral"
        elif change > 0:
            direction = "up"
        else:
            direction = "down"
        return MetricDelta(
            current=round(current, 6),
            previous=round(previous, 6),
            change_percent=round(change, 2),
            direction=direction,
        )

    def _summary(self, ov: AnalyticsOverview, prev: AnalyticsOverview) -> ReportSummary:
        blended_cost_per_1k = (ov.total_cost / ov.total_tokens * 1000.0) if ov.total_tokens else 0.0
        in_out_ratio = (ov.total_input_tokens / ov.total_output_tokens) if ov.total_output_tokens else 0.0
        return ReportSummary(
            cost=self._delta(ov.total_cost, prev.total_cost),
            calls=self._delta(ov.total_calls, prev.total_calls),
            tokens=self._delta(ov.total_tokens, prev.total_tokens),
            # success_rate is vacuous with no calls (overview reports 100 there),
            # so hold the delta neutral unless both windows actually have calls.
            success_rate=(
                self._delta(ov.success_rate, prev.success_rate)
                if ov.total_calls and prev.total_calls
                else self._delta(ov.success_rate, ov.success_rate)
            ),
            avg_latency_ms=self._delta(ov.avg_latency_ms, prev.avg_latency_ms),
            blended_cost_per_1k=round(blended_cost_per_1k, 6),
            in_out_ratio=round(in_out_ratio, 2),
        )

    # ── Cost concentration (Pareto) ───────────────────────────────────────

    @staticmethod
    def _pareto(models: List[ModelStats], total_cost: float, total_models: int) -> ParetoInfo:
        # ``models`` is only the top_n slice, so both the denominator and the
        # model count must come from the full window -- otherwise "3 of 10 models
        # drive 80% of spend" is quoted for a project that actually runs 40, and
        # the 80% is measured against a subtotal rather than real spend.
        if total_cost <= 0 or not models:
            return ParetoInfo(top_count=0, top_share=0.0, total_models=total_models)
        ordered = sorted(models, key=lambda m: m.total_cost, reverse=True)
        cumulative = 0.0
        top_count = 0
        for m in ordered:
            cumulative += m.total_cost
            top_count += 1
            if cumulative / total_cost >= 0.80:
                break
        # When the tail was truncated the listed models may never reach 80%; the
        # reported share then honestly says how much they do cover.
        return ParetoInfo(
            top_count=top_count,
            top_share=round(cumulative / total_cost * 100.0, 1),
            total_models=total_models,
        )

    @staticmethod
    def _agent_cost_share(agents: List[AgentStats], total_cost: float) -> dict[str, float]:
        if total_cost <= 0:
            return {a.agent_name: 0.0 for a in agents}
        return {a.agent_name: round(a.total_cost / total_cost * 100.0, 1) for a in agents}

    # ── Latency percentiles (only section needing raw rows) ───────────────

    async def _latency(
        self, project_id: str, start: datetime, end: datetime
    ) -> LatencyPercentiles:
        """Exact p50/p95/p99 over every call in the window, computed in the
        database rather than over a capped sample of rows."""
        where_clause = (
            Event.project_id == project_id,
            Event.timestamp >= start,
            Event.timestamp <= end,
            Event.latency_ms.isnot(None),
        )

        is_postgres = self._dialect == "postgresql"

        base_q = select(
            func.count(Event.id).label("n"),
            func.avg(Event.latency_ms).label("avg"),
        ).where(*where_clause)

        if is_postgres:
            # One pass in the database: no rows cross the wire.
            base_q = base_q.add_columns(
                *[
                    func.percentile_cont(p).within_group(Event.latency_ms.asc()).label(label)
                    for p, label in ((0.50, "p50"), (0.95, "p95"), (0.99, "p99"))
                ]
            )

        row = (await self.db.execute(base_q)).one()
        total = int(row.n or 0)
        if total == 0:
            return LatencyPercentiles(p50=0.0, p95=0.0, p99=0.0, avg=0.0, sample_size=0)

        avg = float(row.avg)

        if is_postgres:
            p50, p95, p99 = (float(row.p50 or 0), float(row.p95 or 0), float(row.p99 or 0))
        else:
            # SQLite has no percentile function, so seek to the rank instead --
            # still exact, and only the two straddling rows are fetched.
            p50 = await self._percentile_by_offset(where_clause, 0.50, total)
            p95 = await self._percentile_by_offset(where_clause, 0.95, total)
            p99 = await self._percentile_by_offset(where_clause, 0.99, total)

        return LatencyPercentiles(
            p50=round(p50, 2),
            p95=round(p95, 2),
            p99=round(p99, 2),
            avg=round(avg, 2),
            sample_size=total,
        )

    async def _percentile_by_offset(self, where_clause, p: float, total: int) -> float:
        """Linear-interpolated percentile via rank seek (matches percentile_cont)."""
        rank = p * (total - 1)
        lower = int(math.floor(rank))
        fraction = rank - lower

        q = (
            select(Event.latency_ms)
            .where(*where_clause)
            .order_by(Event.latency_ms.asc())
            .offset(lower)
            .limit(2)
        )
        values = [int(v) for (v,) in (await self.db.execute(q)).all() if v is not None]
        if not values:
            return 0.0
        if fraction == 0.0 or len(values) < 2:
            return float(values[0])
        return float(values[0]) + fraction * float(values[1] - values[0])

    # ── Token efficiency ──────────────────────────────────────────────────

    @staticmethod
    def _efficiency(ov: AnalyticsOverview, models: List[ModelStats]) -> TokenEfficiency:
        by_model = []
        for m in models:
            cost_per_1k = (m.total_cost / m.total_tokens * 1000.0) if m.total_tokens else 0.0
            ratio = (m.input_tokens / m.output_tokens) if m.output_tokens else 0.0
            by_model.append(
                ModelEfficiency(
                    model=m.model,
                    cost_per_1k=round(cost_per_1k, 6),
                    in_out_ratio=round(ratio, 2),
                )
            )
        blended = (ov.total_cost / ov.total_tokens * 1000.0) if ov.total_tokens else 0.0
        ratio = (ov.total_input_tokens / ov.total_output_tokens) if ov.total_output_tokens else 0.0
        return TokenEfficiency(
            blended_cost_per_1k=round(blended, 6),
            in_out_ratio=round(ratio, 2),
            total_input_tokens=ov.total_input_tokens,
            total_output_tokens=ov.total_output_tokens,
            by_model=by_model,
        )

    # ── Error / failure breakdown ─────────────────────────────────────────

    async def _errors(
        self, project_id: str, start: datetime, end: datetime
    ) -> tuple[List[ErrorBreakdownRow], List[TopError]]:
        by_model_q = (
            select(
                Event.model,
                func.count(Event.id).label("total_calls"),
                func.sum(case((Event.success == False, 1), else_=0)).label("error_count"),  # noqa: E712
            )
            .where(
                Event.project_id == project_id,
                Event.timestamp >= start,
                Event.timestamp <= end,
            )
            .group_by(Event.model)
        )
        rows = await self.db.execute(by_model_q)
        breakdown: List[ErrorBreakdownRow] = []
        for row in rows:
            total_calls = int(row.total_calls or 0)
            error_count = int(row.error_count or 0)
            rate = (error_count / total_calls * 100.0) if total_calls else 0.0
            breakdown.append(
                ErrorBreakdownRow(
                    model=row.model,
                    total_calls=total_calls,
                    error_count=error_count,
                    error_rate=round(rate, 2),
                )
            )
        breakdown.sort(key=lambda r: (r.error_count, r.error_rate), reverse=True)

        top_q = (
            select(Event.error, func.count(Event.id).label("cnt"))
            .where(
                Event.project_id == project_id,
                Event.timestamp >= start,
                Event.timestamp <= end,
                Event.success == False,  # noqa: E712
                Event.error.isnot(None),
            )
            .group_by(Event.error)
            .order_by(func.count(Event.id).desc())
            .limit(10)
        )
        top_rows = await self.db.execute(top_q)
        top_errors = [
            TopError(error=str(r.error), count=int(r.cnt or 0))
            for r in top_rows
            if r.error
        ]
        return breakdown, top_errors

    # ── Usage cadence (busiest day / hour) ────────────────────────────────

    async def _cadence(self, project_id: str, start: datetime, end: datetime) -> UsageCadence:
        # Bucket in UTC to match the UTC window bounds: extract() on a
        # ``timestamptz`` otherwise answers in the session TimeZone, which moves
        # "busiest hour" (and, at the edges, "busiest day") by the server offset.
        ts = utc_timestamp(self._dialect)
        if self._dialect == "sqlite":
            dow_expr = func.strftime("%w", ts)
            hour_expr = func.strftime("%H", ts)
        else:
            dow_expr = func.extract("dow", ts)
            hour_expr = func.extract("hour", ts)

        base_where = (
            Event.project_id == project_id,
            Event.timestamp >= start,
            Event.timestamp <= end,
        )

        dow_q = (
            select(
                dow_expr.label("bucket"),
                func.count(Event.id).label("calls"),
                func.sum(Event.cost).label("cost"),
            )
            .where(*base_where)
            .group_by(dow_expr)
        )
        hour_q = (
            select(
                hour_expr.label("bucket"),
                func.count(Event.id).label("calls"),
                func.sum(Event.cost).label("cost"),
            )
            .where(*base_where)
            .group_by(hour_expr)
        )

        dow_calls = {i: 0 for i in range(7)}
        dow_cost = {i: 0.0 for i in range(7)}
        for row in await self.db.execute(dow_q):
            if row.bucket is None:
                continue
            idx = int(float(row.bucket))
            if 0 <= idx <= 6:
                dow_calls[idx] = int(row.calls or 0)
                dow_cost[idx] = float(row.cost or 0.0)

        hour_calls = {i: 0 for i in range(24)}
        hour_cost = {i: 0.0 for i in range(24)}
        for row in await self.db.execute(hour_q):
            if row.bucket is None:
                continue
            idx = int(float(row.bucket))
            if 0 <= idx <= 23:
                hour_calls[idx] = int(row.calls or 0)
                hour_cost[idx] = float(row.cost or 0.0)

        by_dow = [
            CadenceBucket(label=_DOW_LABELS[i], index=i, calls=dow_calls[i], cost=round(dow_cost[i], 6))
            for i in range(7)
        ]
        by_hour = [
            CadenceBucket(
                label=f"{i:02d}:00", index=i, calls=hour_calls[i], cost=round(hour_cost[i], 6)
            )
            for i in range(24)
        ]

        busiest_day = None
        if any(b.calls for b in by_dow):
            busiest_day = max(by_dow, key=lambda b: b.calls).label
        busiest_hour = None
        if any(b.calls for b in by_hour):
            busiest_hour = max(by_hour, key=lambda b: b.calls).label

        return UsageCadence(
            busiest_day=busiest_day,
            busiest_hour=busiest_hour,
            by_dow=by_dow,
            by_hour=by_hour,
        )

    # ── Run-rate projection ───────────────────────────────────────────────

    @staticmethod
    def _run_rate(window_cost: float, window_days: float) -> RunRateProjection:
        daily_avg = window_cost / window_days if window_days > 0 else 0.0
        return RunRateProjection(
            daily_avg_cost=round(daily_avg, 6),
            projected_monthly_cost=round(daily_avg * 30.0, 2),
            window_days=round(window_days, 2),
        )

    # ── Budget status ─────────────────────────────────────────────────────

    async def _budget(self, project: Project) -> BudgetStatus:
        try:
            result = await BudgetService(self.db).evaluate(project)
            return BudgetStatus(
                enabled=bool(result.get("enabled")),
                budget=result.get("budget"),
                current_spend=float(result.get("current_spend") or 0.0),
                projected_spend=float(result.get("projected_spend") or 0.0),
                utilization_percent=result.get("utilization_percent"),
                currency=str(result.get("currency") or "USD"),
                fx_rate=float(result.get("fx_rate") or 1.0),
                mode=str(result.get("mode") or "off"),
            )
        except Exception as exc:  # noqa: BLE001 — budget context is best-effort
            logger.warning("Report budget evaluation failed for %s: %s", project.id, exc)
            return BudgetStatus(enabled=False)

    # ── Optimization savings rollup ───────────────────────────────────────

    async def _savings(self, project_id: str, window_days: float) -> SavingsRollup:
        try:
            days = max(1, min(90, round(window_days)))
            summary = await OptimizationService(self.db).get_summary(project_id, days)
            return SavingsRollup(
                total_potential_savings_monthly=float(
                    summary.get("total_potential_savings_monthly") or 0.0
                ),
                total_potential_savings_percent=float(
                    summary.get("total_potential_savings_percent") or 0.0
                ),
                suggestion_count=int(summary.get("suggestion_count") or 0),
                high_priority_count=int(summary.get("high_priority_count") or 0),
                top_suggestions=list(summary.get("suggestions") or [])[:3],
            )
        except Exception as exc:  # noqa: BLE001 — savings rollup is best-effort
            logger.warning("Report savings rollup failed for %s: %s", project_id, exc)
            return SavingsRollup()
