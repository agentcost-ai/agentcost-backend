"""
AgentCost Backend - Trace Analytics

Cost attributed to the shape of a run: per workflow, per step, per tool, and
where the same work is done twice.

Every query filters on ``trace_id IS NOT NULL``. Untraced events have no place
in a tree, and folding them into a workflow's totals would overstate it; they
remain visible in the agent and model analytics.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import Event, TraceOutcome


class TraceService:
    """Analytics over the trace structure attached to events."""

    def __init__(self, db: AsyncSession):
        self.db = db

    def _scoped(self, project_id: str, start_time: datetime, end_time: datetime):
        """The filter every trace query shares."""
        return (
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
            Event.trace_id.isnot(None),
        )

    async def get_workflow_stats(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Cost per workflow, and per *run* of that workflow.

        Aggregated in two stages, per trace then per workflow, so the averages
        divide by runs rather than by calls.
        """
        per_trace = (
            select(
                Event.workflow.label("workflow"),
                Event.trace_id.label("trace_id"),
                func.count(Event.id).label("calls"),
                func.sum(Event.cost).label("cost"),
                func.sum(Event.total_tokens).label("tokens"),
                func.count(func.distinct(Event.step_name)).label("steps"),
                func.max(Event.depth).label("max_depth"),
                func.sum(case((Event.success == False, 1), else_=0)).label("failures"),  # noqa: E712
                func.min(Event.timestamp).label("started_at"),
                func.max(Event.timestamp).label("ended_at"),
            )
            .where(*self._scoped(project_id, start_time, end_time))
            .group_by(Event.workflow, Event.trace_id)
            .subquery()
        )

        query = (
            select(
                per_trace.c.workflow,
                func.count().label("runs"),
                func.sum(per_trace.c.cost).label("total_cost"),
                func.avg(per_trace.c.cost).label("avg_cost_per_run"),
                func.max(per_trace.c.cost).label("max_cost_per_run"),
                func.sum(per_trace.c.tokens).label("total_tokens"),
                func.avg(per_trace.c.calls).label("avg_calls_per_run"),
                func.avg(per_trace.c.steps).label("avg_steps_per_run"),
                func.max(per_trace.c.max_depth).label("max_depth"),
                func.sum(per_trace.c.calls).label("total_calls"),
                func.sum(per_trace.c.failures).label("failed_calls"),
            )
            .group_by(per_trace.c.workflow)
            .order_by(desc("total_cost"))
            .limit(limit)
        )

        result = await self.db.execute(query)

        workflows: List[Dict[str, Any]] = []
        for row in result:
            total_calls = int(row.total_calls or 0)
            failed = int(row.failed_calls or 0)
            workflows.append(
                {
                    "workflow": row.workflow,
                    "runs": int(row.runs or 0),
                    "total_cost": round(float(row.total_cost or 0.0), 6),
                    "avg_cost_per_run": round(float(row.avg_cost_per_run or 0.0), 6),
                    "max_cost_per_run": round(float(row.max_cost_per_run or 0.0), 6),
                    "total_tokens": int(row.total_tokens or 0),
                    "total_calls": total_calls,
                    "avg_calls_per_run": round(float(row.avg_calls_per_run or 0.0), 2),
                    "avg_steps_per_run": round(float(row.avg_steps_per_run or 0.0), 2),
                    "max_depth": int(row.max_depth or 0),
                    # Same empty-window convention as the agent/model analytics.
                    "success_rate": round(
                        ((total_calls - failed) / total_calls * 100) if total_calls else 100.0,
                        2,
                    ),
                }
            )
        return workflows

    async def get_step_stats(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        workflow: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Cost per step, optionally within one workflow.

        ``calls`` far exceeding ``runs`` means the step executed repeatedly
        inside single runs -- a retry or a loop.
        """
        filters = list(self._scoped(project_id, start_time, end_time))
        filters.append(Event.step_name.isnot(None))
        if workflow:
            filters.append(Event.workflow == workflow)

        query = (
            select(
                Event.workflow,
                Event.step_name,
                func.count(Event.id).label("calls"),
                func.count(func.distinct(Event.trace_id)).label("runs"),
                func.sum(Event.cost).label("total_cost"),
                func.sum(Event.total_tokens).label("total_tokens"),
                func.avg(Event.latency_ms).label("avg_latency"),
                func.sum(case((Event.success == False, 1), else_=0)).label("failures"),  # noqa: E712
            )
            .where(*filters)
            .group_by(Event.workflow, Event.step_name)
            .order_by(desc("total_cost"))
            .limit(limit)
        )

        result = await self.db.execute(query)

        steps: List[Dict[str, Any]] = []
        for row in result:
            calls = int(row.calls or 0)
            runs = int(row.runs or 0)
            failures = int(row.failures or 0)
            steps.append(
                {
                    "workflow": row.workflow,
                    "step_name": row.step_name,
                    "calls": calls,
                    "runs": runs,
                    "calls_per_run": round(calls / runs, 2) if runs else 0.0,
                    "total_cost": round(float(row.total_cost or 0.0), 6),
                    "cost_per_run": round(float(row.total_cost or 0.0) / runs, 6) if runs else 0.0,
                    "total_tokens": int(row.total_tokens or 0),
                    "avg_latency_ms": round(float(row.avg_latency or 0.0), 2),
                    "success_rate": round(
                        ((calls - failures) / calls * 100) if calls else 100.0, 2
                    ),
                }
            )
        return steps

    async def get_tool_stats(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """LLM spend incurred underneath each named tool."""
        filters = list(self._scoped(project_id, start_time, end_time))
        filters.append(Event.tool_name.isnot(None))

        query = (
            select(
                Event.tool_name,
                func.count(Event.id).label("calls"),
                func.count(func.distinct(Event.trace_id)).label("runs"),
                func.sum(Event.cost).label("total_cost"),
                func.sum(Event.total_tokens).label("total_tokens"),
                func.avg(Event.latency_ms).label("avg_latency"),
            )
            .where(*filters)
            .group_by(Event.tool_name)
            .order_by(desc("total_cost"))
            .limit(limit)
        )

        result = await self.db.execute(query)
        return [
            {
                "tool_name": row.tool_name,
                "calls": int(row.calls or 0),
                "runs": int(row.runs or 0),
                "total_cost": round(float(row.total_cost or 0.0), 6),
                "total_tokens": int(row.total_tokens or 0),
                "avg_latency_ms": round(float(row.avg_latency or 0.0), 2),
            }
            for row in result
        ]

    async def detect_repeated_work(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Find identical work repeated inside a single run.

        Distinct from the cross-run duplication the caching analyzer reports:
        that argues for a cache, this usually means the control flow is wrong.
        ``wasted_cost`` covers every occurrence beyond the first.
        """
        filters = list(self._scoped(project_id, start_time, end_time))
        filters.append(Event.input_hash.isnot(None))

        query = (
            select(
                Event.trace_id,
                Event.workflow,
                Event.step_name,
                Event.input_hash,
                Event.model,
                func.count(Event.id).label("occurrences"),
                func.sum(Event.cost).label("spend"),
                func.min(Event.timestamp).label("first_seen"),
            )
            .where(*filters)
            .group_by(
                Event.trace_id,
                Event.workflow,
                Event.step_name,
                Event.input_hash,
                Event.model,
            )
            .having(func.count(Event.id) > 1)
            .order_by(desc("spend"))
            .limit(limit)
        )

        result = await self.db.execute(query)

        findings: List[Dict[str, Any]] = []
        for row in result:
            occurrences = int(row.occurrences or 0)
            spend = float(row.spend or 0.0)
            # Cost of the redundant repeats only.
            wasted = spend * (occurrences - 1) / occurrences if occurrences else 0.0
            findings.append(
                {
                    "trace_id": row.trace_id,
                    "workflow": row.workflow,
                    "step_name": row.step_name,
                    "model": row.model,
                    "input_hash": row.input_hash,
                    "occurrences": occurrences,
                    "spend": round(spend, 6),
                    "wasted_cost": round(wasted, 6),
                    "first_seen": row.first_seen.isoformat() if row.first_seen else None,
                }
            )
        return findings

    # Reads the whole window, not a top-N slice. The cap bounds memory; when
    # it bites, ``truncated`` says so.
    DISTRIBUTION_MAX_RUNS = 50_000

    async def get_run_cost_distribution(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        workflow: Optional[str] = None,
        buckets: int = 24,
    ) -> Optional[Dict[str, Any]]:
        """
        How much one run costs, across every run — not just the average.

        A mean hides the runs worth finding. ``tail_share`` reports the share
        of spend consumed by the most expensive 5% of runs; when it is large,
        the fix is bounding the tail rather than shaving the median.
        """
        filters = list(self._scoped(project_id, start_time, end_time))
        if workflow:
            filters.append(Event.workflow == workflow)

        # One row per run. Ordered so the percentile indices below are simply
        # positions in the list.
        per_trace = (
            select(func.sum(Event.cost).label("cost"))
            .where(*filters)
            .group_by(Event.trace_id)
            .order_by("cost")
            .limit(self.DISTRIBUTION_MAX_RUNS)
        )

        result = await self.db.execute(per_trace)
        costs = [float(row.cost or 0.0) for row in result]

        if not costs:
            return None

        n = len(costs)
        total = sum(costs)

        def percentile(fraction: float) -> float:
            # Nearest-rank on the already-sorted list. Exact for the sample,
            # and it never invents a value between two observed runs.
            index = min(n - 1, max(0, int(round(fraction * (n - 1)))))
            return costs[index]

        p50, p90, p95, p99 = (percentile(f) for f in (0.50, 0.90, 0.95, 0.99))
        lo, hi = costs[0], costs[-1]

        # Defined by rank, not by "cost >= p95": on a skewed distribution the
        # 95th percentile can equal the minimum, and a value test would then
        # select every run.
        tail_count = max(1, int(round(n * 0.05)))
        tail_costs = costs[-tail_count:]
        tail_threshold = tail_costs[0]
        tail_share = (sum(tail_costs) / total * 100) if total > 0 else 0.0

        # Buckets span the body only; the tail becomes one overflow bucket.
        # Equal-width bucketing across the full range crushes the body into a
        # couple of bars whenever a long tail exists.
        body = costs[: n - tail_count]
        # The body's own range: tail_threshold is the cheapest tail run, which
        # can sit far above every body run.
        body_hi = body[-1] if body else lo
        span = body_hi - lo

        if not body:
            histogram = []
        elif span <= 0:
            # Every body run cost the same. One bar is the honest picture --
            # but the tail still gets its bar below, which an early return
            # here would have swallowed.
            histogram = [
                {
                    "lower": round(lo, 6),
                    "upper": round(body_hi, 6),
                    "count": len(body),
                    "is_tail": False,
                }
            ]
        else:
            width = span / buckets
            counts = [0] * buckets
            for c in body:
                index = min(buckets - 1, int((c - lo) / width))
                counts[index] += 1
            histogram = [
                {
                    "lower": round(lo + i * width, 6),
                    "upper": round(lo + (i + 1) * width, 6),
                    "count": counts[i],
                    "is_tail": False,
                }
                for i in range(buckets)
            ]

        # Appended in every branch, and skipped only when the tail is
        # indistinguishable from the body.
        if tail_count and hi > body_hi:
            histogram.append(
                {
                    "lower": round(tail_threshold, 6),
                    "upper": round(hi, 6),
                    "count": tail_count,
                    "is_tail": True,
                }
            )
        elif histogram:
            # Body and tail are the same value: fold the tail back in rather
            # than reporting a bucket count that misses runs.
            histogram[-1]["count"] += tail_count

        return {
            "workflow": workflow,
            "runs": n,
            "truncated": n >= self.DISTRIBUTION_MAX_RUNS,
            "total_cost": round(total, 6),
            "min": round(lo, 6),
            "max": round(hi, 6),
            "mean": round(total / n, 6),
            "p50": round(p50, 6),
            "p90": round(p90, 6),
            "p95": round(p95, 6),
            "p99": round(p99, 6),
            "tail_runs": len(tail_costs),
            "tail_threshold": round(tail_threshold, 6),
            "tail_share_percent": round(tail_share, 1),
            # How much more the worst runs cost than the typical one. The
            # single number that says whether the tail is worth chasing.
            "tail_ratio": round(p99 / p50, 1) if p50 > 0 else None,
            "histogram": histogram,
        }

    async def get_outcome_stats(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Cost per completed outcome, per workflow.

        Cost per run averages successes and failures together;
        ``cost_per_success`` charges the failures to the wins. Runs that
        declared no outcome are reported as unknown rather than failed.
        """
        per_trace = (
            select(
                Event.trace_id.label("trace_id"),
                Event.workflow.label("workflow"),
                func.sum(Event.cost).label("cost"),
            )
            .where(*self._scoped(project_id, start_time, end_time))
            .group_by(Event.trace_id, Event.workflow)
            .subquery()
        )

        query = (
            select(
                per_trace.c.workflow,
                func.count().label("runs"),
                func.sum(per_trace.c.cost).label("total_cost"),
                func.sum(
                    case((TraceOutcome.success == True, 1), else_=0)  # noqa: E712
                ).label("succeeded"),
                func.sum(
                    case((TraceOutcome.success == False, 1), else_=0)  # noqa: E712
                ).label("failed"),
                func.sum(
                    case((TraceOutcome.success == True, per_trace.c.cost), else_=0.0)  # noqa: E712
                ).label("cost_on_success"),
                func.sum(
                    case((TraceOutcome.success == False, per_trace.c.cost), else_=0.0)  # noqa: E712
                ).label("cost_on_failure"),
                func.sum(
                    case((TraceOutcome.id.is_(None), 1), else_=0)
                ).label("unknown"),
            )
            .select_from(
                per_trace.outerjoin(
                    TraceOutcome,
                    (TraceOutcome.trace_id == per_trace.c.trace_id)
                    & (TraceOutcome.project_id == project_id),
                )
            )
            .group_by(per_trace.c.workflow)
            .order_by(desc("total_cost"))
            .limit(limit)
        )

        result = await self.db.execute(query)

        rows: List[Dict[str, Any]] = []
        for row in result:
            succeeded = int(row.succeeded or 0)
            failed = int(row.failed or 0)
            declared = succeeded + failed
            cost_on_success = float(row.cost_on_success or 0.0)
            cost_on_failure = float(row.cost_on_failure or 0.0)
            rows.append(
                {
                    "workflow": row.workflow,
                    "runs": int(row.runs or 0),
                    "succeeded": succeeded,
                    "failed": failed,
                    "unknown": int(row.unknown or 0),
                    "total_cost": round(float(row.total_cost or 0.0), 6),
                    "cost_on_success": round(cost_on_success, 6),
                    "cost_on_failure": round(cost_on_failure, 6),
                    # Everything spent on this workflow divided by the results
                    # it produced -- failures included, since they were paid for.
                    "cost_per_success": (
                        round((cost_on_success + cost_on_failure) / succeeded, 6)
                        if succeeded
                        else None
                    ),
                    "success_rate": (
                        round(succeeded / declared * 100, 2) if declared else None
                    ),
                }
            )
        return rows

    async def get_trace_detail(
        self, project_id: str, trace_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Every span of one run, ordered as it executed.

        Flat with parent ids rather than pre-nested, so an event whose parent
        never arrived cannot break the response.
        """
        query = (
            select(Event)
            .where(Event.project_id == project_id, Event.trace_id == trace_id)
            .order_by(Event.step_index.asc().nullslast(), Event.timestamp.asc())
        )
        result = await self.db.execute(query)
        events = result.scalars().all()

        if not events:
            return None

        spans = [
            {
                "span_id": e.span_id,
                "parent_span_id": e.parent_span_id,
                "step_name": e.step_name,
                "tool_name": e.tool_name,
                "step_index": e.step_index,
                "depth": e.depth,
                "agent_name": e.agent_name,
                "model": e.model,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "cost": round(float(e.cost or 0.0), 6),
                "latency_ms": e.latency_ms,
                "success": e.success,
                "error": e.error,
                "input_hash": e.input_hash,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in events
        ]

        total_cost = sum(s["cost"] for s in spans)
        started = min((e.timestamp for e in events if e.timestamp), default=None)
        ended = max((e.timestamp for e in events if e.timestamp), default=None)

        return {
            "trace_id": trace_id,
            "workflow": events[0].workflow,
            "total_cost": round(total_cost, 6),
            "total_calls": len(spans),
            "total_tokens": sum(int(e.total_tokens or 0) for e in events),
            "max_depth": max((int(e.depth or 0) for e in events), default=0),
            "failed_calls": sum(1 for e in events if not e.success),
            "started_at": started.isoformat() if started else None,
            "ended_at": ended.isoformat() if ended else None,
            "duration_ms": (
                int((ended - started).total_seconds() * 1000)
                if started and ended
                else None
            ),
            "spans": spans,
        }

    async def list_traces(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        workflow: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Recent runs, most expensive first."""
        filters = list(self._scoped(project_id, start_time, end_time))
        if workflow:
            filters.append(Event.workflow == workflow)

        query = (
            select(
                Event.trace_id,
                Event.workflow,
                func.count(Event.id).label("calls"),
                func.sum(Event.cost).label("total_cost"),
                func.sum(Event.total_tokens).label("total_tokens"),
                func.max(Event.depth).label("max_depth"),
                func.sum(case((Event.success == False, 1), else_=0)).label("failures"),  # noqa: E712
                func.min(Event.timestamp).label("started_at"),
                func.max(Event.timestamp).label("ended_at"),
            )
            .where(*filters)
            .group_by(Event.trace_id, Event.workflow)
            .order_by(desc("total_cost"))
            .limit(limit)
        )

        result = await self.db.execute(query)
        traces: List[Dict[str, Any]] = []
        for row in result:
            started, ended = row.started_at, row.ended_at
            traces.append(
                {
                    "trace_id": row.trace_id,
                    "workflow": row.workflow,
                    "calls": int(row.calls or 0),
                    "total_cost": round(float(row.total_cost or 0.0), 6),
                    "total_tokens": int(row.total_tokens or 0),
                    "max_depth": int(row.max_depth or 0),
                    "failed_calls": int(row.failures or 0),
                    "started_at": started.isoformat() if started else None,
                    "duration_ms": (
                        int((ended - started).total_seconds() * 1000)
                        if started and ended
                        else None
                    ),
                }
            )
        return traces
