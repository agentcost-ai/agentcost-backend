"""
AgentCost Backend - Session habits

Advice for coding agents. The rest of the optimizer assumes you control the
prompt and the model: swap one, cache the other. In Claude Code or Codex you
control neither. What you do control is how you use sessions -- how long one
runs, whether you come back to an old one, when you start fresh -- and that is
where a coding agent's usage goes.

These are observations with a share of usage attached, not savings in dollars:
most coding-agent usage is on a subscription, where the scarce thing is the
plan's limit rather than a bill. They report ``estimated_savings_monthly`` as
0 so they never inflate the money the optimizer says is recoverable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import Event

# Workflows written by the SDK's coding-agent collectors and the OTLP receiver.
CODING_AGENTS = ("claude-code", "codex", "antigravity")

# A prompt this large is a long-running session: the agent's own instructions
# and tools are a fraction of it, the rest is accumulated conversation.
LARGE_CONTEXT_TOKENS = 200_000

# A call that writes at least this share of a prompt this large to the cache is
# rebuilding it from nothing rather than extending it.
REBUILD_WRITE_SHARE = 0.5
REBUILD_MIN_TOKENS = 50_000

# Below these there is too little usage to call anything a habit.
MIN_CALLS = 20
MIN_SESSIONS = 3
# In USD at API rates. An agent tried once or twice has no habits to speak of.
MIN_COST = 1.0

LABELS = {"claude-code": "Claude Code", "codex": "Codex", "antigravity": "Antigravity"}


def _priority(share: float) -> str:
    return "high" if share >= 60 else "medium" if share >= 35 else "low"


def _money(value: float) -> str:
    return f"${value:,.2f}"


class SessionHabitService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def analyze(
        self, project_id: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        window = (
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
            Event.workflow.in_(CODING_AGENTS),
            Event.trace_id.isnot(None),
        )

        totals = await self.db.execute(
            select(
                Event.workflow,
                func.count(Event.id).label("calls"),
                func.coalesce(func.sum(Event.cost), 0.0).label("cost"),
                func.coalesce(
                    func.sum(case((Event.input_tokens >= LARGE_CONTEXT_TOKENS, Event.cost), else_=0.0)),
                    0.0,
                ).label("large_cost"),
                func.sum(case((Event.input_tokens >= LARGE_CONTEXT_TOKENS, 1), else_=0)).label(
                    "large_calls"
                ),
                func.sum(
                    case(
                        (and_(Event.success == False, Event.error.like("rate_limit%")), 1),  # noqa: E712
                        else_=0,
                    )
                ).label("rate_limited"),
            )
            .where(*window)
            .group_by(Event.workflow)
        )

        suggestions: List[Dict[str, Any]] = []
        for row in totals:
            calls, cost = int(row.calls or 0), float(row.cost or 0.0)
            if calls < MIN_CALLS or cost < MIN_COST:
                continue
            agent = row.workflow
            sessions = await self._sessions(window, agent)

            for suggestion in (
                self._large_context(agent, cost, float(row.large_cost or 0.0), int(row.large_calls or 0)),
                await self._cache_rebuilds(window, agent, cost, sessions),
                await self._dominant_session(project_id, agent, cost, sessions),
                self._rate_limits(agent, int(row.rate_limited or 0)),
            ):
                if suggestion:
                    suggestions.append(suggestion)
        return suggestions

    async def _sessions(self, window, agent: str) -> Dict[str, Dict[str, Any]]:
        rows = await self.db.execute(
            select(
                Event.trace_id,
                func.min(Event.timestamp).label("started"),
                func.coalesce(func.sum(Event.cost), 0.0).label("cost"),
                func.count(Event.id).label("calls"),
            )
            .where(*window, Event.workflow == agent)
            .group_by(Event.trace_id)
        )
        return {
            row.trace_id: {"started": row.started, "cost": float(row.cost), "calls": int(row.calls)}
            for row in rows
        }

    @staticmethod
    def _suggestion(
        kind: str,
        agent: str,
        title: str,
        description: str,
        share: float,
        share_label: str,
        actions: List[str],
        priority: Optional[str] = None,
        headline: Optional[str] = None,
        **metrics: Any,
    ) -> Dict[str, Any]:
        return {
            "type": f"session_{kind}",
            "title": title,
            "description": description,
            # Usage, not money: see the module docstring.
            "estimated_savings_monthly": 0,
            "estimated_savings_percent": round(share, 1),
            "agent_name": agent,
            "model": None,
            "priority": priority or _priority(share),
            "action_items": actions,
            "metrics": {
                # The one figure a card shows beside the title.
                "headline_label": share_label,
                "headline_value": headline or f"{share:.0f}%",
                "share_percent": round(share, 1),
                **metrics,
            },
        }

    def _large_context(
        self, agent: str, cost: float, large_cost: float, large_calls: int
    ) -> Optional[Dict[str, Any]]:
        share = large_cost / cost * 100
        if share < 25 or large_calls < 5:
            return None
        label = LABELS.get(agent, agent)
        return self._suggestion(
            "large_context",
            agent,
            f"{share:.0f}% of {label} usage is spent re-sending very long conversations",
            (
                f"{large_calls:,} calls were made with more than "
                f"{LARGE_CONTEXT_TOKENS // 1000}K tokens of context, worth {_money(large_cost)} "
                f"of {_money(cost)} at API rates. Every call in a session sends the whole "
                "conversation so far, so each extra hour in the same session costs more than "
                "the one before it."
            ),
            share,
            "Share of usage",
            [
                "Start a new session when you move to a different task, instead of continuing the old one.",
                "In Claude Code, run /compact before carrying on with a long session.",
                "Give big, self-contained jobs to a subagent: it starts with a small context of its own.",
                "Set a per-run cost limit on the Guardrails page to cap how far one session can grow.",
            ],
            large_context_calls=large_calls,
            window_cost=round(large_cost, 2),
        )

    async def _cache_rebuilds(
        self, window, agent: str, cost: float, sessions: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        rows = await self.db.execute(
            select(Event.trace_id, Event.timestamp, Event.cost).where(
                *window,
                Event.workflow == agent,
                Event.input_tokens >= REBUILD_MIN_TOKENS,
                Event.cache_write_tokens >= Event.input_tokens * REBUILD_WRITE_SHARE,
            )
        )
        # A session's first call always writes its cache; that is not a rebuild.
        rebuilds = [
            row
            for row in rows
            if row.timestamp != (sessions.get(row.trace_id) or {}).get("started")
        ]
        rebuild_cost = sum(float(row.cost or 0.0) for row in rebuilds)
        share = rebuild_cost / cost * 100
        if len(rebuilds) < 3 or share < 3:
            return None
        label = LABELS.get(agent, agent)
        affected = len({row.trace_id for row in rebuilds})
        return self._suggestion(
            "cache_rebuild",
            agent,
            f"{label} rebuilt its prompt cache mid-session {len(rebuilds):,} times",
            (
                f"In {affected:,} session(s), a call had to write the whole conversation to the "
                f"prompt cache again instead of reading it, worth {_money(rebuild_cost)} at API "
                f"rates ({share:.0f}% of usage). That happens when you come back to a session "
                "after the cache has expired, usually after a break of an hour or more: the "
                "first message back pays for the entire context at the cache-write rate."
            ),
            share,
            "Share of usage",
            [
                "After a long break, start a new session rather than resuming a large old one.",
                "If you need the old context, run /compact first so there is less to re-cache.",
                "Finish a task in one sitting where you can; a session left overnight is the costly case.",
            ],
            priority="medium" if share >= 10 else "low",
            rebuilds=len(rebuilds),
            sessions_affected=affected,
            window_cost=round(rebuild_cost, 2),
        )

    async def _dominant_session(
        self, project_id: str, agent: str, cost: float, sessions: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if len(sessions) < MIN_SESSIONS:
            return None
        top_id, top = max(sessions.items(), key=lambda item: item[1]["cost"])
        share = top["cost"] / cost * 100
        if share < 40:
            return None

        from .event_service import EventService

        name = (await EventService(self.db).session_names(project_id, [top_id])).get(top_id)
        label = LABELS.get(agent, agent)
        called = f'"{name}"' if name else f"session {top_id[:8]}"
        return self._suggestion(
            "dominant",
            agent,
            f"One session is {share:.0f}% of {label} usage",
            (
                f"{called} used {_money(top['cost'])} at API rates across {top['calls']:,} "
                f"calls, out of {_money(cost)} for all {len(sessions):,} sessions in the window. "
                "A single very long session is usually the large-context pattern at its extreme."
            ),
            share,
            "Share of usage",
            [
                "Open the agent's page to see this session call by call.",
                "Break work of this size into several sessions, one per task.",
                "Set a per-run cost limit on the Guardrails page so a session stops before it gets here.",
            ],
            priority="medium" if share >= 60 else "low",
            trace_id=top_id,
            session_name=name,
            window_cost=round(top["cost"], 2),
        )

    def _rate_limits(self, agent: str, hits: int) -> Optional[Dict[str, Any]]:
        if hits < 3:
            return None
        label = LABELS.get(agent, agent)
        return self._suggestion(
            "rate_limited",
            agent,
            f"{label} hit its usage limit {hits:,} times",
            (
                f"{hits:,} requests were refused because the plan's usage limit had been "
                "reached. Work stops until the limit resets. The other habits listed for "
                "this agent are what use the allowance up; fixing the largest is what "
                "moves this number."
            ),
            0.0,
            "Refused requests",
            [
                "Start with the largest habit listed for this agent; long sessions use the allowance fastest.",
                "Use a smaller model for routine edits and searches, and the largest one for hard problems.",
                "Set a per-run cost limit on the Guardrails page so one session cannot use the whole allowance.",
            ],
            priority="high" if hits >= 10 else "medium",
            headline=f"{hits:,}",
            rate_limited=hits,
        )
