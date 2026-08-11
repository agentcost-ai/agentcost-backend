"""
AgentCost Backend - Event Ingestion Service

Business logic for storing and processing events.
"""

from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..models.db_models import Event, Project
from ..models.schemas import EventCreate, OutcomeCreate

# Prices are quoted per 1000 tokens, so a single call's cost runs to small
# fractions of a cent. Round at the precision PricingService prices with.
_COST_PRECISION = 8


@dataclass
class PreparedBatch:
    """Priced, not-yet-persisted events plus the aggregates they imply.

    Splitting "price it" from "write it" is what lets budget enforcement run
    against the cost the server is actually going to store instead of the
    figure the client claimed.
    """

    project_id: str
    rows: List[Event] = field(default_factory=list)
    # (agent_name, input_hash) -> (occurrences, summed cost)
    pattern_counts: Dict[Tuple[str, str], Tuple[int, float]] = field(default_factory=dict)
    total_cost: float = 0.0


class EventService:
    """Service for event ingestion and queries"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def prepare_events_batch(
        self,
        project_id: str,
        events: List[EventCreate],
    ) -> PreparedBatch:
        """Build (but do not write) the Event rows for a batch.

        Nothing is added to the session here.
        """
        db_events: List[Event] = []
        from .pricing_service import PricingService
        pricing_service = PricingService(self.db)

        # pre-fetch pricing for all unique models in the batch
        # to eliminate N+1 per-event DB queries.
        try:
            unique_models = {e.model for e in events}
            pricing_cache: dict[str, dict | None] = {}
            for model_name in unique_models:
                pricing_cache[model_name] = await pricing_service.get_model_pricing(model_name)

            # (agent_name, input_hash) -> (occurrences, summed cost), folded
            # here so the whole batch costs one read + one write pass later.
            pattern_counts: dict[tuple[str, str], tuple[int, float]] = {}
            total_cost = 0.0

            ingested_at = datetime.now(timezone.utc)

            for event_data in events:
                # Pin the zone, then clamp: analytics bounds on
                # `timestamp <= now(UTC)`, so a naive or future-dated row would
                # be stored but invisible in every chart and KPI.
                timestamp = datetime.fromisoformat(
                    event_data.timestamp.replace('Z', '+00:00')
                )
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = timestamp.astimezone(timezone.utc)
                if timestamp > ingested_at:
                    timestamp = ingested_at

                # Always derived, never trusted: the client's total_tokens is
                # optional and has been seen to disagree with its own counts.
                total_tokens = event_data.input_tokens + event_data.output_tokens

                # Use cached pricing instead of per-event DB query
                pricing = pricing_cache.get(event_data.model)
                if pricing is not None:
                    input_cost = (event_data.input_tokens / 1000) * pricing["input"]
                    output_cost = (event_data.output_tokens / 1000) * pricing["output"]
                    calculated_cost = round(input_cost + output_cost, _COST_PRECISION)
                else:
                    calculated_cost = 0.0

                # Use server cost when available; fall back to SDK-provided cost.
                # cost_source records *how* we priced it, so a fuzzy match
                # (a neighbouring model's rates) is distinguishable from an
                # exact one downstream.
                if calculated_cost > 0:
                    final_cost = calculated_cost
                    cost_source = (
                        "database-fuzzy"
                        if pricing.get("match") == "fuzzy"
                        else "database-exact"
                    )
                else:
                    final_cost = float(event_data.cost or 0.0)
                    cost_source = "client-sdk"

                total_cost += final_cost

                db_event = Event(
                    project_id=project_id,
                    agent_name=event_data.agent_name,
                    model=event_data.model,
                    input_tokens=event_data.input_tokens,
                    output_tokens=event_data.output_tokens,
                    total_tokens=total_tokens,
                    cost=final_cost,
                    cost_source=cost_source,
                    latency_ms=event_data.latency_ms,
                    timestamp=timestamp,
                    success=event_data.success,
                    error=event_data.error,
                    extra_data=event_data.metadata,
                    input_hash=event_data.input_hash,
                    trace_id=event_data.trace_id,
                    span_id=event_data.span_id,
                    parent_span_id=event_data.parent_span_id,
                    workflow=event_data.workflow,
                    step_name=event_data.step_name,
                    step_index=event_data.step_index,
                    depth=event_data.depth,
                    tool_name=event_data.tool_name,
                )
                db_events.append(db_event)

                # Fold repeats of the same pattern together instead of
                # emitting one SELECT + flush per event.
                if event_data.input_hash:
                    key = (event_data.agent_name, event_data.input_hash)
                    occurrences, summed = pattern_counts.get(key, (0, 0.0))
                    pattern_counts[key] = (occurrences + 1, summed + final_cost)
        finally:
            await pricing_service.close()

        return PreparedBatch(
            project_id=project_id,
            rows=db_events,
            pattern_counts=pattern_counts,
            total_cost=round(total_cost, _COST_PRECISION),
        )

    async def persist_events_batch(self, prepared: PreparedBatch) -> int:
        """Write a prepared batch. Caller's transaction owns the commit."""
        from .baseline_service import PatternAnalysisService

        if not prepared.rows:
            return 0

        self.db.add_all(prepared.rows)
        await PatternAnalysisService(self.db).record_patterns_bulk(
            project_id=prepared.project_id,
            pattern_counts=prepared.pattern_counts,
        )
        await self.db.flush()  # Let get_db handle the final commit

        return len(prepared.rows)

    async def persist_outcomes(
        self, project_id: str, outcomes: List[OutcomeCreate]
    ) -> int:
        """
        Upsert run outcomes. Caller's transaction owns the commit.

        A run can report an outcome more than once -- an optimistic success
        followed by a late failure -- so the last write for a trace wins.
        """
        from ..models.db_models import TraceOutcome

        if not outcomes:
            return 0

        # Last record per trace, so one batch cannot insert two rows for the
        # same run and trip the unique constraint.
        latest = {o.trace_id: o for o in outcomes}

        existing_rows = await self.db.execute(
            select(TraceOutcome).where(
                TraceOutcome.project_id == project_id,
                TraceOutcome.trace_id.in_(list(latest)),
            )
        )
        existing = {row.trace_id: row for row in existing_rows.scalars().all()}

        now = datetime.now(timezone.utc)
        written = 0
        for trace_id, outcome in latest.items():
            row = existing.get(trace_id)
            if row is None:
                self.db.add(
                    TraceOutcome(
                        project_id=project_id,
                        trace_id=trace_id,
                        workflow=outcome.workflow,
                        success=outcome.success,
                        label=outcome.label,
                        recorded_at=now,
                    )
                )
            else:
                row.success = outcome.success
                row.label = outcome.label
                row.workflow = outcome.workflow or row.workflow
                row.recorded_at = now
            written += 1

        await self.db.flush()
        return written

    async def get_events(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0,
        agent_name: Optional[str] = None,
        model: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Event]:
        """
        Get events with optional filtering.
        
        Args:
            project_id: Project ID
            limit: Max results
            offset: Offset for pagination
            agent_name: Filter by agent
            model: Filter by model
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of events
        """
        query = select(Event).where(Event.project_id == project_id)
        
        if agent_name:
            query = query.where(Event.agent_name == agent_name)
        
        if model:
            query = query.where(Event.model == model)
        
        if start_time:
            query = query.where(Event.timestamp >= start_time)
        
        if end_time:
            query = query.where(Event.timestamp <= end_time)
        
        query = query.order_by(Event.timestamp.desc())
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_event_count(self, project_id: str) -> int:
        """Get total event count for project"""
        query = select(func.count(Event.id)).where(Event.project_id == project_id)
        result = await self.db.execute(query)
        return result.scalar() or 0


class ProjectService:
    """Service for project management"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        query = select(Project).where(Project.id == project_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_api_key(self, api_key: str) -> Optional[Project]:
        """Get project by API key (hashed lookup only)"""
        from ..utils.auth import hash_api_key
        
        hashed_key = hash_api_key(api_key)
        query = select(Project).where(Project.api_key == hashed_key)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create(self, name: str, description: Optional[str] = None, owner_id: Optional[str] = None) -> tuple:
        """
        Create a new project with a secure hashed API key.
        
        Args:
            name: Project name
            description: Optional project description
            owner_id: Optional user ID to link as owner
        
        Returns:
            Tuple of (project, plaintext_api_key)
            The plaintext key should be shown to user ONCE and never stored.
        """
        from ..utils.auth import generate_secure_api_key
        
        plaintext_key, hashed_key = generate_secure_api_key()
        
        project = Project(
            name=name,
            description=description,
            owner_id=owner_id,
        )
        project.api_key = hashed_key
        
        self.db.add(project)
        await self.db.flush()
        
        return project, plaintext_key
    
    async def update(
        self, 
        project_id: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Project]:
        """Update project"""
        project = await self.get_by_id(project_id)
        if not project:
            return None
        
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        if is_active is not None:
            project.is_active = is_active
        
        await self.db.flush()
        return project

    async def regenerate_api_key(self, project_id: str) -> Optional[tuple]:
        """
        Regenerate a project's API key.

        Returns:
            Tuple of (project, plaintext_api_key)
        """
        from ..utils.auth import generate_secure_api_key

        project = await self.get_by_id(project_id)
        if not project:
            return None

        plaintext_key, hashed_key = generate_secure_api_key()
        project.api_key = hashed_key
        await self.db.flush()
        return project, plaintext_key
    
    async def delete(self, project_id: str) -> bool:
        """
        Delete a project and ALL of its dependent data.

        Several child tables (events, daily_aggregates, optimization
        recommendations, baselines, input_pattern_cache, model_feedback_*)
        reference ``projects.id`` without an ON DELETE CASCADE clause —
        adding cascade DDL post-hoc is fragile across SQLite/Postgres, so
        we clean them up explicitly here.

        Tables that already have ON DELETE CASCADE on the DB FK
        (project_members, pending_email_invitations, budget_threshold_alerts,
        notifications) get cleaned up automatically by the DB; we issue the
        DELETEs here too for SQLite test parity and to guarantee no orphans
        on Postgres even if a constraint was added in a non-standard way.
        """
        from sqlalchemy import delete as sa_delete

        from ..models.db_models import (
            BudgetThresholdAlert,
            DailyAggregate,
            Event,
            InputPatternCache,
            Notification,
            OptimizationRecommendation,
            ProjectBaseline,
        )
        from ..models.user_models import PendingEmailInvitation, ProjectMember

        project = await self.get_by_id(project_id)
        if not project:
            return False

        # Order matters only when a child has its own children — but none of
        # these tables fan out further, so any order works.
        for model in (
            Event,
            DailyAggregate,
            OptimizationRecommendation,
            ProjectBaseline,
            InputPatternCache,
            BudgetThresholdAlert,
            Notification,
            ProjectMember,
            PendingEmailInvitation,
        ):
            await self.db.execute(
                sa_delete(model).where(model.project_id == project_id)
            )

        await self.db.delete(project)
        await self.db.flush()
        return True
