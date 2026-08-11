"""
AgentCost Backend - Events API Routes

Endpoints for event ingestion.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from ..database import get_db
from ..models.schemas import EventBatchRequest, EventBatchResponse, EventResponse
from ..models.db_models import Project
from ..services.event_service import EventService
from ..services.budget_service import BudgetService
from ..utils.auth import validate_api_key, validate_project_access
from ..config import get_settings

router = APIRouter(prefix="/v1/events", tags=["Events"])

logger = logging.getLogger(__name__)


def _ingest_failed(stage: str, exc: Exception) -> HTTPException:
    """Generic 500 for a failed ingest: no internals reach the client, and a
    500 is retryable, which is right for a transient pricing or DB fault."""
    logger.exception("Error %s events: %s", stage, exc)
    return HTTPException(
        status_code=500,
        detail="Failed to process events. Please try again later.",
    )


@router.post("/batch", response_model=EventBatchResponse)
async def ingest_events_batch(
    request: EventBatchRequest,
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Ingest a batch of events.

    This is the main endpoint called by the AgentCost SDK.
    Events are stored and processed for analytics.

    Malformed events are dropped individually and reported back; the batch as
    a whole still succeeds so the SDK does not retry a payload it can never
    get accepted.
    """
    # M7 fix: enforce config.max_batch_size at runtime.
    # Counted against what was *sent*, not what survived validation.
    settings = get_settings()
    if request.received_count > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch too large: {request.received_count} events exceeds maximum of {settings.max_batch_size}.",
        )

    # Verify project_id matches
    if request.project_id != project.id:
        raise HTTPException(
            status_code=403,
            detail="Project ID does not match API key.",
        )

    if request.rejected:
        logger.warning(
            "Dropped %d malformed event(s) for project %s: %s",
            len(request.rejected),
            project.id,
            request.rejected[0].reason,
        )

    def _response(stored: int) -> EventBatchResponse:
        return EventBatchResponse(
            status="ok",
            events_stored=stored,
            timestamp=datetime.now(timezone.utc).isoformat(),
            events_received=request.received_count,
            events_rejected=len(request.rejected),
            # Enough to debug with, without echoing a whole junk batch back.
            rejected=request.rejected[:10],
        )

    if not request.events:
        return _response(0)

    event_service = EventService(db)
    budget_service = BudgetService(db)

    try:
        prepared = await event_service.prepare_events_batch(
            project_id=project.id,
            events=request.events,
        )
    except Exception as exc:
        raise _ingest_failed("pricing", exc) from exc

    budget_eval = None
    if BudgetService.has_budget(project):
        # Skipped wholesale for projects without a budget: evaluate() is a
        # month-to-date SUM(cost) plus an FX lookup, on every batch.
        try:
            budget_eval = await budget_service.evaluate(
                project,
                additional_cost=prepared.total_cost,
                hot_path=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Fail open: a broken budget check must not cost the user telemetry.
            logger.warning("Budget evaluation failed for project %s: %s", project.id, exc)

    # Deliberate check-then-act: concurrent batches can overshoot a hard cap by
    # roughly the value of what is in flight, which is cheaper than serializing
    # every ingest on the project row.
    if budget_eval and budget_eval.get("should_block"):
        raise HTTPException(
            status_code=429,
            detail=(
                "Monthly budget hard cap reached. Increase your budget or switch "
                "budget mode to 'warn' to continue ingesting events."
            ),
        )

    try:
        count = await event_service.persist_events_batch(prepared)
        await event_service.persist_outcomes(project.id, request.outcomes)
    except Exception as exc:
        raise _ingest_failed("ingesting", exc) from exc

    # Record newly crossed thresholds for this month (deduplicated).
    # Fan-out to in-app notifications + email for owners/admins.
    # Deliberately outside the ingest try-block and swallowing its own errors:
    # the events are already flushed and an alerting failure must not turn a
    # successful ingest into a 500 that rolls them back.
    if budget_eval and budget_eval.get("enabled") and budget_eval.get("crossed_thresholds"):
        try:
            await budget_service.record_threshold_crossings(
                project_id=project.id,
                period_key=budget_eval["period_key"],
                crossed_thresholds=budget_eval["crossed_thresholds"],
                spent_amount=budget_eval["projected_spend"],
                budget_amount=budget_eval["budget"],
                utilization_percent=budget_eval["utilization_percent"],
                project=project,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Budget threshold recording failed for project %s: %s", project.id, exc
            )

    return _response(count)


@router.get("", response_model=list[EventResponse])
async def list_events(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    agent_name: str = None,
    model: str = None,
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """
    List events for a project.
    
    Supports filtering by agent_name and model.
    """
    event_service = EventService(db)
    events = await event_service.get_events(
        project_id=project.id,
        limit=limit,
        offset=offset,
        agent_name=agent_name,
        model=model,
    )
    
    return events


@router.get("/count")
async def get_event_count(
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Get total event count for project."""
    event_service = EventService(db)
    count = await event_service.get_event_count(project.id)
    
    return {"count": count}
