"""
AgentCost Backend - Events API Routes

Endpoints for event ingestion.
"""

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
    """
    # M7 fix: enforce config.max_batch_size at runtime
    settings = get_settings()
    if len(request.events) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch too large: {len(request.events)} events exceeds maximum of {settings.max_batch_size}.",
        )

    # Verify project_id matches
    if request.project_id != project.id:
        raise HTTPException(
            status_code=403,
            detail="Project ID does not match API key.",
        )

    budget_service = BudgetService(db)
    incoming_cost = sum(max(float(event.cost), 0.0) for event in request.events)
    budget_eval = await budget_service.evaluate(project, additional_cost=incoming_cost)
    if budget_eval.get("should_block"):
        raise HTTPException(
            status_code=429,
            detail=(
                "Monthly budget hard cap reached. Increase your budget or switch "
                "budget mode to 'warn' to continue ingesting events."
            ),
        )
    
    try:
        # Store events
        event_service = EventService(db)
        count = await event_service.create_events_batch(
            project_id=project.id,
            events=request.events,
        )

        # Record newly crossed thresholds for this month (deduplicated).
        # Fan-out to in-app notifications + email for owners/admins.
        if budget_eval.get("enabled") and budget_eval.get("crossed_thresholds"):
            await budget_service.record_threshold_crossings(
                project_id=project.id,
                period_key=budget_eval["period_key"],
                crossed_thresholds=budget_eval["crossed_thresholds"],
                spent_amount=budget_eval["projected_spend"],
                budget_amount=budget_eval["budget"],
                utilization_percent=budget_eval["utilization_percent"],
                project=project,
            )
        
        return EventBatchResponse(
            status="ok",
            events_stored=count,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        # Log error internally for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("Error ingesting events: %s", str(e))
        # Return generic message to client - no internal details exposed
        raise HTTPException(
            status_code=500,
            detail="Failed to process events. Please try again later."
        )


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
