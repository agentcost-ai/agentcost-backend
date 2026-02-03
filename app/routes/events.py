"""
AgentCost Backend - Events API Routes

Endpoints for event ingestion.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from ..database import get_db
from ..models.schemas import EventBatchRequest, EventBatchResponse, EventResponse
from ..models.db_models import Project
from ..services.event_service import EventService
from ..utils.auth import validate_api_key

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
    # Verify project_id matches
    if request.project_id != project.id:
        raise HTTPException(
            status_code=403,
            detail="Project ID does not match API key.",
        )
    
    try:
        # Store events
        event_service = EventService(db)
        count = await event_service.create_events_batch(
            project_id=project.id,
            events=request.events,
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
    limit: int = 100,
    offset: int = 0,
    agent_name: str = None,
    model: str = None,
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    List events for a project.
    
    Supports filtering by agent_name and model.
    """
    event_service = EventService(db)
    events = await event_service.get_events(
        project_id=project.id,
        limit=min(limit, 1000),  # Cap at 1000
        offset=offset,
        agent_name=agent_name,
        model=model,
    )
    
    return events


@router.get("/count")
async def get_event_count(
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """Get total event count for project."""
    event_service = EventService(db)
    count = await event_service.get_event_count(project.id)
    
    return {"count": count}
