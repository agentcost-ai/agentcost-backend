"""
AgentCost Backend - Event Ingestion Service

Business logic for storing and processing events.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timezone
from typing import List, Optional

from ..models.db_models import Event, Project
from ..models.schemas import EventCreate


class EventService:
    """Service for event ingestion and queries"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_events_batch(
        self, 
        project_id: str, 
        events: List[EventCreate]
    ) -> int:
        """
        Create multiple events in a single transaction.
        
        Args:
            project_id: Project ID
            events: List of event data
            
        Returns:
            Number of events created
        """
        from .baseline_service import PatternAnalysisService
        
        db_events = []
        pattern_service = PatternAnalysisService(self.db)
        from .pricing_service import PricingService
        pricing_service = PricingService(self.db)
        
        # pre-fetch pricing for all unique models in the batch
        # to eliminate N+1 per-event DB queries.
        try:
            unique_models = {e.model for e in events}
            pricing_cache: dict[str, dict | None] = {}
            for model_name in unique_models:
                pricing_cache[model_name] = await pricing_service.get_model_pricing(model_name)

            # Collect pattern records to batch after the loop
            pattern_records: list[tuple[str, str, str, float]] = []

            for event_data in events:
                # Parse timestamp
                timestamp = datetime.fromisoformat(
                    event_data.timestamp.replace('Z', '+00:00')
                )
                
                total_tokens = event_data.input_tokens + event_data.output_tokens

                # Use cached pricing instead of per-event DB query
                pricing = pricing_cache.get(event_data.model)
                if pricing is not None:
                    input_cost = (event_data.input_tokens / 1000) * pricing["input"]
                    output_cost = (event_data.output_tokens / 1000) * pricing["output"]
                    calculated_cost = round(input_cost + output_cost, 8)
                else:
                    calculated_cost = 0.0

                # Use server cost when available; fall back to SDK-provided cost
                if calculated_cost > 0:
                    final_cost = calculated_cost
                    cost_source = pricing.get("source", "database")
                else:
                    final_cost = event_data.cost
                    cost_source = "client-sdk"

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
                )
                db_events.append(db_event)
                
                # Collect pattern for later batch recording
                if event_data.input_hash:
                    pattern_records.append(
                        (event_data.agent_name, event_data.input_hash, final_cost)
                    )

            # Batch-record patterns in one pass (still individual inserts but
            # avoids interleaving with pricing queries)
            for agent_name, input_hash, cost in pattern_records:
                await pattern_service.record_pattern(
                    project_id=project_id,
                    agent_name=agent_name,
                    input_hash=input_hash,
                    cost=cost,
                )
            
            self.db.add_all(db_events)
            await self.db.flush()  # Let get_db handle the final commit
        finally:
            await pricing_service.close()
        
        return len(db_events)
    
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
        # Override the default API key with our hashed version
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
        """Delete project"""
        project = await self.get_by_id(project_id)
        if not project:
            return False
        
        await self.db.delete(project)
        await self.db.flush()
        return True
