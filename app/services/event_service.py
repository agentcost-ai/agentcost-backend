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
        
        for event_data in events:
            # Parse timestamp
            timestamp = datetime.fromisoformat(
                event_data.timestamp.replace('Z', '+00:00')
            )
            
            total_tokens = event_data.input_tokens + event_data.output_tokens
            calculated_cost = await pricing_service.calculate_cost(
                event_data.model,
                event_data.input_tokens,
                event_data.output_tokens,
            )

            db_event = Event(
                project_id=project_id,
                agent_name=event_data.agent_name,
                model=event_data.model,
                input_tokens=event_data.input_tokens,
                output_tokens=event_data.output_tokens,
                total_tokens=total_tokens,
                cost=calculated_cost,
                latency_ms=event_data.latency_ms,
                timestamp=timestamp,
                success=event_data.success,
                error=event_data.error,
                extra_data=event_data.metadata,
                input_hash=event_data.input_hash,
            )
            db_events.append(db_event)
            
            # Record pattern for caching opportunity detection
            if event_data.input_hash:
                await pattern_service.record_pattern(
                    project_id=project_id,
                    agent_name=event_data.agent_name,
                    input_hash=event_data.input_hash,
                    cost=calculated_cost,
                )
        
        self.db.add_all(db_events)
        await self.db.commit()  # Commit without needing IDs back
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
        """Get project by API key (supports both legacy plaintext and hashed keys)"""
        from ..utils.auth import hash_api_key
        
        # First try hashed lookup (secure method)
        hashed_key = hash_api_key(api_key)
        query = select(Project).where(Project.api_key == hashed_key)
        result = await self.db.execute(query)
        project = result.scalar_one_or_none()
        
        if project:
            return project
        
        # Fallback: try plaintext lookup for backward compatibility
        # (for existing projects created before hashing was implemented)
        query = select(Project).where(Project.api_key == api_key)
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
