"""
AgentCost Backend - Pydantic Schemas

Request/Response models for API validation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


class EventCreate(BaseModel):
    """Schema for a single event in batch"""
    
    agent_name: str = Field(default="default", max_length=255)
    model: str = Field(..., max_length=100)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    cost: float = Field(..., ge=0)
    latency_ms: int = Field(..., ge=0)
    timestamp: str
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Hash of normalized input text for caching pattern detection
    input_hash: Optional[str] = Field(None, max_length=64)
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate and parse timestamp"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format. Use ISO 8601.')


class EventBatchRequest(BaseModel):
    """Request body for batch event ingestion"""
    
    project_id: str = Field(..., min_length=1)
    events: List[EventCreate] = Field(..., min_length=1, max_length=1000)


class EventBatchResponse(BaseModel):
    """Response for batch event ingestion"""
    
    status: str = "ok"
    events_stored: int
    timestamp: str


class EventResponse(BaseModel):
    """Single event response"""
    
    id: str
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency_ms: int
    timestamp: str
    success: bool
    error: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def serialize_timestamp(cls, v):
        """Convert datetime to UTC ISO string"""
        if isinstance(v, datetime):
            # Ensure it's UTC
            if v.tzinfo is None:
                v = v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc).isoformat()
        return v
    
    model_config = ConfigDict(from_attributes=True)


class AnalyticsOverview(BaseModel):
    """Overview analytics response"""
    
    total_cost: float
    total_calls: int
    total_tokens: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_cost_per_call: float
    avg_tokens_per_call: float = 0.0
    avg_latency_ms: float
    success_rate: float
    period_start: datetime
    period_end: datetime


class AgentStats(BaseModel):
    """Stats for a single agent"""
    
    agent_name: str
    total_calls: int
    total_tokens: int
    total_cost: float
    avg_latency_ms: float
    success_rate: float


class ModelStats(BaseModel):
    """Stats for a single model"""
    
    model: str
    total_calls: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    total_cost: float
    avg_latency_ms: float
    cost_share: float = 0.0


class TimeSeriesPoint(BaseModel):
    """Single point in time series"""
    
    timestamp: datetime
    calls: int
    tokens: int
    cost: float
    avg_latency_ms: float


class AnalyticsResponse(BaseModel):
    """Full analytics response"""
    
    overview: AnalyticsOverview
    agents: List[AgentStats]
    models: List[ModelStats]
    timeseries: List[TimeSeriesPoint]


class ProjectCreate(BaseModel):
    """Create project request"""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    """Project response"""
    
    id: str
    name: str
    description: Optional[str] = None
    api_key: str
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ProjectUpdate(BaseModel):
    """Update project request"""
    
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = "ok"
    version: str
    timestamp: str
