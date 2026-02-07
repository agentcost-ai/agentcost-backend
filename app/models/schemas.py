"""
AgentCost Backend - Pydantic Schemas

Request/Response models for API validation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, EmailStr
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timezone
from pydantic import model_validator


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
    api_key: Optional[str] = None
    key_prefix: Optional[str] = None
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


FeedbackType = Literal[
    "feature_request",
    "bug_report",
    "model_request",
    "general",
    "security_report",
    "performance_issue",
]
FeedbackStatus = Literal[
    "open",
    "under_review",
    "needs_info",
    "in_progress",
    "completed",
    "shipped",
    "rejected",
    "duplicate",
]
FeedbackPriority = Literal["low", "medium", "high", "critical"]


class FeedbackCreate(BaseModel):
    """Submit feedback or a request."""

    type: FeedbackType
    title: str = Field(..., min_length=3, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    model_name: Optional[str] = Field(None, max_length=255)
    model_provider: Optional[str] = Field(None, max_length=100)
    user_email: Optional[EmailStr] = Field(None, max_length=255)
    user_name: Optional[str] = Field(None, max_length=255)

    # Type-specific structured data stored as JSON
    metadata: Optional[Dict[str, Any]] = None
    # Attachment references (list of {url, name?, size?, type?})
    attachments: Optional[List[Dict[str, Any]]] = None
    # Environment context (production, staging, development)
    environment: Optional[str] = Field(None, max_length=50)
    # Client metadata (SDK version, OS, browser)
    client_metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_model_request(self):
        if self.type == "model_request" and not (self.model_name or self.model_provider):
            raise ValueError("Model requests should include a model name or provider")
        return self


class FeedbackUpdate(BaseModel):
    """Admin update payload for feedback status and response."""

    status: FeedbackStatus
    priority: Optional[FeedbackPriority] = None
    admin_response: Optional[str] = Field(None, max_length=5000)


class FeedbackResponse(BaseModel):
    id: str
    type: FeedbackType
    title: str
    description: str
    status: FeedbackStatus
    priority: FeedbackPriority
    upvotes: int
    user_has_upvoted: bool
    model_name: Optional[str]
    model_provider: Optional[str]
    admin_response: Optional[str]
    created_at: datetime
    updated_at: datetime
    comment_count: int
    user_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    environment: Optional[str] = None
    is_confidential: bool = False

    model_config = ConfigDict(from_attributes=True)


class FeedbackListResponse(BaseModel):
    items: List[FeedbackResponse]
    total: int
    limit: int
    offset: int


class FeedbackSummaryResponse(BaseModel):
    total: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]


class FeedbackCreatedResponse(BaseModel):
    id: str
    message: str


class FeedbackCommentCreate(BaseModel):
    comment: str = Field(..., min_length=1, max_length=2000)
    user_name: Optional[str] = Field(None, max_length=255)


class FeedbackCommentResponse(BaseModel):
    id: str
    user_name: Optional[str]
    comment: str
    is_admin: bool
    created_at: datetime


class FeedbackCommentListResponse(BaseModel):
    items: List[FeedbackCommentResponse]
    total: int


class FeedbackEventResponse(BaseModel):
    """Audit trail event for a feedback item."""

    id: str
    feedback_id: str
    event_type: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    actor_id: Optional[str] = None
    created_at: datetime
