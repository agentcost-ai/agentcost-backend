"""
AgentCost Backend - Pydantic Schemas

Request/Response models for API validation.
"""

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    EmailStr,
    PrivateAttr,
    ValidationError,
)
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timezone
from pydantic import model_validator


class EventCreate(BaseModel):
    """Schema for a single event in batch"""

    agent_name: str = Field(default="default", max_length=255)
    # 255, matching events.model: Bedrock inference-profile ARNs exceed 100.
    model: str = Field(..., max_length=255)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    # Optional: the server derives total_tokens and re-prices cost itself, so
    # requiring them would only reject batches it can already handle.
    total_tokens: Optional[int] = Field(default=None, ge=0)
    cost: float = Field(default=0.0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    timestamp: str
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Hash of normalized input text for caching pattern detection
    input_hash: Optional[str] = Field(None, max_length=64)

    # Trace structure, emitted by SDKs that use workflow()/step()/tool().
    # Every field is optional: older SDKs, and calls made outside a workflow,
    # send none of them and must keep ingesting exactly as before.
    trace_id: Optional[str] = Field(None, max_length=32)
    span_id: Optional[str] = Field(None, max_length=32)
    parent_span_id: Optional[str] = Field(None, max_length=32)
    workflow: Optional[str] = Field(None, max_length=255)
    step_name: Optional[str] = Field(None, max_length=255)
    step_index: Optional[int] = Field(None, ge=0)
    depth: Optional[int] = Field(None, ge=0)
    tool_name: Optional[str] = Field(None, max_length=255)


    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate and parse timestamp"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format. Use ISO 8601.')


class RejectedEvent(BaseModel):
    """An event that failed validation, echoed back so clients can fix it."""

    index: int
    reason: str


# Enough for a client to fix the event without echoing its whole error list back.
_MAX_REASONS_PER_EVENT = 3

# Upper safety net on a single batch. Kept as a constant because it is enforced
# in two places that must not drift: the field constraint below and the
# early size check in partition_events.
_MAX_EVENTS_PER_BATCH = 1000


def _describe(exc: ValidationError) -> str:
    """One-line, client-facing summary of why an event failed validation."""
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or 'event'}: {err['msg']}"
        for err in exc.errors()[:_MAX_REASONS_PER_EVENT]
    )


class OutcomeCreate(BaseModel):
    """How one run ended, as reported by the SDK."""

    trace_id: str = Field(..., min_length=1, max_length=32)
    workflow: Optional[str] = Field(None, max_length=255)
    success: bool = True
    label: Optional[str] = Field(None, max_length=255)


class EventBatchRequest(BaseModel):
    """Request body for batch event ingestion

    Note: The effective max batch size is enforced by config.max_batch_size
    (default 100) at the route level against ``received_count``. The schema
    allows up to 1000 as an upper safety net.
    """

    project_id: str = Field(..., min_length=1)
    events: List[EventCreate] = Field(..., max_length=_MAX_EVENTS_PER_BATCH)
    # Absent from older SDKs, and from any batch whose runs declared none.
    outcomes: List[OutcomeCreate] = Field(
        default_factory=list, max_length=_MAX_EVENTS_PER_BATCH
    )

    # Outputs of the partitioning validator, not things a client sends: private
    # so they stay out of the generated request-body schema. ``events`` may end
    # up empty here even though the request must carry at least one event.
    _rejected: List[RejectedEvent] = PrivateAttr(default_factory=list)
    _received_count: int = PrivateAttr(default=0)

    @property
    def rejected(self) -> List[RejectedEvent]:
        """The events that failed validation and were dropped."""
        return self._rejected

    @property
    def received_count(self) -> int:
        """How many events the client sent, before any were dropped."""
        return self._received_count

    @model_validator(mode="wrap")
    @classmethod
    def partition_events(cls, data: Any, handler) -> "EventBatchRequest":
        """Validate events one by one instead of all-or-nothing.

        With a plain ``List[EventCreate]`` a single malformed event 422s the
        batch and stores *nothing*. The SDK re-queues the identical payload
        every flush interval (agentcost-sdk/batcher.py), so one permanently
        invalid event blocks the retry queue forever and real events are
        dropped once that queue hits max_retry_batches. Take what parses and
        report the rest.
        """
        if not isinstance(data, dict):
            return handler(data)

        raw = data.get("events")
        if raw is None or (isinstance(raw, list) and not raw):
            raise ValueError("At least one event is required.")
        if not isinstance(raw, list):
            return handler(data)  # wrong type entirely — let field validation say so

        # Size-check before the loop, not after. A wrap validator runs ahead of
        # field validation, so the ``max_length`` above is only consulted at the
        # handler() call below — by which point every event in an oversized
        # batch has already been parsed individually. Before this validator
        # existed, pydantic-core rejected an over-long list without looking at
        # its items; this check restores that.
        if len(raw) > _MAX_EVENTS_PER_BATCH:
            raise ValueError(
                f"Batch too large: {len(raw)} events (max {_MAX_EVENTS_PER_BATCH})."
            )

        valid: List[EventCreate] = []
        rejected: List[RejectedEvent] = []
        for index, item in enumerate(raw):
            try:
                valid.append(EventCreate.model_validate(item))
            except ValidationError as exc:
                rejected.append(RejectedEvent(index=index, reason=_describe(exc)))

        request = handler({**data, "events": valid})
        request._rejected = rejected
        request._received_count = len(raw)
        return request


class EventBatchResponse(BaseModel):
    """Response for batch event ingestion

    ``status``/``events_stored``/``timestamp`` are the contract the SDK parses
    (it only treats ``status == "ok"`` as success); the rejection fields are
    additive so older SDKs keep working.
    """

    status: str = "ok"
    events_stored: int
    timestamp: str
    events_received: int = 0
    events_rejected: int = 0
    rejected: List[RejectedEvent] = Field(default_factory=list)


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

    # Null for calls made outside a workflow().
    trace_id: Optional[str] = None
    workflow: Optional[str] = None
    step_name: Optional[str] = None
    tool_name: Optional[str] = None

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


# ── Executive Report ──────────────────────────────────────────────────────


class MetricDelta(BaseModel):
    """A headline metric with its prior-period comparison."""

    current: float
    previous: float
    change_percent: float  # signed; 0.0 when previous is 0
    direction: Literal["up", "down", "neutral"]


class ReportSummary(BaseModel):
    """Executive-summary band: KPIs with period-over-period deltas."""

    cost: MetricDelta
    calls: MetricDelta
    tokens: MetricDelta
    success_rate: MetricDelta
    avg_latency_ms: MetricDelta
    blended_cost_per_1k: float
    in_out_ratio: float  # input_tokens / output_tokens


class LatencyPercentiles(BaseModel):
    p50: float
    p95: float
    p99: float
    avg: float
    sample_size: int
    approximate: bool = False


class ModelEfficiency(BaseModel):
    model: str
    cost_per_1k: float
    in_out_ratio: float


class TokenEfficiency(BaseModel):
    blended_cost_per_1k: float
    in_out_ratio: float
    total_input_tokens: int
    total_output_tokens: int
    by_model: List[ModelEfficiency]


class ParetoInfo(BaseModel):
    """Cost concentration: how few models drive ≥80% of spend."""

    top_count: int
    top_share: float  # percent of spend held by top_count models
    total_models: int


class CadenceBucket(BaseModel):
    label: str
    index: int
    calls: int
    cost: float


class UsageCadence(BaseModel):
    busiest_day: Optional[str] = None
    busiest_hour: Optional[str] = None
    by_dow: List[CadenceBucket]
    by_hour: List[CadenceBucket]


class ErrorBreakdownRow(BaseModel):
    model: str
    total_calls: int
    error_count: int
    error_rate: float  # percent


class TopError(BaseModel):
    error: str
    count: int


class RunRateProjection(BaseModel):
    daily_avg_cost: float
    projected_monthly_cost: float
    window_days: float


class BudgetStatus(BaseModel):
    enabled: bool
    budget: Optional[float] = None
    current_spend: float = 0.0
    projected_spend: float = 0.0
    utilization_percent: Optional[float] = None
    currency: str = "USD"
    fx_rate: float = 1.0
    mode: str = "off"


class SavingsRollup(BaseModel):
    total_potential_savings_monthly: float = 0.0
    total_potential_savings_percent: float = 0.0
    suggestion_count: int = 0
    high_priority_count: int = 0
    top_suggestions: List[Dict[str, Any]] = Field(default_factory=list)


class ExecutiveReport(BaseModel):
    """Board-ready cost & usage report: executive summary + deep breakdowns."""

    generated_at: datetime
    range_label: str
    period_start: datetime
    period_end: datetime
    previous_period_start: datetime
    previous_period_end: datetime
    is_custom_range: bool = False
    project_name: str
    currency: str = "USD"

    summary: ReportSummary
    overview: AnalyticsOverview
    timeseries: List[TimeSeriesPoint]

    models: List[ModelStats]
    model_pareto: ParetoInfo

    agents: List[AgentStats]
    agent_cost_share: Dict[str, float]  # agent_name -> percent of spend

    latency: LatencyPercentiles
    efficiency: TokenEfficiency
    errors: List[ErrorBreakdownRow]
    top_errors: List[TopError]
    cadence: UsageCadence
    run_rate: RunRateProjection
    budget: BudgetStatus
    savings: SavingsRollup


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
    monthly_budget_usd: Optional[float] = None
    budget_enforcement_mode: Optional[str] = "off"
    budget_alert_thresholds: Optional[List[float]] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ProjectUpdate(BaseModel):
    """Update project request"""
    
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    is_active: Optional[bool] = None


SupportedCurrency = Literal["USD", "INR"]


class ProjectBudgetUpdate(BaseModel):
    """Update project budget settings.

    ``monthly_budget_usd`` accepts the budget amount in ``budget_currency``
    (the name is retained for backward compatibility with prior clients).
    """

    monthly_budget_usd: Optional[float] = Field(None, ge=0)
    budget_enforcement_mode: Literal["off", "warn", "hard_cap"] = "warn"
    budget_alert_thresholds: List[float] = Field(default_factory=lambda: [50.0, 80.0, 100.0])
    budget_currency: SupportedCurrency = "USD"

    @field_validator("budget_alert_thresholds")
    @classmethod
    def validate_thresholds(cls, values: List[float]) -> List[float]:
        cleaned: list[float] = []
        for item in values:
            if item < 1 or item > 100:
                raise ValueError("Threshold values must be between 1 and 100")
            cleaned.append(round(float(item), 2))

        if not cleaned:
            raise ValueError("At least one threshold is required")

        return sorted(set(cleaned))


class ProjectBudgetResponse(BaseModel):
    """Project budget settings with current utilization snapshot.

    All monetary values are expressed in ``budget_currency``. The raw
    USD-denominated spend (from event costs) is also returned for clients
    that want to display it.
    """

    project_id: str
    monthly_budget_usd: Optional[float] = None
    budget_enforcement_mode: Literal["off", "warn", "hard_cap"] = "off"
    budget_alert_thresholds: List[float]
    current_month_spend: float
    current_month_spend_usd: float = 0.0
    utilization_percent: Optional[float] = None
    period_key: str
    budget_currency: SupportedCurrency = "USD"
    fx_rate: float = 1.0


class NotificationResponse(BaseModel):
    """A single in-app notification."""

    id: str
    type: str
    severity: Literal["info", "warning", "critical"] = "info"
    title: str
    body: Optional[str] = None
    link: Optional[str] = None
    project_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    is_read: bool
    read_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class NotificationListResponse(BaseModel):
    items: List[NotificationResponse]
    total: int
    unread_count: int


class NotificationCountResponse(BaseModel):
    unread_count: int


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
