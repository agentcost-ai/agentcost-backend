"""
AgentCost Backend - Database Models

SQLAlchemy models for all database tables.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey,
    Index, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from ..database import Base
from ..common import generate_uuid


def generate_api_key():
    return f"sk_{uuid.uuid4().hex}"


class Project(Base):
    """Projects table - one per user/organization"""
    
    __tablename__ = "projects"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    api_key = Column(String(255), unique=True, nullable=False, default=generate_api_key)
    
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    events = relationship("Event", back_populates="project", lazy="dynamic")
    owner = relationship("User", back_populates="owned_projects", foreign_keys=[owner_id])
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project {self.name}>"


class Event(Base):
    """Events table - stores every LLM call"""
    
    __tablename__ = "events"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    
    agent_name = Column(String(255), nullable=False, default="default")
    model = Column(String(100), nullable=False)
    
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    
    cost = Column(Float, nullable=False)
    # Track where the cost figure came from ('database', 'defaults', 'client-sdk')
    cost_source = Column(String(50), nullable=True)
    latency_ms = Column(Integer, nullable=False)
    
    success = Column(Boolean, default=True)
    error = Column(Text, nullable=True)
    
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    # using 'extra_data' instead of 'metadata' (reserved by SQLAlchemy)
    extra_data = Column(JSON, nullable=True)
    
    # Hash of normalized input for caching pattern detection
    input_hash = Column(String(64), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    project = relationship("Project", back_populates="events")
    
    __table_args__ = (
        Index("idx_events_project_time", "project_id", "timestamp"),
        Index("idx_events_agent", "project_id", "agent_name", "timestamp"),
        Index("idx_events_model", "project_id", "model", "timestamp"),
        Index("idx_events_input_hash", "project_id", "input_hash"),
    )
    
    def __repr__(self):
        return f"<Event {self.id} - {self.agent_name}/{self.model}>"


class DailyAggregate(Base):
    """
    Pre-calculated daily aggregates for fast dashboard queries.
    
    TODO: Future Performance Optimization
    This table is designed for pre-computing daily analytics to improve dashboard
    performance when the events table grows large (100K+ events).
    
    plan:
    1. Create a background job to aggregate daily data
    2. Modify analytics_service.py to use this table for historical data
    3. Query events table only for today's incomplete data
    
    Currently, analytics_service.py queries the events table directly.
    Implement this optimization when dashboard response time exceeds 500ms.
    """
    
    __tablename__ = "daily_aggregates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    
    agent_name = Column(String(255), nullable=True)
    model = Column(String(100), nullable=True)
    
    total_calls = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    total_input_tokens = Column(Integer, nullable=False, default=0)
    total_output_tokens = Column(Integer, nullable=False, default=0)
    total_cost = Column(Float, nullable=False, default=0.0)
    avg_latency_ms = Column(Float, nullable=False, default=0.0)
    
    success_count = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index("idx_daily_project_date", "project_id", "date"),
        Index("idx_daily_agent", "project_id", "agent_name", "date"),
    )
    
    def __repr__(self):
        return f"<DailyAggregate {self.date} - {self.agent_name}>"


class ModelPricing(Base):
    """
    Dynamic model pricing table.
    
    Stores pricing for 1600+ models synced from LiteLLM.
    SDK fetches latest prices from this table via /v1/pricing endpoint.
    """
    
    __tablename__ = "model_pricing"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(255), unique=True, nullable=False)
    
    input_price_per_1k = Column(Float, nullable=False, default=0.0)
    output_price_per_1k = Column(Float, nullable=False, default=0.0)
    
    provider = Column(String(50), nullable=False, default="unknown")
    is_active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Additional fields for capability tracking
    max_tokens = Column(Integer, nullable=True)
    supports_vision = Column(Boolean, default=False)
    supports_function_calling = Column(Boolean, default=False)
    supports_streaming = Column(Boolean, default=True)
    
    # Source tracking for automated updates
    pricing_source = Column(String(50), default="manual")  # manual, litellm, openrouter
    source_updated_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index("idx_pricing_provider", "provider"),
    )
    
    def __repr__(self):
        return f"<ModelPricing {self.model_name} - ${self.input_price_per_1k}/${self.output_price_per_1k}>"


class ModelAlternative(Base):
    """
    Model alternatives with learned scores for cost optimization.
    Auto-populated from pricing data + user feedback.
    
    The system learns which alternatives work based on:
    1. Implementation/dismissal rates
    2. Actual vs estimated savings accuracy
    3. User feedback patterns
    """
    
    __tablename__ = "model_alternatives"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    source_model = Column(String(255), nullable=False)
    alternative_model = Column(String(255), nullable=False)
    
    # When to suggest this alternative
    max_output_tokens_threshold = Column(Integer, nullable=True)
    max_input_tokens_threshold = Column(Integer, nullable=True)
    min_success_rate_required = Column(Float, default=0.95)
    
    # Capability requirements - only suggest if alternative has these
    requires_vision = Column(Boolean, default=False)
    requires_function_calling = Column(Boolean, default=False)
    
    # Learned scoring (0.0-1.0, starts neutral at 0.5)
    confidence_score = Column(Float, default=0.5)
    times_suggested = Column(Integer, default=0)
    times_implemented = Column(Integer, default=0)
    times_dismissed = Column(Integer, default=0)
    
    # Actual outcome tracking
    total_estimated_savings = Column(Float, default=0.0)
    total_actual_savings = Column(Float, default=0.0)
    avg_accuracy = Column(Float, default=0.0)  # actual/estimated ratio
    
    # Quality tier (1=best, 5=lowest) - auto-calculated from price ratio
    quality_tier = Column(Integer, default=3)
    price_ratio = Column(Float, default=1.0)  # alt_price / source_price
    
    # Provider matching
    source_provider = Column(String(50), nullable=True)
    alternative_provider = Column(String(50), nullable=True)
    same_provider = Column(Boolean, default=False)
    
    # Source tracking
    source = Column(String(20), default="auto")  # auto, manual, benchmark
    
    is_active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index("idx_alternatives_source", "source_model"),
        Index("idx_alternatives_lookup", "source_model", "is_active"),
        Index("idx_alternatives_confidence", "source_model", "confidence_score"),
    )
    
    def __repr__(self):
        return f"<ModelAlternative {self.source_model} -> {self.alternative_model} (conf: {self.confidence_score:.2f})>"


class OptimizationRecommendation(Base):
    """
    Tracks optimization recommendations shown to users.
    Used to measure effectiveness and learn from outcomes.
    """
    
    __tablename__ = "optimization_recommendations"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    
    recommendation_type = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    
    agent_name = Column(String(255), nullable=True)
    model = Column(String(100), nullable=True)
    alternative_model = Column(String(100), nullable=True)
    
    # Calculated savings at time of recommendation
    estimated_monthly_savings = Column(Float, default=0.0)
    estimated_savings_percent = Column(Float, default=0.0)
    
    # Metrics snapshot at recommendation time
    metrics_snapshot = Column(JSON, nullable=True)
    
    # User interaction tracking
    status = Column(String(20), default="pending")  # pending, implemented, dismissed, expired
    user_feedback = Column(Text, nullable=True)
    implemented_at = Column(DateTime(timezone=True), nullable=True)
    dismissed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Outcome tracking (populated after implementation)
    actual_savings = Column(Float, nullable=True)
    outcome_measured_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index("idx_recommendations_project", "project_id", "created_at"),
        Index("idx_recommendations_status", "project_id", "status"),
    )
    
    def __repr__(self):
        return f"<OptimizationRecommendation {self.recommendation_type} - {self.title[:30]}>"


class ProjectBaseline(Base):
    """
    Statistical baselines for anomaly detection.
    Stores rolling averages and standard deviations per agent/model.
    """
    
    __tablename__ = "project_baselines"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    
    agent_name = Column(String(255), nullable=True)
    model = Column(String(100), nullable=True)
    
    # Cost baselines
    avg_cost_per_call = Column(Float, default=0.0)
    stddev_cost_per_call = Column(Float, default=0.0)
    
    # Token baselines
    avg_input_tokens = Column(Float, default=0.0)
    stddev_input_tokens = Column(Float, default=0.0)
    avg_output_tokens = Column(Float, default=0.0)
    stddev_output_tokens = Column(Float, default=0.0)
    
    # Latency baselines
    avg_latency_ms = Column(Float, default=0.0)
    stddev_latency_ms = Column(Float, default=0.0)
    p95_latency_ms = Column(Float, default=0.0)
    
    # Call frequency
    avg_daily_calls = Column(Float, default=0.0)
    stddev_daily_calls = Column(Float, default=0.0)
    
    # Error rate
    avg_error_rate = Column(Float, default=0.0)
    
    # Sample info
    sample_count = Column(Integer, default=0)
    sample_days = Column(Integer, default=0)
    
    last_calculated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_baselines_project", "project_id"),
        Index("idx_baselines_lookup", "project_id", "agent_name", "model"),
    )
    
    def __repr__(self):
        return f"<ProjectBaseline {self.project_id} - {self.agent_name}/{self.model}>"


class InputPatternCache(Base):
    """
    Stores hashed input patterns for caching opportunity detection.
    Uses a hash of normalized input to find duplicate/similar queries.
    """
    
    __tablename__ = "input_pattern_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    
    agent_name = Column(String(255), nullable=False)
    input_hash = Column(String(64), nullable=False)  # SHA256 of normalized input
    
    occurrence_count = Column(Integer, default=1)
    first_seen_at = Column(DateTime(timezone=True), server_default=func.now())
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Cost impact
    total_cost_for_pattern = Column(Float, default=0.0)
    avg_cost_per_occurrence = Column(Float, default=0.0)
    
    __table_args__ = (
        Index("idx_patterns_project_agent", "project_id", "agent_name"),
        Index("idx_patterns_hash", "project_id", "input_hash"),
        Index("idx_patterns_count", "project_id", "occurrence_count"),
    )
    
    def __repr__(self):
        return f"<InputPatternCache {self.agent_name} - {self.occurrence_count} occurrences>"


class Feedback(Base):
    """
    User feedback and requests.

    Supports feature requests, bug reports, model requests, and general feedback.
    """

    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True, default=generate_uuid)

    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    user_email = Column(String(255), nullable=True)
    user_name = Column(String(255), nullable=True)

    type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)

    model_name = Column(String(255), nullable=True)
    model_provider = Column(String(100), nullable=True)

    # Mapped to column "metadata" in the DB; attribute renamed to dodge
    # SQLAlchemy's reserved Base.metadata namespace.
    type_metadata = Column("metadata", JSON, nullable=True)
    attachments = Column(JSON, nullable=True)
    environment = Column(String(50), nullable=True)  # e.g. production, staging, local
    client_metadata = Column(JSON, nullable=True)     # SDK version, OS, browser, etc.
    is_confidential = Column(Boolean, default=False, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

    status = Column(String(50), default="open", nullable=False)
    priority = Column(String(50), default="medium", nullable=False)

    upvotes = Column(Integer, default=0, nullable=False)

    admin_response = Column(Text, nullable=True)
    admin_responded_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    comments = relationship("FeedbackComment", back_populates="feedback", cascade="all, delete-orphan")
    upvote_entries = relationship("FeedbackUpvote", back_populates="feedback", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_feedback_type", "type"),
        Index("idx_feedback_status", "status"),
        Index("idx_feedback_priority", "priority"),
        Index("idx_feedback_created", "created_at"),
        Index("idx_feedback_upvotes", "upvotes"),
        Index("idx_feedback_user", "user_id"),
    )

    def __repr__(self):
        return f"<Feedback {self.type} - {self.title[:30]}>"


class FeedbackUpvote(Base):
    """Tracks upvotes for feedback items (prevents duplicates)."""

    __tablename__ = "feedback_upvotes"

    feedback_id = Column(String(36), ForeignKey("feedback.id", ondelete="CASCADE"), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    feedback = relationship("Feedback", back_populates="upvote_entries")

    __table_args__ = (
        Index("idx_feedback_upvotes_user", "user_id"),
    )


class FeedbackComment(Base):
    """Comments on feedback items."""

    __tablename__ = "feedback_comments"

    id = Column(String(36), primary_key=True, default=generate_uuid)

    feedback_id = Column(String(36), ForeignKey("feedback.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    user_name = Column(String(255), nullable=True)

    comment = Column(Text, nullable=False)
    is_admin = Column(Boolean, default=False)
    is_internal = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    feedback = relationship("Feedback", back_populates="comments")

    __table_args__ = (
        Index("idx_feedback_comments_feedback", "feedback_id", "created_at"),
    )


class FeedbackEvent(Base):
    """
    Audit trail for feedback lifecycle changes.

    Every status change, priority change, or admin action emits
    an immutable event record for full traceability.
    """

    __tablename__ = "feedback_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    feedback_id = Column(String(36), ForeignKey("feedback.id", ondelete="CASCADE"), nullable=False)

    event_type = Column(String(50), nullable=False)  # status_change, priority_change, admin_note
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    actor_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_feedback_events_feedback", "feedback_id", "created_at"),
    )

    def __repr__(self):
        return f"<FeedbackEvent {self.event_type} on {self.feedback_id}>"


class AdminActivityLog(Base):
    """
    Immutable audit trail for all admin-initiated actions.

    Every admin operation (user toggle, project freeze, feedback update,
    key rotation, email dispatch, etc.) is recorded here for compliance
    and operational visibility.
    """

    __tablename__ = "admin_activity_log"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    admin_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    action_type = Column(String(100), nullable=False)  # e.g. user_disabled, project_frozen, feedback_updated
    target_type = Column(String(50), nullable=True)     # user, project, feedback, system
    target_id = Column(String(36), nullable=True)

    details = Column(JSON, nullable=True)  # Serialized change details
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_admin_log_admin", "admin_id", "created_at"),
        Index("idx_admin_log_action", "action_type", "created_at"),
        Index("idx_admin_log_target", "target_type", "target_id"),
    )

    def __repr__(self):
        return f"<AdminActivityLog {self.action_type} by {self.admin_id}>"


class PricingSyncLog(Base):
    """
    Detailed audit trail for pricing sync operations.

    Captures every sync (LiteLLM / OpenRouter) with full change breakdown:
    new models added, prices changed, capabilities updated, and errors.
    """

    __tablename__ = "pricing_sync_log"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    admin_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    source = Column(String(50), nullable=False)  # litellm, openrouter
    status = Column(String(20), nullable=False)   # ok, error, partial

    models_created = Column(Integer, default=0)
    models_updated = Column(Integer, default=0)
    models_skipped = Column(Integer, default=0)

    # Detailed change records stored as JSON arrays
    new_models = Column(JSON, nullable=True)          # [{model, provider, input_price, output_price, ...}]
    price_changes = Column(JSON, nullable=True)       # [{model, old_input, new_input, change_pct, ...}]
    capability_changes = Column(JSON, nullable=True)  # [{model, change, old, new}]

    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_sync_log_source", "source", "created_at"),
    )

    def __repr__(self):
        return f"<PricingSyncLog {self.source} - {self.status} at {self.created_at}>"


class UserMilestone(Base):
    """
    Records of milestone achievements for users.

    Each row is an immutable record of a milestone a user has earned,
    such as being among the first N registrants or reaching usage
    thresholds.
    """

    __tablename__ = "user_milestones"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    milestone_type = Column(String(100), nullable=False)  # signup_position, first_event, power_user, etc.
    milestone_name = Column(String(255), nullable=False)
    milestone_description = Column(Text, nullable=True)

    achieved_at = Column(DateTime(timezone=True), server_default=func.now())
    notified = Column(Boolean, default=False)

    metadata_json = Column("metadata", JSON, nullable=True)

    __table_args__ = (
        Index("idx_user_milestones_user", "user_id", "achieved_at"),
        Index("idx_user_milestones_type", "user_id", "milestone_type"),
    )

    def __repr__(self):
        return f"<UserMilestone {self.milestone_type} for {self.user_id}>"
