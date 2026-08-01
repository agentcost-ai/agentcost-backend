"""
AgentCost Backend - Models

Database models and Pydantic schemas.

Do not prune the db_models imports below to match __all__. Importing every
model here is what registers it on Base.metadata, which create_all relies on;
a linter will call several of them unused, but dropping them silently stops
their tables from being created.
"""

from .db_models import (
    Project, 
    Event, 
    DailyAggregate, 
    ModelPricing,
    ModelAlternative,
    OptimizationRecommendation,
    ProjectBaseline,
    InputPatternCache,
    Feedback,
    FeedbackUpvote,
    FeedbackComment,
    FeedbackEvent,
)
from .user_models import User, UserSession, ProjectMember, UserRole
from .schemas import (
    EventCreate,
    EventBatchRequest,
    EventBatchResponse,
    EventResponse,
    AnalyticsOverview,
    AgentStats,
    ModelStats,
    TimeSeriesPoint,
    AnalyticsResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    HealthResponse,
    FeedbackCreate,
    FeedbackUpdate,
    FeedbackResponse,
    FeedbackListResponse,
    FeedbackSummaryResponse,
    FeedbackEventResponse,
)
from .auth_schemas import (
    UserRegister,
    UserLogin,
    UserResponse,
    TokenResponse,
    RegisterResponse,
    PasswordChangeRequest,
    PasswordResetRequest,
    PasswordResetConfirm,
    SessionInfo,
    SessionListResponse,
    ProfileUpdate,
    ProjectMemberCreate,
    ProjectMemberResponse,
    ProjectMemberUpdate,
)

__all__ = [
    # Database models
    "Project",
    "Event",
    "DailyAggregate",
    "Feedback",
    "FeedbackUpvote",
    "FeedbackComment",
    "FeedbackEvent",
    
    # User models
    "User",
    "UserSession",
    "ProjectMember",
    "UserRole",
    
    # Event schemas
    "EventCreate",
    "EventBatchRequest",
    "EventBatchResponse",
    "EventResponse",
    
    # Analytics schemas
    "AnalyticsOverview",
    "AgentStats",
    "ModelStats",
    "TimeSeriesPoint",
    "AnalyticsResponse",
    
    # Project schemas
    "ProjectCreate",
    "ProjectResponse",
    "ProjectUpdate",
    
    # Auth schemas
    "UserRegister",
    "UserLogin",
    "UserResponse",
    "TokenResponse",
    "RegisterResponse",
    "PasswordChangeRequest",
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "SessionInfo",
    "SessionListResponse",
    "ProfileUpdate",
    "ProjectMemberCreate",
    "ProjectMemberResponse",
    "ProjectMemberUpdate",
    
    # Feedback schemas
    "FeedbackCreate",
    "FeedbackUpdate",
    "FeedbackResponse",
    "FeedbackListResponse",
    "FeedbackSummaryResponse",
    "FeedbackEventResponse",
    
    # Health
    "HealthResponse",
]
