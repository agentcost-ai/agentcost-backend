"""
AgentCost Backend - Models

Database models and Pydantic schemas.
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
    
    # Health
    "HealthResponse",
]
