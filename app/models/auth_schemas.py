"""
AgentCost Backend - Authentication Schemas

Pydantic models for auth requests and responses.
"""

from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict
from typing import Optional, List
from datetime import datetime
import re


#  Policy Consent 

class PolicyConsentInput(BaseModel):
    """Single policy consent record"""
    
    policy_type: str = Field(..., description="Policy type: 'terms' or 'privacy'")
    policy_version: str = Field(..., description="Version accepted, e.g., '1.0'")
    
    @field_validator('policy_type')
    @classmethod
    def validate_policy_type(cls, v):
        if v not in ('terms', 'privacy'):
            raise ValueError("Policy type must be 'terms' or 'privacy'")
        return v


class PolicyConsentStatus(BaseModel):
    """Status of a user's policy consent"""
    
    policy_type: str
    current_version: str
    accepted_version: Optional[str] = None
    accepted_at: Optional[datetime] = None
    is_current: bool = False  # True if accepted version matches current


class PolicyCheckResponse(BaseModel):
    """Response for policy version check"""
    
    policies_accepted: bool  # True if all policies are current
    terms: PolicyConsentStatus
    privacy: PolicyConsentStatus


#  Registration 

class UserRegister(BaseModel):
    """User registration request"""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=100, description="Password (min 8 chars)")
    name: Optional[str] = Field(None, max_length=255, description="Display name")
    
    # Legal consent - REQUIRED for registration
    accept_terms: bool = Field(..., description="User accepts Terms of Service")
    accept_privacy: bool = Field(..., description="User accepts Privacy Policy")
    terms_version: str = Field(..., description="Terms of Service version accepted")
    privacy_version: str = Field(..., description="Privacy Policy version accepted")
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Ensure password meets security requirements"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        return v
    
    @field_validator('accept_terms', 'accept_privacy')
    @classmethod
    def validate_consent(cls, v, info):
        """Ensure consent is explicitly given"""
        if not v:
            field_name = info.field_name
            policy = "Terms of Service" if field_name == "accept_terms" else "Privacy Policy"
            raise ValueError(f'You must accept the {policy} to create an account')
        return v


class UserResponse(BaseModel):
    """User response (no password)"""
    
    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    email_verified: bool
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class RegisterResponse(BaseModel):
    """Registration response"""
    
    user: UserResponse
    message: str = "Registration successful. Please check your email to verify your account."


#  Login 

class UserLogin(BaseModel):
    """User login request"""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(default=False, description="Extended session duration")


class TokenResponse(BaseModel):
    """JWT token response"""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    
    refresh_token: str


#  Password Management 

class PasswordChangeRequest(BaseModel):
    """Change password request (when logged in)"""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    
    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        return v


class PasswordResetRequest(BaseModel):
    """Request password reset email"""
    
    email: EmailStr = Field(..., description="Email address for password reset")


class PasswordResetConfirm(BaseModel):
    """Confirm password reset with token"""
    
    token: str = Field(..., description="Password reset token from email")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    
    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        return v


#  Email Verification 

class EmailVerificationRequest(BaseModel):
    """Email verification request"""
    
    token: str = Field(..., description="Email verification token")


class ResendVerificationRequest(BaseModel):
    """Resend verification email request"""
    
    email: EmailStr = Field(..., description="Email address to resend verification")


#  Session Management 

class SessionInfo(BaseModel):
    """Active session information"""
    
    id: str
    device_info: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime
    is_current: bool = False
    
    model_config = ConfigDict(from_attributes=True)


class SessionListResponse(BaseModel):
    """List of active sessions"""
    
    sessions: List[SessionInfo]
    total: int


#  Profile 

class ProfileUpdate(BaseModel):
    """Update user profile"""
    
    name: Optional[str] = Field(None, max_length=255)
    avatar_url: Optional[str] = Field(None, max_length=512)


#  Project Membership 

class ProjectMemberCreate(BaseModel):
    """Invite user to project"""
    
    email: EmailStr = Field(..., description="Email of user to invite")
    role: str = Field(default="member", description="Role: admin, member, or viewer")


class ProjectMemberResponse(BaseModel):
    """Project member info"""
    
    id: str
    user_id: str
    user_email: str
    user_name: Optional[str] = None
    role: str
    invited_at: datetime
    accepted_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class ProjectMemberUpdate(BaseModel):
    """Update member role"""
    
    role: str = Field(..., description="New role: admin, member, or viewer")
