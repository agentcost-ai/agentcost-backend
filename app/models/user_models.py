"""
AgentCost Backend - User Database Models

SQLAlchemy models for user authentication and management.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey,
    Index, Enum
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
import enum

from ..database import Base
from ..common import generate_uuid


class UserRole(str, enum.Enum):
    """User roles for access control"""
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class User(Base):
    """Users table - authentication and profile"""
    
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # Nullable for Google OAuth users
    
    # OAuth provider tracking
    auth_provider = Column(String(20), default="email", nullable=False)  # email, google
    google_id = Column(String(255), unique=True, nullable=True, index=True)  # Google sub claim
    
    name = Column(String(255), nullable=True)
    avatar_url = Column(String(512), nullable=True)
    
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255), nullable=True)
    email_verification_sent_at = Column(DateTime(timezone=True), nullable=True)
    
    password_reset_token = Column(String(255), nullable=True)
    password_reset_expires = Column(DateTime(timezone=True), nullable=True)
    
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    admin_notes = Column(Text, nullable=True)  # Internal notes visible only to admins
    
    # Milestone tracking
    user_number = Column(Integer, unique=True, nullable=True)  # Sequential registration order
    milestone_badge = Column(String(50), nullable=True)  # top_20, top_50, top_100, top_1000, early_adopter
    
    # Soft-delete support
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_active_at = Column(DateTime(timezone=True), nullable=True)
    
    owned_projects = relationship("Project", back_populates="owner", foreign_keys="Project.owner_id")
    project_memberships = relationship("ProjectMember", back_populates="user", foreign_keys="ProjectMember.user_id", passive_deletes=True)
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.email}>"


class UserSession(Base):
    """
    Active user sessions (JWT tokens).
    Allows session management and revocation.
    """
    
    __tablename__ = "user_sessions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    token_hash = Column(String(255), nullable=False, index=True)
    
    device_info = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_revoked = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index("idx_sessions_user", "user_id", "expires_at"),
    )
    
    def __repr__(self):
        return f"<UserSession {self.id[:8]}... for user {self.user_id[:8]}...>"


class ProjectMember(Base):
    """
    Project membership - links users to projects with roles.
    Allows multiple users per project with different permissions.
    """
    
    __tablename__ = "project_members"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    role = Column(String(20), default=UserRole.MEMBER.value, nullable=False)
    
    invited_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    invited_at = Column(DateTime(timezone=True), server_default=func.now())
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships", foreign_keys=[user_id])
    invited_by = relationship("User", foreign_keys=[invited_by_id])
    
    __table_args__ = (
        Index("idx_project_members_unique", "project_id", "user_id", unique=True),
    )
    
    def __repr__(self):
        return f"<ProjectMember {self.user_id} in {self.project_id} as {self.role}>"


class PendingEmailInvitation(Base):
    """
    Pending invitations for users who haven't registered yet.
    When they register, these get converted to ProjectMember entries.
    """
    
    __tablename__ = "pending_email_invitations"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    
    role = Column(String(20), default=UserRole.MEMBER.value, nullable=False)
    
    invited_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    invited_at = Column(DateTime(timezone=True), server_default=func.now())
    
    project = relationship("Project")
    invited_by = relationship("User", foreign_keys=[invited_by_id])
    
    __table_args__ = (
        Index("idx_pending_invites_unique", "project_id", "email", unique=True),
    )
    
    def __repr__(self):
        return f"<PendingEmailInvitation {self.email} to {self.project_id} as {self.role}>"


class PolicyType(str, enum.Enum):
    """Types of legal policies requiring consent"""
    TERMS_OF_SERVICE = "terms"
    PRIVACY_POLICY = "privacy"


class PolicyConsent(Base):
    """
    Audit-grade policy consent tracking.
    
    Stores explicit user consent to Terms of Service and Privacy Policy.
    Supports policy versioning - when policies update, users must re-accept.
    
    GDPR/Legal Compliance Features:
    - Stores exact timestamp of consent
    - Records IP address and user agent for audit trail
    - Tracks specific policy version accepted
    - Immutable records (no updates, only new inserts)
    """
    
    __tablename__ = "policy_consents"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Policy identification
    policy_type = Column(String(20), nullable=False)  # 'terms' or 'privacy'
    policy_version = Column(String(20), nullable=False)  # e.g., '1.0', '1.1', '2.0'
    
    # Consent metadata (for legal audit trail)
    ip_address = Column(String(45), nullable=True)  # IPv6 max length
    user_agent = Column(Text, nullable=True)  # Full user agent string
    
    # Timestamp - immutable
    consented_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationship
    user = relationship("User", backref=backref("policy_consents", passive_deletes=True))
    
    __table_args__ = (
        # Index for efficient policy version checks
        Index("idx_policy_consent_user_type", "user_id", "policy_type"),
        # Index for finding latest consent per user/policy
        Index("idx_policy_consent_lookup", "user_id", "policy_type", "policy_version"),
    )
    
    def __repr__(self):
        return f"<PolicyConsent {self.user_id} accepted {self.policy_type} v{self.policy_version}>"
