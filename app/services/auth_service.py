"""
AgentCost Backend - Authentication Service

Handles password hashing, JWT generation/validation, and session management.
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from ..models.user_models import User, UserSession, ProjectMember, PolicyConsent
from ..models.auth_schemas import (
    UserRegister, UserLogin, UserResponse, TokenResponse,
    SessionInfo, ProfileUpdate, PolicyConsentStatus, PolicyCheckResponse
)
from ..config import get_settings


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


settings = get_settings()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", settings.secret_key)
JWT_ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = settings.refresh_token_expire_days
EXTENDED_SESSION_DAYS = settings.extended_session_days


def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> Tuple[str, datetime]:
    """
    Create a JWT access token.
    
    Returns:
        Tuple of (token, expiry_datetime)
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, expire


def create_refresh_token(
    user_id: str,
    remember_me: bool = False
) -> Tuple[str, datetime]:
    """
    Create a refresh token for obtaining new access tokens.
    
    Returns:
        Tuple of (token, expiry_datetime)
    """
    if remember_me:
        expire = datetime.now(timezone.utc) + timedelta(days=EXTENDED_SESSION_DAYS)
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_hex(16),
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, expire


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.
    
    Returns:
        Decoded payload dict or None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def hash_token(token: str) -> str:
    """Hash a token for secure storage in database"""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_verification_token() -> str:
    """Generate a secure email verification token"""
    return secrets.token_urlsafe(32)


def generate_password_reset_token() -> str:
    """Generate a secure password reset token"""
    return secrets.token_urlsafe(32)


class AuthService:
    """Service class for authentication operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def create_user(
        self, 
        data: UserRegister,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> User:
        """
        Create a new user account with legal consent tracking.
        
        Args:
            data: Registration data including consent flags
            ip_address: Client IP for consent audit trail
            user_agent: Client user agent for consent audit trail
        
        Raises:
            ValueError: If email already exists or consent not given
        """
        # Validate consent flags (should be caught by Pydantic, but double-check)
        if not data.accept_terms or not data.accept_privacy:
            raise ValueError("You must accept the Terms of Service and Privacy Policy")
        
        # Normalize email to lowercase for consistent comparison
        normalized_email = data.email.lower().strip()
        
        # Check if email already exists
        existing = await self.get_user_by_email(normalized_email)
        if existing:
            raise ValueError("Email already registered")
        
        user = User(
            email=normalized_email,
            password_hash=hash_password(data.password),
            name=data.name,
            email_verification_token=generate_verification_token(),
        )
        
        try:
            self.db.add(user)
            await self.db.flush()  # Get user ID without committing
            
            # Record consent for Terms of Service
            terms_consent = PolicyConsent(
                user_id=user.id,
                policy_type="terms",
                policy_version=data.terms_version,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            self.db.add(terms_consent)
            
            # Record consent for Privacy Policy
            privacy_consent = PolicyConsent(
                user_id=user.id,
                policy_type="privacy",
                policy_version=data.privacy_version,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            self.db.add(privacy_consent)
            
            await self.db.commit()
            await self.db.refresh(user)
        except IntegrityError:
            # Race condition: someone registered this email between our check and insert
            await self.db.rollback()
            raise ValueError("Email already registered")
        
        return user
    
    async def record_policy_consent(
        self,
        user_id: str,
        policy_type: str,
        policy_version: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> PolicyConsent:
        """
        Record a new policy consent (for re-acceptance after policy updates).
        
        Args:
            user_id: The user accepting the policy
            policy_type: 'terms' or 'privacy'
            policy_version: Version being accepted
            ip_address: Client IP for audit
            user_agent: Client user agent for audit
        
        Returns:
            The created PolicyConsent record
        """
        consent = PolicyConsent(
            user_id=user_id,
            policy_type=policy_type,
            policy_version=policy_version,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.db.add(consent)
        await self.db.commit()
        await self.db.refresh(consent)
        return consent
    
    async def get_policy_consent_status(self, user_id: str) -> PolicyCheckResponse:
        """
        Check if user has accepted current versions of all policies.
        
        Returns:
            PolicyCheckResponse with status for each policy
        """
        from ..config import get_settings
        settings = get_settings()
        
        # Get latest consent for each policy type
        terms_consent = await self._get_latest_consent(user_id, "terms")
        privacy_consent = await self._get_latest_consent(user_id, "privacy")
        
        terms_status = PolicyConsentStatus(
            policy_type="terms",
            current_version=settings.terms_of_service_version,
            accepted_version=terms_consent.policy_version if terms_consent else None,
            accepted_at=terms_consent.consented_at if terms_consent else None,
            is_current=(
                terms_consent is not None and 
                terms_consent.policy_version == settings.terms_of_service_version
            ),
        )
        
        privacy_status = PolicyConsentStatus(
            policy_type="privacy",
            current_version=settings.privacy_policy_version,
            accepted_version=privacy_consent.policy_version if privacy_consent else None,
            accepted_at=privacy_consent.consented_at if privacy_consent else None,
            is_current=(
                privacy_consent is not None and 
                privacy_consent.policy_version == settings.privacy_policy_version
            ),
        )
        
        return PolicyCheckResponse(
            policies_accepted=terms_status.is_current and privacy_status.is_current,
            terms=terms_status,
            privacy=privacy_status,
        )
    
    async def _get_latest_consent(
        self, 
        user_id: str, 
        policy_type: str
    ) -> Optional[PolicyConsent]:
        """Get the most recent consent record for a user and policy type."""
        result = await self.db.execute(
            select(PolicyConsent)
            .where(
                PolicyConsent.user_id == user_id,
                PolicyConsent.policy_type == policy_type
            )
            .order_by(PolicyConsent.consented_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def authenticate_user(
        self,
        email: str,
        password: str
    ) -> Optional[User]:
        """
        Authenticate user with email and password.
        
        Returns:
            User if authentication successful, None otherwise
        """
        user = await self.get_user_by_email(email)
        
        if not user:
            return None
        
        if not verify_password(password, user.password_hash):
            return None
        
        if not user.is_active:
            return None
        
        return user
    
    async def login_user(
        self,
        user: User,
        remember_me: bool = False,
        device_info: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> TokenResponse:
        """
        Generate tokens and create session for user login.
        """
        access_token, access_expires = create_access_token(
            user_id=user.id,
            email=user.email
        )
        
        refresh_token, refresh_expires = create_refresh_token(
            user_id=user.id,
            remember_me=remember_me
        )
        
        session = UserSession(
            user_id=user.id,
            token_hash=hash_token(refresh_token),
            device_info=device_info,
            ip_address=ip_address,
            expires_at=refresh_expires,
        )
        
        self.db.add(session)
        
        user.last_login_at = datetime.now(timezone.utc)
        
        await self.db.commit()
        
        expires_in = int((access_expires - datetime.now(timezone.utc)).total_seconds())
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in,
            user=UserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                email_verified=user.email_verified,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login_at=user.last_login_at,
            )
        )
    
    async def logout_user(self, token: str) -> bool:
        """Revoke a specific session by token"""
        token_hash = hash_token(token)
        
        result = await self.db.execute(
            update(UserSession)
            .where(UserSession.token_hash == token_hash)
            .values(is_revoked=True)
        )
        
        await self.db.commit()
        return result.rowcount > 0
    
    async def logout_all_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        result = await self.db.execute(
            update(UserSession)
            .where(
                UserSession.user_id == user_id,
                UserSession.is_revoked == False
            )
            .values(is_revoked=True)
        )
        
        await self.db.commit()
        return result.rowcount
    
    async def verify_email(self, token: str) -> Optional[User]:
        """Verify user email with token"""
        result = await self.db.execute(
            select(User).where(
                User.email_verification_token == token,
                User.email_verified == False
            )
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        user.email_verified = True
        user.email_verification_token = None
        
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def regenerate_verification_token(self, user_id: str) -> Optional[str]:
        """Generate a fresh verification token for resend requests"""
        user = await self.get_user_by_id(user_id)
        
        if not user or user.email_verified:
            return None
        
        new_token = generate_verification_token()
        user.email_verification_token = new_token
        
        await self.db.commit()
        await self.db.refresh(user)
        
        return new_token
    
    async def request_password_reset(self, email: str) -> Optional[str]:
        """
        Generate password reset token.
        
        Returns:
            Reset token if user exists, None otherwise
        """
        user = await self.get_user_by_email(email)
        
        if not user:
            return None
        
        token = generate_password_reset_token()
        user.password_reset_token = token
        user.password_reset_expires = datetime.now(timezone.utc) + timedelta(hours=24)
        
        await self.db.commit()
        
        return token
    
    async def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password using token"""
        result = await self.db.execute(
            select(User).where(
                User.password_reset_token == token,
                User.password_reset_expires > datetime.now(timezone.utc)
            )
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return False
        
        user.password_hash = hash_password(new_password)
        user.password_reset_token = None
        user.password_reset_expires = None
        
        await self.logout_all_sessions(user.id)
        
        await self.db.commit()
        return True
    
    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """Change password for logged-in user"""
        user = await self.get_user_by_id(user_id)
        
        if not user:
            return False
        
        if not verify_password(current_password, user.password_hash):
            return False
        
        user.password_hash = hash_password(new_password)
        
        await self.db.commit()
        return True
    
    async def update_profile(
        self,
        user_id: str,
        data: ProfileUpdate
    ) -> Optional[User]:
        """Update user profile"""
        user = await self.get_user_by_id(user_id)
        
        if not user:
            return None
        
        if data.name is not None:
            user.name = data.name
        if data.avatar_url is not None:
            user.avatar_url = data.avatar_url
        
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def get_active_sessions(self, user_id: str) -> list[UserSession]:
        """Get all active sessions for a user"""
        result = await self.db.execute(
            select(UserSession)
            .where(
                UserSession.user_id == user_id,
                UserSession.is_revoked == False,
                UserSession.expires_at > datetime.now(timezone.utc)
            )
            .order_by(UserSession.last_used_at.desc())
        )
        
        return result.scalars().all()
    
    async def revoke_session(self, user_id: str, session_id: str) -> bool:
        """Revoke a specific session"""
        result = await self.db.execute(
            update(UserSession)
            .where(
                UserSession.id == session_id,
                UserSession.user_id == user_id
            )
            .values(is_revoked=True)
        )
        
        await self.db.commit()
        return result.rowcount > 0
    
    async def refresh_session(
        self,
        refresh_token: str,
        device_info: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Optional[TokenResponse]:
        """
        Refresh access token using refresh token.
        """
        payload = decode_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("sub")
        token_hash = hash_token(refresh_token)
        
        result = await self.db.execute(
            select(UserSession)
            .where(
                UserSession.token_hash == token_hash,
                UserSession.is_revoked == False,
                UserSession.expires_at > datetime.now(timezone.utc)
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            return None
        
        user = await self.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            return None
        
        session.last_used_at = datetime.now(timezone.utc)
        if device_info:
            session.device_info = device_info
        if ip_address:
            session.ip_address = ip_address
        
        access_token, access_expires = create_access_token(
            user_id=user.id,
            email=user.email
        )
        
        await self.db.commit()
        
        expires_in = int((access_expires - datetime.now(timezone.utc)).total_seconds())
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            user=UserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                email_verified=user.email_verified,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login_at=user.last_login_at,
            )
        )


async def get_current_user(
    db: AsyncSession,
    token: str
) -> Optional[User]:
    """
    Get current user from JWT token.
    Used as a dependency in protected routes.
    """
    payload = decode_token(token)
    
    if not payload or payload.get("type") != "access":
        return None
    
    user_id = payload.get("sub")
    
    if not user_id:
        return None
    
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)
    )
    
    return result.scalar_one_or_none()
