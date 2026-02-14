# AgentCost Backend - Authentication Routes - API endpoints for user authentication, registration, and session management.

import time
from collections import defaultdict
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Annotated

from ..database import get_db
from ..models.auth_schemas import (
    UserRegister, UserLogin, UserResponse, TokenResponse, RegisterResponse,
    PasswordChangeRequest, PasswordResetRequest, PasswordResetConfirm,
    EmailVerificationRequest, ResendVerificationRequest,
    SessionListResponse, SessionInfo, ProfileUpdate,
    RefreshTokenRequest, PolicyCheckResponse, PolicyConsentInput,
    GoogleAuthRequest
)
from ..services.auth_service import AuthService, get_current_user, decode_token, verify_google_id_token
from ..services.email_service import send_verification_email, send_password_reset_email, send_welcome_email
from ..services.member_service import MemberService
from ..models.user_models import User
from ..models.db_models import UserMilestone
from ..config import get_settings
from ..utils.auth import get_required_user, get_optional_user


router = APIRouter(prefix="/v1/auth", tags=["Authentication"])

# Security scheme for JWT bearer token
security = HTTPBearer(auto_error=False)

# Simple in-memory rate limiter for sensitive endpoints
_reset_attempts: dict[str, list[float]] = defaultdict(list)
_RESET_MAX_ATTEMPTS = 3
_RESET_WINDOW_SECONDS = 3600  # 1 hour


def _check_rate_limit(key: str) -> None:
    """Raise 429 if key has exceeded rate limit within the window."""
    now = time.monotonic()
    attempts = _reset_attempts[key]
    # Prune expired entries
    _reset_attempts[key] = [t for t in attempts if now - t < _RESET_WINDOW_SECONDS]
    if len(_reset_attempts[key]) >= _RESET_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later.",
        )
    _reset_attempts[key].append(now)


# ============== Dependencies ==============

async def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    """Get auth service instance"""
    return AuthService(db)


def get_client_info(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Extract device info and IP from request"""
    device_info = request.headers.get("User-Agent", "Unknown")
    
    # Try to get real IP from proxy headers
    ip_address = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
        request.headers.get("X-Real-IP") or
        request.client.host if request.client else None
    )
    
    return device_info, ip_address


# ============== Registration ==============

@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    data: UserRegister,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    - **email**: Valid email address (must be unique)
    - **password**: Min 8 chars, must include uppercase, lowercase, and number
    - **name**: Optional display name
    - **accept_terms**: Must be true - explicit consent to Terms of Service
    - **accept_privacy**: Must be true - explicit consent to Privacy Policy
    - **terms_version**: Version of Terms being accepted (e.g., "1.0")
    - **privacy_version**: Version of Privacy Policy being accepted (e.g., "1.0")
    
    Legal consent is recorded with timestamp, IP address, and user agent for audit compliance.
    """
    # Extract client info for consent audit trail
    device_info, ip_address = get_client_info(request)
    
    try:
        user = await auth_service.create_user(
            data,
            ip_address=ip_address,
            user_agent=device_info
        )
        
        # Process any pending project invitations for this email
        member_service = MemberService(db)
        pending_count = await member_service.process_pending_invitations_for_user(user)
        if pending_count > 0:
            print(f"[AUTH] Processed {pending_count} pending invitation(s) for {user.email}")
        
        # Fire off verification email (non-blocking, won't fail registration if email fails)
        plaintext_token = getattr(user, '_plaintext_verification_token', None)
        if plaintext_token:
            await send_verification_email(user.email, plaintext_token, user.name)
            # Track when verification email was sent
            user.email_verification_sent_at = datetime.now(timezone.utc)
            await db.flush()
        
        message = "Registration successful. Please check your email to verify your account."
        if pending_count > 0:
            message += f" You have {pending_count} pending project invitation(s) waiting for you."
        
        return RegisterResponse(
            user=UserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                email_verified=user.email_verified,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login_at=user.last_login_at,
                auth_provider=getattr(user, 'auth_provider', 'email'),
                user_number=user.user_number,
                milestone_badge=user.milestone_badge,
            ),
            message=message
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ============== Google OAuth ==============

@router.post("/google", response_model=TokenResponse)
async def google_auth(
    data: GoogleAuthRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate or register with Google.
    
    Accepts a Google ID token (credential) from Google Identity Services.
    - If the user exists, signs them in.
    - If the user doesn't exist, creates a new account and signs them in.
    
    Google users get auto-verified email and no password is required.
    """
    settings = get_settings()
    
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google sign-in is not configured on this server",
        )
    
    # Verify the Google ID token
    google_info = verify_google_id_token(data.credential)
    
    if not google_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google credentials. Please try again.",
        )
    
    device_info, ip_address = get_client_info(request)
    
    try:
        user, is_new_user = await auth_service.google_authenticate(
            google_id_info=google_info,
            terms_version=data.terms_version,
            privacy_version=data.privacy_version,
            ip_address=ip_address,
            user_agent=device_info,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    # Process pending invitations for new users
    if is_new_user:
        member_service = MemberService(db)
        pending_count = await member_service.process_pending_invitations_for_user(user)
        if pending_count > 0:
            print(f"[AUTH] Processed {pending_count} pending invitation(s) for {user.email}")
        
        # Send welcome email with early-adopter badge (skip for superusers/admins)
        if user.user_number and not getattr(user, 'is_superuser', False):
            try:
                sent = await send_welcome_email(
                    email=user.email,
                    name=user.name,
                    user_number=user.user_number,
                    milestone_badge=user.milestone_badge,
                )
                if sent:
                    from sqlalchemy import update
                    await db.execute(
                        update(UserMilestone)
                        .where(
                            UserMilestone.user_id == user.id,
                            UserMilestone.milestone_type == "signup_position",
                        )
                        .values(notified=True)
                    )
                    await db.flush()
            except Exception:
                pass  # Don't fail auth if welcome email fails
    
    # Generate tokens and create session (same as regular login)
    return await auth_service.login_user(
        user=user,
        remember_me=True,  # Google users get extended session by default
        device_info=device_info,
        ip_address=ip_address,
    )


# ============== Policy Consent ==============

@router.get("/policies/current")
async def get_current_policy_versions():
    """
    Get current policy versions.
    
    Use this to display correct versions to users during signup
    and to check if policies have been updated.
    """
    settings = get_settings()
    return {
        "terms_version": settings.terms_of_service_version,
        "privacy_version": settings.privacy_policy_version,
    }


@router.get("/policies/status", response_model=PolicyCheckResponse)
async def check_policy_status(
    user: User = Depends(get_required_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Check if current user has accepted the latest policy versions.
    
    Returns status for both Terms of Service and Privacy Policy,
    including whether re-acceptance is required.
    """
    return await auth_service.get_policy_consent_status(user.id)


@router.post("/policies/accept", response_model=PolicyCheckResponse)
async def accept_policies(
    consents: list[PolicyConsentInput],
    request: Request,
    user: User = Depends(get_required_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Accept updated policy versions.
    
    Use this when policies have been updated and user needs to re-accept.
    Records consent with timestamp, IP, and user agent for audit trail.
    """
    device_info, ip_address = get_client_info(request)
    
    for consent in consents:
        await auth_service.record_policy_consent(
            user_id=user.id,
            policy_type=consent.policy_type,
            policy_version=consent.policy_version,
            ip_address=ip_address,
            user_agent=device_info
        )
    
    return await auth_service.get_policy_consent_status(user.id)


# ============== Login ==============

@router.post("/login", response_model=TokenResponse)
async def login(
    data: UserLogin,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate user and get access token.
    
    - **email**: User's email address
    - **password**: User's password
    - **remember_me**: If true, session lasts 30 days instead of 7
    """
    user = await auth_service.authenticate_user(data.email, data.password)
    
    if not user:
        # Check if this is a Google-only user trying to log in with password
        google_check = await auth_service.get_user_by_email(data.email)
        if google_check and google_check.auth_provider == "google" and not google_check.password_hash:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This account uses Google sign-in. Please use the 'Continue with Google' button.",
            )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Block login if email is not verified
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email address before logging in. Check your inbox for the verification link.",
        )
    
    device_info, ip_address = get_client_info(request)
    
    return await auth_service.login_user(
        user=user,
        remember_me=data.remember_me,
        device_info=device_info,
        ip_address=ip_address
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    data: RefreshTokenRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Get a new access token using a refresh token.
    
    Use this when your access token expires.
    """
    device_info, ip_address = get_client_info(request)
    
    result = await auth_service.refresh_session(
        refresh_token=data.refresh_token,
        device_info=device_info,
        ip_address=ip_address
    )
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return result


# ============== Logout ==============

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Logout and invalidate current session.
    """
    if credentials:
        await auth_service.logout_user(credentials.credentials)
    
    return None


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
async def logout_all(
    user: User = Depends(get_required_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Logout from all devices and invalidate all sessions.
    
    Requires authentication.
    """
    await auth_service.logout_all_sessions(user.id)
    return None


# ============== Current User ==============

@router.get("/me", response_model=UserResponse)
async def get_me(
    user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current authenticated user's profile.
    
    Requires authentication.
    Also updates last_active_at to track user activity (debounced to 5 mins).
    """
    # Update last_active_at at most once every 5 minutes to reduce write load
    now = datetime.now(timezone.utc)
    should_update = True
    
    if user.last_active_at:
        elapsed = (now - user.last_active_at).total_seconds()
        if elapsed < 300:  # 5 minutes
            should_update = False
            
    if should_update:
        user.last_active_at = now
        await db.commit()
    
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        email_verified=user.email_verified,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        auth_provider=getattr(user, 'auth_provider', 'email'),
        user_number=user.user_number,
        milestone_badge=user.milestone_badge,
    )


@router.patch("/me", response_model=UserResponse)
async def update_me(
    data: ProfileUpdate,
    user: User = Depends(get_required_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Update current user's profile.
    
    Requires authentication.
    """
    updated = await auth_service.update_profile(user.id, data)
    
    return UserResponse(
        id=updated.id,
        email=updated.email,
        name=updated.name,
        avatar_url=updated.avatar_url,
        email_verified=updated.email_verified,
        is_active=updated.is_active,
        created_at=updated.created_at,
        last_login_at=updated.last_login_at,
        auth_provider=getattr(updated, 'auth_provider', 'email'),
        user_number=updated.user_number,
        milestone_badge=updated.milestone_badge,
    )


# ============== Password Management ==============

@router.post("/password/change", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    data: PasswordChangeRequest,
    user: User = Depends(get_required_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Change password for current user.
    
    Requires authentication and current password.
    """
    # Google-only users cannot change a password they don't have
    if getattr(user, 'auth_provider', 'email') == 'google' and not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account uses Google sign-in and has no password to change. Use 'Forgot Password' to set one."
        )
    
    success = await auth_service.change_password(
        user_id=user.id,
        current_password=data.current_password,
        new_password=data.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    return None


@router.post("/password/reset-request", status_code=status.HTTP_204_NO_CONTENT)
async def request_password_reset(
    data: PasswordResetRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Request a password reset email.
    
    For security, always returns 204 even if email doesn't exist.
    Rate limited to 3 requests per email per hour.
    """
    _check_rate_limit(f"reset:{data.email.lower().strip()}")

    user = await auth_service.get_user_by_email(data.email)
    token = await auth_service.request_password_reset(data.email)
    
    if token and user:
        # Send password reset email with user's name for personalization
        await send_password_reset_email(data.email, token, user.name)
    
    return None


@router.post("/password/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_password(
    data: PasswordResetConfirm,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Reset password using token from email.
    """
    success = await auth_service.reset_password(
        token=data.token,
        new_password=data.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    return None


# ============== Email Verification ==============

@router.post("/verify-email", status_code=status.HTTP_204_NO_CONTENT)
async def verify_email(
    data: EmailVerificationRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Verify email address using token from email.
    """
    user = await auth_service.verify_email(data.token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    # Send welcome email now that the address is confirmed (skip for superusers/admins)
    if user.user_number and not getattr(user, 'is_superuser', False):
        try:
            sent = await send_welcome_email(
                email=user.email,
                name=user.name,
                user_number=user.user_number,
                milestone_badge=user.milestone_badge,
            )
            if sent:
                from sqlalchemy import update, select
                await auth_service.db.execute(
                    update(UserMilestone)
                    .where(
                        UserMilestone.user_id == user.id,
                        UserMilestone.milestone_type == "signup_position",
                    )
                    .values(notified=True)
                )
                await auth_service.db.flush()
        except Exception:
            pass  # Don't fail verification if welcome email fails
    
    return None


@router.post("/resend-verification", status_code=status.HTTP_204_NO_CONTENT)
async def resend_verification(
    data: ResendVerificationRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Resend email verification link.
    
    For security, always returns 204 even if email doesn't exist.
    """
    user = await auth_service.get_user_by_email(data.email)
    
    if user and not user.email_verified:
        # Regenerate token and send fresh verification email
        new_token = await auth_service.regenerate_verification_token(user.id)
        if new_token:
            await send_verification_email(user.email, new_token, user.name)
            # Track when verification email was sent
            user.email_verification_sent_at = datetime.now(timezone.utc)
            await auth_service.db.flush()
    
    return None


# ============== Session Management ==============

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user: User = Depends(get_required_user),
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)] = None,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    List all active sessions for current user.
    
    Requires authentication.
    """
    sessions = await auth_service.get_active_sessions(user.id)
    
    # Determine current session
    current_token_hash = None
    if credentials:
        payload = decode_token(credentials.credentials)
        if payload:
            # We can't easily identify the current session from access token
            # This would need the refresh token instead
            pass
    
    session_list = [
        SessionInfo(
            id=s.id,
            device_info=s.device_info,
            ip_address=s.ip_address,
            created_at=s.created_at,
            last_used_at=s.last_used_at,
            expires_at=s.expires_at,
            is_current=False,  # Would need refresh token to determine
        )
        for s in sessions
    ]
    
    return SessionListResponse(
        sessions=session_list,
        total=len(session_list)
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_session(
    session_id: str,
    user: User = Depends(get_required_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Revoke a specific session.
    
    Requires authentication.
    """
    success = await auth_service.revoke_session(user.id, session_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return None
