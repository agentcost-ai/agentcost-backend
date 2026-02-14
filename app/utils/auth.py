"""
AgentCost Backend - Authentication Utilities

API key validation and JWT token handling.

Shared get_required_user / get_optional_user dependencies live here
to eliminate duplication across route modules.
"""

import hashlib
import secrets
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Tuple

from ..database import get_db
from ..services.event_service import ProjectService
from ..services.auth_service import get_current_user
from ..models.db_models import Project
from ..models.user_models import User


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using SHA256.
    
    This is used for secure storage - never store plaintext API keys.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_secure_api_key() -> Tuple[str, str]:
    """
    Generate a new secure API key.
    
    Returns:
        Tuple of (plaintext_key, hashed_key)
        - plaintext_key: Show to user ONCE (sk_...)
        - hashed_key: Store in database
    """
    # Generate 32 random bytes = 64 hex chars for strong entropy
    random_part = secrets.token_hex(32)
    plaintext_key = f"sk_{random_part}"
    hashed_key = hash_api_key(plaintext_key)
    return plaintext_key, hashed_key


def verify_api_key(plaintext_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against its stored hash.
    
    Uses constant-time comparison to prevent timing attacks.
    """
    computed_hash = hash_api_key(plaintext_key)
    return secrets.compare_digest(computed_hash, stored_hash)

# API Key in header
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Bearer token
bearer_scheme = HTTPBearer(auto_error=False)


async def get_api_key(
    authorization: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """Extract API key from Authorization header"""
    if not authorization:
        return None
    
    # Handle "Bearer sk_xxx" format
    if authorization.startswith("Bearer "):
        return authorization[7:]
    
    return authorization


async def validate_api_key(
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
) -> Project:
    """
    Validate API key and return associated project.
    
    Raises HTTPException if invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'Authorization: Bearer sk_xxx' header.",
        )
    
    project_service = ProjectService(db)
    project = await project_service.get_by_api_key(api_key)
    
    if not project:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )
    
    if not project.is_active:
        raise HTTPException(
            status_code=403,
            detail="Project is disabled.",
        )
    
    return project


async def optional_api_key(
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
) -> Optional[Project]:
    """
    Optional API key validation.
    Returns None if no API key provided.
    """
    if not api_key:
        return None
    
    project_service = ProjectService(db)
    return await project_service.get_by_api_key(api_key)


# Shared JWT user dependencies

async def get_required_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get the current authenticated user or raise 401."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_current_user(db, credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None

    try:
        user = await get_current_user(db, credentials.credentials)
    except Exception:
        # Token was provided but is invalid/expired — treat as unauthenticated
        return None
    return user  # May be None if user no longer exists
