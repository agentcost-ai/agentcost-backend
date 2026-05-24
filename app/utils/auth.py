"""
AgentCost Backend - Authentication Utilities

API key validation and JWT token handling.

Shared get_required_user / get_optional_user dependencies live here
to eliminate duplication across route modules.
"""

import hashlib
import secrets
from fastapi import HTTPException, Security, Depends, Query, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Tuple

from ..database import get_db
from ..services.event_service import ProjectService
from ..services.auth_service import get_current_user
from ..services.permission_service import PermissionService, Permission
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


async def _resolve_project_access(
    project_id: Optional[str],
    api_key: Optional[str],
    db: AsyncSession,
) -> Project:
    """Shared dual-auth resolution logic — see validate_project_access."""
    # Path 1: SDK API key. sk_ prefix lets us distinguish reliably from JWTs.
    if api_key and api_key.startswith("sk_"):
        project_service = ProjectService(db)
        project = await project_service.get_by_api_key(api_key)
        if not project:
            raise HTTPException(status_code=401, detail="Invalid API key.")
        if not project.is_active:
            raise HTTPException(status_code=403, detail="Project is disabled.")
        # When the route also has a path-scoped project_id, ensure it matches.
        if project_id and project_id != project.id:
            raise HTTPException(
                status_code=403,
                detail="API key does not match the requested project.",
            )
        return project

    # Path 2: JWT + project_id. The ``api_key`` variable carries the JWT in
    # this case (Authorization header is shared between the two paths).
    if api_key and project_id:
        user = await get_current_user(db, api_key)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        permission_service = PermissionService(db)
        try:
            await permission_service.require_permission(
                user.id, project_id, Permission.VIEW_PROJECT
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))

        project_service = ProjectService(db)
        project = await project_service.get_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")
        if not project.is_active:
            raise HTTPException(status_code=403, detail="Project is disabled.")
        return project

    # Neither auth path succeeded.
    if api_key and not project_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "When authenticating with a session token, the project_id "
                "query parameter is required."
            ),
        )
    raise HTTPException(
        status_code=401,
        detail=(
            "Authentication required: provide either an SDK API key "
            "(Authorization: Bearer sk_…) or a session token plus project_id."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


async def validate_project_access(
    project_id: Optional[str] = Query(
        None,
        description=(
            "Project ID. Required for JWT-authenticated requests; ignored "
            "for SDK API-key requests (the project is derived from the key)."
        ),
    ),
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
) -> Project:
    """
    Dual-auth dependency for project-scoped read endpoints (no path id).

    Use this on routes like ``/v1/analytics/overview`` where ``project_id``
    arrives as a query parameter. For routes that already have ``project_id``
    in their path (``/v1/projects/{project_id}``), use
    :func:`validate_project_access_for_path_id` instead to avoid FastAPI's
    "path vs query" parameter conflict.

    Resolution rules:

    1. **API-key path** — if the Authorization header carries an ``sk_`` API
       key, validate it and return the project it belongs to. The query
       ``project_id`` is ignored except as a safety check.
    2. **JWT path** — if the Authorization header carries a JWT,
       ``project_id`` must be provided and the caller must have
       ``VIEW_PROJECT`` permission on that project.
    """
    return await _resolve_project_access(project_id, api_key, db)


async def validate_project_access_for_path_id(
    project_id: str,  # resolved from the route's path parameter
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
) -> Project:
    """
    Dual-auth variant for routes whose URL already contains ``project_id``
    as a path parameter (``/v1/projects/{project_id}``). Same authentication
    rules as :func:`validate_project_access`; project_id is read from the
    path instead of a query parameter.
    """
    return await _resolve_project_access(project_id, api_key, db)


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
