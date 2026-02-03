"""
AgentCost Backend - Projects API Routes

Endpoints for project management.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_db
from ..models.schemas import ProjectCreate, ProjectResponse, ProjectUpdate
from ..models.db_models import Project
from ..models.user_models import User
from ..services.event_service import ProjectService
from ..services.auth_service import get_current_user
from ..utils.auth import validate_api_key, optional_api_key

router = APIRouter(prefix="/v1/projects", tags=["Projects"])

# Optional bearer token for project creation
bearer_scheme = HTTPBearer(auto_error=False)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Get user from JWT token if provided, otherwise None"""
    if not credentials:
        return None
    try:
        return await get_current_user(db, credentials.credentials)
    except Exception:
        return None


@router.post("")
async def create_project(
    request: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Create a new project.
    
    If authenticated with JWT, the project will be linked to the user.
    
    Returns the project with its API key.
    
    IMPORTANT: Store the API key securely - it will NEVER be shown again!
    The API key is hashed before storage for security.
    """
    project_service = ProjectService(db)
    owner_id = current_user.id if current_user else None
    project, plaintext_api_key = await project_service.create(
        name=request.name,
        description=request.description,
        owner_id=owner_id,
    )
    
    # Return project with the plaintext API key (only time it's shown)
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "api_key": plaintext_api_key,  # Show ONCE, then never again
        "is_active": project.is_active,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
        "owner_id": owner_id,
        "warning": "Save this API key now! It cannot be retrieved later."
    }


@router.get("/me")
async def get_current_project(
    project: Project = Depends(validate_api_key),
):
    """
    Get the current project (based on API key).
    
    Note: API key is not returned for security reasons.
    """
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "api_key": "sk_***" + project.api_key[-8:] if len(project.api_key) > 8 else "sk_***",  # Masked
        "is_active": project.is_active,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    }


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    auth_project: Project = Depends(validate_api_key),
):
    """
    Get project by ID.
    
    Only returns project if API key matches.
    """
    if project_id != auth_project.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this project.",
        )
    
    return auth_project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    request: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
    auth_project: Project = Depends(validate_api_key),
):
    """
    Update project settings.
    """
    if project_id != auth_project.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to update this project.",
        )
    
    project_service = ProjectService(db)
    project = await project_service.update(
        project_id=project_id,
        name=request.name,
        description=request.description,
        is_active=request.is_active,
    )
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    return project


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    auth_project: Project = Depends(validate_api_key),
):
    """
    Delete a project.
    
    WARNING: This will delete all associated events!
    """
    if project_id != auth_project.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to delete this project.",
        )
    
    project_service = ProjectService(db)
    success = await project_service.delete(project_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    return {"status": "deleted"}
