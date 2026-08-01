"""
AgentCost Backend - Projects API Routes

Endpoints for project management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.schemas import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ProjectBudgetUpdate,
    ProjectBudgetResponse,
)
from ..models.db_models import Project
from ..models.user_models import User
from ..services.event_service import ProjectService
from ..services.budget_service import BudgetService
from ..services.permission_service import PermissionService, Permission
from ..utils.auth import (
    validate_api_key,
    get_required_user,
    validate_project_access_for_path_id,
)

router = APIRouter(prefix="/v1/projects", tags=["Projects"])

# Optional bearer token for project creation
bearer_scheme = HTTPBearer(auto_error=False)


@router.get("", response_model=list[dict])
async def list_my_projects(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """
    List every project the authenticated user can access, including projects
    they own and projects they were invited to (pending and accepted).

    Used by the dashboard's project switcher so invited members can find and
    open the projects they have access to without ever needing the project's
    raw API key.
    """
    permission_service = PermissionService(db)
    accessible = await permission_service.get_user_projects(
        current_user.id, include_pending=True
    )

    items: list[dict] = []
    for entry in accessible:
        project: Project = entry["project"]
        items.append(
            {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "is_active": project.is_active,
                "role": entry["role"].value if entry["role"] else "viewer",
                "is_owner": entry["is_owner"],
                "is_pending": entry["is_pending"],
                "created_at": project.created_at.isoformat() if project.created_at else None,
            }
        )

    # Stable ordering: active first, then owned first, then by name
    items.sort(
        key=lambda p: (
            not p["is_active"],
            not p["is_owner"],
            (p["name"] or "").lower(),
        )
    )
    return items


@router.post("")
async def create_project(
    request: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """
    Create a new project.
    
    Requires authentication. The project will be linked to the authenticated user.
    
    Returns the project with its API key.
    
    IMPORTANT: Store the API key securely - it will NEVER be shown again!
    The API key is hashed before storage for security.
    """
    project_service = ProjectService(db)
    owner_id = current_user.id
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
        "key_prefix": plaintext_api_key[:8] if plaintext_api_key else None,
        "is_active": project.is_active,
        "monthly_budget_usd": project.monthly_budget_usd,
        "budget_enforcement_mode": project.budget_enforcement_mode,
        "budget_alert_thresholds": project.budget_alert_thresholds,
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
        "api_key": None,
        "key_prefix": None,
        "is_active": project.is_active,
        "monthly_budget_usd": project.monthly_budget_usd,
        "budget_enforcement_mode": project.budget_enforcement_mode,
        "budget_alert_thresholds": project.budget_alert_thresholds,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    }


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    auth_project: Project = Depends(validate_project_access_for_path_id),
):
    """
    Get project by ID.

    Dual-auth: accepts either an SDK API key (returned project must match
    ``project_id``) or a JWT + ``project_id`` query param that the caller
    has VIEW_PROJECT permission on.
    """
    # validate_project_access already enforces that the resolved project
    # matches the requested id (API-key path) or that the JWT user has
    # permission on the requested project (JWT path). Guard against the
    # unlikely API-key/path mismatch anyway.
    if auth_project.id != project_id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this project.",
        )

    return {
        "id": auth_project.id,
        "name": auth_project.name,
        "description": auth_project.description,
        "api_key": None,
        "key_prefix": None,
        "is_active": auth_project.is_active,
        "monthly_budget_usd": auth_project.monthly_budget_usd,
        "budget_enforcement_mode": auth_project.budget_enforcement_mode,
        "budget_alert_thresholds": auth_project.budget_alert_thresholds,
        "created_at": auth_project.created_at,
    }


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    request: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """
    Update project settings.

    Requires JWT authentication and admin/owner role on the project.
    """
    permission_service = PermissionService(db)
    try:
        await permission_service.require_permission(
            current_user.id, project_id, Permission.EDIT_PROJECT
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    project_service = ProjectService(db)
    project = await project_service.update(
        project_id=project_id,
        name=request.name,
        description=request.description,
        is_active=request.is_active,
    )
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "api_key": None,
        "key_prefix": None,
        "is_active": project.is_active,
        "monthly_budget_usd": project.monthly_budget_usd,
        "budget_enforcement_mode": project.budget_enforcement_mode,
        "budget_alert_thresholds": project.budget_alert_thresholds,
        "created_at": project.created_at,
    }


@router.get("/{project_id}/budget", response_model=ProjectBudgetResponse)
async def get_project_budget(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """
    Get budget settings and current month utilization for a project.
    """
    permission_service = PermissionService(db)
    try:
        await permission_service.require_permission(
            current_user.id, project_id, Permission.VIEW_PROJECT
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    project_service = ProjectService(db)
    project = await project_service.get_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    budget_service = BudgetService(db)
    evaluation = await budget_service.evaluate(project)

    return ProjectBudgetResponse(
        project_id=project.id,
        monthly_budget_usd=project.monthly_budget_usd,
        budget_enforcement_mode=(project.budget_enforcement_mode or "off"),
        budget_alert_thresholds=budget_service.normalize_thresholds(
            project.budget_alert_thresholds
        ),
        current_month_spend=float(evaluation.get("current_spend") or 0.0),
        current_month_spend_usd=float(evaluation.get("current_spend_usd") or 0.0),
        utilization_percent=evaluation.get("utilization_percent"),
        period_key=evaluation.get("period_key"),
        budget_currency=evaluation.get("currency") or "USD",
        fx_rate=float(evaluation.get("fx_rate") or 1.0),
    )


@router.put("/{project_id}/budget", response_model=ProjectBudgetResponse)
async def update_project_budget(
    project_id: str,
    request: ProjectBudgetUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """
    Update project budget guardrail settings.
    """
    permission_service = PermissionService(db)
    try:
        await permission_service.require_permission(
            current_user.id, project_id, Permission.EDIT_PROJECT
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    project_service = ProjectService(db)
    project = await project_service.get_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    # Treat zero budget as disabled to avoid accidental hard lockouts.
    project.monthly_budget_usd = (
        None if request.monthly_budget_usd in (None, 0) else request.monthly_budget_usd
    )
    project.budget_enforcement_mode = request.budget_enforcement_mode
    project.budget_alert_thresholds = request.budget_alert_thresholds
    project.budget_currency = request.budget_currency
    await db.flush()

    budget_service = BudgetService(db)
    evaluation = await budget_service.evaluate(project)

    return ProjectBudgetResponse(
        project_id=project.id,
        monthly_budget_usd=project.monthly_budget_usd,
        budget_enforcement_mode=(project.budget_enforcement_mode or "off"),
        budget_alert_thresholds=budget_service.normalize_thresholds(
            project.budget_alert_thresholds
        ),
        current_month_spend=float(evaluation.get("current_spend") or 0.0),
        current_month_spend_usd=float(evaluation.get("current_spend_usd") or 0.0),
        utilization_percent=evaluation.get("utilization_percent"),
        period_key=evaluation.get("period_key"),
        budget_currency=evaluation.get("currency") or "USD",
        fx_rate=float(evaluation.get("fx_rate") or 1.0),
    )


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """
    Delete a project.

    Requires JWT authentication and admin/owner role on the project.
    WARNING: This will delete all associated events!
    """
    permission_service = PermissionService(db)
    try:
        await permission_service.require_permission(
            current_user.id, project_id, Permission.DELETE_PROJECT
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    project_service = ProjectService(db)
    success = await project_service.delete(project_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    return {"status": "deleted"}


@router.post("/{project_id}/api-key/rotate")
async def rotate_api_key(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_required_user),
):
    """
    Rotate the project's API key.

    Returns the new API key ONCE. Requires regenerate_api_key permission.
    """
    permission_service = PermissionService(db)
    try:
        await permission_service.require_permission(
            user.id, project_id, Permission.REGENERATE_API_KEY
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    project_service = ProjectService(db)
    result = await project_service.regenerate_api_key(project_id)
    if not result:
        raise HTTPException(status_code=404, detail="Project not found.")

    project, plaintext_key = result
    await db.commit()

    return {
        "status": "ok",
        "project_id": project.id,
        "api_key": plaintext_key,
        "key_prefix": plaintext_key[:8] if plaintext_key else None,
        "message": "Save this API key now. It cannot be retrieved later.",
    }
