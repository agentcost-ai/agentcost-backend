"""
AgentCost Backend - Guardrail API Routes

Declared agent policy (permitted tools, read-only) and the compliance view
that judges observed tool usage against it. Reads follow the analytics auth
model (project API key or member JWT); mutations require EDIT_PROJECT, the
same bar as budget guardrail settings.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Literal

from ..database import get_db
from ..models.db_models import Project
from ..models.user_models import User
from ..models.schemas import (
    GuardrailUpsert,
    GuardrailResponse,
    GuardrailComplianceResponse,
    ToolAccessTagUpsert,
    ToolAccessTagResponse,
)
from ..services.guardrail_service import GuardrailService
from ..services.permission_service import PermissionService, Permission
from ..utils.auth import validate_project_access, get_required_user
from .analytics import parse_time_range

router = APIRouter(tags=["Guardrails"])


async def _require_edit(db: AsyncSession, user: User, project_id: str) -> None:
    permission_service = PermissionService(db)
    try:
        await permission_service.require_permission(
            user.id, project_id, Permission.EDIT_PROJECT
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.get("/v1/guardrails", response_model=List[GuardrailResponse])
async def list_guardrails(
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Declared guardrails for every agent in the project."""
    return await GuardrailService(db).list_guardrails(project.id)


@router.get("/v1/guardrails/compliance", response_model=GuardrailComplianceResponse)
async def guardrail_compliance(
    range: Literal["1h", "24h", "7d", "30d", "90d"] = Query(
        "7d", description="Time range: 1h, 24h, 7d, 30d, 90d"
    ),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_project_access),
):
    """Observed tool usage judged against each agent's declared guardrail.

    Coverage caveat: only calls wrapped in ``track_costs.tool(...)`` carry a
    tool name, so compliance is judged on instrumented calls — the response
    reports ``tool_tracked_calls`` against ``total_calls`` for that reason.
    """
    start_time, end_time = parse_time_range(range)
    return await GuardrailService(db).compliance(project.id, start_time, end_time)


@router.put(
    "/v1/projects/{project_id}/guardrails", response_model=GuardrailResponse
)
async def upsert_guardrail(
    project_id: str,
    payload: GuardrailUpsert,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Create or replace the guardrail for one agent."""
    await _require_edit(db, current_user, project_id)
    return await GuardrailService(db).upsert_guardrail(project_id, payload)


@router.delete(
    "/v1/projects/{project_id}/guardrails/{agent_name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_guardrail(
    project_id: str,
    agent_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Remove the guardrail for one agent; its status returns to no_guardrail."""
    await _require_edit(db, current_user, project_id)
    deleted = await GuardrailService(db).delete_guardrail(project_id, agent_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Guardrail not found.")


@router.put(
    "/v1/projects/{project_id}/guardrails/tool-tags",
    response_model=ToolAccessTagResponse,
)
async def upsert_tool_tag(
    project_id: str,
    payload: ToolAccessTagUpsert,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Tag a tool name as read or write for read-only guardrail evaluation."""
    await _require_edit(db, current_user, project_id)
    return await GuardrailService(db).upsert_tool_tag(
        project_id, payload.tool_name, payload.access
    )


@router.delete(
    "/v1/projects/{project_id}/guardrails/tool-tags/{tool_name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_tool_tag(
    project_id: str,
    tool_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Remove a tool's read/write tag; read-only checks treat it as unknown again."""
    await _require_edit(db, current_user, project_id)
    deleted = await GuardrailService(db).delete_tool_tag(project_id, tool_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tool tag not found.")
