"""
Project Members API Routes

Endpoints for managing project team members: invitations, roles, removals.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime

from ..database import get_db
from ..models.user_models import User, UserRole
from ..services.auth_service import get_current_user
from ..services.member_service import MemberService
from ..services.permission_service import PermissionService, Permission
from ..services.email_service import send_invitation_email, send_new_user_invitation_email


router = APIRouter(prefix="/v1/projects", tags=["Project Members"])
security = HTTPBearer(auto_error=False)


# Request/Response models

class InviteMemberRequest(BaseModel):
    email: EmailStr = Field(..., description="Email of user to invite")
    role: str = Field(default="member", description="Role: admin, member, or viewer")


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., description="New role: admin, member, or viewer")


class MemberResponse(BaseModel):
    id: str
    user_id: str
    email: str
    name: Optional[str]
    role: str
    is_owner: bool
    is_pending: bool
    invited_at: Optional[datetime]
    accepted_at: Optional[datetime]


class MembersListResponse(BaseModel):
    members: List[MemberResponse]
    total: int


class InvitationResponse(BaseModel):
    project_id: str
    project_name: str
    role: str
    invited_by: Optional[dict]
    invited_at: datetime


class InvitationsListResponse(BaseModel):
    invitations: List[InvitationResponse]
    total: int


# Dependencies

async def get_required_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current user, raise 401 if not authenticated"""
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


# Routes

@router.get("/{project_id}/members", response_model=MembersListResponse)
async def list_members(
    project_id: str,
    include_pending: bool = True,
    user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all members of a project.
    
    Requires VIEW_MEMBERS permission (all roles have this).
    """
    permission_service = PermissionService(db)
    
    # Check user has access to view members
    try:
        await permission_service.require_permission(
            user.id, project_id, Permission.VIEW_MEMBERS
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    
    members_data = await permission_service.get_project_members(
        project_id, include_pending
    )
    
    members = [
        MemberResponse(
            id=m.get("membership_id") or m["user"].id,
            user_id=m["user"].id,
            email=m["user"].email,
            name=m["user"].name,
            role=m["role"].value if m["role"] else "member",
            is_owner=m["is_owner"],
            is_pending=m["is_pending"],
            invited_at=m.get("invited_at"),
            accepted_at=m.get("accepted_at"),
        )
        for m in members_data
    ]
    
    return MembersListResponse(members=members, total=len(members))


@router.post("/{project_id}/members", status_code=status.HTTP_201_CREATED)
async def invite_member(
    project_id: str,
    request: InviteMemberRequest,
    user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Invite a user to the project.
    
    Requires INVITE_MEMBERS permission (admin only).
    If the user has an account, they'll receive an invitation to accept.
    If the user doesn't have an account, they'll receive an email with instructions to register.
    """
    member_service = MemberService(db)
    
    result, error, is_new_user = await member_service.invite_member(
        project_id=project_id,
        inviter_id=user.id,
        invitee_email=request.email,
        role=request.role,
    )
    
    if error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    
    # Get project name for email
    from ..models.db_models import Project
    from sqlalchemy import select
    
    project_result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = project_result.scalar_one_or_none()
    
    # Send appropriate email based on whether user exists
    if project:
        if is_new_user:
            # User doesn't have an account - send registration invitation
            await send_new_user_invitation_email(
                email=request.email,
                project_name=project.name,
                inviter_name=user.name or user.email,
                role=request.role,
            )
            return {
                "message": f"Invitation sent to {request.email}. They will need to create an account to accept.",
                "pending_registration": True,
                "role": request.role,
            }
        else:
            # User has an account - send normal invitation
            invitee_result = await db.execute(
                select(User).where(User.id == result.user_id)
            )
            invitee = invitee_result.scalar_one_or_none()
            
            await send_invitation_email(
                email=request.email,
                project_name=project.name,
                inviter_name=user.name or user.email,
                role=request.role,
                invitee_name=invitee.name if invitee else None,
            )
            return {
                "message": f"Invitation sent to {request.email}",
                "membership_id": result.id,
                "role": result.role,
            }


@router.patch("/{project_id}/members/{user_id}")
async def update_member_role(
    project_id: str,
    user_id: str,
    request: UpdateRoleRequest,
    current_user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a member's role.
    
    Requires CHANGE_ROLES permission (admin only).
    Only project owners can promote to admin.
    """
    member_service = MemberService(db)
    
    membership, error = await member_service.update_member_role(
        project_id=project_id,
        actor_id=current_user.id,
        target_user_id=user_id,
        new_role=request.role,
    )
    
    if error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    
    return {
        "message": "Role updated successfully",
        "user_id": user_id,
        "new_role": membership.role,
    }


@router.delete("/{project_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    project_id: str,
    user_id: str,
    current_user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a member from the project.
    
    Requires REMOVE_MEMBERS permission (admin only).
    Admins cannot remove other admins (only the owner can).
    """
    member_service = MemberService(db)
    
    success, error = await member_service.remove_member(
        project_id=project_id,
        actor_id=current_user.id,
        target_user_id=user_id,
    )
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    
    return None


@router.post("/{project_id}/leave", status_code=status.HTTP_204_NO_CONTENT)
async def leave_project(
    project_id: str,
    user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Leave a project voluntarily.
    
    Project owners cannot leave - they must transfer ownership or delete the project.
    """
    member_service = MemberService(db)
    
    success, error = await member_service.leave_project(
        user_id=user.id,
        project_id=project_id,
    )
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    
    return None


# Invitation management routes

@router.get("/invitations/pending", response_model=InvitationsListResponse)
async def get_pending_invitations(
    user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all pending project invitations for the current user.
    """
    member_service = MemberService(db)
    
    invitations = await member_service.get_pending_invitations(user.id)
    
    return InvitationsListResponse(
        invitations=[
            InvitationResponse(
                project_id=inv["project_id"],
                project_name=inv["project_name"],
                role=inv["role"],
                invited_by=inv["invited_by"],
                invited_at=inv["invited_at"],
            )
            for inv in invitations
        ],
        total=len(invitations),
    )


@router.post("/{project_id}/invitations/accept")
async def accept_invitation(
    project_id: str,
    user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a project invitation.
    """
    member_service = MemberService(db)
    
    membership, error = await member_service.accept_invitation(
        user_id=user.id,
        project_id=project_id,
    )
    
    if error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    
    return {
        "message": "Invitation accepted",
        "project_id": project_id,
        "role": membership.role,
    }


@router.post("/{project_id}/invitations/decline", status_code=status.HTTP_204_NO_CONTENT)
async def decline_invitation(
    project_id: str,
    user: User = Depends(get_required_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Decline a project invitation.
    """
    member_service = MemberService(db)
    
    success, error = await member_service.decline_invitation(
        user_id=user.id,
        project_id=project_id,
    )
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    
    return None
