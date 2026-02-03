"""
Member Service

Handles project membership operations: invitations, role changes, removals.
"""

from datetime import datetime, timezone
from typing import Optional, List, Tuple, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, delete
from sqlalchemy.exc import IntegrityError

from ..models.user_models import User, ProjectMember, UserRole, PendingEmailInvitation
from ..models.db_models import Project
from .permission_service import PermissionService, Permission


class MemberService:
    """Service for managing project members"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.permissions = PermissionService(db)
    
    async def invite_member(
        self,
        project_id: str,
        inviter_id: str,
        invitee_email: str,
        role: str = "member"
    ) -> Tuple[Optional[Union[ProjectMember, PendingEmailInvitation]], Optional[str], bool]:
        """
        Invite a user to a project.
        
        Returns:
            Tuple of (membership_or_pending, error_message, is_new_user)
            - If successful: (membership/pending_invite, None, is_new_user)
            - If failed: (None, error_message, False)
        """
        # Validate role
        try:
            parsed_role = UserRole(role.lower())
        except ValueError:
            return None, f"Invalid role: {role}. Must be admin, member, or viewer.", False
        
        # Check inviter has permission
        try:
            await self.permissions.require_permission(
                inviter_id, project_id, Permission.INVITE_MEMBERS
            )
        except PermissionError as e:
            return None, str(e), False
        
        # Get inviter's email to prevent self-invitation
        inviter_result = await self.db.execute(
            select(User).where(User.id == inviter_id)
        )
        inviter = inviter_result.scalar_one_or_none()
        
        if inviter and inviter.email.lower() == invitee_email.lower().strip():
            return None, "You cannot invite yourself to the project", False
        
        # Find the user by email
        user_result = await self.db.execute(
            select(User).where(User.email == invitee_email.lower().strip())
        )
        invitee = user_result.scalar_one_or_none()
        
        # Get project for checks
        project_result = await self.db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = project_result.scalar_one_or_none()
        
        if not project:
            return None, "Project not found", False
        
        # Cannot invite admins unless you're the owner
        if parsed_role == UserRole.ADMIN:
            if project.owner_id != inviter_id:
                return None, "Only the project owner can invite admins", False
        
        if not invitee:
            # User doesn't exist - create pending invitation
            return await self._create_pending_invitation(
                project_id=project_id,
                email=invitee_email.lower().strip(),
                role=parsed_role.value,
                inviter_id=inviter_id,
            )
        
        # User exists - proceed with normal invitation flow
        
        # Check if user is already a member
        existing_result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == invitee.id
                )
            )
        )
        existing = existing_result.scalar_one_or_none()
        
        if existing:
            if existing.accepted_at:
                return None, "User is already a member of this project", False
            else:
                return None, "User already has a pending invitation", False
        
        # Check if user is the owner
        if project.owner_id == invitee.id:
            return None, "Cannot invite the project owner", False
        
        # Create the membership
        membership = ProjectMember(
            project_id=project_id,
            user_id=invitee.id,
            role=parsed_role.value,
            invited_by_id=inviter_id,
        )
        
        try:
            self.db.add(membership)
            await self.db.commit()
            await self.db.refresh(membership)
            return membership, None, False  # Not a new user
        except IntegrityError:
            await self.db.rollback()
            return None, "Failed to create invitation", False
    
    async def _create_pending_invitation(
        self,
        project_id: str,
        email: str,
        role: str,
        inviter_id: str,
    ) -> Tuple[Optional[PendingEmailInvitation], Optional[str], bool]:
        """
        Create a pending invitation for a user who hasn't registered yet.
        """
        # Check if already invited
        existing_result = await self.db.execute(
            select(PendingEmailInvitation).where(
                and_(
                    PendingEmailInvitation.project_id == project_id,
                    PendingEmailInvitation.email == email
                )
            )
        )
        existing = existing_result.scalar_one_or_none()
        
        if existing:
            return None, "This email already has a pending invitation", False
        
        # Create pending invitation
        pending = PendingEmailInvitation(
            email=email,
            project_id=project_id,
            role=role,
            invited_by_id=inviter_id,
        )
        
        try:
            self.db.add(pending)
            await self.db.commit()
            await self.db.refresh(pending)
            return pending, None, True  # Is a new user (not registered)
        except IntegrityError:
            await self.db.rollback()
            return None, "Failed to create invitation", False
    
    async def process_pending_invitations_for_user(
        self,
        user: User
    ) -> int:
        """
        Convert pending email invitations to actual memberships when a user registers.
        Called during user registration.
        
        Returns the number of invitations processed.
        """
        result = await self.db.execute(
            select(PendingEmailInvitation).where(
                PendingEmailInvitation.email == user.email.lower()
            )
        )
        pending_invitations = result.scalars().all()
        
        if not pending_invitations:
            return 0
        
        processed = 0
        memberships_to_add = []
        invitations_to_delete = []
        
        for pending in pending_invitations:
            # Check if already a member (avoid IntegrityError)
            existing = await self.db.execute(
                select(ProjectMember).where(
                    and_(
                        ProjectMember.project_id == pending.project_id,
                        ProjectMember.user_id == user.id
                    )
                )
            )
            if existing.scalar_one_or_none():
                invitations_to_delete.append(pending)
                continue
            
            membership = ProjectMember(
                project_id=pending.project_id,
                user_id=user.id,
                role=pending.role,
                invited_by_id=pending.invited_by_id,
                accepted_at=None,
            )
            memberships_to_add.append(membership)
            invitations_to_delete.append(pending)
            processed += 1
        
        # Batch add and delete
        for membership in memberships_to_add:
            self.db.add(membership)
        for pending in invitations_to_delete:
            await self.db.delete(pending)
        
        if memberships_to_add or invitations_to_delete:
            await self.db.commit()
        
        return processed
    
    async def accept_invitation(
        self,
        user_id: str,
        project_id: str
    ) -> Tuple[Optional[ProjectMember], Optional[str]]:
        """
        Accept a pending project invitation.
        
        Returns:
            Tuple of (membership, error_message)
        """
        # Find the pending invitation
        result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == user_id,
                    ProjectMember.accepted_at.is_(None)
                )
            )
        )
        membership = result.scalar_one_or_none()
        
        if not membership:
            return None, "No pending invitation found"
        
        membership.accepted_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(membership)
        
        return membership, None
    
    async def decline_invitation(
        self,
        user_id: str,
        project_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Decline a pending project invitation.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Find and delete the pending invitation
        result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == user_id,
                    ProjectMember.accepted_at.is_(None)
                )
            )
        )
        membership = result.scalar_one_or_none()
        
        if not membership:
            return False, "No pending invitation found"
        
        await self.db.delete(membership)
        await self.db.commit()
        
        return True, None
    
    async def update_member_role(
        self,
        project_id: str,
        actor_id: str,
        target_user_id: str,
        new_role: str
    ) -> Tuple[Optional[ProjectMember], Optional[str]]:
        """
        Update a member's role in a project.
        
        Returns:
            Tuple of (membership, error_message)
        """
        # Validate role
        try:
            parsed_role = UserRole(new_role.lower())
        except ValueError:
            return None, f"Invalid role: {new_role}. Must be admin, member, or viewer."
        
        # Check actor has permission to change roles
        try:
            await self.permissions.require_permission(
                actor_id, project_id, Permission.CHANGE_ROLES
            )
        except PermissionError as e:
            return None, str(e)
        
        # Check if actor can modify this target
        can_modify = await self.permissions.can_modify_member(
            actor_id, target_user_id, project_id
        )
        
        if not can_modify:
            return None, "You cannot modify this member's role"
        
        # Cannot promote to admin unless you're the owner
        project_result = await self.db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = project_result.scalar_one_or_none()
        
        if parsed_role == UserRole.ADMIN:
            if not project or project.owner_id != actor_id:
                return None, "Only the project owner can promote members to admin"
        
        # Find the membership
        result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == target_user_id
                )
            )
        )
        membership = result.scalar_one_or_none()
        
        if not membership:
            return None, "User is not a member of this project"
        
        membership.role = parsed_role.value
        await self.db.commit()
        await self.db.refresh(membership)
        
        return membership, None
    
    async def remove_member(
        self,
        project_id: str,
        actor_id: str,
        target_user_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Remove a member from a project.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Check actor has permission
        try:
            await self.permissions.require_permission(
                actor_id, project_id, Permission.REMOVE_MEMBERS
            )
        except PermissionError as e:
            return False, str(e)
        
        # Check if actor can modify this target
        can_modify = await self.permissions.can_modify_member(
            actor_id, target_user_id, project_id
        )
        
        if not can_modify:
            return False, "You cannot remove this member"
        
        # Find and delete the membership
        result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == target_user_id
                )
            )
        )
        membership = result.scalar_one_or_none()
        
        if not membership:
            return False, "User is not a member of this project"
        
        # Get the user's email to also delete any pending email invitations
        user_result = await self.db.execute(
            select(User).where(User.id == target_user_id)
        )
        target_user = user_result.scalar_one_or_none()
        
        await self.db.delete(membership)
        
        # Also delete any pending email invitations for this user in this project
        if target_user:
            pending_result = await self.db.execute(
                select(PendingEmailInvitation).where(
                    and_(
                        PendingEmailInvitation.project_id == project_id,
                        PendingEmailInvitation.email == target_user.email.lower()
                    )
                )
            )
            pending_invitations = pending_result.scalars().all()
            for pending in pending_invitations:
                await self.db.delete(pending)
        
        await self.db.commit()
        
        return True, None
    
    async def leave_project(
        self,
        user_id: str,
        project_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Leave a project voluntarily.
        Owners cannot leave their own projects.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Check if user is the owner
        project_result = await self.db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = project_result.scalar_one_or_none()
        
        if not project:
            return False, "Project not found"
        
        if project.owner_id == user_id:
            return False, "Project owners cannot leave their own projects. Transfer ownership or delete the project instead."
        
        # Find and delete the membership
        result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == user_id
                )
            )
        )
        membership = result.scalar_one_or_none()
        
        if not membership:
            return False, "You are not a member of this project"
        
        await self.db.delete(membership)
        await self.db.commit()
        
        return True, None
    
    async def get_pending_invitations(
        self,
        user_id: str
    ) -> List[dict]:
        """
        Get all pending invitations for a user.
        """
        result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.user_id == user_id,
                    ProjectMember.accepted_at.is_(None)
                )
            )
        )
        
        invitations = []
        for membership in result.scalars():
            # Get project info
            project_result = await self.db.execute(
                select(Project).where(Project.id == membership.project_id)
            )
            project = project_result.scalar_one_or_none()
            
            # Get inviter info
            inviter = None
            if membership.invited_by_id:
                inviter_result = await self.db.execute(
                    select(User).where(User.id == membership.invited_by_id)
                )
                inviter = inviter_result.scalar_one_or_none()
            
            if project:
                invitations.append({
                    "project_id": project.id,
                    "project_name": project.name,
                    "role": membership.role,
                    "invited_by": {
                        "id": inviter.id,
                        "email": inviter.email,
                        "name": inviter.name,
                    } if inviter else None,
                    "invited_at": membership.invited_at,
                })
        
        return invitations
