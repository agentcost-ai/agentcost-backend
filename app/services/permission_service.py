"""
Permission Service

Role-based access control for projects and resources.
Handles permission checks, role hierarchy, and access validation.
"""

from enum import Enum
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..models.user_models import User, ProjectMember, UserRole
from ..models.db_models import Project


class Permission(str, Enum):
    """Granular permissions for different actions"""
    
    # Project-level permissions
    VIEW_PROJECT = "view_project"
    EDIT_PROJECT = "edit_project"
    DELETE_PROJECT = "delete_project"
    
    # Member management
    VIEW_MEMBERS = "view_members"
    INVITE_MEMBERS = "invite_members"
    REMOVE_MEMBERS = "remove_members"
    CHANGE_ROLES = "change_roles"
    
    # Event/Data access
    VIEW_EVENTS = "view_events"
    CREATE_EVENTS = "create_events"
    DELETE_EVENTS = "delete_events"
    
    # Analytics
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"
    
    # API Key management
    VIEW_API_KEY = "view_api_key"
    REGENERATE_API_KEY = "regenerate_api_key"


# Role to permissions mapping
# Higher roles inherit all permissions from lower roles
ROLE_PERMISSIONS = {
    UserRole.VIEWER: {
        Permission.VIEW_PROJECT,
        Permission.VIEW_MEMBERS,
        Permission.VIEW_EVENTS,
        Permission.VIEW_ANALYTICS,
    },
    UserRole.MEMBER: {
        # Inherits all viewer permissions
        Permission.VIEW_PROJECT,
        Permission.VIEW_MEMBERS,
        Permission.VIEW_EVENTS,
        Permission.VIEW_ANALYTICS,
        # Plus member-specific permissions
        Permission.CREATE_EVENTS,
        Permission.EXPORT_DATA,
        Permission.VIEW_API_KEY,
    },
    UserRole.ADMIN: {
        # Full access to everything
        Permission.VIEW_PROJECT,
        Permission.EDIT_PROJECT,
        Permission.DELETE_PROJECT,
        Permission.VIEW_MEMBERS,
        Permission.INVITE_MEMBERS,
        Permission.REMOVE_MEMBERS,
        Permission.CHANGE_ROLES,
        Permission.VIEW_EVENTS,
        Permission.CREATE_EVENTS,
        Permission.DELETE_EVENTS,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_DATA,
        Permission.VIEW_API_KEY,
        Permission.REGENERATE_API_KEY,
    },
}


def get_role_permissions(role: UserRole) -> set[Permission]:
    """Get all permissions for a given role"""
    return ROLE_PERMISSIONS.get(role, set())


def role_has_permission(role: UserRole, permission: Permission) -> bool:
    """Check if a role has a specific permission"""
    return permission in get_role_permissions(role)


def parse_role(role_str: str) -> Optional[UserRole]:
    """Safely parse a role string to UserRole enum"""
    try:
        return UserRole(role_str.lower())
    except ValueError:
        return None


class PermissionService:
    """Service for checking user permissions on projects"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_user_role_in_project(
        self, 
        user_id: str, 
        project_id: str
    ) -> Optional[UserRole]:
        """
        Get the user's role in a project.
        
        Returns UserRole if user has access, None otherwise.
        Project owners automatically get ADMIN role.
        """
        # First check if user is the project owner
        project_result = await self.db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = project_result.scalar_one_or_none()
        
        if not project:
            return None
        
        # Owner always has admin access
        if project.owner_id == user_id:
            return UserRole.ADMIN
        
        # Check project membership
        member_result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == user_id,
                    ProjectMember.accepted_at.isnot(None)  # Must have accepted invite
                )
            )
        )
        member = member_result.scalar_one_or_none()
        
        if not member:
            return None
        
        return parse_role(member.role)
    
    async def check_permission(
        self,
        user_id: str,
        project_id: str,
        permission: Permission
    ) -> bool:
        """
        Check if a user has a specific permission on a project.
        
        Returns True if user has the permission, False otherwise.
        """
        role = await self.get_user_role_in_project(user_id, project_id)
        
        if not role:
            return False
        
        return role_has_permission(role, permission)
    
    async def require_permission(
        self,
        user_id: str,
        project_id: str,
        permission: Permission
    ) -> UserRole:
        """
        Check permission and raise exception if not allowed.
        
        Returns the user's role if permission is granted.
        Raises PermissionError if denied.
        """
        role = await self.get_user_role_in_project(user_id, project_id)
        
        if not role:
            raise PermissionError("You don't have access to this project")
        
        if not role_has_permission(role, permission):
            raise PermissionError(
                f"Your role ({role.value}) doesn't have permission to perform this action"
            )
        
        return role
    
    async def get_user_projects(
        self,
        user_id: str,
        include_pending: bool = False
    ) -> List[dict]:
        """
        Get all projects a user has access to.
        
        Returns list of dicts with project info and user's role.
        """
        projects = []
        
        # Get owned projects
        owned_result = await self.db.execute(
            select(Project).where(Project.owner_id == user_id)
        )
        for project in owned_result.scalars():
            projects.append({
                "project": project,
                "role": UserRole.ADMIN,
                "is_owner": True,
                "is_pending": False,
            })
        
        # Get projects where user is a member
        if include_pending:
            member_query = select(ProjectMember).where(
                ProjectMember.user_id == user_id
            )
        else:
            member_query = select(ProjectMember).where(
                and_(
                    ProjectMember.user_id == user_id,
                    ProjectMember.accepted_at.isnot(None)
                )
            )
        
        member_result = await self.db.execute(member_query)
        
        for membership in member_result.scalars():
            # Skip if already in list (owner)
            project_ids = [p["project"].id for p in projects]
            if membership.project_id in project_ids:
                continue
            
            # Get the project
            project_result = await self.db.execute(
                select(Project).where(Project.id == membership.project_id)
            )
            project = project_result.scalar_one_or_none()
            
            if project:
                projects.append({
                    "project": project,
                    "role": parse_role(membership.role),
                    "is_owner": False,
                    "is_pending": membership.accepted_at is None,
                })
        
        return projects
    
    async def get_project_members(
        self,
        project_id: str,
        include_pending: bool = True
    ) -> List[dict]:
        """
        Get all members of a project with their roles.
        """
        members = []
        
        # Get the project and owner
        project_result = await self.db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = project_result.scalar_one_or_none()
        
        if not project:
            return []
        
        # Add owner if exists
        if project.owner_id:
            owner_result = await self.db.execute(
                select(User).where(User.id == project.owner_id)
            )
            owner = owner_result.scalar_one_or_none()
            if owner:
                members.append({
                    "user": owner,
                    "role": UserRole.ADMIN,
                    "is_owner": True,
                    "is_pending": False,
                    "membership_id": None,
                })
        
        # Get all memberships
        if include_pending:
            member_query = select(ProjectMember).where(
                ProjectMember.project_id == project_id
            )
        else:
            member_query = select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.accepted_at.isnot(None)
                )
            )
        
        member_result = await self.db.execute(member_query)
        
        for membership in member_result.scalars():
            # Skip owner (already added)
            if membership.user_id == project.owner_id:
                continue
            
            # Get user
            user_result = await self.db.execute(
                select(User).where(User.id == membership.user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if user:
                members.append({
                    "user": user,
                    "role": parse_role(membership.role),
                    "is_owner": False,
                    "is_pending": membership.accepted_at is None,
                    "membership_id": membership.id,
                    "invited_at": membership.invited_at,
                    "accepted_at": membership.accepted_at,
                })
        
        return members
    
    async def can_modify_member(
        self,
        actor_id: str,
        target_user_id: str,
        project_id: str
    ) -> bool:
        """
        Check if actor can modify target member in a project.
        
        Rules:
        - Owner can modify anyone except themselves
        - Admin can modify members and viewers, but not other admins
        - Members and viewers cannot modify anyone
        """
        actor_role = await self.get_user_role_in_project(actor_id, project_id)
        
        # Get target membership to determine their role (even if pending)
        target_membership_result = await self.db.execute(
            select(ProjectMember).where(
                and_(
                    ProjectMember.project_id == project_id,
                    ProjectMember.user_id == target_user_id
                )
            )
        )
        target_membership = target_membership_result.scalar_one_or_none()
        
        if target_membership:
            target_role = parse_role(target_membership.role)
        else:
            target_role = None
        
        if not actor_role or not target_role:
            return False
        
        # Cannot modify yourself
        if actor_id == target_user_id:
            return False
        
        # Get project to check ownership
        project_result = await self.db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = project_result.scalar_one_or_none()
        
        if not project:
            return False
        
        # Owner can modify anyone except themselves
        if project.owner_id == actor_id:
            return True
        
        # Admin can modify non-admins
        if actor_role == UserRole.ADMIN and target_role != UserRole.ADMIN:
            return True
        
        return False
