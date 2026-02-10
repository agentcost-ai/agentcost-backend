"""
Admin routes -- auth verification.
"""

from fastapi import APIRouter, Depends

from ...models.user_models import User
from ._deps import require_superuser

router = APIRouter()


@router.get("/auth/verify")
async def verify_admin(admin: User = Depends(require_superuser)):
    """Verify that the current token belongs to a superuser."""
    return {
        "id": admin.id,
        "email": admin.email,
        "name": admin.name,
        "is_superuser": True,
    }
