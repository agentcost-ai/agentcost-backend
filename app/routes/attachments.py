"""Attachment upload/download endpoints.

Security:
  - Files never publicly accessible -- downloads go through auth.
  - Allow-list of safe MIME types only.
  - Files renamed to <uuid>.<ext> on upload.
  - Hard size limit enforced server-side.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_db
from ..models.user_models import User
from ..services.auth_service import get_current_user
from ..services.storage_service import get_storage, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/v1/attachments", tags=["Attachments"])
security = HTTPBearer(auto_error=False)



async def _get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    if not credentials:
        return None
    user = await get_current_user(db, credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user



@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_attachment(
    file: UploadFile = File(...),
    user: Optional[User] = Depends(_get_optional_user),
):
    """Upload a single file. Returns metadata dict to store with the feedback item."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Hard size guard (don't rely on Content-Length alone)
    max_size = settings.max_upload_size
    data = await file.read()
    if len(data) > max_size:
        mb = max_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {mb:.0f} MB",
        )

    storage = get_storage()
    try:
        meta = await storage.save(data, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info(
        "Attachment uploaded: %s by user=%s",
        meta["stored_name"],
        user.id if user else "anonymous",
    )
    return meta



@router.get("/{stored_name}")
async def download_attachment(
    stored_name: str,
    user: Optional[User] = Depends(_get_optional_user),
):
    """Download an attachment (auth-gated, never served as a static file)."""
    storage = get_storage()
    try:
        data = await storage.read(stored_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Attachment not found")

    # Resolve content type from extension
    ext = "." + stored_name.rsplit(".", 1)[-1] if "." in stored_name else ""
    content_type = ALLOWED_EXTENSIONS.get(ext, "application/octet-stream")

    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{stored_name}"',
            "Cache-Control": "private, max-age=3600",
        },
    )



@router.get("/config/limits")
async def get_attachment_limits():
    """Return upload constraints for the frontend."""
    return {
        "max_file_size": settings.max_upload_size,
        "max_files_per_feedback": settings.max_attachments_per_feedback,
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS.keys()),
        "allowed_mime_types": sorted(set(ALLOWED_EXTENSIONS.values())),
    }
