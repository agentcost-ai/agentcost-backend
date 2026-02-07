"""AgentCost - File Storage Service

Abstracts away the backing store so the app can run on local disk today
and swap to S3 / R2 later without touching the schema or API layer.

Rules:
- Files are never publicly accessible; downloads go through an auth endpoint.
- Every file is renamed to <uuid>.<ext> on ingest.
- Only an allow-list of safe MIME types is accepted.
- The DB stores lightweight JSON metadata, never binary data.
"""

from __future__ import annotations

import logging
import mimetypes
import uuid
from pathlib import Path

import aiofiles

from ..config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Safe MIME types keyed by extension
ALLOWED_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".json": "application/json",
    ".csv": "text/csv",
}

# Reject outright regardless of MIME sniffing
BLOCKED_EXTENSIONS = {
    ".exe", ".sh", ".bat", ".cmd", ".ps1", ".js", ".py", ".rb",
    ".php", ".html", ".htm", ".svg", ".msi", ".dll", ".so",
}


def _sanitise_extension(original_name: str) -> str:
    """Return a lower-cased extension from the original filename."""
    ext = Path(original_name).suffix.lower()
    if not ext:
        ext = ".bin"
    return ext


def _validate_file(original_name: str, size: int) -> str:
    """Check type and size, return the safe extension or raise ValueError."""
    ext = _sanitise_extension(original_name)

    if ext in BLOCKED_EXTENSIONS:
        raise ValueError(f"File type '{ext}' is not allowed")

    if ext not in ALLOWED_EXTENSIONS:
        guessed = mimetypes.guess_type(original_name)[0]
        if guessed and guessed.split("/")[0] in ("image", "text", "application"):
            pass  # generic safe MIME family -- allow through
        else:
            raise ValueError(
                f"File type '{ext}' is not supported. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

    max_size = settings.max_upload_size
    if size > max_size:
        mb = max_size / (1024 * 1024)
        raise ValueError(f"File exceeds maximum size of {mb:.0f} MB")

    return ext


# Abstract base -- swap the concrete class to change backend
class StorageBackend:
    """Minimal interface every storage backend must implement."""

    async def save(self, file_data: bytes, original_name: str) -> dict:
        """
        Persist *file_data* and return metadata dict::

            {
                "id": "<uuid>",
                "name": "<original name>",
                "stored_name": "<uuid>.<ext>",
                "type": "<mime type>",
                "size": <int>,
                "storage": "local",
                "path": "uploads/<uuid>.<ext>",
            }
        """
        raise NotImplementedError

    async def read(self, stored_name: str) -> bytes:
        """Return file bytes or raise FileNotFoundError."""
        raise NotImplementedError

    async def delete(self, stored_name: str) -> None:
        """Best-effort delete.  Must not raise if file already gone."""
        raise NotImplementedError


# Local-disk implementation
class LocalStorage(StorageBackend):
    """Store files on the local filesystem."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir or settings.upload_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, file_data: bytes, original_name: str) -> dict:
        ext = _validate_file(original_name, len(file_data))
        file_id = str(uuid.uuid4())
        stored_name = f"{file_id}{ext}"
        dest = self.base_dir / stored_name

        async with aiofiles.open(dest, "wb") as f:
            await f.write(file_data)

        mime = ALLOWED_EXTENSIONS.get(ext) or mimetypes.guess_type(original_name)[0] or "application/octet-stream"

        logger.info("Attachment saved: %s (%d bytes)", stored_name, len(file_data))
        return {
            "id": file_id,
            "name": original_name,
            "stored_name": stored_name,
            "type": mime,
            "size": len(file_data),
            "storage": "local",
            "path": f"uploads/{stored_name}",
        }

    async def read(self, stored_name: str) -> bytes:
        path = self.base_dir / stored_name
        if not path.exists():
            raise FileNotFoundError(f"Attachment not found: {stored_name}")
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def delete(self, stored_name: str) -> None:
        path = self.base_dir / stored_name
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete attachment: %s", stored_name)


# Singleton factory
_instance: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """Return the singleton storage backend (lazy-initialised)."""
    global _instance
    if _instance is None:
        backend = settings.storage_backend
        if backend == "local":
            _instance = LocalStorage()
        else:
            raise ValueError(f"Unknown storage backend: {backend}")
    return _instance
