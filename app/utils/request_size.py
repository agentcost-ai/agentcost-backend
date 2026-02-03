"""
AgentCost Backend - Request Size Limit Middleware

Lightweight middleware to protect against oversized request payloads.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import get_settings


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.
    
    Protects against denial-of-service via large payloads.
    Returns 413 Payload Too Large if exceeded.
    """
    
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        max_size_bytes = settings.max_request_size_mb * 1024 * 1024
        
        # Check Content-Length header if present
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_size_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Request body too large. Maximum size is {settings.max_request_size_mb}MB."
                        }
                    )
            except ValueError:
                pass  # Invalid content-length header, continue
        
        return await call_next(request)
