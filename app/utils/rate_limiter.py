"""
AgentCost Backend - Rate Limiting Middleware

In-memory rate limiter with sliding window algorithm.
For production with multiple instances, use Redis instead.
"""

import time
from collections import defaultdict
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import get_settings


class RateLimiter:
    """
    In-memory rate limiter using sliding window.
    
    Thread-safe for single-process deployments.
    For multi-process, use Redis-backed implementation.
    """
    
    def __init__(self, requests_per_window: int = 100, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
    
    def _cleanup_old_requests(self, key: str, now: float) -> None:
        """Remove requests outside the current window"""
        cutoff = now - self.window_seconds
        self._requests[key] = [ts for ts in self._requests[key] if ts > cutoff]
    
    def is_allowed(self, key: str) -> tuple[bool, int, int]:
        """
        Check if request is allowed for the given key.
        
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_in_seconds)
        """
        now = time.time()
        self._cleanup_old_requests(key, now)
        
        current_count = len(self._requests[key])
        remaining = max(0, self.requests_per_window - current_count)
        
        # Calculate when the oldest request will expire
        if self._requests[key]:
            oldest = min(self._requests[key])
            reset_in = int(oldest + self.window_seconds - now)
        else:
            reset_in = self.window_seconds
        
        if current_count >= self.requests_per_window:
            return False, 0, reset_in
        
        # Record this request
        self._requests[key].append(now)
        return True, remaining - 1, reset_in
    
    def get_key_from_request(self, request: Request) -> str:
        """
        Extract rate limit key from request.
        
        Priority:
        1. API key from Authorization header
        2. Client IP address
        """
        # Try to get API key
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return f"api_key:{auth_header[7:20]}..."  # Use prefix for privacy
        
        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"


# Global rate limiter instance
# NOTE: Currently uses in-memory storage (single-process only).
# For multi-instance deployments, set RATE_LIMIT_BACKEND=redis in .env
# and implement Redis-backed rate limiting.
settings = get_settings()
rate_limiter = RateLimiter(
    requests_per_window=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_period,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Applies to all /v1/ endpoints.
    Adds standard rate limit headers to responses.
    """
    
    # Paths that are exempt from rate limiting
    EXEMPT_PATHS = {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/v1/health",
    }
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Skip rate limiting for non-API paths
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)
        
        # Check rate limit
        key = rate_limiter.get_key_from_request(request)
        is_allowed, remaining, reset_in = rate_limiter.is_allowed(key)
        
        # Add rate limit headers to all responses
        headers = {
            "X-RateLimit-Limit": str(settings.rate_limit_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_in),
        }
        
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "retry_after": reset_in,
                    "limit": settings.rate_limit_requests,
                    "period": f"{settings.rate_limit_period} seconds",
                },
                headers={
                    **headers,
                    "Retry-After": str(reset_in),
                },
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value
        
        return response
