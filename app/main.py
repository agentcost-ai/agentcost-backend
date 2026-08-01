# AgentCost Backend - Main Application

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from datetime import datetime, timezone
import asyncio

from .config import get_settings
from .database import create_tables, get_db_session
from .routes import (
    events_router,
    analytics_router,
    projects_router,
    optimizations_router,
    pricing_router,
    feedback_router,
    attachments_router,
    notifications_router,
    currency_router,
)
from .routes.auth import router as auth_router
from .routes.members import router as members_router
from .routes.admin import router as admin_router
from .routes.demo import router as demo_router
from .models.schemas import HealthResponse
from .utils.rate_limiter import RateLimitMiddleware
from .utils.request_size import RequestSizeLimitMiddleware

import logging
import os
import sys

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting AgentCost Backend...")
    await create_tables()
    logger.info("Database tables created")
    
    # Create upload directory if missing
    upload_dir = Path(settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Upload directory: %s", upload_dir)

    # Start background cron jobs
    from .services.cron import cron_loop
    cron_task = asyncio.create_task(cron_loop())

    # Pricing sync is owned entirely by cron_loop, which already evaluates it on
    # its first tick at startup. A second task here called the same function at
    # the same moment, and since a full sync takes minutes, both passed the
    # is-it-due check and ran overlapping syncs on every boot.

    # Auto-seed superuser from environment variables (for Docker / first-time setup)
    admin_email = os.getenv("ADMIN_EMAIL", "").strip()
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
    if admin_email and admin_password:
        try:
            from sqlalchemy import select
            from .models.user_models import User
            from .services.auth_service import hash_password
            from .common import validate_password_strength

            # Validate admin password against policy
            try:
                validate_password_strength(admin_password)
            except ValueError as pwd_err:
                logger.warning("ADMIN_PASSWORD does not meet security policy: %s", pwd_err)
                admin_password = None

            if not admin_password:
                logger.warning("Skipping admin auto-seed: password doesn't meet requirements")
            else:
                async for db in get_db_session():
                    existing = (await db.execute(
                        select(User).where(User.email == admin_email.lower())
                    )).scalar_one_or_none()

                    if existing:
                        if not existing.is_superuser:
                            existing.is_superuser = True
                            existing.is_active = True
                            await db.commit()
                            logger.info("Existing user %s promoted to superuser", admin_email)
                        else:
                            logger.info("Superuser %s already exists", admin_email)
                    else:
                        admin_name = os.getenv("ADMIN_NAME", "Admin").strip()
                        user = User(
                            email=admin_email.lower(),
                            password_hash=hash_password(admin_password),
                            name=admin_name,
                            is_superuser=True,
                            is_active=True,
                            email_verified=True,
                        )
                        db.add(user)
                        await db.commit()
                        logger.info("Superuser %s created from environment variables", admin_email)
                    break
        except Exception as e:
            logger.warning("Admin auto-seed failed: %s", e)
    
    yield
    
    # Shutdown
    logger.info("Shutting down AgentCost Backend...")
    cron_task.cancel()
    try:
        await cron_task
    except asyncio.CancelledError:
        pass

    from .utils.rate_limiter import redis_rate_limiter
    if redis_rate_limiter is not None:
        await redis_rate_limiter.close()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Track LLM costs in your AI applications",
    lifespan=lifespan,
)

# Middleware order matters here. add_middleware inserts at position 0, so the
# LAST one added is the OUTERMOST. CORS therefore has to be registered after the
# two middlewares that short-circuit: RateLimitMiddleware's 429 and
# RequestSizeLimitMiddleware's 413 return without calling the rest of the stack,
# so if CORS sat inside them those responses would reach the browser with no
# Access-Control-Allow-Origin and the dashboard would show an opaque network
# error instead of the real status.

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Request size limit middleware
app.add_middleware(RequestSizeLimitMiddleware)

# CORS middleware - added last so it wraps everything above
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "Accept"],
)

# Register routes
app.include_router(auth_router)
app.include_router(members_router)
app.include_router(events_router)
app.include_router(analytics_router)
app.include_router(projects_router)
app.include_router(optimizations_router)
app.include_router(pricing_router)
app.include_router(feedback_router)
app.include_router(attachments_router)
app.include_router(notifications_router)
app.include_router(currency_router)
app.include_router(admin_router)
app.include_router(demo_router)


@app.get("/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns server status and version.
    """
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/v1/health",
    }


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    # api.agentcost.tech is crawled by Googlebot (it currently 404s here);
    # the API surface has no indexable content, so opt the whole host out.
    return PlainTextResponse("User-agent: *\nDisallow: /\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
