# AgentCost Backend - Main Application

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime, timezone
import asyncio
import re

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
    integrations_router,
    metrics_router,
)
from .routes.auth import router as auth_router
from .routes.members import router as members_router
from .routes.admin import router as admin_router
from .routes.demo import router as demo_router
from .models.schemas import HealthResponse
from .utils.errors import (
    ErrorEnvelopeMiddleware,
    error_body,
    summarize_validation_errors,
)
from .utils.rate_limiter import RateLimitMiddleware
from .utils.request_size import RequestSizeLimitMiddleware

import logging
import os
import sys
import time

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
    # Log what the bootstrap actually did and how long it took -- the old
    # unconditional "Database tables created" was wrong on every boot but the first.
    schema_started = time.monotonic()
    schema_summary = await create_tables()
    logger.info(
        "Database schema: %s (%.2fs)", schema_summary, time.monotonic() - schema_started
    )
    
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
                            logger.debug("Superuser %s already exists", admin_email)
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


# Tag descriptions. Without these the published spec lists bare tag names, which
# tells an agent nothing about which group of operations it wants.
OPENAPI_TAGS = [
    {"name": "Pricing", "description": (
        "Public model catalogue. Per-1k input/output/cached rates, provider, mode and "
        "announced retirement dates for every model AgentCost can bill. No credentials "
        "required -- this is the read surface SDKs and agents call."
    )},
    {"name": "Health", "description": "Liveness and version reporting. No credentials required."},
    {"name": "Root", "description": "Service banner pointing at the docs and health endpoints."},
    {"name": "Authentication", "description": (
        "Account registration, sign-in, OAuth exchange, email verification, password "
        "reset and session management. Issues the bearer tokens the rest of the API uses."
    )},
    {"name": "Events", "description": (
        "LLM call ingestion. Batched writes from the SDK, plus read-back of stored events. "
        "Authenticated with a project API key ('Authorization: Bearer sk_...')."
    )},
    {"name": "Analytics", "description": (
        "Aggregated spend: overview totals, per-agent and per-dimension breakdowns, "
        "time series, cache savings, workflow and trace analysis."
    )},
    {"name": "Optimizations", "description": (
        "Cost-reduction recommendations, model-swap baselines, caching opportunities and "
        "effectiveness tracking for recommendations already applied."
    )},
    {"name": "Projects", "description": "Project lifecycle, budgets, webhooks and API-key rotation."},
    {"name": "Project Members", "description": "Team membership, roles and invitations on a project."},
    {"name": "Feedback", "description": "Public feedback board: submissions, comments and upvotes."},
    {"name": "Attachments", "description": "Upload limits and auth-gated attachment download."},
    {"name": "Notifications", "description": "In-app notification feed and read state."},
    {"name": "Currency", "description": "USD conversion rates for non-USD cost display."},
    {"name": "Integrations", "description": "Provider-side cost import from OpenAI and Anthropic billing APIs."},
    {"name": "Metrics", "description": "Prometheus exposition of per-project ingestion and spend metrics."},
    {"name": "Admin", "description": "Superuser-only operations. Not part of the public contract."},
    {"name": "Demo", "description": "Anonymous demo-mode telemetry. No PII."},
]

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Track LLM costs in your AI applications.\n\n"
        "AgentCost records every LLM call with model, tokens, cost, latency and status, "
        "then reports spend by agent, workflow, project and model.\n\n"
        "**Public, no credentials required:** everything under `/v1/pricing` (the model "
        "catalogue and its per-1k rates) and `/v1/health`.\n\n"
        "**Everything else** needs either a project API key "
        "(`Authorization: Bearer sk_...`) or a session token plus a `project_id`.\n\n"
        "Errors return a structured envelope: `error.code`, `error.message`, `error.hint`."
    ),
    terms_of_service="https://agentcost.tech/terms",
    contact={
        "name": "AgentCost",
        "url": "https://agentcost.tech/contact",
        "email": "hello@agentcost.tech",
    },
    license_info={
        "name": "MIT",
        "url": "https://github.com/agentcost-ai/agentcost-sdk/blob/main/LICENSE",
    },
    servers=[
        {"url": "https://api.agentcost.tech", "description": "Production"},
        {"url": "http://localhost:8000", "description": "Local development"},
    ],
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

# Middleware order matters here. add_middleware inserts at position 0, so the
# LAST one added is the OUTERMOST. CORS therefore has to be registered after the
# two middlewares that short-circuit: RateLimitMiddleware's 429 and
# RequestSizeLimitMiddleware's 413 return without calling the rest of the stack,
# so if CORS sat inside them those responses would reach the browser with no
# Access-Control-Allow-Origin and the dashboard would show an opaque network
# error instead of the real status.

# Added FIRST, so it ends up INNERMOST: an unhandled route error becomes the JSON
# envelope here and the response still travels back out through CORSMiddleware.
# Starlette's own ServerErrorMiddleware sits outside everything and would answer
# with CORS-less plain text instead.
app.add_middleware(ErrorEnvelopeMiddleware)

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

# Exception handlers -----------------------------------------------------------
# Without these, FastAPI answers HTTPException with a bare {"detail": "..."} and
# validation failures with {"detail": [ ...objects... ]} -- two different shapes,
# neither carrying a code an agent can branch on.


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.status_code, detail),
        # WWW-Authenticate on 401 and Retry-After on 429 are part of the contract.
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    summary, fields = summarize_validation_errors(exc.errors())
    return JSONResponse(
        status_code=422,
        content=error_body(422, summary, fields=fields),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Backstop for failures raised outside the router (e.g. in an outer
    # middleware), which ErrorEnvelopeMiddleware never sees.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=error_body(500, "Internal server error."),
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
app.include_router(integrations_router)
app.include_router(metrics_router)
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


# The machine-readable surface an agent looks for -- the spec and the public
# catalogue -- has to be fetchable. Everything else on this host is either
# auth-gated or has no indexable content, so it stays disallowed.
ROBOTS_TXT = "\n".join(
    [
        "User-agent: *",
        "Allow: /openapi.json",
        "Allow: /docs",
        "Allow: /redoc",
        "Allow: /v1/pricing",
        "Allow: /v1/health",
        "Disallow: /",
        "",
        "Sitemap: https://agentcost.tech/sitemap.xml",
        "",
    ]
)


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    return PlainTextResponse(ROBOTS_TXT)


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", value)).strip("_").lower()


def assign_operation_ids(fastapi_app: FastAPI) -> None:
    """Give every operation a readable, unique operationId.

    FastAPI's default is derived from the function name plus the path and method
    ("get_all_pricing_v1_pricing_get"), and that string is what an LLM
    function-calling client shows as the tool name. Tag + function name reads far
    better; collisions (the admin routers reuse function names) fall back to
    appending the HTTP method, then a counter, so ids stay stable and unambiguous.
    """
    used: set[str] = set()
    for route in fastapi_app.routes:
        if not isinstance(route, APIRoute):
            continue
        tag = route.tags[0] if route.tags else "api"
        candidate = _slug(f"{tag}_{route.name}")
        if candidate in used:
            methods = sorted(route.methods - {"HEAD", "OPTIONS"})
            candidate = _slug(f"{candidate}_{methods[0] if methods else 'get'}")
        base, counter = candidate, 2
        while candidate in used:
            candidate = f"{base}_{counter}"
            counter += 1
        used.add(candidate)
        route.operation_id = candidate


assign_operation_ids(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
