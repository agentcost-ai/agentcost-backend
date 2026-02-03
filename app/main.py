# AgentCost Backend - Main Application

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

from .config import get_settings
from .database import create_tables, get_db_session
from .routes import events_router, analytics_router, projects_router, optimizations_router, pricing_router
from .routes.auth import router as auth_router
from .routes.members import router as members_router
from .models.schemas import HealthResponse
from .utils.rate_limiter import RateLimitMiddleware
from .utils.request_size import RequestSizeLimitMiddleware
from .services.pricing_service import PricingService

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("Starting AgentCost Backend...")
    await create_tables()
    print("Database tables created")
    
    # Optional: Auto-sync pricing on startup
    if settings.auto_sync_pricing_on_startup:
        print("Auto-syncing pricing from LiteLLM...")
        try:
            async for db in get_db_session():
                pricing_service = PricingService(db)
                result = await pricing_service.sync_from_litellm(track_changes=False)
                await pricing_service.close()
                print(f"Pricing sync complete: {result.get('models_created', 0)} created, "
                      f"{result.get('models_updated', 0)} updated")
                break
        except Exception as e:
            print(f"Warning: Pricing sync failed: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down AgentCost Backend...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Track LLM costs in your AI applications",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Request size limit middleware
app.add_middleware(RequestSizeLimitMiddleware)

# Register routes
app.include_router(auth_router)
app.include_router(members_router)
app.include_router(events_router)
app.include_router(analytics_router)
app.include_router(projects_router)
app.include_router(optimizations_router)
app.include_router(pricing_router)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
