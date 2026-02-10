"""
AgentCost Backend - Admin API Routes (Package)

Assembles all admin sub-routers into a single router
mounted at /v1/admin.  Each sub-module owns one domain:

  auth.py       -- token verification
  overview.py   -- platform-wide stats and timeseries
  users.py      -- user / tenant management
  projects.py   -- project and API key governance
  pricing.py    -- model pricing CRUD and sync
  system.py     -- health checks and ingestion stats
  analytics.py  -- cross-tenant analytics
  incidents.py  -- error and incident logs
  feedback.py   -- feedback triage and response
  audit_log.py  -- immutable admin action trail
"""

from fastapi import APIRouter

from .auth import router as auth_router
from .overview import router as overview_router
from .users import router as users_router
from .projects import router as projects_router
from .pricing import router as pricing_router
from .system import router as system_router
from .analytics import router as analytics_router
from .incidents import router as incidents_router
from .feedback import router as feedback_router
from .audit_log import router as audit_log_router

router = APIRouter(prefix="/v1/admin", tags=["Admin"])

router.include_router(auth_router)
router.include_router(overview_router)
router.include_router(users_router)
router.include_router(projects_router)
router.include_router(pricing_router)
router.include_router(system_router)
router.include_router(analytics_router)
router.include_router(incidents_router)
router.include_router(feedback_router)
router.include_router(audit_log_router)
