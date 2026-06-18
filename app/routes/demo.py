"""
AgentCost Backend - Demo Tracking Routes

Anonymous, unauthenticated pings from the dashboard's demo mode. The demo
itself is fully client-side; these events only exist so the admin panel can
report demo adoption and demo -> signup conversion.

Deliberately forgiving: tracking must never break the demo, so failures
return 200 with an "ignored" status rather than erroring.
"""

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.db_models import DemoSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/demo", tags=["Demo"])

MAX_TRACKED_PAGES = 50


class DemoTrackRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=64)
    event_type: Literal[
        "demo_start", "page_view", "signup_click", "signup_completed", "demo_exit"
    ]
    page: Optional[str] = Field(None, max_length=255)
    source: Optional[str] = Field(None, max_length=50)
    referrer: Optional[str] = Field(None, max_length=512)


async def _get_session(
    db: AsyncSession, session_id: str
) -> Optional[DemoSession]:
    return (
        await db.execute(
            select(DemoSession).where(DemoSession.session_id == session_id)
        )
    ).scalar_one_or_none()


def _apply_event(session: DemoSession, payload: DemoTrackRequest) -> None:
    """Mutate a session row according to the incoming event."""
    if payload.event_type == "demo_start":
        # Entry attributes are captured on first sight only.
        if payload.source and not session.source:
            session.source = payload.source
        if payload.referrer and not session.referrer:
            session.referrer = payload.referrer

    elif payload.event_type == "page_view":
        session.page_views = (session.page_views or 0) + 1
        if payload.page:
            pages = list(session.pages or [])
            if payload.page not in pages and len(pages) < MAX_TRACKED_PAGES:
                pages.append(payload.page)
                session.pages = pages

    elif payload.event_type == "signup_click":
        session.signup_clicked = True

    elif payload.event_type == "signup_completed":
        session.converted = True
        session.converted_at = datetime.now(timezone.utc)

    # demo_exit only refreshes last_seen_at (set below for every event).
    session.last_seen_at = datetime.now(timezone.utc)


@router.post("/track")
async def track_demo_event(
    payload: DemoTrackRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Record a demo-mode event. Public endpoint, no auth, no PII."""
    try:
        session = await _get_session(db, payload.session_id)

        if session is None:
            # New session. On demo entry two pings (demo_start + the first
            # page_view) fire near-simultaneously and can both reach here
            # before either commits — they then race on the INSERT and the
            # unique constraint on session_id rejects the loser. Catch that
            # and fall back to updating the row the winner just created, so
            # no event is dropped.
            session = DemoSession(
                session_id=payload.session_id,
                user_agent=(request.headers.get("user-agent") or "")[:512]
                or None,
                page_views=0,
                pages=[],
            )
            db.add(session)
            _apply_event(session, payload)
            try:
                await db.commit()
                return {"status": "ok"}
            except IntegrityError:
                await db.rollback()
                session = await _get_session(db, payload.session_id)
                if session is None:
                    # Genuinely shouldn't happen — never raise from tracking.
                    return {"status": "ignored"}
                # Fall through to the update path on the now-existing row.

        _apply_event(session, payload)
        await db.commit()
        return {"status": "ok"}
    except Exception as e:  # noqa: BLE001 — tracking must never 500 the client
        logger.warning("Demo tracking failed: %s", e)
        await db.rollback()
        return {"status": "ignored"}
