"""
AgentCost Backend - Docs Feedback Route

The public "Was this page helpful?" vote. No credentials: the docs are public
and the vote carries nothing about the voter. The global rate limiter applies.
"""

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.schemas import DocsFeedbackCreate
from ..services.docs_feedback_service import record_vote

router = APIRouter(prefix="/v1/docs", tags=["Docs"])


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def vote_docs_page(payload: DocsFeedbackCreate, db: AsyncSession = Depends(get_db)):
    """Record an anonymous yes/no vote for one documentation page."""
    await record_vote(db, payload.page, payload.helpful)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
