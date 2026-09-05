"""Anonymous docs votes: stored, validated, and aggregated for admin."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.db_models import DocsFeedback
from app.services.docs_feedback_service import summarize


@pytest.mark.asyncio
async def test_vote_is_stored_without_credentials(client: AsyncClient, test_session):
    client.headers.pop("Authorization", None)
    response = await client.post("/v1/docs/feedback", json={"page": "/docs/sdk", "helpful": True})
    assert response.status_code == 204

    rows = (await test_session.execute(select(DocsFeedback))).scalars().all()
    assert len(rows) == 1
    assert rows[0].page == "/docs/sdk"
    assert rows[0].helpful is True


@pytest.mark.asyncio
async def test_only_docs_pages_are_accepted(client: AsyncClient):
    for page in ("/dashboard", "/docs/../etc", "https://evil.example/docs", "x" * 300):
        response = await client.post("/v1/docs/feedback", json={"page": page, "helpful": False})
        assert response.status_code == 422, page


@pytest.mark.asyncio
async def test_summary_tallies_per_page(client: AsyncClient, test_session):
    votes = (("/docs/api", True), ("/docs/api", True), ("/docs/api", False), ("/docs", True))
    for page, helpful in votes:
        r = await client.post("/v1/docs/feedback", json={"page": page, "helpful": helpful})
        assert r.status_code == 204

    summary = await summarize(test_session)
    assert summary.total_votes == 4
    assert summary.votes_last_30d == 4
    api = next(p for p in summary.pages if p.page == "/docs/api")
    assert (api.helpful, api.not_helpful, api.total, api.score) == (2, 1, 3, 66.7)
    assert summary.pages[0].page == "/docs/api"  # most voted first
