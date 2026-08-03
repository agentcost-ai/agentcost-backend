"""
Tests for the dual-auth ``validate_project_access`` dependency and the
new ``GET /v1/projects`` listing endpoint.

These tests use the real FastAPI app via the in-memory SQLite fixture so we
exercise the dependency exactly the way production code does (header parsing,
permission service, query-param handling, error responses).
"""

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app
from app.models.db_models import Project
from app.models.user_models import ProjectMember, User, UserRole
from app.services.auth_service import create_access_token, hash_password
from app.utils.auth import hash_api_key


@pytest.fixture
async def engine_and_session():
    """In-memory engine + sessionmaker, schema created via metadata.create_all."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", echo=False, future=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield engine, Session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(engine_and_session):
    _, Session = engine_and_session
    async with Session() as session:
        yield session


@pytest.fixture
async def http_client(engine_and_session):
    """Test client wired up so every request uses our in-memory SQLite session."""
    _, Session = engine_and_session

    async def override_get_db():
        async with Session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


# ─────────────────────────── helpers ───────────────────────────


async def _seed_user(
    session: AsyncSession, *, email: str, name: str = "U"
) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        name=name,
        password_hash=hash_password("Aaaa1234!"),
        is_active=True,
        email_verified=True,
    )
    session.add(user)
    await session.commit()
    return user


async def _seed_project_with_api_key(
    session: AsyncSession, *, owner: User, name: str = "Proj", api_key: str = "sk_test_abc"
) -> tuple[Project, str]:
    """
    Returns (project, plaintext_api_key). The DB stores a hash; we return
    the plaintext key for header use.
    """
    project = Project(
        id=str(uuid.uuid4()),
        name=name,
        api_key=hash_api_key(api_key),
        owner_id=owner.id,
        is_active=True,
    )
    session.add(project)
    await session.commit()
    return project, api_key


async def _add_member(
    session: AsyncSession, *, project: Project, user: User, role: UserRole
):
    membership = ProjectMember(
        project_id=project.id,
        user_id=user.id,
        role=role.value,
        accepted_at=datetime.now(timezone.utc),
    )
    session.add(membership)
    await session.commit()


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _jwt_for(user: User) -> str:
    token, _ = create_access_token(user.id, user.email)
    return token


# ─────────────────────────── /v1/projects listing ───────────────────────────


@pytest.mark.asyncio
async def test_list_projects_returns_owned_and_member_projects(
    db_session, http_client
):
    owner = await _seed_user(db_session, email="owner@example.com", name="Owner")
    member = await _seed_user(db_session, email="member@example.com", name="Member")

    owned, _ = await _seed_project_with_api_key(
        db_session, owner=owner, name="Owned Project", api_key="sk_owner"
    )
    invited, _ = await _seed_project_with_api_key(
        db_session, owner=owner, name="Invited Project", api_key="sk_invited"
    )
    other_user_only, _ = await _seed_project_with_api_key(
        db_session, owner=owner, name="Stranger", api_key="sk_stranger"
    )
    await _add_member(
        db_session, project=invited, user=member, role=UserRole.VIEWER
    )

    response = await http_client.get("/v1/projects", headers=_bearer(_jwt_for(member)))
    assert response.status_code == 200
    items = response.json()
    names = {item["name"] for item in items}
    assert "Invited Project" in names
    assert "Stranger" not in names

    # Owner sees their own projects
    response = await http_client.get("/v1/projects", headers=_bearer(_jwt_for(owner)))
    assert response.status_code == 200
    items = response.json()
    names = {item["name"] for item in items}
    assert {"Owned Project", "Invited Project", "Stranger"} == names

    # No auth -> 401
    response = await http_client.get("/v1/projects")
    assert response.status_code == 401


# ─────────────────────────── dual-auth dependency ───────────────────────────


@pytest.mark.asyncio
async def test_analytics_overview_works_with_api_key(db_session, http_client):
    owner = await _seed_user(db_session, email="o1@example.com")
    project, api_key = await _seed_project_with_api_key(
        db_session, owner=owner, api_key="sk_apikey_path"
    )

    response = await http_client.get(
        "/v1/analytics/overview?range=7d",
        headers=_bearer(api_key),
    )
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_analytics_overview_works_with_jwt_and_project_id_for_member(
    db_session, http_client
):
    owner = await _seed_user(db_session, email="o2@example.com")
    member = await _seed_user(db_session, email="m2@example.com")
    project, _ = await _seed_project_with_api_key(
        db_session, owner=owner, api_key="sk_unused"
    )
    await _add_member(
        db_session, project=project, user=member, role=UserRole.MEMBER
    )

    response = await http_client.get(
        f"/v1/analytics/overview?range=7d&project_id={project.id}",
        headers=_bearer(_jwt_for(member)),
    )
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_analytics_overview_rejects_jwt_without_project_id(
    db_session, http_client
):
    member = await _seed_user(db_session, email="m3@example.com")
    response = await http_client.get(
        "/v1/analytics/overview?range=7d",
        headers=_bearer(_jwt_for(member)),
    )
    assert response.status_code == 400
    assert "project_id" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_analytics_overview_rejects_jwt_user_without_membership(
    db_session, http_client
):
    owner = await _seed_user(db_session, email="o4@example.com")
    stranger = await _seed_user(db_session, email="s4@example.com")
    project, _ = await _seed_project_with_api_key(
        db_session, owner=owner, api_key="sk_unused4"
    )

    response = await http_client.get(
        f"/v1/analytics/overview?range=7d&project_id={project.id}",
        headers=_bearer(_jwt_for(stranger)),
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_analytics_overview_no_auth_returns_401(http_client):
    response = await http_client.get("/v1/analytics/overview?range=7d")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_analytics_overview_invalid_api_key_returns_401(http_client):
    response = await http_client.get(
        "/v1/analytics/overview?range=7d",
        headers=_bearer("sk_nope_nope"),
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_events_list_works_with_jwt_for_viewer(db_session, http_client):
    owner = await _seed_user(db_session, email="o5@example.com")
    viewer = await _seed_user(db_session, email="v5@example.com")
    project, _ = await _seed_project_with_api_key(
        db_session, owner=owner, api_key="sk_unused5"
    )
    await _add_member(
        db_session, project=project, user=viewer, role=UserRole.VIEWER
    )

    response = await http_client.get(
        f"/v1/events?project_id={project.id}",
        headers=_bearer(_jwt_for(viewer)),
    )
    assert response.status_code == 200, response.text
    assert response.json() == []


@pytest.mark.asyncio
async def test_event_ingestion_still_requires_api_key(db_session, http_client):
    """POST /v1/events/batch must remain API-key-only (SDK path)."""
    owner = await _seed_user(db_session, email="o6@example.com")
    project, _ = await _seed_project_with_api_key(
        db_session, owner=owner, api_key="sk_ingest_only"
    )

    body = {
        "project_id": project.id,
        "events": [
            {
                "agent_name": "a",
                "model": "gpt-4",
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "cost": 0.01,
                "latency_ms": 100,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True,
            }
        ],
    }

    # JWT must NOT be accepted on the ingestion endpoint.
    response = await http_client.post(
        "/v1/events/batch",
        json=body,
        headers=_bearer(_jwt_for(owner)),
    )
    assert response.status_code in (401, 403), response.text

    # API key is accepted.
    response = await http_client.post(
        "/v1/events/batch",
        json=body,
        headers=_bearer("sk_ingest_only"),
    )
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_get_project_by_id_works_with_jwt_for_member(db_session, http_client):
    """
    /v1/projects/{id} must accept JWT (with project_id query) so that
    invited members can fetch project metadata without the raw API key.
    """
    owner = await _seed_user(db_session, email="op@example.com")
    member = await _seed_user(db_session, email="mp@example.com")
    project, _ = await _seed_project_with_api_key(
        db_session, owner=owner, api_key="sk_proj_jwt"
    )
    await _add_member(
        db_session, project=project, user=member, role=UserRole.VIEWER
    )

    # JWT path: project_id comes from the URL path; no query needed.
    response = await http_client.get(
        f"/v1/projects/{project.id}",
        headers=_bearer(_jwt_for(member)),
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["id"] == project.id

    # API-key path must still work (back-compat)
    response = await http_client.get(
        f"/v1/projects/{project.id}",
        headers=_bearer("sk_proj_jwt"),
    )
    assert response.status_code == 200, response.text

    # Stranger gets 403
    stranger = await _seed_user(db_session, email="strangerp@example.com")
    response = await http_client.get(
        f"/v1/projects/{project.id}",
        headers=_bearer(_jwt_for(stranger)),
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_project_delete_cleans_up_all_dependent_rows(db_session, http_client):
    """
    Deleting a project must remove every dependent row across child tables
    so a future project (or re-creation) doesn't surface stale data.
    """
    from sqlalchemy import select, func as sa_func
    from app.models.db_models import (
        BudgetThresholdAlert,
        Event,
        Notification,
        OptimizationRecommendation,
    )
    from app.services.event_service import ProjectService

    owner = await _seed_user(db_session, email="del@example.com")
    project, _ = await _seed_project_with_api_key(
        db_session, owner=owner, api_key="sk_del"
    )

    # Seed several child rows across different tables.
    for _ in range(4):
        db_session.add(
            Event(
                id=str(uuid.uuid4()),
                project_id=project.id,
                agent_name="a",
                model="gpt-4",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cost=0.01,
                latency_ms=100,
                timestamp=datetime.now(timezone.utc),
                success=True,
            )
        )
    db_session.add(
        OptimizationRecommendation(
            id=str(uuid.uuid4()),
            project_id=project.id,
            recommendation_type="downgrade",
            title="t",
        )
    )
    db_session.add(
        BudgetThresholdAlert(
            id=str(uuid.uuid4()),
            project_id=project.id,
            period_key="2026-05",
            threshold_percent=50.0,
            spent_amount=10.0,
            budget_amount=100.0,
            utilization_percent=10.0,
        )
    )
    db_session.add(
        Notification(
            id=str(uuid.uuid4()),
            user_id=owner.id,
            type="budget_threshold",
            severity="warning",
            title="t",
            project_id=project.id,
        )
    )
    await db_session.commit()

    ok = await ProjectService(db_session).delete(project.id)
    await db_session.commit()
    assert ok is True

    for model in (Event, OptimizationRecommendation, BudgetThresholdAlert, Notification):
        count = (
            await db_session.execute(
                select(sa_func.count())
                .select_from(model)
                .where(model.project_id == project.id)
            )
        ).scalar()
        assert count == 0, f"{model.__tablename__} still has {count} rows after delete"


@pytest.mark.asyncio
async def test_disabled_project_returns_403_for_jwt(db_session, http_client):
    owner = await _seed_user(db_session, email="o7@example.com")
    member = await _seed_user(db_session, email="m7@example.com")
    project, _ = await _seed_project_with_api_key(
        db_session, owner=owner, api_key="sk_disabled"
    )
    await _add_member(
        db_session, project=project, user=member, role=UserRole.VIEWER
    )
    project.is_active = False
    await db_session.commit()

    response = await http_client.get(
        f"/v1/analytics/overview?range=7d&project_id={project.id}",
        headers=_bearer(_jwt_for(member)),
    )
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_revoking_a_session_kills_its_access_token(
    test_session: AsyncSession, test_user
):
    """The gap this closes: revoking a session only killed the refresh token,
    so the access token kept working until exp."""
    from app.services.auth_service import AuthService, get_current_user

    auth = AuthService(test_session)
    tokens = await auth.login_user(test_user)
    await test_session.commit()

    assert (await get_current_user(test_session, tokens.access_token)) is not None

    sessions = await auth.get_active_sessions(test_user.id)
    assert sessions, "login must have created a session"
    await auth.revoke_session(test_user.id, sessions[0].id)
    await test_session.commit()

    assert (await get_current_user(test_session, tokens.access_token)) is None
