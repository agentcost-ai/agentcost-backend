"""
Test fixtures for AgentCost Backend tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from datetime import datetime, timezone

from httpx import AsyncClient, ASGITransport
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.main import app
from app.database import Base, get_db
from app.models.db_models import Project
from app.utils.auth import hash_api_key


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Plaintext SDK key sent by the test client. The project row stores its hash,
# and the key carries the ``sk_`` prefix the dual-auth resolver requires.
TEST_API_KEY = "sk_test_12345"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def no_outbound_email(monkeypatch, request):
    """Never let the suite reach api.resend.com.

    resend.api_key is bound at import time from .env -- the production key on a
    dev machine -- and email_service._send swallows failures, so tests touching
    email paths made live sends without anything turning red. Autouse so no
    test can forget; assert on captures via the ``sent_emails`` fixture.
    """
    import resend

    sent = _SENT_EMAILS
    sent.clear()

    def _capture(params, *args, **kwargs):
        sent.append(params)
        return {"id": f"test-{len(sent)}"}

    monkeypatch.setattr(resend.Emails, "send", staticmethod(_capture))
    yield sent


_SENT_EMAILS: list = []


@pytest.fixture
def sent_emails(no_outbound_email):
    """The emails this test asked to send, in order. See no_outbound_email."""
    return no_outbound_email


@pytest.fixture(scope="function")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
        # A single shared connection so the in-memory DB persists across the
        # many short-lived sessions a request cycle opens. Without StaticPool
        # each connection gets its own empty ``:memory:`` database.
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # SQLite ignores foreign keys unless asked; PostgreSQL always enforces them.
    # Without this the suite silently accepts writes that production rejects —
    # it is how a purge job that violated a FK on every run stayed green here.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture(scope="function")
async def test_project(test_session: AsyncSession) -> Project:
    """Create a test project"""
    import uuid
    
    project = Project(
        id=str(uuid.uuid4()),
        name="Test Project",
        api_key=hash_api_key(TEST_API_KEY),
        created_at=datetime.now(timezone.utc),
    )
    test_session.add(project)
    await test_session.commit()
    await test_session.refresh(project)
    
    return project


@pytest.fixture(scope="function")
async def client(test_engine, test_project) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client with overridden dependencies"""
    
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async def override_get_db():
        # Mirror the real get_db: commit on success, roll back on error.
        # Without the commit, writes (e.g. event ingestion) are discarded
        # when the per-request session closes.
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Add auth header in correct format
        ac.headers["Authorization"] = f"Bearer {TEST_API_KEY}"
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def test_user(test_session: AsyncSession):
    """Create an authenticated user (for JWT-protected endpoints like project creation)."""
    import uuid
    from app.models.user_models import User
    from app.services.auth_service import hash_password

    user = User(
        id=str(uuid.uuid4()),
        email="owner@example.com",
        password_hash=hash_password("hashedpassword123"),
        name="Owner User",
        is_active=True,
        is_deleted=False,
        email_verified=True,
        auth_provider="email",
    )
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Bearer JWT headers for the test user — use on JWT-scoped endpoints."""
    from app.services.auth_service import create_access_token

    token, _ = create_access_token(test_user.id, test_user.email)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_events(test_project):
    """Sample event data for testing"""
    return {
        "project_id": test_project.id,
        "events": [
            {
                "agent_name": "research-agent",
                "model": "gpt-4",
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "cost": 0.015,
                "latency_ms": 1200,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True,
                "error": None,
                "metadata": {"task": "research"},
            },
            {
                "agent_name": "writer-agent",
                "model": "gpt-3.5-turbo",
                "input_tokens": 500,
                "output_tokens": 800,
                "total_tokens": 1300,
                "cost": 0.003,
                "latency_ms": 800,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True,
                "error": None,
                "metadata": {"task": "writing"},
            },
        ],
    }
