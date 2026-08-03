"""
AgentCost Backend - Database Setup

SQLAlchemy async database configuration.
"""

import hashlib
import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData, text, inspect
from typing import AsyncGenerator, List

from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Naming convention for constraints (helps with migrations)
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}


class Base(DeclarativeBase):
    """Base class for all database models"""
    metadata = MetaData(naming_convention=convention)


database_url = settings.database_url
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    logger.info("Converted DATABASE_URL to use asyncpg driver")


def _utc_connect_args(url: str) -> dict:
    """Pin PostgreSQL sessions to UTC.

    date_trunc/date/extract on a timestamptz bucket in the session's TimeZone,
    so without this they would split days at whatever offset the server happens
    to be configured for, while the window bounds around them are UTC.
    """
    if "postgresql" not in url:
        return {}
    return {"server_settings": {"timezone": "UTC"}}


# Create async engine
# Note: echo=False to prevent verbose SQL logging in terminal
# For SQL debugging, use logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)
engine = create_async_engine(
    database_url,
    echo=False,
    future=True,
    connect_args=_utc_connect_args(database_url),
    # Managed Postgres (and any pooler in front of it) closes idle backends, so
    # a pooled connection can be dead by the time it is handed out. pre_ping
    # costs one round trip and turns that 500 into a transparent reconnect;
    # recycle retires connections before the server does it for us.
    pool_pre_ping=True,
    pool_recycle=1800,
)


# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Always commits on success, rolls back on error.
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session for use outside of FastAPI dependency injection.
    
    Use this for startup tasks, background jobs, etc.
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Arbitrary but stable key for the PostgreSQL advisory lock that serializes
# schema bootstrap across workers.
_SCHEMA_LOCK_KEY = 8474921003


def _schema_fingerprint() -> str:
    """A stable hash of every table, column and type the models declare.

    Also covers the hand-written ADD COLUMN table in _apply_column_migrations,
    so editing either one invalidates the cache and the next boot re-applies.
    """
    parts: List[str] = []
    for table_name in sorted(Base.metadata.tables):
        table = Base.metadata.tables[table_name]
        cols = sorted(f"{c.name}:{c.type!s}:{c.nullable}" for c in table.columns)
        parts.append(f"{table_name}({','.join(cols)})")
    parts.append(repr(_DESIRED_COLUMNS))
    parts.append(repr(_WIDEN_COLUMNS))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


async def create_tables() -> str:
    """Bring the live schema up to what the models declare.

    Returns a one-line summary for the startup log. The fingerprint row turns
    the steady-state boot into a single SELECT; the full introspection this
    replaces cost ~6s of every cold start against a remote Postgres.
    """
    async with engine.begin() as conn:
        is_postgres = conn.dialect.name == "postgresql"
        if is_postgres:
            # Serialize ALL schema DDL across workers, including the state
            # table itself -- CREATE TABLE IF NOT EXISTS can still raise under
            # concurrency on PG. Held until the transaction commits.
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SCHEMA_LOCK_KEY}
            )
        await conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_bootstrap_state ("
            "fingerprint VARCHAR(64) PRIMARY KEY, applied_at TIMESTAMP)"
        ))

        fingerprint = _schema_fingerprint()
        already = (await conn.execute(
            text("SELECT 1 FROM schema_bootstrap_state WHERE fingerprint = :fp"),
            {"fp": fingerprint},
        )).scalar()
        if already:
            return "schema already matches models; no changes needed"

        before = set(await conn.run_sync(lambda c: set(inspect(c).get_table_names())))
        await conn.run_sync(Base.metadata.create_all)
        after = set(await conn.run_sync(lambda c: set(inspect(c).get_table_names())))
        added_columns = await _apply_column_migrations(conn)

        # Only the current fingerprint is kept: an older row would let a
        # rollback to a previous build skip the migrations it still needs.
        await conn.execute(text("DELETE FROM schema_bootstrap_state"))
        await conn.execute(
            text("INSERT INTO schema_bootstrap_state (fingerprint, applied_at) "
                 "VALUES (:fp, :at)"),
            # ISO string: binding a datetime to this raw-DDL column trips
            # SQLite's deprecated adapter. Nothing reads it back.
            {"fp": fingerprint, "at": datetime.now(timezone.utc).isoformat()},
        )

        created = sorted(after - before)
        if not created and not added_columns:
            return "schema verified; no changes needed"
        return (
            f"schema updated: {len(created)} table(s) created"
            f"{' (' + ', '.join(created) + ')' if created else ''}, "
            f"{added_columns} column(s) added"
        )


# Columns added to already-existing tables after their CREATE TABLE shipped.
# Module level so _schema_fingerprint can hash it: editing this table must
# invalidate the bootstrap cache, or the new column would never be applied.
_DESIRED_COLUMNS = {
    "projects": {
        "monthly_budget_usd": {"type": "FLOAT"},
        "budget_alert_thresholds": {"type": "JSON"},
        "budget_enforcement_mode": {
            "type": "VARCHAR(20)",
            "default": "'off'",
            "nullable": False,
        },
        "budget_currency": {
            "type": "VARCHAR(3)",
            "default": "'USD'",
            "nullable": False,
        },
    },
    "events": {
        # Track where the cost figure came from (database / defaults / client-sdk)
        "cost_source": {"type": "VARCHAR(50)"},
        # SHA256 of normalized input for caching pattern detection
        "input_hash": {"type": "VARCHAR(64)"},
    },
    "users": {
        "admin_notes":    {"type": "TEXT"},
        "user_number":    {"type": "INTEGER"},
        "milestone_badge": {"type": "VARCHAR(50)"},
        "last_active_at": {"type": "TIMESTAMP"},
        "auth_provider":  {"type": "VARCHAR(20)", "default": "'email'", "nullable": False},
        "google_id":      {"type": "VARCHAR(255)"},
        "github_id":      {"type": "VARCHAR(255)"},
        "is_deleted":     {"type": "BOOLEAN", "default": "false", "nullable": False},
        "deleted_at":     {"type": "TIMESTAMP"},
    },
    "feedback": {
        "metadata":        {"type": "JSON"},
        "attachments":     {"type": "JSON"},
        "environment":     {"type": "VARCHAR(50)"},
        "client_metadata": {"type": "JSON"},
        "is_confidential": {"type": "BOOLEAN", "default": "false", "nullable": False},
        "ip_address":      {"type": "VARCHAR(45)"},
        "user_agent":      {"type": "TEXT"},
    },
    "feedback_comments": {
        "is_internal": {"type": "BOOLEAN", "default": "false"},
    },
    "model_pricing": {
        # Context/input cap, distinct from max_tokens (the output cap).
        "max_input_tokens": {"type": "INTEGER"},
    },
}

# Existing VARCHAR columns to widen: (table, column, new_length). Widening is
# metadata-only on PostgreSQL; SQLite ignores VARCHAR lengths, so it is skipped.
_WIDEN_COLUMNS = [
    # Bedrock inference-profile ARNs exceed the old 100-char cap.
    ("events", "model", 255),
    ("daily_aggregates", "model", 255),
]


async def _apply_column_migrations(conn):
    """
    Patch existing tables with any columns the models define but the DB lacks.

    create_all() only creates new tables -- it won't ALTER existing ones.
    This introspects the live schema and issues ALTER TABLE ADD COLUMN
    for each missing column.  Works for both PostgreSQL and SQLite.
    """
    def _get_missing_columns(sync_conn):
        insp = inspect(sync_conn)
        migrations = []

        desired = _DESIRED_COLUMNS

        for table_name, columns in desired.items():
            if not insp.has_table(table_name):
                continue
            existing = {col["name"] for col in insp.get_columns(table_name)}
            for col_name, spec in columns.items():
                if col_name not in existing:
                    migrations.append((table_name, col_name, spec))

        return migrations

    missing = await conn.run_sync(_get_missing_columns)
    applied = 0

    if conn.dialect.name == "postgresql":
        def _too_narrow(sync_conn):
            insp = inspect(sync_conn)
            out = []
            for table, column, target in _WIDEN_COLUMNS:
                if not insp.has_table(table):
                    continue
                for col in insp.get_columns(table):
                    current = getattr(col["type"], "length", None)
                    if col["name"] == column and current and current < target:
                        out.append((table, column, target))
            return out

        for table, column, target in await conn.run_sync(_too_narrow):
            stmt = f"ALTER TABLE {table} ALTER COLUMN {column} TYPE VARCHAR({target})"
            logger.info("Migration: %s", stmt)
            await conn.execute(text(stmt))
            applied += 1

    if missing:
        # Detect dialect for boolean default syntax from the live connection,
        # not the configured URL (tests bind SQLite while DATABASE_URL is PG).
        is_sqlite = conn.dialect.name == "sqlite"

        for table_name, col_name, spec in missing:
            col_type = spec["type"]
            default = spec.get("default")
            nullable = spec.get("nullable", True)

            # SQLite uses 0/1 for booleans, PostgreSQL uses true/false
            if default is not None and col_type == "BOOLEAN" and is_sqlite:
                default = "0" if default == "false" else "1"

            parts = [f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"]
            if default is not None:
                parts.append(f"DEFAULT {default}")
            if not nullable:
                parts.append("NOT NULL")

            stmt = " ".join(parts)
            logger.info("Migration: %s", stmt)
            # Each ALTER runs inside its own SAVEPOINT: introspection and DDL are
            # not atomic together, so a concurrent worker (or a pooler that
            # bypassed the advisory lock) can add the column in between. On
            # PostgreSQL a failed statement poisons the whole transaction, which
            # would abort this worker's startup -- the savepoint contains it so a
            # column that already exists is simply skipped.
            try:
                async with conn.begin_nested():
                    await conn.execute(text(stmt))
                applied += 1
            except Exception as exc:  # noqa: BLE001 — see _is_duplicate_column
                if _is_duplicate_column(exc):
                    logger.info(
                        "Migration skipped (column already present): %s.%s",
                        table_name,
                        col_name,
                    )
                else:
                    raise

    return applied


def _is_duplicate_column(exc: Exception) -> bool:
    """
    True when a failed ALTER means "another worker already added this column".

    PostgreSQL raises SQLSTATE 42701 (duplicate_column); SQLite has no codes,
    so its message is matched instead.
    """
    orig = getattr(exc, "orig", exc)
    if getattr(orig, "sqlstate", None) == "42701" or getattr(orig, "pgcode", None) == "42701":
        return True
    message = str(orig).lower()
    return "duplicate column" in message or "already exists" in message
