"""
AgentCost Backend - Database Setup

SQLAlchemy async database configuration.
"""

import logging

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData, text, inspect
from typing import AsyncGenerator

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


# Create async engine
# Note: echo=False to prevent verbose SQL logging in terminal
# For SQL debugging, use logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
)

# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
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


async def create_tables():
    """Create all tables (for development)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Apply column-level migrations for existing tables
        await _apply_column_migrations(conn)


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

        desired = {
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
        }

        for table_name, columns in desired.items():
            if not insp.has_table(table_name):
                continue
            existing = {col["name"] for col in insp.get_columns(table_name)}
            for col_name, spec in columns.items():
                if col_name not in existing:
                    migrations.append((table_name, col_name, spec))

        return migrations

    missing = await conn.run_sync(_get_missing_columns)

    if missing:
        # Detect dialect for boolean default syntax
        is_sqlite = "sqlite" in settings.database_url

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
            await conn.execute(text(stmt))


async def drop_tables():
    """Drop all tables (for testing)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
