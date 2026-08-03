"""
Tests for schema-bootstrap resilience and the UTC session pinning.

Workers boot concurrently and all of them run create_all + ALTER TABLE; a loser
used to die on the duplicate object and take its startup with it.
"""

from sqlalchemy import inspect, text

from app.database import _apply_column_migrations, _is_duplicate_column, _utc_connect_args


def test_postgres_connections_are_pinned_to_utc():
    asyncpg = _utc_connect_args("postgresql+asyncpg://u:p@host/db")
    assert asyncpg["server_settings"]["timezone"] == "UTC"

    # SQLite has no session timezone to pin.
    assert _utc_connect_args("sqlite+aiosqlite:///./agentcost.db") == {}


class _Orig(Exception):
    def __init__(self, message, sqlstate=None):
        super().__init__(message)
        self.sqlstate = sqlstate


class _Wrapped(Exception):
    def __init__(self, orig):
        super().__init__(str(orig))
        self.orig = orig


def test_duplicate_column_detection():
    assert _is_duplicate_column(_Wrapped(_Orig('column "x" of relation "y" already exists', "42701")))
    assert _is_duplicate_column(_Wrapped(_Orig("duplicate column name: is_internal")))
    assert not _is_duplicate_column(_Wrapped(_Orig("no such table: events", "42P01")))


class _HidesCostSource:
    """Inspector that reports events.cost_source as missing when it is present.

    Stands in for the real race: another worker adds the column between this
    one's introspection and its ALTER.
    """

    def __init__(self, real):
        self._real = real

    def has_table(self, name: str) -> bool:
        return name == "events" and self._real.has_table(name)

    def get_columns(self, name: str) -> list:
        return [c for c in self._real.get_columns(name) if c["name"] != "cost_source"]


async def test_duplicate_alter_does_not_abort_the_migration(test_engine, monkeypatch):
    """A column added by another worker must be skipped, not kill startup."""
    monkeypatch.setattr(
        "app.database.inspect", lambda conn: _HidesCostSource(inspect(conn))
    )

    async with test_engine.begin() as conn:
        # The ALTER this issues fails as a duplicate; it must be swallowed and
        # the surrounding transaction must stay usable.
        await _apply_column_migrations(conn)

        columns = await conn.execute(text("PRAGMA table_info(events)"))
        assert [row[1] for row in columns].count("cost_source") == 1


async def test_column_migrations_are_idempotent(test_engine):
    """Re-running the bootstrap on an up-to-date schema is a no-op."""
    async with test_engine.begin() as conn:
        await _apply_column_migrations(conn)
        await _apply_column_migrations(conn)

        columns = await conn.execute(text("PRAGMA table_info(events)"))
        names = [row[1] for row in columns]
        assert names.count("cost_source") == 1
        assert names.count("input_hash") == 1


async def test_first_boot_creates_then_second_boot_skips_the_work():
    """The bootstrap cost 87% of a cold start by re-introspecting every table on
    every boot only to find nothing to do."""
    import app.database as db_module
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    original = db_module.engine
    db_module.engine = engine
    try:
        first = await db_module.create_tables()
        assert "created" in first, first

        second = await db_module.create_tables()
        assert second == "schema already matches models; no changes needed"

        # The tables really are there -- the skip is a cache hit, not a no-op boot.
        async with engine.begin() as conn:
            tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
        assert "users" in tables and "model_pricing" in tables
    finally:
        db_module.engine = original
        await engine.dispose()


async def test_changing_the_models_invalidates_the_bootstrap_cache():
    """The cache must never let a schema change go unapplied."""
    import app.database as db_module
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    original = db_module.engine
    original_desired = db_module._DESIRED_COLUMNS
    db_module.engine = engine
    try:
        await db_module.create_tables()
        assert await db_module.create_tables() == (
            "schema already matches models; no changes needed"
        )

        before = db_module._schema_fingerprint()
        db_module._DESIRED_COLUMNS = {
            **original_desired,
            "users": {**original_desired["users"],
                      "brand_new_column": {"type": "VARCHAR(10)"}},
        }
        assert db_module._schema_fingerprint() != before, "fingerprint must move"

        summary = await db_module.create_tables()
        assert "column(s) added" in summary

        async with engine.begin() as conn:
            cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("users")}
            )
        assert "brand_new_column" in cols
    finally:
        db_module._DESIRED_COLUMNS = original_desired
        db_module.engine = original
        await engine.dispose()
