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
