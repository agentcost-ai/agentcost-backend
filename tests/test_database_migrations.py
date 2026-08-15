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


def test_no_table_is_listed_twice_in_the_migration_map():
    """A repeated table key silently discards the earlier entry's columns.

    Python keeps the last key in a dict literal, so the columns declared under
    the first occurrence never reach production -- and nothing fails, because
    tests build the schema from the models rather than migrating it. Detecting
    this needs the source, not the evaluated dict, since the duplicate is gone
    by the time the module is imported.
    """
    import ast
    import pathlib

    import app.database as db_module

    source = pathlib.Path(db_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    literal = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_DESIRED_COLUMNS" for t in node.targets
        ):
            literal = node.value
            break

    assert isinstance(literal, ast.Dict), "_DESIRED_COLUMNS must be a dict literal"

    keys = [k.value for k in literal.keys if isinstance(k, ast.Constant)]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    assert not duplicates, (
        f"_DESIRED_COLUMNS lists these tables more than once: {duplicates}. "
        "Merge them -- the earlier entry's columns are being discarded."
    )


async def test_every_model_column_is_reachable_on_an_existing_deployment():
    """Every column the models declare must be creatable on a live database.

    ``create_all()`` only creates *tables*. A column added to an existing table
    reaches production solely through ``_DESIRED_COLUMNS``, so a column present
    in the model and absent from that table exists in tests -- which always
    build a fresh schema -- and is missing in production, where every INSERT
    referencing it then fails. That is an ingest outage, not a degradation.

    This test is the one that fails when someone adds a column and forgets the
    migration entry.
    """
    import app.database as db_module

    # Tables that predate the migration table and are still evolving. A brand
    # new table is created wholesale by create_all() and needs no entry.
    LIVE_TABLES = ("events", "projects", "model_pricing", "trace_outcomes")

    missing: list[str] = []
    for table_name in LIVE_TABLES:
        table = db_module.Base.metadata.tables[table_name]
        declared = {c.name for c in table.columns}
        # Columns present when the table was first created need no ALTER.
        migratable = set(db_module._DESIRED_COLUMNS.get(table_name, {}))
        for column in sorted(declared - migratable):
            missing.append(f"{table_name}.{column}")

    # The baseline: columns that shipped with the original CREATE TABLE.
    ORIGINAL = {
        "events.id", "events.project_id", "events.agent_name", "events.model",
        "events.input_tokens", "events.output_tokens", "events.total_tokens",
        "events.cost", "events.latency_ms", "events.success", "events.error",
        "events.timestamp", "events.extra_data", "events.created_at",
        "projects.id", "projects.name", "projects.description", "projects.api_key",
        "projects.owner_id", "projects.is_active", "projects.created_at",
        "projects.updated_at", "projects.budget_enforcement_mode",
        "projects.budget_currency",
        "model_pricing.id", "model_pricing.model_name",
        "model_pricing.input_price_per_1k", "model_pricing.output_price_per_1k",
        "model_pricing.provider", "model_pricing.is_active", "model_pricing.notes",
        "model_pricing.created_at", "model_pricing.updated_at",
        "model_pricing.max_tokens", "model_pricing.max_input_tokens",
        "model_pricing.supports_vision", "model_pricing.supports_function_calling",
        "model_pricing.supports_streaming", "model_pricing.pricing_source",
        "model_pricing.source_updated_at",
        "trace_outcomes.id", "trace_outcomes.project_id", "trace_outcomes.trace_id",
        "trace_outcomes.success", "trace_outcomes.label",
        "trace_outcomes.recorded_at", "trace_outcomes.created_at",
    }

    unaccounted = sorted(set(missing) - ORIGINAL)
    assert not unaccounted, (
        "These columns exist in the models but would never be added to an "
        "existing production database. Add them to _DESIRED_COLUMNS in "
        f"app/database.py: {unaccounted}"
    )


async def test_widened_columns_cover_the_declared_trace_id_width():
    """A model that widens a column must widen it in the live schema too.

    SQLite ignores VARCHAR lengths, so a too-narrow production column is
    invisible to this suite. The check is therefore on the declaration, not on
    behaviour: every trace-id column the models widened must appear in
    _WIDEN_COLUMNS with the same target.
    """
    import app.database as db_module

    widen_targets = {(t, c): n for t, c, n in db_module._WIDEN_COLUMNS}

    for table_name, column_name in (
        ("events", "trace_id"),
        ("events", "span_id"),
        ("events", "parent_span_id"),
        ("trace_outcomes", "trace_id"),
    ):
        declared = db_module.Base.metadata.tables[table_name].columns[column_name]
        target = widen_targets.get((table_name, column_name))
        assert target == declared.type.length, (
            f"{table_name}.{column_name} is VARCHAR({declared.type.length}) in the "
            f"model but _WIDEN_COLUMNS says {target}. An existing deployment "
            f"would keep the old width and silently reject longer ids."
        )


async def test_new_indexes_are_declared_for_existing_tables():
    """create_all() only builds indexes for tables it creates."""
    import app.database as db_module

    declared = {
        index.name
        for index in db_module.Base.metadata.tables["events"].indexes
    }
    migrated = {name for name, _table, _cols in db_module._DESIRED_INDEXES}
    migrated |= {
        name for name, _table, _cols, _where in db_module._DESIRED_PARTIAL_UNIQUE_INDEXES
    }

    # idx_events_project_time and friends shipped with the original table.
    ORIGINAL_INDEXES = {
        "idx_events_project_time", "idx_events_agent",
        "idx_events_model", "idx_events_input_hash",
    }
    unaccounted = sorted(declared - migrated - ORIGINAL_INDEXES)
    assert not unaccounted, (
        "These indexes would never be created on an existing database. Add "
        f"them to _DESIRED_INDEXES in app/database.py: {unaccounted}"
    )


async def test_upgrading_a_legacy_events_table_adds_every_new_column():
    """End-to-end rehearsal of the deploy, against a pre-upgrade schema.

    Builds an ``events`` table shaped the way production's is *before* this
    release, runs the bootstrap over it, then writes a row using the new
    columns. Every other migration test starts from a schema create_all() just
    built with all the columns already present, so none of them can catch a
    missing _DESIRED_COLUMNS entry.
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    import app.database as db_module

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    try:
        async with engine.begin() as conn:
            # The events table as it exists in production today.
            await conn.execute(text(
                "CREATE TABLE events ("
                " id VARCHAR(36) PRIMARY KEY,"
                " project_id VARCHAR(36) NOT NULL,"
                " agent_name VARCHAR(255) NOT NULL,"
                " model VARCHAR(255) NOT NULL,"
                " input_tokens INTEGER NOT NULL,"
                " output_tokens INTEGER NOT NULL,"
                " total_tokens INTEGER NOT NULL,"
                " cost FLOAT NOT NULL,"
                " cost_source VARCHAR(50),"
                " latency_ms INTEGER NOT NULL,"
                " success BOOLEAN,"
                " error TEXT,"
                " timestamp TIMESTAMP NOT NULL,"
                " extra_data JSON,"
                " input_hash VARCHAR(64),"
                " trace_id VARCHAR(32),"
                " span_id VARCHAR(32),"
                " parent_span_id VARCHAR(32),"
                " workflow VARCHAR(255),"
                " step_name VARCHAR(255),"
                " step_index INTEGER,"
                " depth INTEGER,"
                " tool_name VARCHAR(255),"
                " created_at TIMESTAMP)"
            ))

            # A historical row whose metadata carries the dimensions that are
            # about to be promoted to columns.
            await conn.execute(text(
                "INSERT INTO events (id, project_id, agent_name, model,"
                " input_tokens, output_tokens, total_tokens, cost, latency_ms,"
                " timestamp, extra_data)"
                " VALUES ('legacy-1','p1','a','m',10,1,11,0.1,50,"
                " '2026-08-01T00:00:00',"
                " '{\"user_id\": \"alice\", \"session_id\": 7, \"other\": true}')"
            ))

            await db_module._apply_column_migrations(conn)
            await db_module._apply_index_migrations(conn)

            columns = {
                row[1]
                for row in await conn.execute(text("PRAGMA table_info(events)"))
            }
            indexes = {
                row[0]
                for row in await conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ))
            }
            backfilled = (await conn.execute(text(
                "SELECT user_id, session_id FROM events WHERE id='legacy-1'"
            ))).one()

        for column in (
            "cached_tokens", "cache_write_tokens", "streaming",
            "event_id", "user_id", "session_id",
        ):
            assert column in columns, (
                f"events.{column} was not added to a pre-existing table. "
                "Every INSERT naming it would fail in production."
            )

        assert "uq_events_project_event_id" in indexes, (
            "The idempotency unique index was not created on an existing "
            "table; concurrent replays would duplicate rows."
        )

        # Historical metadata was promoted, with the ingest path's coercion
        # rules: numbers become strings, booleans and objects are skipped.
        assert backfilled == ("alice", "7")

        # The columns are not merely present -- a write using them succeeds.
        async with engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO events (id, project_id, agent_name, model,"
                " input_tokens, output_tokens, total_tokens, cost, latency_ms,"
                " timestamp, cached_tokens, cache_write_tokens, streaming,"
                " event_id, user_id, session_id)"
                " VALUES ('e1','p1','a','m',100,10,110,0.5,200,"
                " '2026-08-13T00:00:00', 90, 5, 1, 'idem-1', 'alice', 's1')"
            ))
            stored = (await conn.execute(
                text("SELECT cached_tokens, user_id FROM events WHERE id='e1'")
            )).one()
        assert stored == (90, "alice")
    finally:
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
