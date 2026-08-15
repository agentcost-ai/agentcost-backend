"""
PostgreSQL rehearsal of the schema migrations.

The SQLite suite cannot execute _WIDEN_COLUMNS — ALTER COLUMN TYPE only runs
on PostgreSQL — so without this file the widening path ships unverified.

Gated on TEST_PG_URL (an asyncpg URL to a disposable database, e.g. the
container that scripts/rehearse_migrations_pg.py starts). Everything here may
DROP and recreate tables in that database; never point it at real data.
"""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

PG_URL = os.environ.get("TEST_PG_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="TEST_PG_URL not set; run scripts/rehearse_migrations_pg.py"
)


async def test_full_upgrade_of_a_pre_release_schema():
    """Build the pre-release events table on PostgreSQL, upgrade it, use it."""
    import app.database as db_module

    engine = create_async_engine(PG_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS events CASCADE"))
            # The events table as production had it before this release:
            # 32-char trace ids, none of the cache/dimension/idempotency columns.
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
            await conn.execute(text(
                "INSERT INTO events (id, project_id, agent_name, model,"
                " input_tokens, output_tokens, total_tokens, cost, latency_ms,"
                " timestamp, extra_data)"
                " VALUES ('legacy-1','p1','a','m',10,1,11,0.1,50,"
                " '2026-08-01T00:00:00',"
                " '{\"user_id\": \"alice\", \"session_id\": 7}')"
            ))

            await db_module._apply_column_migrations(conn)
            await db_module._apply_index_migrations(conn)

        async with engine.begin() as conn:
            widths = {
                row[0]: row[1]
                for row in await conn.execute(text(
                    "SELECT column_name, character_maximum_length"
                    " FROM information_schema.columns"
                    " WHERE table_name = 'events'"
                ))
            }
            indexes = {
                row[0]
                for row in await conn.execute(text(
                    "SELECT indexname FROM pg_indexes WHERE tablename = 'events'"
                ))
            }
            backfilled = (await conn.execute(text(
                "SELECT user_id, session_id FROM events WHERE id='legacy-1'"
            ))).one()

            # The path SQLite can never exercise: real ALTER COLUMN TYPE.
            assert widths["trace_id"] == 64
            assert widths["span_id"] == 64
            assert widths["parent_span_id"] == 64

            for column in (
                "cached_tokens", "cache_write_tokens", "streaming",
                "event_id", "user_id", "session_id",
            ):
                assert column in widths, f"events.{column} missing after upgrade"

            assert "uq_events_project_event_id" in indexes
            assert backfilled == ("alice", "7")

            # A 36-char foreign run id and the new columns are writable.
            await conn.execute(text(
                "INSERT INTO events (id, project_id, agent_name, model,"
                " input_tokens, output_tokens, total_tokens, cost, latency_ms,"
                " timestamp, trace_id, cached_tokens, event_id, user_id)"
                " VALUES ('e-new','p1','a','m',100,10,110,0.5,200,"
                " '2026-08-15T00:00:00',"
                " '0532f9c4-a022-4e98-a543-d8e17c5b90a6', 90, 'idem-1', 'bob')"
            ))

            # The unique index actually enforces: a replayed event_id fails.
            with pytest.raises(Exception):
                async with conn.begin_nested():
                    await conn.execute(text(
                        "INSERT INTO events (id, project_id, agent_name, model,"
                        " input_tokens, output_tokens, total_tokens, cost,"
                        " latency_ms, timestamp, event_id)"
                        " VALUES ('e-dup','p1','a','m',1,1,2,0.0,0,"
                        " '2026-08-15T00:00:00', 'idem-1')"
                    ))

            await conn.execute(text("DROP TABLE events CASCADE"))
    finally:
        await engine.dispose()
