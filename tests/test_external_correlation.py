"""
Correlating a run with an external control plane.

Covers the three things that had to change for another system -- a policy
layer, an orchestrator -- to join its own records to AgentCost's cost data
using a run id it minted:

  * trace ids wide enough for a UUID (they were capped at 32 chars, so every
    event of a correlated run was silently rejected),
  * outcomes accepted without any events attached (a run ended by a denial
    produces no LLM call, but that is the most important thing to record),
  * idempotency, so a retried delivery is a no-op rather than a duplicate.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

from app.models.db_models import Event, TraceOutcome


def _event(**overrides):
    base = {
        "agent_name": "coding-agent",
        "model": "gpt-4",
        "input_tokens": 100,
        "output_tokens": 50,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    base.update(overrides)
    return base


class TestForeignTraceIds:
    async def test_canonical_uuid_is_accepted(self, client, test_project, test_session):
        """36 chars with dashes -- the shape most systems mint."""
        run_id = str(uuid.uuid4())
        assert len(run_id) == 36

        response = await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(trace_id=run_id, workflow="refactor-run")],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["events_stored"] == 1
        assert response.json()["events_rejected"] == 0

        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.trace_id == run_id

    async def test_trace_detail_is_retrievable_by_the_foreign_id(self, client, test_project):
        run_id = str(uuid.uuid4())
        await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(trace_id=run_id, workflow="refactor-run", step_name="plan")],
            },
        )

        detail = await client.get(f"/v1/analytics/traces/{run_id}")
        assert detail.status_code == 200, detail.text
        assert detail.json()

    async def test_id_longer_than_the_column_is_rejected_not_truncated(
        self, client, test_project
    ):
        """Truncating would silently merge two distinct runs into one trace."""
        response = await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [_event(trace_id="x" * 65)]},
        )
        assert response.status_code == 200
        assert response.json()["events_stored"] == 0
        assert response.json()["events_rejected"] == 1


class TestOutcomeOnlyBatches:
    async def test_outcome_with_no_events_is_recorded(self, client, test_project, test_session):
        """A run killed by a policy denial has an outcome but no LLM calls."""
        run_id = str(uuid.uuid4())
        response = await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [],
                "outcomes": [
                    {
                        "trace_id": run_id,
                        "workflow": "refactor-run",
                        "success": False,
                        "label": "denied:postgres.query",
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["outcomes_recorded"] == 1

        row = (await test_session.execute(select(TraceOutcome))).scalars().one()
        assert row.trace_id == run_id
        assert row.success is False
        assert row.label == "denied:postgres.query"

    async def test_outcomes_field_may_be_the_only_key(self, client, test_project):
        """Omitting `events` entirely, not just sending an empty list."""
        response = await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "outcomes": [{"trace_id": str(uuid.uuid4()), "success": True}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["outcomes_recorded"] == 1

    async def test_outcomes_survive_a_batch_of_entirely_invalid_events(
        self, client, test_project, test_session
    ):
        """The route used to return before persisting outcomes in this case."""
        run_id = str(uuid.uuid4())
        response = await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                # Missing `model` and `timestamp`: dropped per-event.
                "events": [{"agent_name": "x", "input_tokens": 1, "output_tokens": 1}],
                "outcomes": [{"trace_id": run_id, "success": False, "label": "denied"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["events_stored"] == 0
        assert response.json()["events_rejected"] == 1
        assert response.json()["outcomes_recorded"] == 1

        row = (await test_session.execute(select(TraceOutcome))).scalars().one()
        assert row.trace_id == run_id

    async def test_a_wholly_empty_batch_is_still_an_error(self, client, test_project):
        response = await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [], "outcomes": []},
        )
        assert response.status_code == 422

    async def test_outcome_resend_updates_rather_than_duplicates(
        self, client, test_project, test_session
    ):
        run_id = str(uuid.uuid4())
        for success in (True, False):
            await client.post(
                "/v1/events/batch",
                json={
                    "project_id": test_project.id,
                    "outcomes": [
                        {"trace_id": run_id, "success": success, "label": "final" if not success else "optimistic"}
                    ],
                },
            )

        rows = (await test_session.execute(select(TraceOutcome))).scalars().all()
        assert len(rows) == 1
        assert rows[0].success is False
        assert rows[0].label == "final"


class TestIdempotency:
    async def test_replayed_event_id_is_not_stored_twice(
        self, client, test_project, test_session
    ):
        payload = {
            "project_id": test_project.id,
            "events": [_event(event_id="delivery-1")],
        }

        first = await client.post("/v1/events/batch", json=payload)
        second = await client.post("/v1/events/batch", json=payload)

        assert first.json()["events_stored"] == 1
        assert second.json()["events_stored"] == 0
        assert second.json()["events_duplicate"] == 1

        count = (await test_session.execute(select(func.count(Event.id)))).scalar()
        assert count == 1

    async def test_duplicates_within_one_batch_are_collapsed(
        self, client, test_project, test_session
    ):
        response = await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(event_id="same"), _event(event_id="same")],
            },
        )
        assert response.json()["events_stored"] == 1
        assert response.json()["events_duplicate"] == 1

    async def test_events_without_an_id_are_never_deduplicated(
        self, client, test_project, test_session
    ):
        """Two genuinely identical calls are two calls unless told otherwise."""
        payload = {"project_id": test_project.id, "events": [_event()]}
        await client.post("/v1/events/batch", json=payload)
        await client.post("/v1/events/batch", json=payload)

        count = (await test_session.execute(select(func.count(Event.id)))).scalar()
        assert count == 2

    async def test_a_duplicate_landing_mid_flight_is_dropped_not_stored(
        self, test_project, test_session
    ):
        """The race the lookup dedup cannot see.

        A concurrent delivery commits the same event_id between this request's
        duplicate lookup and its insert. The partial unique index turns that
        into an IntegrityError, and persist retries with the survivors.
        """
        from app.models.schemas import EventCreate
        from app.services.event_service import EventService

        service = EventService(test_session)
        prepared = await service.prepare_events_batch(
            project_id=test_project.id,
            events=[
                EventCreate(**_event(event_id="race-1")),
                EventCreate(**_event(event_id="race-2")),
            ],
        )

        # The concurrent request's row lands after prepare, before persist.
        test_session.add(
            Event(
                project_id=test_project.id,
                agent_name="concurrent-sender",
                model="gpt-4",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cost=0.0,
                latency_ms=0,
                success=True,
                timestamp=datetime.now(timezone.utc),
                event_id="race-1",
            )
        )
        await test_session.flush()

        stored = await service.persist_events_batch(prepared)
        await test_session.commit()

        assert stored == 1
        assert prepared.duplicates == 1

        rows = (
            await test_session.execute(
                select(Event.event_id).where(Event.event_id.isnot(None))
            )
        ).scalars().all()
        assert sorted(rows) == ["race-1", "race-2"]
