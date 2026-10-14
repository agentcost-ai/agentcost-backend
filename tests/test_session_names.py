"""
Session names: what a coding agent's chat is called, shared only when the
client opts in, and shown wherever a session or run is listed.
"""

from datetime import datetime, timezone

from sqlalchemy import select

from app.models.db_models import SessionName

SESSION = "0532f9c4-a022-4e98-a543-d8e17c5b90a6"
OTHER = "9f4752e8-f7ba-46ee-8009-a61e75031214"


def _event(session, cost):
    return {
        "agent_name": "claude-code",
        "model": "unpriced-model",
        "input_tokens": 1000,
        "output_tokens": 100,
        "cost": cost,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": session,
        "workflow": "claude-code",
        "metadata": {"session_id": session},
    }


async def _post(client, project, **body):
    return await client.post("/v1/events/batch", json={"project_id": project.id, **body})


async def test_names_arrive_on_their_own_and_the_last_one_wins(
    client, test_project, test_session
):
    first = await _post(
        client, test_project, sessions=[{"session_id": SESSION, "name": "Fix the  billing\nbug"}]
    )
    again = await _post(
        client, test_project, sessions=[{"session_id": SESSION, "name": "Fix the billing export"}]
    )

    assert first.status_code == 200 and first.json()["sessions_named"] == 1
    assert first.json()["events_stored"] == 0
    assert again.status_code == 200
    rows = (await test_session.execute(select(SessionName))).scalars().all()
    assert [(row.session_id, row.name) for row in rows] == [(SESSION, "Fix the billing export")]


async def test_a_name_is_one_tidy_line_of_bounded_length(client, test_project, test_session):
    await _post(
        client, test_project, sessions=[{"session_id": SESSION, "name": "  a\n\tb  " + "x" * 400}]
    )

    name = (await test_session.execute(select(SessionName.name))).scalar_one()
    assert name.startswith("a b x") and len(name) == 255


async def test_sessions_and_runs_are_listed_with_their_names(client, test_project):
    await _post(
        client,
        test_project,
        events=[_event(SESSION, 4.0), _event(SESSION, 1.0), _event(OTHER, 2.0)],
        sessions=[{"session_id": SESSION, "name": "Fix the billing export"}],
    )

    by_session = (await client.get("/v1/analytics/by/session?range=24h")).json()
    assert [(row["key"], row["name"], row["total_cost"]) for row in by_session] == [
        (SESSION, "Fix the billing export", 5.0),
        (OTHER, None, 2.0),
    ]

    traces = (await client.get("/v1/analytics/traces?range=24h")).json()
    assert {trace["trace_id"]: trace["name"] for trace in traces} == {
        SESSION: "Fix the billing export",
        OTHER: None,
    }

    detail = (await client.get(f"/v1/analytics/traces/{SESSION}")).json()
    assert detail["name"] == "Fix the billing export"
    assert (await client.get(f"/v1/analytics/traces/{OTHER}")).json()["name"] is None


async def test_other_dimensions_carry_no_name(client, test_project):
    await _post(client, test_project, events=[_event(SESSION, 1.0)])

    by_agent = (await client.get("/v1/analytics/by/agent?range=24h")).json()
    assert by_agent[0]["name"] is None


async def test_a_batch_still_needs_something_in_it(client, test_project):
    assert (await _post(client, test_project, sessions=[])).status_code == 422
    assert (
        await _post(client, test_project, sessions=[{"session_id": SESSION, "name": ""}])
    ).status_code == 422
