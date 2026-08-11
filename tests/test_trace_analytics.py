"""
Tests for trace analytics: cost attributed to the shape of a run.
"""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient


def _event(**overrides):
    """A minimal valid event; overrides carry the trace structure."""
    base = {
        "agent_name": "researcher",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.01,
        "latency_ms": 200,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    base.update(overrides)
    return base


def _batch(project_id, events):
    return {"project_id": project_id, "events": events}


@pytest.fixture
def traced_run(test_project):
    """
    One run of "triage": a classify step, a tool that made two calls, and an
    answer step. The tool repeats an identical call, which is the loop signal.
    """
    return _batch(
        test_project.id,
        [
            _event(
                trace_id="trace1", span_id="s1", workflow="triage",
                step_name="classify", step_index=0, depth=1,
                input_hash="hash_a", cost=0.01,
            ),
            _event(
                trace_id="trace1", span_id="s2", parent_span_id="s1",
                workflow="triage", step_name="search", tool_name="web_search",
                step_index=1, depth=2, input_hash="hash_b", cost=0.02,
            ),
            _event(
                trace_id="trace1", span_id="s3", parent_span_id="s1",
                workflow="triage", step_name="search", tool_name="web_search",
                step_index=2, depth=2, input_hash="hash_b", cost=0.02,
            ),
            _event(
                trace_id="trace1", span_id="s4", workflow="triage",
                step_name="answer", step_index=3, depth=1,
                input_hash="hash_c", cost=0.05,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_workflow_stats_empty(client: AsyncClient):
    response = await client.get("/v1/analytics/workflows")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_trace_fields_survive_ingestion(client: AsyncClient, traced_run):
    """The columns must persist, not just validate."""
    ingest = await client.post("/v1/events/batch", json=traced_run)
    assert ingest.status_code == 200

    response = await client.get("/v1/analytics/traces/trace1")
    assert response.status_code == 200
    detail = response.json()

    assert detail["workflow"] == "triage"
    assert detail["total_calls"] == 4
    assert detail["max_depth"] == 2
    assert detail["total_cost"] == pytest.approx(0.10, rel=1e-3)

    spans = {s["span_id"]: s for s in detail["spans"]}
    assert spans["s2"]["parent_span_id"] == "s1"
    assert spans["s2"]["tool_name"] == "web_search"
    assert spans["s1"]["tool_name"] is None


@pytest.mark.asyncio
async def test_workflow_stats_aggregate_per_run(client: AsyncClient, traced_run):
    await client.post("/v1/events/batch", json=traced_run)

    response = await client.get("/v1/analytics/workflows")
    assert response.status_code == 200
    rows = response.json()

    assert len(rows) == 1
    triage = rows[0]
    assert triage["workflow"] == "triage"
    assert triage["runs"] == 1
    assert triage["total_calls"] == 4
    assert triage["total_cost"] == pytest.approx(0.10, rel=1e-3)
    # One run, so the per-run average equals the total.
    assert triage["avg_cost_per_run"] == pytest.approx(0.10, rel=1e-3)
    assert triage["max_depth"] == 2


@pytest.mark.asyncio
async def test_avg_cost_per_run_divides_by_runs_not_calls(
    client: AsyncClient, test_project
):
    """The distinction the two-stage aggregation exists to preserve."""
    events = [
        _event(trace_id="t1", workflow="w", step_name="a", cost=0.02),
        _event(trace_id="t1", workflow="w", step_name="b", cost=0.02),
        _event(trace_id="t2", workflow="w", step_name="a", cost=0.02),
        _event(trace_id="t2", workflow="w", step_name="b", cost=0.02),
    ]
    await client.post("/v1/events/batch", json=_batch(test_project.id, events))

    rows = (await client.get("/v1/analytics/workflows")).json()
    assert rows[0]["runs"] == 2
    assert rows[0]["total_cost"] == pytest.approx(0.08, rel=1e-3)
    # 0.08 over 2 runs, not over 4 calls.
    assert rows[0]["avg_cost_per_run"] == pytest.approx(0.04, rel=1e-3)


@pytest.mark.asyncio
async def test_step_stats_expose_calls_per_run(client: AsyncClient, traced_run):
    await client.post("/v1/events/batch", json=traced_run)

    rows = (await client.get("/v1/analytics/workflows/steps")).json()
    steps = {r["step_name"]: r for r in rows}

    assert steps["search"]["calls"] == 2
    assert steps["search"]["runs"] == 1
    # Twice in one run is the retry/loop signature.
    assert steps["search"]["calls_per_run"] == pytest.approx(2.0)
    assert steps["classify"]["calls_per_run"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_step_stats_filter_by_workflow(client: AsyncClient, test_project):
    events = [
        _event(trace_id="t1", workflow="alpha", step_name="a"),
        _event(trace_id="t2", workflow="beta", step_name="b"),
    ]
    await client.post("/v1/events/batch", json=_batch(test_project.id, events))

    rows = (
        await client.get("/v1/analytics/workflows/steps", params={"workflow": "alpha"})
    ).json()
    assert [r["step_name"] for r in rows] == ["a"]


@pytest.mark.asyncio
async def test_tool_stats_attribute_spend_to_the_tool(
    client: AsyncClient, traced_run
):
    await client.post("/v1/events/batch", json=traced_run)

    rows = (await client.get("/v1/analytics/workflows/tools")).json()
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "web_search"
    assert rows[0]["calls"] == 2
    assert rows[0]["total_cost"] == pytest.approx(0.04, rel=1e-3)


@pytest.mark.asyncio
async def test_repeated_work_charges_only_the_redundant_repeats(
    client: AsyncClient, traced_run
):
    await client.post("/v1/events/batch", json=traced_run)

    rows = (await client.get("/v1/analytics/workflows/repeated-work")).json()
    assert len(rows) == 1

    finding = rows[0]
    assert finding["trace_id"] == "trace1"
    assert finding["step_name"] == "search"
    assert finding["occurrences"] == 2
    assert finding["spend"] == pytest.approx(0.04, rel=1e-3)
    # A correct run pays for one of the two.
    assert finding["wasted_cost"] == pytest.approx(0.02, rel=1e-3)


@pytest.mark.asyncio
async def test_same_hash_across_different_runs_is_not_a_loop(
    client: AsyncClient, test_project
):
    """Cross-run duplication argues for a cache; only within-run is a loop."""
    events = [
        _event(trace_id="t1", workflow="w", step_name="a", input_hash="same"),
        _event(trace_id="t2", workflow="w", step_name="a", input_hash="same"),
    ]
    await client.post("/v1/events/batch", json=_batch(test_project.id, events))

    rows = (await client.get("/v1/analytics/workflows/repeated-work")).json()
    assert rows == []


@pytest.mark.asyncio
async def test_untraced_events_are_excluded_not_blended(
    client: AsyncClient, test_project
):
    """Folding untraced spend into a workflow would misreport its cost."""
    events = [
        _event(trace_id="t1", workflow="w", step_name="a", cost=0.01),
        _event(cost=99.0),  # no trace structure at all
    ]
    await client.post("/v1/events/batch", json=_batch(test_project.id, events))

    rows = (await client.get("/v1/analytics/workflows")).json()
    assert len(rows) == 1
    assert rows[0]["total_cost"] == pytest.approx(0.01, rel=1e-3)

    # ...but the untraced event still exists in the ordinary analytics.
    overview = (await client.get("/v1/analytics/overview")).json()
    assert overview["total_calls"] == 2


@pytest.mark.asyncio
async def test_events_without_trace_fields_still_ingest(
    client: AsyncClient, test_project
):
    """Older SDKs must keep working untouched."""
    response = await client.post(
        "/v1/events/batch", json=_batch(test_project.id, [_event()])
    )
    assert response.status_code == 200
    assert response.json().get("events_rejected", 0) == 0


@pytest.mark.asyncio
async def test_list_traces_orders_by_cost(client: AsyncClient, test_project):
    events = [
        _event(trace_id="cheap", workflow="w", step_name="a", cost=0.01),
        _event(trace_id="pricey", workflow="w", step_name="a", cost=0.50),
    ]
    await client.post("/v1/events/batch", json=_batch(test_project.id, events))

    rows = (await client.get("/v1/analytics/traces")).json()
    assert [r["trace_id"] for r in rows] == ["pricey", "cheap"]


@pytest.mark.asyncio
async def test_unknown_trace_returns_404(client: AsyncClient):
    response = await client.get("/v1/analytics/traces/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_trace_detail_is_scoped_to_the_project(
    client: AsyncClient, test_session, test_project
):
    """A trace id from another project must not be readable."""
    import uuid

    from app.models.db_models import Event, Project
    from app.utils.auth import hash_api_key

    other_project = Project(
        id=str(uuid.uuid4()),
        name="Someone Else's Project",
        api_key=hash_api_key("sk_other_key"),
        created_at=datetime.now(timezone.utc),
    )
    test_session.add(other_project)
    await test_session.flush()

    test_session.add(
        Event(
            id="foreign-event",
            project_id=other_project.id,
            agent_name="a",
            model="gpt-4o",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cost=1.0,
            latency_ms=1,
            timestamp=datetime.now(timezone.utc),
            trace_id="foreign-trace",
            workflow="theirs",
        )
    )
    await test_session.commit()

    response = await client.get("/v1/analytics/traces/foreign-trace")
    assert response.status_code == 404


# ── Run-cost distribution ─────────────────────────────────────────────────


async def _seed_runs(client, project_id, costs, workflow="w"):
    """One single-call run per cost, so run cost == the given cost."""
    events = [
        _event(trace_id=f"t{i}", workflow=workflow, step_name="s", cost=c)
        for i, c in enumerate(costs)
    ]
    # Batches cap at 1000 events; chunk so large samples still ingest.
    for start in range(0, len(events), 500):
        await client.post(
            "/v1/events/batch", json=_batch(project_id, events[start : start + 500])
        )


@pytest.mark.asyncio
async def test_distribution_is_empty_without_data(client: AsyncClient):
    response = await client.get("/v1/analytics/workflows/distribution")
    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio
async def test_percentiles_are_exact_over_every_run(client: AsyncClient, test_project):
    """1..100 makes each percentile checkable by hand."""
    await _seed_runs(client, test_project.id, [i / 100 for i in range(1, 101)])

    d = (await client.get("/v1/analytics/workflows/distribution")).json()

    assert d["runs"] == 100
    assert d["min"] == pytest.approx(0.01, abs=1e-6)
    assert d["max"] == pytest.approx(1.00, abs=1e-6)
    assert d["p50"] == pytest.approx(0.50, abs=0.011)
    assert d["p90"] == pytest.approx(0.90, abs=0.011)
    assert d["p95"] == pytest.approx(0.95, abs=0.011)
    assert d["p99"] == pytest.approx(0.99, abs=0.011)


@pytest.mark.asyncio
async def test_tail_share_quantifies_the_expensive_runs(
    client: AsyncClient, test_project
):
    """
    The headline claim: 95 cheap runs and 5 costly ones, where the costly 5
    hold most of the spend. A mean would hide this entirely.
    """
    costs = [0.01] * 95 + [1.00] * 5
    await _seed_runs(client, test_project.id, costs)

    d = (await client.get("/v1/analytics/workflows/distribution")).json()

    # 5.00 of 5.95 total.
    assert d["tail_runs"] == 5
    assert d["tail_share_percent"] == pytest.approx(84.0, abs=1.0)
    assert d["p50"] == pytest.approx(0.01, abs=1e-6)
    assert d["tail_ratio"] == pytest.approx(100.0, abs=0.1)


@pytest.mark.asyncio
async def test_run_cost_sums_every_call_in_the_run(client: AsyncClient, test_project):
    """A run is the unit, not a call — three calls of 0.01 is one 0.03 run."""
    events = [
        _event(trace_id="multi", workflow="w", step_name=f"s{i}", cost=0.01)
        for i in range(3)
    ]
    await client.post("/v1/events/batch", json=_batch(test_project.id, events))

    d = (await client.get("/v1/analytics/workflows/distribution")).json()
    assert d["runs"] == 1
    assert d["max"] == pytest.approx(0.03, abs=1e-6)


@pytest.mark.asyncio
async def test_histogram_counts_every_run_exactly_once(
    client: AsyncClient, test_project
):
    await _seed_runs(client, test_project.id, [i / 100 for i in range(1, 51)])

    d = (await client.get("/v1/analytics/workflows/distribution")).json()
    # Body buckets plus the single overflow bucket must still account for
    # every run exactly once.
    assert sum(b["count"] for b in d["histogram"]) == d["runs"] == 50


@pytest.mark.asyncio
async def test_identical_runs_collapse_to_one_bucket(client: AsyncClient, test_project):
    """Zero range must not divide by zero."""
    await _seed_runs(client, test_project.id, [0.05] * 10)

    d = (await client.get("/v1/analytics/workflows/distribution")).json()
    assert len(d["histogram"]) == 1
    assert d["histogram"][0]["count"] == 10
    assert d["tail_ratio"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_tail_buckets_are_marked_server_side(client: AsyncClient, test_project):
    await _seed_runs(client, test_project.id, [0.01] * 95 + [1.00] * 5)

    d = (await client.get("/v1/analytics/workflows/distribution")).json()
    tail = [b for b in d["histogram"] if b["is_tail"]]
    # The tail is exactly one overflow bucket, holding exactly the tail runs.
    assert len(tail) == 1
    assert tail[0]["count"] == d["tail_runs"]
    assert tail[0]["lower"] == pytest.approx(d["tail_threshold"], abs=1e-6)
    assert len(tail) < len(d["histogram"])


@pytest.mark.asyncio
async def test_distribution_scopes_to_one_workflow(client: AsyncClient, test_project):
    await _seed_runs(client, test_project.id, [0.01] * 10, workflow="cheap")
    await _seed_runs(client, test_project.id, [5.00] * 4, workflow="pricey")

    # No workflow given: defaults to the highest-spend one.
    default = (await client.get("/v1/analytics/workflows/distribution")).json()
    assert default["workflow"] == "pricey"
    assert default["runs"] == 4

    explicit = (
        await client.get(
            "/v1/analytics/workflows/distribution", params={"workflow": "cheap"}
        )
    ).json()
    assert explicit["workflow"] == "cheap"
    assert explicit["runs"] == 10


@pytest.mark.asyncio
async def test_untraced_events_never_enter_the_distribution(
    client: AsyncClient, test_project
):
    await _seed_runs(client, test_project.id, [0.01] * 5)
    await client.post(
        "/v1/events/batch", json=_batch(test_project.id, [_event(cost=99.0)])
    )

    d = (await client.get("/v1/analytics/workflows/distribution")).json()
    assert d["runs"] == 5
    assert d["max"] == pytest.approx(0.01, abs=1e-6)


@pytest.mark.asyncio
async def test_body_buckets_are_not_crushed_by_the_tail(
    client: AsyncClient, test_project
):
    """
    The reason the axis stops at the tail threshold.

    With 95 runs between $0.01-$0.02 and 5 at $1.00, bucketing across the full
    range would pile every ordinary run into the first bar or two. The body
    must stay spread across the buckets instead.
    """
    body = [0.01 + (i % 20) * 0.0005 for i in range(95)]
    await _seed_runs(client, test_project.id, body + [1.00] * 5)

    d = (await client.get("/v1/analytics/workflows/distribution")).json()
    body_buckets = [b for b in d["histogram"] if not b["is_tail"]]
    occupied = [b for b in body_buckets if b["count"] > 0]

    # Spread over many bars, not crammed into one or two.
    assert len(occupied) >= 6, f"body occupies only {len(occupied)} buckets"
    # And the body axis stops before the outliers.
    assert max(b["upper"] for b in body_buckets) < 0.5


@pytest.mark.asyncio
async def test_events_endpoint_returns_trace_fields(client: AsyncClient, traced_run):
    """The Events page cannot show what the endpoint does not send."""
    await client.post("/v1/events/batch", json=traced_run)

    events = (await client.get("/v1/events")).json()
    tool_call = next(e for e in events if e["tool_name"])

    assert tool_call["workflow"] == "triage"
    assert tool_call["step_name"] == "search"
    assert tool_call["trace_id"] == "trace1"


@pytest.mark.asyncio
async def test_untraced_events_return_null_trace_fields(
    client: AsyncClient, test_project
):
    await client.post("/v1/events/batch", json=_batch(test_project.id, [_event()]))

    event = (await client.get("/v1/events")).json()[0]
    assert event["trace_id"] is None
    assert event["workflow"] is None
