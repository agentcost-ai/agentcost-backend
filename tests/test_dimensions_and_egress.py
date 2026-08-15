"""
Cost dimensions, budget state and machine-readable egress.

Three capabilities a consumer outside the dashboard needs:
  * grouping cost by who incurred it, not only by agent or model,
  * a pollable budget position that costs nothing to read,
  * Prometheus exposition so cost lands on the same dashboards as everything else.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.models.db_models import Event
from app.services.webhook_service import sign_payload


def _event(**overrides):
    base = {
        "agent_name": "coding-agent",
        "model": "gpt-4",
        "input_tokens": 1000,
        "output_tokens": 100,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
    }
    base.update(overrides)
    return base


@pytest.fixture
async def tagged_events(client, test_project):
    """Two developers with unequal spend, plus one untagged call.

    Costs are stated explicitly rather than derived: the test catalogue has no
    row for this model, so server-side repricing yields 0.0 for every event and
    any ordering assertion would be testing nothing.
    """
    await client.post(
        "/v1/events/batch",
        json={
            "project_id": test_project.id,
            "events": [
                _event(metadata={"user_id": "alice", "session_id": "s1"}, cost=0.40),
                _event(metadata={"user_id": "alice", "session_id": "s1"}, cost=0.40),
                _event(metadata={"user_id": "bob", "session_id": "s2"}, cost=0.10),
                _event(cost=0.99),  # untagged, and the most expensive
            ],
        },
    )


class TestDimensionPromotion:
    async def test_user_and_session_are_promoted_to_columns(
        self, client, test_project, test_session
    ):
        await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(metadata={"user_id": "alice", "session_id": "s1"})],
            },
        )
        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.user_id == "alice"
        assert row.session_id == "s1"
        # Still present in the raw metadata; promotion copies, it does not move.
        assert row.extra_data["user_id"] == "alice"

    async def test_non_string_ids_are_coerced(self, client, test_project, test_session):
        """An integer user id must not split one person across two buckets."""
        await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [_event(metadata={"user_id": 42})]},
        )
        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.user_id == "42"

    async def test_structured_metadata_values_are_ignored(
        self, client, test_project, test_session
    ):
        await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(metadata={"user_id": {"nested": "object"}})],
            },
        )
        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.user_id is None

    async def test_missing_metadata_leaves_dimensions_null(
        self, client, test_project, test_session
    ):
        await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [_event()]},
        )
        row = (await test_session.execute(select(Event))).scalars().one()
        assert row.user_id is None


class TestDimensionAnalytics:
    async def test_cost_per_user(self, client, tagged_events):
        response = await client.get("/v1/analytics/by/user?range=24h")
        assert response.status_code == 200, response.text
        rows = response.json()

        keys = [r["key"] for r in rows]
        assert keys == ["alice", "bob"], "ordered by cost descending"
        assert rows[0]["total_calls"] == 2
        assert rows[0]["total_cost"] > rows[1]["total_cost"]

    async def test_untagged_events_are_excluded_not_bucketed(self, client, tagged_events):
        """An untagged call is not the property of a user called 'unknown'."""
        rows = (await client.get("/v1/analytics/by/user?range=24h")).json()
        assert all(r["key"] is not None for r in rows)
        assert sum(r["total_calls"] for r in rows) == 3  # not 4

    async def test_session_dimension(self, client, tagged_events):
        rows = (await client.get("/v1/analytics/by/session?range=24h")).json()
        assert {r["key"] for r in rows} == {"s1", "s2"}

    async def test_unknown_dimension_is_rejected(self, client, tagged_events):
        response = await client.get("/v1/analytics/by/passwords?range=24h")
        assert response.status_code == 422


class TestBudgetState:
    async def test_reports_position_and_staleness(self, client, test_project, test_session):
        test_project.monthly_budget_usd = 100.0
        test_project.budget_enforcement_mode = "warn"
        test_session.add(test_project)
        await test_session.commit()

        response = await client.get(f"/v1/projects/{test_project.id}/budget-state")
        assert response.status_code == 200, response.text
        state = response.json()

        assert state["enabled"] is True
        assert state["budget"] == 100.0
        assert state["remaining"] == pytest.approx(100.0 - state["spend_mtd"])
        assert state["exhausted"] is False
        assert state["as_of"], "consumers need to reason about staleness"
        assert state["period_ends_at"]

    async def test_authenticated_with_the_project_api_key(self, client, test_project):
        """An enforcement point polls with the same credential it sends events with."""
        response = await client.get(f"/v1/projects/{test_project.id}/budget-state")
        assert response.status_code == 200

    async def test_reading_state_records_no_alerts(self, client, test_project, test_session):
        from app.models.db_models import BudgetThresholdAlert

        test_project.monthly_budget_usd = 0.000001
        test_project.budget_enforcement_mode = "warn"
        test_session.add(test_project)
        await test_session.commit()

        await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [_event()]},
        )
        before = len((await test_session.execute(select(BudgetThresholdAlert))).scalars().all())
        await client.get(f"/v1/projects/{test_project.id}/budget-state")
        after = len((await test_session.execute(select(BudgetThresholdAlert))).scalars().all())
        assert before == after, "budget-state must be side-effect free"

    async def test_project_without_budget_reports_disabled(self, client, test_project):
        state = (await client.get(f"/v1/projects/{test_project.id}/budget-state")).json()
        assert state["enabled"] is False
        assert state["remaining"] is None


class TestPrometheusExport:
    async def test_exposition_format(self, client, tagged_events):
        response = await client.get("/v1/metrics")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/plain")

        body = response.text
        assert "# HELP agentcost_cost_usd" in body
        assert "# TYPE agentcost_cost_usd gauge" in body
        assert "agentcost_calls{project=" in body
        assert "agentcost_cost_usd_by_model{" in body
        assert "agentcost_cost_usd_by_agent{" in body
        # Windowed gauges must not carry the counter-reserved suffix.
        assert "_total" not in body

    async def test_label_values_are_escaped(self, client, test_project):
        """A model name with a quote must not break the exposition."""
        await client.post(
            "/v1/events/batch",
            json={"project_id": test_project.id, "events": [_event(model='we"ird\\model')]},
        )
        body = (await client.get("/v1/metrics")).text
        assert 'we\\"ird\\\\model' in body

    async def test_budget_series_appear_only_when_a_budget_exists(
        self, client, test_project, test_session
    ):
        assert "agentcost_budget_utilization_percent" not in (await client.get("/v1/metrics")).text

        test_project.monthly_budget_usd = 50.0
        test_session.add(test_project)
        await test_session.commit()

        assert "agentcost_budget_utilization_percent" in (await client.get("/v1/metrics")).text


class TestWebhookSigning:
    def test_signature_binds_the_timestamp_to_the_body(self):
        """Replaying a capture with a fresh timestamp must invalidate it."""
        body = '{"event":"budget.threshold_crossed"}'
        first = sign_payload("secret", "1000", body)
        assert sign_payload("secret", "1000", body) == first
        assert sign_payload("secret", "2000", body) != first
        assert sign_payload("other", "1000", body) != first


@pytest.fixture
async def owned_project(test_project, test_user, test_session):
    """The test project with an owner, so permission checks can pass."""
    test_project.owner_id = test_user.id
    test_session.add(test_project)
    await test_session.commit()
    return test_project


class TestWebhookConfig:
    async def test_secret_without_url_is_rejected(self, client, owned_project, auth_headers):
        """A secret-only PUT must not silently disable the webhook."""
        response = await client.put(
            f"/v1/projects/{owned_project.id}/webhook",
            json={"secret": "rotate-me"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_rotation_restates_url_and_keeps_working(
        self, client, owned_project, auth_headers
    ):
        base = f"/v1/projects/{owned_project.id}/webhook"

        first = await client.put(
            base, json={"url": "https://example.com/hook", "secret": "s1"}, headers=auth_headers
        )
        assert first.status_code == 200, first.text
        assert first.json()["secret_set"] is True

        # URL restated without a secret: the existing secret stays.
        kept = await client.put(
            base, json={"url": "https://example.com/hook"}, headers=auth_headers
        )
        assert kept.json()["secret_set"] is True

        # Explicit null disables and clears the secret with it.
        off = await client.put(base, json={"url": None}, headers=auth_headers)
        assert off.json()["url"] is None
        assert off.json()["secret_set"] is False

    async def test_plain_http_is_rejected_for_non_local_hosts(
        self, client, owned_project, auth_headers
    ):
        response = await client.put(
            f"/v1/projects/{owned_project.id}/webhook",
            json={"url": "http://example.com/hook"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_test_delivery_sends_a_signed_sample(
        self, client, owned_project, auth_headers, monkeypatch
    ):
        from app.services import webhook_service

        await client.put(
            f"/v1/projects/{owned_project.id}/webhook",
            json={"url": "https://example.com/hook", "secret": "s1"},
            headers=auth_headers,
        )

        captured = {}

        async def fake_deliver(url, event_type, payload, *, secret=None):
            captured.update(url=url, event_type=event_type, payload=payload, secret=secret)
            return webhook_service.DeliveryResult(delivered=True, status_code=200)

        monkeypatch.setattr(webhook_service, "deliver", fake_deliver)

        response = await client.post(
            f"/v1/projects/{owned_project.id}/webhook/test", headers=auth_headers
        )
        assert response.status_code == 200, response.text
        assert response.json()["delivered"] is True
        assert captured["event_type"] == "webhook.test"
        assert captured["url"] == "https://example.com/hook"
        assert captured["secret"] == "s1"

    async def test_test_delivery_without_configuration_is_a_400(
        self, client, owned_project, auth_headers
    ):
        response = await client.post(
            f"/v1/projects/{owned_project.id}/webhook/test", headers=auth_headers
        )
        assert response.status_code == 400


class TestWebhookDeliveryGuards:
    """Delivery-time SSRF and status handling, on the real deliver()."""

    async def test_private_destination_is_refused(self):
        from app.services import webhook_service

        result = await webhook_service.deliver("https://127.0.0.1/hook", "t", {})
        assert result.delivered is False
        assert "non-public" in (result.error or "")

    async def test_unresolvable_host_is_refused(self):
        from app.services import webhook_service

        result = await webhook_service.deliver(
            "https://agentcost-does-not-exist.invalid/hook", "t", {}
        )
        assert result.delivered is False
        assert "resolve" in (result.error or "")

    async def test_non_2xx_is_not_delivered(self, monkeypatch):
        """A redirect is not followed, so it must not count as delivered."""
        import httpx

        from app.services import webhook_service

        async def no_block(url):
            return None

        monkeypatch.setattr(webhook_service, "_destination_blocked", no_block)

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, content=None, headers=None):
                return httpx.Response(302, request=httpx.Request("POST", url))

        monkeypatch.setattr(webhook_service.httpx, "AsyncClient", FakeClient)
        result = await webhook_service.deliver("https://example.com/hook", "t", {})
        assert result.delivered is False
        assert result.status_code == 302


class TestCacheAnalytics:
    @pytest.fixture
    async def cached_model(self, test_session):
        from app.models.db_models import ModelPricing

        test_session.add(
            ModelPricing(
                model_name="cache-model",
                input_price_per_1k=10.0,
                output_price_per_1k=30.0,
                cached_input_price_per_1k=1.0,
                cache_write_price_per_1k=12.5,
                provider="test",
            )
        )
        await test_session.commit()

    async def test_cache_savings_are_priced_per_model(
        self, client, test_project, cached_model
    ):
        await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [
                    _event(
                        model="cache-model",
                        input_tokens=1000,
                        cached_tokens=900,
                        cache_write_tokens=200,
                    )
                ],
            },
        )

        data = (await client.get("/v1/analytics/cache?range=24h")).json()
        assert data["cached_tokens"] == 900
        assert data["cache_write_tokens"] == 200
        assert data["cache_hit_rate"] == pytest.approx(90.0)
        assert data["events_with_cache"] == 1
        # 900/1000 * (10 - 1) saved on reads; 200/1000 * (12.5 - 10) paid on writes.
        assert data["read_savings"] == pytest.approx(8.1)
        assert data["write_premium"] == pytest.approx(0.5)
        assert data["net_savings"] == pytest.approx(7.6)

    async def test_no_published_cache_rate_claims_no_savings(self, client, test_project):
        await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [_event(input_tokens=1000, cached_tokens=900)],
            },
        )
        data = (await client.get("/v1/analytics/cache?range=24h")).json()
        assert data["cached_tokens"] == 900
        assert data["read_savings"] == 0.0
        assert data["net_savings"] == 0.0
