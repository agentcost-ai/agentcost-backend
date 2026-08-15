"""
The air-gapped pricing path: POST /v1/pricing/import.

Egress-restricted deployments (banks, defence) load the LiteLLM catalogue
from an uploaded bundle instead of the network. These tests prove the bundle
path applies the same parsing, sanity bounds and logging as the online sync,
with no network in sight.
"""

import uuid

import pytest
from sqlalchemy import select

from app.common import MAX_PRICE_PER_1K
from app.models.db_models import ModelPricing, PricingSyncLog


@pytest.fixture
async def admin_headers(test_session):
    """Bearer JWT for a superuser — /v1/pricing/import is admin-only."""
    from app.models.user_models import User
    from app.services.auth_service import create_access_token, hash_password

    admin = User(
        id=str(uuid.uuid4()),
        email="admin@example.com",
        password_hash=hash_password("adminpassword123"),
        name="Admin",
        is_active=True,
        is_deleted=False,
        email_verified=True,
        auth_provider="email",
        is_superuser=True,
    )
    test_session.add(admin)
    await test_session.commit()

    token, _ = create_access_token(admin.id, admin.email)
    return {"Authorization": f"Bearer {token}"}


def _bundle() -> dict:
    """A miniature model_prices_and_context_window.json, LiteLLM-shaped."""
    return {
        "sample_spec": {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        "gpt-4o-mini": {
            "input_cost_per_token": 1.5e-07,
            "output_cost_per_token": 6e-07,
            "cache_read_input_token_cost": 7.5e-08,
            "litellm_provider": "openai",
            "max_tokens": 16384,
            "max_input_tokens": 128000,
            "supports_vision": True,
            "supports_function_calling": True,
        },
        "claude-sonnet-4": {
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
            "cache_read_input_token_cost": 3e-07,
            "cache_creation_input_token_cost": 3.75e-06,
            "litellm_provider": "anthropic",
            "max_tokens": 64000,
        },
    }


class TestImportBundle:
    async def test_bundle_creates_models_with_cache_rates(
        self, client, admin_headers, test_session
    ):
        response = await client.post(
            "/v1/pricing/import", json=_bundle(), headers=admin_headers
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "ok"
        assert result["models_created"] == 2

        rows = {
            row.model_name: row
            for row in (await test_session.execute(select(ModelPricing))).scalars()
        }
        gpt = rows["gpt-4o-mini"]
        assert gpt.input_price_per_1k == pytest.approx(0.00015)
        assert gpt.output_price_per_1k == pytest.approx(0.0006)
        assert gpt.cached_input_price_per_1k == pytest.approx(0.000075)
        assert gpt.cache_write_price_per_1k is None
        assert gpt.provider == "openai"
        assert gpt.max_input_tokens == 128000
        assert gpt.supports_vision is True

        claude = rows["claude-sonnet-4"]
        assert claude.cached_input_price_per_1k == pytest.approx(0.0003)
        assert claude.cache_write_price_per_1k == pytest.approx(0.00375)

    async def test_import_prices_subsequent_ingest(
        self, client, admin_headers, test_project, test_session
    ):
        """The point of the path: an offline catalogue prices real events."""
        from datetime import datetime, timezone

        from app.models.db_models import Event

        await client.post("/v1/pricing/import", json=_bundle(), headers=admin_headers)
        await client.post(
            "/v1/events/batch",
            json={
                "project_id": test_project.id,
                "events": [{
                    "agent_name": "a",
                    "model": "claude-sonnet-4",
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            },
        )
        row = (await test_session.execute(select(Event))).scalars().one()
        # 1000/1000 * 0.003 + 100/1000 * 0.015
        assert row.cost == pytest.approx(0.0045)
        assert row.cost_source == "database-exact"

    async def test_reimport_updates_rather_than_duplicates(
        self, client, admin_headers, test_session
    ):
        await client.post("/v1/pricing/import", json=_bundle(), headers=admin_headers)

        changed = _bundle()
        changed["gpt-4o-mini"]["input_cost_per_token"] = 3e-07
        response = await client.post(
            "/v1/pricing/import", json=changed, headers=admin_headers
        )
        assert response.json()["models_updated"] >= 1

        rows = (
            await test_session.execute(
                select(ModelPricing).where(ModelPricing.model_name == "gpt-4o-mini")
            )
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].input_price_per_1k == pytest.approx(0.0003)

    async def test_unit_error_prices_are_rejected(
        self, client, admin_headers, test_session
    ):
        """The wandb-class failure: a per-million price uploaded as per-token."""
        bad = {
            "broken-model": {
                "input_cost_per_token": MAX_PRICE_PER_1K,  # 1000x over the bound once per-1k
                "output_cost_per_token": MAX_PRICE_PER_1K,
                "litellm_provider": "openai",
            }
        }
        response = await client.post("/v1/pricing/import", json=bad, headers=admin_headers)
        assert response.status_code == 200

        rows = (await test_session.execute(select(ModelPricing))).scalars().all()
        assert rows == [], "an implausible price must not produce a live row"

    async def test_sync_log_records_the_run(self, client, admin_headers, test_session):
        await client.post("/v1/pricing/import", json=_bundle(), headers=admin_headers)
        log = (
            await test_session.execute(
                select(PricingSyncLog).order_by(PricingSyncLog.created_at.desc())
            )
        ).scalars().first()
        assert log is not None
        assert log.status == "ok"
        assert log.models_created == 2

    async def test_empty_bundle_is_rejected(self, client, admin_headers):
        response = await client.post("/v1/pricing/import", json={}, headers=admin_headers)
        assert response.status_code == 422

    async def test_requires_admin(self, client, test_project, auth_headers):
        """A signed-in but non-admin user gets 403; the SDK key gets 401/403."""
        response = await client.post(
            "/v1/pricing/import", json=_bundle(), headers=auth_headers
        )
        assert response.status_code == 403

    async def test_requires_authentication(self, client):
        response = await client.post(
            "/v1/pricing/import",
            json=_bundle(),
            headers={"Authorization": ""},
        )
        assert response.status_code in (401, 403)