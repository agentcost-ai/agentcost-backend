"""
Capability inference for model-downgrade suggestions.

Regression cover for a fail-open bug: the requirement check read
``state == "true"``, so an *unknown* requirement -- the normal case, because
nothing populated the metadata it inspected -- was treated as "not required".
The optimizer would then propose a text-only model for a workload sending
images, or a model with no tool support for an agent that calls tools.

Unknown now means "assume required", which narrows candidates to a capability
superset. That withholds a suggestion at worst; the old behaviour proposed a
switch that breaks production.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models.db_models import Event, ModelPricing
from app.services.optimization_service import CAPABILITY_KEY, OptimizationService


CALLS = 12  # above the analyzer's 10-call cutoff


def _events(project_id: str, *, metadata: dict | None, model: str = "premium-model"):
    now = datetime.now(timezone.utc)
    return [
        Event(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent_name="vision-agent",
            model=model,
            input_tokens=2000,
            output_tokens=500,
            total_tokens=2500,
            cost=1.0,
            latency_ms=800,
            success=True,
            timestamp=now - timedelta(hours=1),
            extra_data=metadata,
        )
        for _ in range(CALLS)
    ]


@pytest.fixture
async def catalogue(test_session):
    """An expensive vision model and two cheaper candidates, one text-only."""
    test_session.add_all(
        [
            ModelPricing(
                model_name="premium-model",
                input_price_per_1k=10.0,
                output_price_per_1k=30.0,
                provider="test",
                supports_vision=True,
                supports_function_calling=True,
            ),
            ModelPricing(
                model_name="cheap-text-only",
                input_price_per_1k=0.1,
                output_price_per_1k=0.3,
                provider="test",
                supports_vision=False,
                supports_function_calling=False,
            ),
            ModelPricing(
                model_name="cheap-multimodal",
                input_price_per_1k=1.0,
                output_price_per_1k=3.0,
                provider="test",
                supports_vision=True,
                supports_function_calling=True,
            ),
        ]
    )
    await test_session.commit()


async def _suggested_models(test_session, project_id) -> set[str]:
    service = OptimizationService(test_session)
    suggestions = await service.get_suggestions(
        project_id, days=7, persist_recommendations=False
    )
    return {
        s["alternative_model"]
        for s in suggestions
        if s.get("type") == "model_downgrade"
    }


class TestCapabilityFingerprint:
    async def test_vision_workload_never_offered_a_text_only_model(
        self, test_session, test_project, catalogue
    ):
        test_session.add_all(
            _events(test_project.id, metadata={CAPABILITY_KEY: {"vision": True}})
        )
        await test_session.commit()

        models = await _suggested_models(test_session, test_project.id)
        assert "cheap-text-only" not in models, "would break every image request"

    async def test_tool_workload_never_offered_a_model_without_tools(
        self, test_session, test_project, catalogue
    ):
        test_session.add_all(
            _events(test_project.id, metadata={CAPABILITY_KEY: {"tools": True, "tool_count": 3}})
        )
        await test_session.commit()

        models = await _suggested_models(test_session, test_project.id)
        assert "cheap-text-only" not in models

    async def test_plain_text_workload_may_be_offered_the_cheapest_model(
        self, test_session, test_project, catalogue
    ):
        """A measured 'needs nothing special' is what unlocks the big saving."""
        test_session.add_all(
            _events(test_project.id, metadata={CAPABILITY_KEY: {}, "note": "plain"})
        )
        await test_session.commit()

        models = await _suggested_models(test_session, test_project.id)
        assert "cheap-text-only" in models


class TestFailClosedOnUnknown:
    async def test_unknown_requirements_do_not_yield_a_text_only_downgrade(
        self, test_session, test_project, catalogue
    ):
        """No fingerprint at all: assume the capabilities are needed."""
        test_session.add_all(_events(test_project.id, metadata=None))
        await test_session.commit()

        models = await _suggested_models(test_session, test_project.id)
        assert "cheap-text-only" not in models

    async def test_unknown_still_permits_a_capability_superset(
        self, test_session, test_project, catalogue
    ):
        """Failing closed must not mean failing silent -- savings still surface."""
        test_session.add_all(_events(test_project.id, metadata=None))
        await test_session.commit()

        models = await _suggested_models(test_session, test_project.id)
        assert "cheap-multimodal" in models

    async def test_suggestions_report_whether_capabilities_were_measured(
        self, test_session, test_project, catalogue
    ):
        test_session.add_all(_events(test_project.id, metadata=None))
        await test_session.commit()

        service = OptimizationService(test_session)
        suggestions = await service.get_suggestions(
            test_project.id, days=7, persist_recommendations=False
        )
        downgrades = [s for s in suggestions if s.get("type") == "model_downgrade"]
        assert downgrades, "expected at least one suggestion to inspect"
        assert all(
            s["metrics"]["capabilities_verified"] is False for s in downgrades
        ), "an unverified suggestion must say so, so a consumer can gate on it"

    async def test_measured_suggestions_are_marked_verified(
        self, test_session, test_project, catalogue
    ):
        test_session.add_all(
            _events(test_project.id, metadata={CAPABILITY_KEY: {}})
        )
        await test_session.commit()

        service = OptimizationService(test_session)
        suggestions = await service.get_suggestions(
            test_project.id, days=7, persist_recommendations=False
        )
        downgrades = [s for s in suggestions if s.get("type") == "model_downgrade"]
        assert downgrades
        assert all(s["metrics"]["capabilities_verified"] is True for s in downgrades)


class TestLegacyMetadataHeuristics:
    async def test_raw_tools_key_is_still_honoured(
        self, test_session, test_project, catalogue
    ):
        """Callers who hand-tagged metadata before the fingerprint existed."""
        test_session.add_all(
            _events(test_project.id, metadata={"tools": [{"name": "search"}]})
        )
        await test_session.commit()

        models = await _suggested_models(test_session, test_project.id)
        assert "cheap-text-only" not in models

    async def test_structured_output_blocks_downgrades_entirely(
        self, test_session, test_project, catalogue
    ):
        """The catalogue has no per-model JSON-mode flag, so nothing is safe."""
        test_session.add_all(
            _events(
                test_project.id,
                metadata={CAPABILITY_KEY: {"structured_output": True}},
            )
        )
        await test_session.commit()

        models = await _suggested_models(test_session, test_project.id)
        assert models == set()
