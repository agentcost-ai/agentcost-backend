"""Bounds on admin-supplied model pricing.

These rates are re-applied to every ingested event's stored cost, so an
out-of-range value is not a bad estimate — it is permanently written into
customers' history. Both pricing mutators must reject the same things.
"""

import pytest
from pydantic import ValidationError

from app.common import MAX_PRICE_PER_1K
from app.routes.admin.pricing import ModelPricingUpdate
from app.routes.pricing import PricingEntry


@pytest.mark.parametrize("model", [ModelPricingUpdate, PricingEntry])
@pytest.mark.parametrize("bad", [-5.0, -1e9, MAX_PRICE_PER_1K + 0.01, 1e9])
def test_out_of_range_rates_are_rejected(model, bad):
    field = "input_price_per_1k" if model is ModelPricingUpdate else "input"
    with pytest.raises(ValidationError):
        model(**{field: bad})


@pytest.mark.parametrize("model", [ModelPricingUpdate, PricingEntry])
def test_unknown_fields_are_rejected(model):
    """extra='forbid' — a typo must fail loudly rather than silently no-op."""
    with pytest.raises(ValidationError):
        model(inputt_price=0.01)


@pytest.mark.parametrize("model", [ModelPricingUpdate, PricingEntry])
def test_realistic_rates_are_accepted(model):
    field = "input_price_per_1k" if model is ModelPricingUpdate else "input"
    for good in (0.0, 0.00025, 0.6, MAX_PRICE_PER_1K):
        assert model(**{field: good}) is not None


def test_omitted_fields_stay_unset():
    """exclude_unset is what stops an omitted rate overwriting a stored one."""
    assert ModelPricingUpdate(is_active=False).model_dump(exclude_unset=True) == {
        "is_active": False
    }
    assert PricingEntry(provider="openai").model_dump(exclude_unset=True) == {
        "provider": "openai"
    }


def test_provider_is_reachable_on_the_bulk_endpoint():
    """The body used to be Dict[str, float], so `provider` could never be set
    and every model created here was stored as 'unknown'."""
    assert PricingEntry(input=0.01, provider="anthropic").provider == "anthropic"
