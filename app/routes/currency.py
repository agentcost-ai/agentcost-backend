"""
Public live currency rate endpoint.

Used by the dashboard to render the live USD->X conversion rate the moment a
user toggles the budget currency picker, without waiting for them to save.

Authentication: JWT-only — kept behind login because there's no reason for
unauthenticated callers to hit it and we avoid being a free FX proxy.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Literal

from ..models.user_models import User
from ..services.currency_service import (
    CurrencyService,
    SUPPORTED_CURRENCIES,
)
from ..utils.auth import get_required_user


router = APIRouter(prefix="/v1/currency", tags=["Currency"])


class FxRateResponse(BaseModel):
    base: Literal["USD"] = "USD"
    target: str
    rate: float
    source: str = "frankfurter.dev"


@router.get("/rate", response_model=FxRateResponse)
async def get_fx_rate(
    target: str = Query(..., description="Target currency code (e.g. INR, USD)"),
    _: User = Depends(get_required_user),
):
    """
    Return the cached USD -> ``target`` exchange rate.

    Always returns 1.0 when ``target=USD``. Falls back to a sensible static
    rate if the upstream FX provider is unreachable.
    """
    normalized = CurrencyService.normalize(target)
    if target.upper().strip() not in SUPPORTED_CURRENCIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported currency '{target}'. "
                f"Supported: {', '.join(SUPPORTED_CURRENCIES)}."
            ),
        )

    rate = await CurrencyService.usd_to(normalized)
    return FxRateResponse(target=normalized, rate=rate)
