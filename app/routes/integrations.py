"""
AgentCost Backend - Integrations API Routes

Endpoints for importing retroactive cost data from external providers.
"""

import time
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..models.user_models import User
from ..utils.auth import get_required_user

router = APIRouter(prefix="/v1/integrations", tags=["Integrations"])

OPENAI_COSTS_URL = "https://api.openai.com/v1/organization/costs"
ANTHROPIC_COSTS_URL = "https://api.anthropic.com/v1/organizations/cost_report"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_TIMEOUT_SECONDS = 10.0


class OpenAICostImportRequest(BaseModel):
    """Request to import retroactive costs from OpenAI's organization costs API"""

    # SECURITY: this key is used for a single upstream call and is NEVER
    # persisted, logged, or echoed back in any error detail.
    api_key: str = Field(..., description="OpenAI Admin API key (used once, never stored)")
    days: int = Field(default=30, description="How many days back to import (clamped to 1-90)")

    @field_validator('days')
    @classmethod
    def clamp_days(cls, v):
        """Clamp to the supported window instead of rejecting"""
        return max(1, min(90, v))


class DailyCost(BaseModel):
    """Cost total for a single day"""

    date: str  # YYYY-MM-DD
    amount_usd: float


class OpenAICostImportResponse(BaseModel):
    """Aggregated daily costs from OpenAI"""

    total_usd: float
    days: list[DailyCost]


@router.post("/openai/costs", response_model=OpenAICostImportResponse)
async def import_openai_costs(
    data: OpenAICostImportRequest,
    user: User = Depends(get_required_user),
):
    """
    Import retroactive costs from OpenAI's organization costs API.

    Requires an OpenAI **Admin** key (starts with sk-admin-) from
    platform.openai.com/settings/organization/admin-keys. The key is used for
    this request only - it is never stored or logged.
    """
    start_time = int(time.time()) - data.days * 86400
    headers = {"Authorization": f"Bearer {data.api_key}"}

    # bucket start_time (unix) -> summed amount in USD
    buckets: dict[int, float] = {}

    try:
        async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT_SECONDS) as client:
            page: str | None = None
            while True:
                params: dict = {
                    "start_time": start_time,
                    "bucket_width": "1d",
                    "limit": data.days,
                }
                if page:
                    params["page"] = page

                response = await client.get(OPENAI_COSTS_URL, params=params, headers=headers)

                if response.status_code == 401:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            "OpenAI rejected this key. You need an Admin key "
                            "(starts with sk-admin-) from "
                            "platform.openai.com/settings/organization/admin-keys."
                        ),
                    )
                if response.status_code == 403:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            "This key doesn't have permission to read organization "
                            "costs — use an Admin key."
                        ),
                    )
                if response.status_code != 200:
                    # 5xx or any other unexpected upstream status
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Couldn't reach OpenAI. Try again.",
                    )

                payload = response.json()
                for bucket in payload.get("data", []):
                    bucket_start = bucket.get("start_time")
                    if bucket_start is None:
                        continue
                    amount = 0.0
                    for result in bucket.get("results", []):
                        value = (result.get("amount") or {}).get("value")
                        if value is not None:
                            amount += float(value)
                    buckets[int(bucket_start)] = buckets.get(int(bucket_start), 0.0) + amount

                if payload.get("has_more") and payload.get("next_page"):
                    page = payload["next_page"]
                else:
                    break
    except (httpx.TimeoutException, httpx.HTTPError):
        # Deliberately generic: must never include the request (or its
        # Authorization header) in the error detail.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Couldn't reach OpenAI. Try again.",
        )

    # Re-key buckets by calendar day (UTC) and fill missing days with 0.0
    daily: dict[str, float] = {}
    for bucket_start, amount in buckets.items():
        day = datetime.fromtimestamp(bucket_start, tz=timezone.utc).date().isoformat()
        daily[day] = daily.get(day, 0.0) + amount

    first_day = datetime.fromtimestamp(start_time, tz=timezone.utc).date()
    last_day = datetime.now(timezone.utc).date()
    days_out: list[DailyCost] = []
    current = first_day
    while current <= last_day:
        key = current.isoformat()
        days_out.append(DailyCost(date=key, amount_usd=round(daily.get(key, 0.0), 6)))
        current += timedelta(days=1)

    total = round(sum(d.amount_usd for d in days_out), 6)
    return OpenAICostImportResponse(total_usd=total, days=days_out)


class AnthropicCostImportRequest(BaseModel):
    """Request to import retroactive costs from Anthropic's cost report API"""

    # SECURITY: this key is used for a single upstream call and is NEVER
    # persisted, logged, or echoed back in any error detail.
    api_key: str = Field(..., description="Anthropic Admin API key (used once, never stored)")
    days: int = Field(default=30, description="How many days back to import (clamped to 1-90)")

    @field_validator('days')
    @classmethod
    def clamp_days(cls, v):
        """Clamp to the supported window instead of rejecting"""
        return max(1, min(90, v))


def _fill_days(daily: dict[str, float], first_day, last_day) -> list[DailyCost]:
    """Ascending per-day series with missing days zero-filled."""
    days_out: list[DailyCost] = []
    current = first_day
    while current <= last_day:
        key = current.isoformat()
        days_out.append(DailyCost(date=key, amount_usd=round(daily.get(key, 0.0), 6)))
        current += timedelta(days=1)
    return days_out


def _parse_amount(raw) -> float:
    """Anthropic reports amounts as decimal strings; tolerate numbers and
    OpenAI-style {"value": ...} objects so upstream shape drift degrades to a
    parse of 0.0 for one entry, not a 500."""
    if raw is None:
        return 0.0
    if isinstance(raw, dict):
        raw = raw.get("value")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


@router.post("/anthropic/costs", response_model=OpenAICostImportResponse)
async def import_anthropic_costs(
    data: AnthropicCostImportRequest,
    user: User = Depends(get_required_user),
):
    """
    Import retroactive costs from Anthropic's organization cost report API.

    Requires an Anthropic **Admin** key (starts with sk-ant-admin-) from
    console.anthropic.com -> Settings -> Admin keys. The key is used for this
    request only - it is never stored or logged.
    """
    now = datetime.now(timezone.utc)
    first_day = (now - timedelta(days=data.days)).date()
    starting_at = datetime(
        first_day.year, first_day.month, first_day.day, tzinfo=timezone.utc
    ).isoformat().replace("+00:00", "Z")

    headers = {
        "x-api-key": data.api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }

    daily: dict[str, float] = {}

    try:
        async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT_SECONDS) as client:
            page: str | None = None
            while True:
                params: dict = {
                    "starting_at": starting_at,
                    "bucket_width": "1d",
                    # The cost report caps buckets per page well below 90;
                    # stay under it and let pagination do the rest.
                    "limit": min(data.days, 31),
                }
                if page:
                    params["page"] = page

                response = await client.get(
                    ANTHROPIC_COSTS_URL, params=params, headers=headers
                )

                if response.status_code in (401, 403):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            "Anthropic rejected this key. You need an Admin key "
                            "(starts with sk-ant-admin-) from console.anthropic.com "
                            "-> Settings -> Admin keys."
                        ),
                    )
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Couldn't reach Anthropic. Try again.",
                    )

                payload = response.json()
                for bucket in payload.get("data", []):
                    bucket_start = bucket.get("starting_at")
                    if not bucket_start:
                        continue
                    day = bucket_start[:10]  # RFC3339 -> YYYY-MM-DD
                    amount = 0.0
                    for result in bucket.get("results", []):
                        amount += _parse_amount(result.get("amount"))
                    daily[day] = daily.get(day, 0.0) + amount

                if payload.get("has_more") and payload.get("next_page"):
                    page = payload["next_page"]
                else:
                    break
    except (httpx.TimeoutException, httpx.HTTPError):
        # Deliberately generic: must never include the request (or its
        # x-api-key header) in the error detail.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Couldn't reach Anthropic. Try again.",
        )

    days_out = _fill_days(daily, first_day, now.date())
    total = round(sum(d.amount_usd for d in days_out), 6)
    return OpenAICostImportResponse(total_usd=total, days=days_out)
