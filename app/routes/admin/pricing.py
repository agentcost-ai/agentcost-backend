"""
Admin routes -- pricing intelligence and sync.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ...database import get_db
from ...models.user_models import User
from ...models.db_models import ModelPricing
from ...services.pricing_service import PricingService
from ._deps import require_superuser

router = APIRouter()


@router.get("/pricing/models")
async def list_pricing_models(
    search: Optional[str] = Query(None),
    provider: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """List all model pricing records with filters."""
    query = select(ModelPricing)

    if search:
        query = query.where(ModelPricing.model_name.ilike(f"%{search}%"))
    if provider:
        query = query.where(ModelPricing.provider == provider)
    if source:
        query = query.where(ModelPricing.pricing_source == source)

    count_q = select(func.count(ModelPricing.id))
    if search:
        count_q = count_q.where(ModelPricing.model_name.ilike(f"%{search}%"))
    if provider:
        count_q = count_q.where(ModelPricing.provider == provider)
    if source:
        count_q = count_q.where(ModelPricing.pricing_source == source)

    total = (await db.execute(count_q)).scalar() or 0

    query = query.order_by(ModelPricing.provider, ModelPricing.model_name)
    query = query.limit(limit).offset(offset)

    models = (await db.execute(query)).scalars().all()

    return {
        "items": [
            {
                "id": m.id,
                "model_name": m.model_name,
                "provider": m.provider,
                "input_price_per_1k": m.input_price_per_1k,
                "output_price_per_1k": m.output_price_per_1k,
                "is_active": m.is_active,
                "pricing_source": m.pricing_source,
                "max_tokens": m.max_tokens,
                "supports_vision": m.supports_vision,
                "supports_function_calling": m.supports_function_calling,
                "source_updated_at": m.source_updated_at.isoformat() if m.source_updated_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
            for m in models
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/pricing/providers")
async def list_pricing_providers(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Get provider summary with model counts and avg pricing."""
    rows = (await db.execute(
        select(
            ModelPricing.provider,
            func.count(ModelPricing.id).label("model_count"),
            func.avg(ModelPricing.input_price_per_1k).label("avg_input"),
            func.avg(ModelPricing.output_price_per_1k).label("avg_output"),
        )
        .where(ModelPricing.is_active == True)
        .group_by(ModelPricing.provider)
        .order_by(desc(func.count(ModelPricing.id)))
    )).all()

    return [
        {
            "provider": r.provider,
            "model_count": int(r.model_count),
            "avg_input_price": round(float(r.avg_input), 6),
            "avg_output_price": round(float(r.avg_output), 6),
        }
        for r in rows
    ]


@router.patch("/pricing/models/{model_id}")
async def update_model_pricing(
    model_id: int,
    body: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """
    Admin override for model pricing.
    Supports: input_price_per_1k, output_price_per_1k, is_active, notes.
    Sets pricing_source to 'admin_override'.
    """
    model = (await db.execute(
        select(ModelPricing).where(ModelPricing.id == model_id)
    )).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model pricing not found")

    allowed = {"input_price_per_1k", "output_price_per_1k", "is_active", "notes"}
    for field in allowed:
        if field in body:
            setattr(model, field, body[field])

    model.pricing_source = "admin_override"
    model.source_updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(model)

    return {
        "id": model.id,
        "model_name": model.model_name,
        "pricing_source": model.pricing_source,
        "message": "Pricing updated",
    }


@router.post("/pricing/sync/litellm")
async def sync_litellm_pricing(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Trigger manual LiteLLM pricing sync."""
    pricing_service = PricingService(db)
    try:
        result = await pricing_service.sync_from_litellm(track_changes=True)
        return result
    finally:
        await pricing_service.close()


@router.post("/pricing/sync/openrouter")
async def sync_openrouter_pricing(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Trigger manual OpenRouter pricing sync."""
    pricing_service = PricingService(db)
    try:
        result = await pricing_service.sync_from_openrouter(track_changes=True)
        return result
    finally:
        await pricing_service.close()
