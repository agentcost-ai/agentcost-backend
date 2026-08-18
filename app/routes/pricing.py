"""
Provides endpoints for model pricing management.

To sync pricing, call POST /v1/pricing/sync/litellm which fetches from:
https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

from fastapi import APIRouter, Depends, Query, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..common import MAX_PRICE_PER_1K
from ..database import get_db
from ..models.db_models import ModelPricing, PricingSyncLog
from ..services.admin_service import log_admin_action
from ..services.pricing_service import PricingService
from ..services.auth_service import get_current_user
from ..models.user_models import User

router = APIRouter(prefix="/v1/pricing", tags=["Pricing"])
security = HTTPBearer(auto_error=False)


async def get_admin_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_current_user(db, credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return user


# Fallback pricing when database is empty.
# Synced with SDK's DEFAULT_PRICING. Prices per 1,000 tokens in USD.
DEFAULT_PRICING = {
    # OpenAI
    'gpt-4': {'input': 0.03, 'output': 0.06, 'provider': 'openai'},
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03, 'provider': 'openai'},
    'gpt-4-turbo-preview': {'input': 0.01, 'output': 0.03, 'provider': 'openai'},
    'gpt-4o': {'input': 0.0025, 'output': 0.01, 'provider': 'openai'},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006, 'provider': 'openai'},
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015, 'provider': 'openai'},
    'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004, 'provider': 'openai'},
    'o1': {'input': 0.015, 'output': 0.06, 'provider': 'openai'},
    'o1-preview': {'input': 0.015, 'output': 0.06, 'provider': 'openai'},
    'o1-mini': {'input': 0.003, 'output': 0.012, 'provider': 'openai'},
    
    # Anthropic
    'claude-3-opus': {'input': 0.015, 'output': 0.075, 'provider': 'anthropic'},
    'claude-3-sonnet': {'input': 0.003, 'output': 0.015, 'provider': 'anthropic'},
    'claude-3-haiku': {'input': 0.00025, 'output': 0.00125, 'provider': 'anthropic'},
    'claude-3-5-sonnet': {'input': 0.003, 'output': 0.015, 'provider': 'anthropic'},
    'claude-3-5-haiku': {'input': 0.0008, 'output': 0.004, 'provider': 'anthropic'},
    'claude-4-opus': {'input': 0.015, 'output': 0.075, 'provider': 'anthropic'},
    
    # Groq
    'llama-3.1-8b-instant': {'input': 0.00005, 'output': 0.00008, 'provider': 'groq'},
    'llama-3.1-70b-versatile': {'input': 0.00059, 'output': 0.00079, 'provider': 'groq'},
    'llama-3.2-3b-preview': {'input': 0.00006, 'output': 0.00006, 'provider': 'groq'},
    'llama-3.3-70b-versatile': {'input': 0.00059, 'output': 0.00079, 'provider': 'groq'},
    'mixtral-8x7b-32768': {'input': 0.00024, 'output': 0.00024, 'provider': 'groq'},
    
    # Google
    'gemini-pro': {'input': 0.00025, 'output': 0.0005, 'provider': 'google'},
    'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005, 'provider': 'google'},
    'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003, 'provider': 'google'},
    'gemini-2.0-flash': {'input': 0.0001, 'output': 0.0004, 'provider': 'google'},
    
    # DeepSeek
    'deepseek-chat': {'input': 0.00014, 'output': 0.00028, 'provider': 'deepseek'},
    'deepseek-coder': {'input': 0.00014, 'output': 0.00028, 'provider': 'deepseek'},
    'deepseek-reasoner': {'input': 0.00055, 'output': 0.00219, 'provider': 'deepseek'},
    
    # Mistral
    'mistral-small': {'input': 0.001, 'output': 0.003, 'provider': 'mistral'},
    'mistral-medium': {'input': 0.00275, 'output': 0.0081, 'provider': 'mistral'},
    'mistral-large': {'input': 0.004, 'output': 0.012, 'provider': 'mistral'},
    
    # Cohere
    'command': {'input': 0.001, 'output': 0.002, 'provider': 'cohere'},
    'command-light': {'input': 0.0003, 'output': 0.0006, 'provider': 'cohere'},
    'command-r': {'input': 0.0005, 'output': 0.0015, 'provider': 'cohere'},
    'command-r-plus': {'input': 0.003, 'output': 0.015, 'provider': 'cohere'},
    
    # Together AI
    'meta-llama/Llama-3-70b-chat-hf': {'input': 0.0009, 'output': 0.0009, 'provider': 'together'},
    'meta-llama/Llama-3-8b-chat-hf': {'input': 0.0002, 'output': 0.0002, 'provider': 'together'},
}


async def _last_synced_at(db: AsyncSession) -> Optional[datetime]:
    """When the catalogue was last refreshed from a pricing source.

    Deliberately not max(ModelPricing.updated_at): that moves only when a price
    actually changes, so a quiet week upstream would report the catalogue as a
    week stale when it had in fact just been checked. Callers asking "is this
    current?" mean the sync, not the last price movement.
    """
    synced_at = (await db.execute(
        select(func.max(PricingSyncLog.created_at))
        .where(PricingSyncLog.status == "ok")
    )).scalar()
    if synced_at is not None:
        return synced_at

    # A populated catalogue with no sync logged (hand-seeded, or pre-dating
    # pricing_sync_log) is stale-dated, not unknown -- returning None here made
    # the public /docs/models page render "Never" over thousands of models.
    return (await db.execute(select(func.max(ModelPricing.updated_at)))).scalar()


@router.get("/sync/status")
async def get_sync_status(db: AsyncSession = Depends(get_db)):
    """Get pricing sync status."""
    total_query = select(func.count(ModelPricing.id)).where(ModelPricing.is_active == True)
    total_result = await db.execute(total_query)
    total_models = total_result.scalar() or 0

    last_updated = await _last_synced_at(db)

    provider_query = select(
        ModelPricing.provider, 
        func.count(ModelPricing.id)
    ).where(ModelPricing.is_active == True).group_by(ModelPricing.provider)
    provider_result = await db.execute(provider_query)
    providers = {row[0]: row[1] for row in provider_result.all()}
    
    source_query = select(
        ModelPricing.pricing_source, 
        func.count(ModelPricing.id)
    ).where(ModelPricing.is_active == True).group_by(ModelPricing.pricing_source)
    source_result = await db.execute(source_query)
    sources = {row[0] or "unknown": row[1] for row in source_result.all()}
    
    # Determine status message based on model count
    if total_models == 0:
        status = "not_synced"
        message = "No models in database. Run POST /v1/pricing/sync/litellm to sync 3500+ models."
    elif total_models < 100:
        status = "partial"
        message = f"Only {total_models} models synced. Run POST /v1/pricing/sync/litellm for full sync."
    else:
        status = "synced"
        message = f"Database contains {total_models} models with up-to-date pricing."
    
    return {
        "status": status,
        "message": message,
        "total_models": total_models,
        "fallback_models": len(DEFAULT_PRICING),
        "last_updated": last_updated.isoformat() if last_updated else None,
        "models_by_provider": providers,
        "models_by_source": sources,
        "database_populated": total_models > 0,
        "sync_endpoints": {
            "litellm": "POST /v1/pricing/sync/litellm",
            "openrouter": "POST /v1/pricing/sync/openrouter",
        }
    }


@router.get("")
async def get_all_pricing(
    provider: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Get all model pricing. Public endpoint for SDKs."""
    query = select(ModelPricing).where(ModelPricing.is_active == True)
    if provider:
        query = query.where(ModelPricing.provider == provider)
    
    result = await db.execute(query)
    db_pricing = result.scalars().all()
    
    if db_pricing:
        pricing = {}
        for model in db_pricing:
            pricing[model.model_name] = {
                'input': model.input_price_per_1k,
                'output': model.output_price_per_1k,
                # The SDK's cost calculator reads these two; omitting them made
                # every client-side estimate bill cached tokens at full rate.
                'cached_input': model.cached_input_price_per_1k,
                'cache_write': model.cache_write_price_per_1k,
                'provider': model.provider,
                'updated_at': model.updated_at.isoformat() if model.updated_at else None,
            }
        # last_updated = when the catalogue was last refreshed; per-model
        # updated_at above still reports when that model's price last moved.
        # _last_synced_at already falls back to the catalogue high-water mark,
        # so only a completely empty history reaches the default here.
        synced_at = await _last_synced_at(db) or datetime.now(timezone.utc)
        return {
            "pricing": pricing,
            "source": "database",
            "last_updated": synced_at.isoformat(),
        }
    
    pricing = DEFAULT_PRICING
    if provider:
        pricing = {k: v for k, v in pricing.items() if v.get('provider') == provider}
    
    return {
        "pricing": pricing,
        "source": "defaults",
        "last_updated": None,
    }


@router.get("/deprecations")
async def list_deprecations(db: AsyncSession = Depends(get_db)):
    """Active models with an upstream-announced retirement date, soonest first.

    Public. The dates come from LiteLLM's deprecation_date and refresh with
    every pricing sync. Registered before /{model_name} so the literal path
    wins over the catch-all.
    """
    rows = (await db.execute(
        select(ModelPricing)
        .where(
            ModelPricing.is_active == True,  # noqa: E712
            ModelPricing.deprecation_date.isnot(None),
        )
        .order_by(ModelPricing.deprecation_date.asc(), ModelPricing.model_name.asc())
    )).scalars().all()

    return {
        "deprecations": [
            {
                "model": r.model_name,
                "provider": r.provider,
                "deprecation_date": r.deprecation_date,
                "mode": r.mode,
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/{model_name}")
async def get_model_pricing(model_name: str, db: AsyncSession = Depends(get_db)):
    """Get pricing for a specific model.

    Answers with the same resolver event ingestion bills with (exact then
    deterministic fuzzy against the full catalogue). This route used to run its
    own substring match over the small DEFAULT_PRICING dict and could quote a
    different price than the one events were actually costed at.
    """
    service = PricingService(db)
    pricing = await service.get_model_pricing(model_name)

    if pricing:
        return {
            "model": model_name,
            "matched_to": pricing["matched_model"],
            "input": pricing["input"],
            "output": pricing["output"],
            "provider": pricing["provider"],
            "source": "database" if pricing["match"] == "exact" else "database-fuzzy",
        }

    return {
        "model": model_name,
        "input": 0.0,
        "output": 0.0,
        "provider": "unknown",
        "source": "fallback",
    }


class PricingEntry(BaseModel):
    """One model's rates, in USD per 1,000 tokens."""

    model_config = ConfigDict(extra="forbid")

    input: Optional[float] = Field(None, ge=0, le=MAX_PRICE_PER_1K)
    output: Optional[float] = Field(None, ge=0, le=MAX_PRICE_PER_1K)
    provider: Optional[str] = Field(None, max_length=100)


@router.post("")
async def update_pricing(
    pricing_updates: Dict[str, PricingEntry],
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """Update pricing for models (Admin).

    Bounded and audited like the admin dashboard's per-model override: these
    rates are re-applied to every ingested event, so an unchecked value is
    written permanently into customers' cost history.
    """
    if not pricing_updates:
        raise HTTPException(status_code=422, detail="No pricing entries supplied.")

    updated_count = 0
    created_count = 0
    audited: Dict[str, Any] = {}

    for model_name, prices in pricing_updates.items():
        existing = (await db.execute(
            select(ModelPricing).where(ModelPricing.model_name == model_name)
        )).scalar_one_or_none()

        # exclude_unset so an omitted rate keeps its stored value instead of
        # being reset to the field default.
        supplied = prices.model_dump(exclude_unset=True)

        if existing:
            before = {}
            for field, column in (("input", "input_price_per_1k"),
                                  ("output", "output_price_per_1k"),
                                  ("provider", "provider")):
                if field not in supplied:
                    continue
                current = getattr(existing, column)
                if current != supplied[field]:
                    before[column] = current
                    setattr(existing, column, supplied[field])
            if before:
                existing.updated_at = datetime.now(timezone.utc)
                audited[model_name] = {"before": before}
                updated_count += 1
        else:
            db.add(ModelPricing(
                model_name=model_name,
                input_price_per_1k=supplied.get("input", 0.0),
                output_price_per_1k=supplied.get("output", 0.0),
                provider=supplied.get("provider", "unknown"),
            ))
            audited[model_name] = {"created": supplied}
            created_count += 1

    if audited:
        await log_admin_action(
            db,
            admin_id=admin.id,
            action_type="model_pricing_bulk_updated",
            target_type="model_pricing",
            details={"models": audited},
            ip_address=request.client.host if request.client else None,
        )

    await db.commit()

    return {
        "status": "ok",
        "models_updated": updated_count,
        "models_created": created_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _mark_sync_failed(db: AsyncSession, log_id: str, exc: Exception) -> None:
    """Record a failed sync so its claim is not left 'running' forever."""
    from sqlalchemy import update as sa_update

    await db.rollback()
    await db.execute(
        sa_update(PricingSyncLog)
        .where(PricingSyncLog.id == log_id)
        .values(status="error", error_message=str(exc)[:500])
    )
    await db.commit()


@router.post("/sync/litellm")
async def sync_from_litellm(
    track_changes: bool = Query(False),
    auto_regenerate_alternatives: bool = Query(True, description="Automatically regenerate model alternatives after sync"),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    """
    Sync pricing from LiteLLM database.
    
    By default, also regenerates model alternatives to reflect new pricing.
    """
    from ..services.cron import claim_pricing_sync

    # Claim like every other entry point, and log the run -- without the log
    # entry, deployments that only ever sync manually reported last_updated
    # as null forever.
    log_entry = await claim_pricing_sync(db)
    if log_entry is None:
        raise HTTPException(
            status_code=409,
            detail="A pricing sync is already running. Try again in a few minutes.",
        )

    log_id = log_entry.id
    pricing_service = PricingService(db)
    try:
        result = await pricing_service.sync_from_litellm(track_changes=track_changes)

        # Auto-regenerate alternatives if requested
        if auto_regenerate_alternatives and result.get("status") == "ok":
            from ..services.alternative_learning_service import AlternativeLearningService
            learning_service = AlternativeLearningService(db)
            alt_result = await learning_service.generate_alternatives_from_pricing()
            result["alternatives_regenerated"] = True
            result["alternatives_created"] = alt_result.get("alternatives_created", 0)
            result["alternatives_updated"] = alt_result.get("alternatives_updated", 0)

        log_entry.status = "error" if result.get("status") == "error" else "ok"
        log_entry.models_created = result.get("models_created", 0)
        log_entry.models_updated = result.get("models_updated", 0)
        log_entry.models_skipped = result.get("models_skipped", 0)
        log_entry.error_message = result.get("error")
        await db.commit()
        return result
    except Exception as exc:
        # HTTPExceptions included: any exit without a status write would
        # leave the claim "running" and block the next sync.
        await _mark_sync_failed(db, log_id, exc)
        raise
    finally:
        await pricing_service.close()


@router.post("/import")
async def import_pricing_bundle(
    bundle: dict,
    track_changes: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    """
    Load a pricing catalogue from an uploaded bundle instead of the network.

    For deployments that cannot reach GitHub — air-gapped, egress-restricted,
    or simply unwilling to depend on a third party being up at sync time. The
    bundle is LiteLLM's `model_prices_and_context_window.json` verbatim, so it
    can be fetched on a connected machine, reviewed, and carried across.

    Same parsing, sanity bounds and change tracking as the network sync; only
    the source of the JSON differs.
    """
    from ..services.cron import claim_pricing_sync

    if not isinstance(bundle, dict) or not bundle:
        raise HTTPException(
            status_code=422,
            detail="Bundle must be a non-empty JSON object of model_name -> pricing.",
        )

    log_entry = await claim_pricing_sync(db)
    if log_entry is None:
        raise HTTPException(
            status_code=409,
            detail="A pricing sync is already running. Try again in a few minutes.",
        )

    log_id = log_entry.id
    pricing_service = PricingService(db)
    try:
        result = await pricing_service.sync_from_litellm(
            track_changes=track_changes, bundle=bundle
        )
        log_entry.status = "error" if result.get("status") == "error" else "ok"
        log_entry.models_created = result.get("models_created", 0)
        log_entry.models_updated = result.get("models_updated", 0)
        log_entry.models_skipped = result.get("models_skipped", 0)
        log_entry.error_message = result.get("error")
        await db.commit()
        return result
    except Exception as exc:
        await _mark_sync_failed(db, log_id, exc)
        raise
    finally:
        await pricing_service.close()


@router.post("/sync/openrouter")
async def sync_from_openrouter(
    auto_regenerate_alternatives: bool = Query(True, description="Automatically regenerate model alternatives after sync"),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    """
    Sync pricing from OpenRouter API.
    
    By default, also regenerates model alternatives to reflect new pricing.
    """
    pricing_service = PricingService(db)
    try:
        result = await pricing_service.sync_from_openrouter()
        
        # Auto-regenerate alternatives if requested
        if auto_regenerate_alternatives and result.get("status") == "ok":
            from ..services.alternative_learning_service import AlternativeLearningService
            learning_service = AlternativeLearningService(db)
            alt_result = await learning_service.generate_alternatives_from_pricing()
            result["alternatives_regenerated"] = True
            result["alternatives_created"] = alt_result.get("alternatives_created", 0)
            result["alternatives_updated"] = alt_result.get("alternatives_updated", 0)
        
        return result
    finally:
        await pricing_service.close()


@router.get("/discover/{model_name}")
async def discover_alternatives(
    model_name: str,
    avg_input_tokens: Optional[int] = Query(None),
    avg_output_tokens: Optional[int] = Query(None),
    requires_vision: bool = Query(False),
    requires_function_calling: bool = Query(False),
    same_provider_only: bool = Query(False),
    max_results: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Find cheaper model alternatives based on pricing and capabilities."""
    pricing_service = PricingService(db)
    
    source_pricing = await pricing_service.get_model_pricing(model_name)
    if not source_pricing:
        return {
            "source_model": model_name,
            "error": "Model not found. Run /sync/litellm first.",
            "alternatives": [],
        }
    
    alternatives = await pricing_service.discover_alternatives(
        model=model_name,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        requires_vision=requires_vision,
        requires_function_calling=requires_function_calling,
        same_provider_only=same_provider_only,
        max_results=max_results,
    )
    
    return {
        "source_model": model_name,
        "source_pricing": {
            "input_per_1k": source_pricing["input"],
            "output_per_1k": source_pricing["output"],
            "provider": source_pricing["provider"],
        },
        "alternatives_count": len(alternatives),
        "alternatives": alternatives,
    }


@router.post("/alternatives/generate")
async def generate_alternatives(
    max_alternatives_per_model: int = Query(5, ge=1, le=20),
    min_savings_percent: float = Query(10.0, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    """
    Auto-generate model alternatives by analyzing pricing data.
    
    This analyzes all 3500+ models in the pricing table and creates
    alternatives based on:
    - Same provider models (safer swaps)
    - Cross-provider models (for expensive models)
    - Capability matching (vision, function calling)
    - Price ratios and quality tier estimation
    
    Run this after syncing pricing data to populate the alternatives table.
    """
    from ..services.alternative_learning_service import AlternativeLearningService
    
    learning_service = AlternativeLearningService(db)
    result = await learning_service.generate_alternatives_from_pricing(
        max_alternatives_per_model=max_alternatives_per_model,
        min_savings_percent=min_savings_percent,
    )
    
    return result


@router.get("/alternatives/stats")
async def get_alternatives_stats(db: AsyncSession = Depends(get_db)):
    """
    Get statistics about the model alternatives learning system.
    
    Returns:
    - Total alternatives in database
    - How many have learning feedback
    - Confidence distribution
    - Total estimated vs actual savings tracked
    """
    from ..services.alternative_learning_service import AlternativeLearningService
    
    learning_service = AlternativeLearningService(db)
    stats = await learning_service.get_alternative_stats()
    
    return {
        "status": "ok",
        **stats,
    }


@router.get("/alternatives/{source_model:path}")
async def get_model_alternatives(
    source_model: str,
    min_confidence: float = Query(0.3, ge=0, le=1.0),
    max_results: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Get learned alternatives for a specific model.
    
    Returns alternatives ranked by confidence score and value
    (confidence × savings potential).
    """
    # Prevent matching reserved routes
    reserved_names = {"generate", "stats"}
    if source_model.lower() in reserved_names:
        return {
            "error": f"'{source_model}' is a reserved endpoint. Use POST /alternatives/generate instead.",
            "source_model": source_model,
            "alternatives_count": 0,
            "alternatives": [],
        }
    
    from ..services.alternative_learning_service import AlternativeLearningService
    
    learning_service = AlternativeLearningService(db)
    alternatives = await learning_service.get_learned_alternatives(
        source_model=source_model,
        min_confidence=min_confidence,
        max_results=max_results,
    )
    
    return {
        "source_model": source_model,
        "alternatives_count": len(alternatives),
        "alternatives": [
            {
                "alternative_model": alt.alternative_model,
                "confidence_score": round(alt.confidence_score, 3),
                "times_suggested": alt.times_suggested,
                "times_implemented": alt.times_implemented,
                "times_dismissed": alt.times_dismissed,
                "quality_tier": alt.quality_tier,
                "price_ratio": round(alt.price_ratio, 3),
                "savings_accuracy": round(alt.avg_accuracy * 100, 1) if alt.avg_accuracy else None,
                "same_provider": alt.same_provider,
                "source": alt.source,
            }
            for alt in alternatives
        ],
    }

