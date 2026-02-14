"""
Provides endpoints for model pricing management.

To sync pricing, call POST /v1/pricing/sync/litellm which fetches from:
https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timezone
from typing import Dict, Optional

from ..database import get_db
from ..models.db_models import ModelPricing
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


@router.get("/sync/status")
async def get_sync_status(db: AsyncSession = Depends(get_db)):
    """Get pricing sync status."""
    total_query = select(func.count(ModelPricing.id)).where(ModelPricing.is_active == True)
    total_result = await db.execute(total_query)
    total_models = total_result.scalar() or 0
    
    last_update_query = select(func.max(ModelPricing.updated_at))
    last_update_result = await db.execute(last_update_query)
    last_updated = last_update_result.scalar()
    
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
        message = "No models in database. Run POST /v1/pricing/sync/litellm to sync 1600+ models."
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
                'provider': model.provider,
                'updated_at': model.updated_at.isoformat() if model.updated_at else None,
            }
        return {
            "pricing": pricing,
            "source": "database",
            "last_updated": max(
                (m.updated_at for m in db_pricing if m.updated_at),
                default=datetime.now(timezone.utc)
            ).isoformat(),
        }
    
    pricing = DEFAULT_PRICING
    if provider:
        pricing = {k: v for k, v in pricing.items() if v.get('provider') == provider}
    
    return {
        "pricing": pricing,
        "source": "defaults",
        "last_updated": None,
    }


@router.get("/{model_name}")
async def get_model_pricing(model_name: str, db: AsyncSession = Depends(get_db)):
    """Get pricing for a specific model."""
    query = select(ModelPricing).where(
        ModelPricing.model_name == model_name,
        ModelPricing.is_active == True
    )
    result = await db.execute(query)
    model = result.scalar_one_or_none()
    
    if model:
        return {
            "model": model_name,
            "input": model.input_price_per_1k,
            "output": model.output_price_per_1k,
            "provider": model.provider,
            "source": "database",
        }
    
    model_lower = model_name.lower()
    
    # Sort by length descending to match most specific model first (e.g. "gpt-4-turbo" before "gpt-4")
    sorted_pricing = sorted(DEFAULT_PRICING.items(), key=lambda x: len(x[0]), reverse=True)
    
    for name, pricing in sorted_pricing:
        # Check if known model name is contained in requested model name
        # e.g. "gpt-4" in "gpt-4-0613" -> Match
        # But "gpt-4-turbo" NOT in "gpt-4" -> No match (correct)
        if name in model_lower:
            return {
                "model": model_name,
                "matched_to": name,
                "input": pricing['input'],
                "output": pricing['output'],
                "provider": pricing['provider'],
                "source": "defaults",
            }
    
    return {
        "model": model_name,
        "input": 0.0,
        "output": 0.0,
        "provider": "unknown",
        "source": "fallback",
    }


@router.post("")
async def update_pricing(
    pricing_updates: Dict[str, Dict[str, float]],
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    """Update pricing for models (Admin)."""
    updated_count = 0
    
    for model_name, prices in pricing_updates.items():
        query = select(ModelPricing).where(ModelPricing.model_name == model_name)
        result = await db.execute(query)
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.input_price_per_1k = prices.get('input', existing.input_price_per_1k)
            existing.output_price_per_1k = prices.get('output', existing.output_price_per_1k)
            existing.provider = prices.get('provider', existing.provider)
            existing.updated_at = datetime.now(timezone.utc)
        else:
            new_pricing = ModelPricing(
                model_name=model_name,
                input_price_per_1k=prices.get('input', 0.0),
                output_price_per_1k=prices.get('output', 0.0),
                provider=prices.get('provider', 'unknown'),
            )
            db.add(new_pricing)
        
        updated_count += 1
    
    await db.commit()
    
    return {
        "status": "ok",
        "models_updated": updated_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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
        
        return result
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
    
    This analyzes all 2000+ models in the pricing table and creates
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

