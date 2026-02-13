"""
AgentCost Backend - Pricing Service
"""

import httpx
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.db_models import ModelPricing
from ..config import get_settings

# Get configurable URLs from settings (with same defaults as fallback)
_settings = get_settings()
LITELLM_PRICING_URL = _settings.litellm_pricing_url
OPENROUTER_MODELS_URL = _settings.openrouter_models_url

PROVIDER_PREFIXES = {
    "openai/": "openai",
    "anthropic/": "anthropic",
    "google/": "google",
    "vertex_ai/": "google",
    "groq/": "groq",
    "mistral/": "mistral",
    "cohere/": "cohere",
    "deepseek/": "deepseek",
    "together_ai/": "together",
    "fireworks_ai/": "fireworks",
    "azure/": "azure",
    "bedrock/": "aws",
}


class PricingService:
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    async def get_model_pricing(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get pricing for a specific model."""
        query = select(ModelPricing).where(
            ModelPricing.model_name == model_name,
            ModelPricing.is_active == True
        )
        result = await self.db.execute(query)
        model = result.scalar_one_or_none()
        
        if model:
            return {
                "input": model.input_price_per_1k,
                "output": model.output_price_per_1k,
                "provider": model.provider,
            }
        
        # Fuzzy match using SQL LIKE instead of loading all models into memory
        model_lower = model_name.lower()
        fuzzy_query = select(ModelPricing).where(
            ModelPricing.is_active == True,
            ModelPricing.model_name.ilike(f"%{model_lower}%")
        ).limit(1)
        result = await self.db.execute(fuzzy_query)
        m = result.scalar_one_or_none()
        
        if m:
            return {
                "input": m.input_price_per_1k,
                "output": m.output_price_per_1k,
                "provider": m.provider,
            }
        
        return None
    
    async def get_all_pricing(self, provider: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get all active model pricing."""
        query = select(ModelPricing).where(ModelPricing.is_active == True)
        if provider:
            query = query.where(ModelPricing.provider == provider)
        
        result = await self.db.execute(query)
        models = result.scalars().all()
        
        pricing = {}
        for m in models:
            pricing[m.model_name] = {
                "input": m.input_price_per_1k,
                "output": m.output_price_per_1k,
                "provider": m.provider,
                "max_tokens": m.max_tokens,
                "supports_vision": m.supports_vision,
                "supports_function_calling": m.supports_function_calling,
                "pricing_source": m.pricing_source,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
        
        return pricing
    
    async def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a model call."""
        pricing = await self.get_model_pricing(model)
        if pricing is None:
            return 0.0
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 8)
    
    async def calculate_potential_savings(
        self,
        current_model: str,
        alternative_model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Tuple[float, float]:
        """Calculate savings when switching models. Returns (absolute, percentage)."""
        current_cost = await self.calculate_cost(current_model, input_tokens, output_tokens)
        alternative_cost = await self.calculate_cost(alternative_model, input_tokens, output_tokens)
        
        if current_cost == 0:
            return (0.0, 0.0)
        
        absolute_savings = current_cost - alternative_cost
        percentage_savings = (absolute_savings / current_cost) * 100
        return (round(absolute_savings, 8), round(percentage_savings, 2))
    
    async def sync_from_litellm(self, track_changes: bool = False) -> Dict[str, Any]:
        """Sync pricing from LiteLLM's pricing database."""
        client = await self._get_client()
        
        try:
            response = await client.get(LITELLM_PRICING_URL)
            response.raise_for_status()
            pricing_data = response.json()
        except Exception as e:
            return {"status": "error", "error": str(e), "models_updated": 0}
        
        updated_count = 0
        created_count = 0
        skipped_count = 0
        changes = {"new_models": [], "price_changes": [], "capability_changes": []}
        
        for model_key, model_data in pricing_data.items():
            if not isinstance(model_data, dict):
                skipped_count += 1
                continue
            
            input_price = model_data.get("input_cost_per_token", 0)
            output_price = model_data.get("output_cost_per_token", 0)
            
            if input_price == 0 and output_price == 0:
                skipped_count += 1
                continue
            
            input_price_per_1k = input_price * 1000
            output_price_per_1k = output_price * 1000
            
            # Use litellm_provider from the data if available, otherwise parse from key
            litellm_provider = model_data.get("litellm_provider")
            if litellm_provider:
                provider = self._normalize_provider(litellm_provider)
                # Keep the full model key as the name for uniqueness
                model_name = model_key
            else:
                model_name, provider = self._parse_litellm_model_key(model_key)
            
            max_tokens = model_data.get("max_tokens") or model_data.get("max_output_tokens")
            supports_vision = model_data.get("supports_vision", False)
            supports_function_calling = model_data.get("supports_function_calling", False)
            
            query = select(ModelPricing).where(ModelPricing.model_name == model_name)
            result = await self.db.execute(query)
            existing = result.scalar_one_or_none()
            
            if existing:
                if track_changes:
                    old_input = existing.input_price_per_1k
                    old_output = existing.output_price_per_1k
                    
                    input_change_pct = ((input_price_per_1k - old_input) / old_input * 100) if old_input > 0 else (100 if input_price_per_1k > 0 else 0)
                    output_change_pct = ((output_price_per_1k - old_output) / old_output * 100) if old_output > 0 else (100 if output_price_per_1k > 0 else 0)
                    
                    if abs(input_change_pct) > 1 or abs(output_change_pct) > 1:
                        changes["price_changes"].append({
                            "model": model_name,
                            "provider": provider,
                            "old_input": round(old_input, 6),
                            "new_input": round(input_price_per_1k, 6),
                            "input_change_pct": round(input_change_pct, 2),
                            "old_output": round(old_output, 6),
                            "new_output": round(output_price_per_1k, 6),
                            "output_change_pct": round(output_change_pct, 2),
                        })
                    
                    if existing.supports_vision != supports_vision:
                        changes["capability_changes"].append({
                            "model": model_name, "change": "vision",
                            "old": existing.supports_vision, "new": supports_vision,
                        })
                    if existing.supports_function_calling != supports_function_calling:
                        changes["capability_changes"].append({
                            "model": model_name, "change": "function_calling",
                            "old": existing.supports_function_calling, "new": supports_function_calling,
                        })
                
                existing.input_price_per_1k = input_price_per_1k
                existing.output_price_per_1k = output_price_per_1k
                existing.provider = provider
                existing.max_tokens = max_tokens
                existing.supports_vision = supports_vision
                existing.supports_function_calling = supports_function_calling
                existing.pricing_source = "litellm"
                existing.source_updated_at = datetime.now(timezone.utc)
                existing.updated_at = datetime.now(timezone.utc)
                updated_count += 1
            else:
                if track_changes:
                    changes["new_models"].append({
                        "model": model_name,
                        "provider": provider,
                        "input_price": round(input_price_per_1k, 6),
                        "output_price": round(output_price_per_1k, 6),
                        "supports_vision": supports_vision,
                        "supports_function_calling": supports_function_calling,
                    })
                
                new_pricing = ModelPricing(
                    model_name=model_name,
                    input_price_per_1k=input_price_per_1k,
                    output_price_per_1k=output_price_per_1k,
                    provider=provider,
                    max_tokens=max_tokens,
                    supports_vision=supports_vision,
                    supports_function_calling=supports_function_calling,
                    pricing_source="litellm",
                    source_updated_at=datetime.now(timezone.utc),
                )
                self.db.add(new_pricing)
                created_count += 1
        
        await self.db.flush()
        
        result = {
            "status": "ok",
            "source": "litellm",
            "models_created": created_count,
            "models_updated": updated_count,
            "models_skipped": skipped_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if track_changes:
            result["changes"] = changes
            result["has_changes"] = bool(changes["new_models"] or changes["price_changes"] or changes["capability_changes"])
        
        return result
    
    async def sync_from_openrouter(self) -> Dict[str, Any]:
        """Sync pricing from OpenRouter API."""
        client = await self._get_client()
        
        try:
            response = await client.get(OPENROUTER_MODELS_URL)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
        except Exception as e:
            return {"status": "error", "error": str(e), "models_updated": 0}
        
        updated_count = 0
        created_count = 0
        
        for model_data in models:
            model_id = model_data.get("id", "")
            pricing = model_data.get("pricing", {})
            
            try:
                input_price = float(pricing.get("prompt", "0"))
                output_price = float(pricing.get("completion", "0"))
            except (ValueError, TypeError):
                continue
            
            if input_price == 0 and output_price == 0:
                continue
            
            input_price_per_1k = input_price * 1000
            output_price_per_1k = output_price * 1000
            provider = model_id.split("/")[0] if "/" in model_id else "unknown"
            model_name = model_id.split("/")[-1] if "/" in model_id else model_id
            context_length = model_data.get("context_length")
            
            query = select(ModelPricing).where(ModelPricing.model_name == model_name)
            result = await self.db.execute(query)
            existing = result.scalar_one_or_none()
            
            if existing:
                if existing.pricing_source != "litellm":
                    existing.input_price_per_1k = input_price_per_1k
                    existing.output_price_per_1k = output_price_per_1k
                    existing.provider = provider
                    existing.max_tokens = context_length
                    existing.pricing_source = "openrouter"
                    existing.source_updated_at = datetime.now(timezone.utc)
                    updated_count += 1
            else:
                new_pricing = ModelPricing(
                    model_name=model_name,
                    input_price_per_1k=input_price_per_1k,
                    output_price_per_1k=output_price_per_1k,
                    provider=provider,
                    max_tokens=context_length,
                    pricing_source="openrouter",
                    source_updated_at=datetime.now(timezone.utc),
                )
                self.db.add(new_pricing)
                created_count += 1
        
        await self.db.flush()
        
        return {
            "status": "ok",
            "source": "openrouter",
            "models_created": created_count,
            "models_updated": updated_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _parse_litellm_model_key(self, model_key: str) -> Tuple[str, str]:
        """Parse model key into (name, provider)."""
        for prefix, prov in PROVIDER_PREFIXES.items():
            if model_key.startswith(prefix):
                return (model_key[len(prefix):], prov)
        
        if "/" in model_key:
            parts = model_key.split("/", 1)
            return (parts[1], parts[0])
        
        return (model_key, "unknown")
    
    def _normalize_provider(self, litellm_provider: str) -> str:
        """Normalize litellm_provider to a clean provider name."""
        provider_lower = litellm_provider.lower()
        
        # Handle vertex_ai-* patterns (e.g., vertex_ai-anthropic_models -> google)
        # All vertex_ai models are served via Google Cloud, so normalize to google
        if provider_lower.startswith("vertex_ai"):
            return "google"
        
        # Handle bedrock_* patterns (e.g., bedrock_converse -> aws)
        if provider_lower.startswith("bedrock"):
            return "aws"
        
        # Handle azure_* patterns
        if provider_lower.startswith("azure"):
            return "azure"
        
        # Handle fireworks_ai-* patterns
        if provider_lower.startswith("fireworks_ai"):
            return "fireworks"
        
        # Handle cohere_* patterns
        if provider_lower.startswith("cohere"):
            return "cohere"
        
        # Handle text-completion-openai
        if provider_lower == "text-completion-openai":
            return "openai"
        
        provider_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google": "google",
            "gemini": "google",
            "groq": "groq",
            "mistral": "mistral",
            "deepseek": "deepseek",
            "together_ai": "together",
            "replicate": "replicate",
            "openrouter": "openrouter",
            "perplexity": "perplexity",
            "xai": "xai",
            "novita": "novita",
            "vercel_ai_gateway": "vercel",
            "gradient_ai": "gradient",
            "amazon-nova": "amazon",
            "amazon_nova": "amazon",
            "anyscale": "anyscale",
            "cerebras": "cerebras",
            "cloudflare": "cloudflare",
            "dashscope": "dashscope",
            "databricks": "databricks",
            "deepinfra": "deepinfra",
            "friendliai": "friendliai",
            "gmi": "gmi",
            "hyperbolic": "hyperbolic",
            "jina_ai": "jina",
            "lambda_ai": "lambda",
            "llamagate": "llamagate",
            "minimax": "minimax",
            "moonshot": "moonshot",
            "morph": "morph",
            "nlp_cloud": "nlp_cloud",
            "nscale": "nscale",
            "oci": "oracle",
            "ovhcloud": "ovhcloud",
            "palm": "google",
            "sambanova": "sambanova",
            "v0": "vercel",
            "voyage": "voyage",
            "wandb": "wandb",
            "watsonx": "ibm",
            "zai": "zai",
            "ai21": "ai21",
            "aleph_alpha": "aleph_alpha",
        }
        return provider_map.get(provider_lower, provider_lower)
    
    async def discover_alternatives(
        self,
        model: str,
        avg_input_tokens: Optional[int] = None,
        avg_output_tokens: Optional[int] = None,
        requires_vision: bool = False,
        requires_function_calling: bool = False,
        same_provider_only: bool = False,
        max_results: int = 5,
        use_learned: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find cheaper model alternatives using learned data first, then dynamic discovery.
        
        The system prioritizes alternatives that have been:
        1. Implemented by users with good outcomes
        2. High confidence scores from feedback
        3. Good savings accuracy (actual vs estimated)
        
        Falls back to dynamic price-based discovery if no learned data exists.
        """
        source_pricing = await self.get_model_pricing(model)
        if not source_pricing:
            return []
        
        source_total_cost = source_pricing["input"] + source_pricing["output"]
        source_provider = source_pricing.get("provider", "unknown")
        
        # Step 1: Try learned alternatives first
        if use_learned:
            learned_alternatives = await self._get_learned_alternatives(
                model=model,
                avg_input_tokens=avg_input_tokens,
                avg_output_tokens=avg_output_tokens,
                requires_vision=requires_vision,
                requires_function_calling=requires_function_calling,
                same_provider_only=same_provider_only,
                source_provider=source_provider,
                max_results=max_results,
            )
            
            if learned_alternatives:
                # Format learned alternatives with confidence data
                formatted = []
                for alt in learned_alternatives:
                    # Get pricing for the alternative
                    alt_pricing = await self.get_model_pricing(alt.alternative_model)
                    if not alt_pricing:
                        continue
                    
                    alt_total = alt_pricing["input"] + alt_pricing["output"]
                    input_savings = source_pricing["input"] - alt_pricing["input"]
                    output_savings = source_pricing["output"] - alt_pricing["output"]
                    total_savings = input_savings + output_savings
                    savings_pct = (total_savings / source_total_cost * 100) if source_total_cost > 0 else 0
                    
                    # Do not infer quality from price-based tiers
                    quality_impact = None
                    
                    formatted.append({
                        "model": alt.alternative_model,
                        "provider": alt.alternative_provider or alt_pricing.get("provider", "unknown"),
                        "pricing": {
                            "input_per_1k": round(alt_pricing["input"], 6),
                            "output_per_1k": round(alt_pricing["output"], 6),
                        },
                        "savings": {
                            "input_per_1k": round(input_savings, 6),
                            "output_per_1k": round(output_savings, 6),
                            "total_per_1k": round(total_savings, 6),
                            "percentage": round(savings_pct, 2),
                        },
                        "quality_impact": quality_impact,
                        "same_provider": alt.same_provider,
                        "capabilities": {
                            "vision": alt.requires_vision,
                            "function_calling": alt.requires_function_calling,
                            "max_tokens": alt.max_input_tokens_threshold,
                        },
                        # Learned data
                        "source": "learned",
                        "confidence_score": round(alt.confidence_score, 3),
                        "times_implemented": alt.times_implemented,
                        "times_dismissed": alt.times_dismissed,
                        "savings_accuracy": round(alt.avg_accuracy * 100, 1) if alt.avg_accuracy else None,
                    })
                
                if formatted:
                    return formatted[:max_results]
        
        # Step 2: Fall back to dynamic discovery
        return await self._discover_dynamically(
            model=model,
            source_pricing=source_pricing,
            source_total_cost=source_total_cost,
            source_provider=source_provider,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            requires_vision=requires_vision,
            requires_function_calling=requires_function_calling,
            same_provider_only=same_provider_only,
            max_results=max_results,
        )
    
    async def _get_learned_alternatives(
        self,
        model: str,
        avg_input_tokens: Optional[int],
        avg_output_tokens: Optional[int],
        requires_vision: bool,
        requires_function_calling: bool,
        same_provider_only: bool,
        source_provider: str,
        max_results: int,
        min_confidence: float = 0.3,
    ) -> List:
        """Get learned alternatives from ModelAlternative table."""
        from ..models.db_models import ModelAlternative
        
        query = select(ModelAlternative).where(
            ModelAlternative.source_model == model,
            ModelAlternative.is_active == True,
            ModelAlternative.confidence_score >= min_confidence,
        )
        
        if requires_vision:
            query = query.where(ModelAlternative.requires_vision == True)
        if requires_function_calling:
            query = query.where(ModelAlternative.requires_function_calling == True)
        if same_provider_only:
            query = query.where(ModelAlternative.same_provider == True)
        
        # Improved ranking:
        # 1. same_provider DESC (True=1 comes before False=0)
        # 2. quality_tier ASC (tier 1 is best)
        # 3. confidence_score DESC (higher is better)
        # 4. price_ratio ASC (lower = more savings)
        query = query.order_by(
            ModelAlternative.same_provider.desc(),
            ModelAlternative.confidence_score.desc(),
            ModelAlternative.price_ratio.asc(),
        ).limit(max_results)
        
        result = await self.db.execute(query)
        alternatives = result.scalars().all()

        if avg_input_tokens is None and avg_output_tokens is None:
            return alternatives

        total_tokens = (avg_input_tokens or 0) + (avg_output_tokens or 0)
        if total_tokens <= 0:
            return alternatives

        # Filter alternatives that cannot support observed token usage
        filtered = [
            alt for alt in alternatives
            if not alt.max_input_tokens_threshold or alt.max_input_tokens_threshold >= total_tokens
        ]

        return filtered
    
    def _tier_to_quality_impact(self, tier: int) -> str:
        """Convert quality tier to impact string."""
        if not tier:
            return None
        if tier <= 2:
            return "minimal"
        elif tier <= 3:
            return "moderate"
        else:
            return "significant"
    
    async def _discover_dynamically(
        self,
        model: str,
        source_pricing: Dict,
        source_total_cost: float,
        source_provider: str,
        avg_input_tokens: Optional[int],
        avg_output_tokens: Optional[int],
        requires_vision: bool,
        requires_function_calling: bool,
        same_provider_only: bool,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Dynamic discovery of alternatives based on pricing (original logic)."""
        query = select(ModelPricing).where(
            ModelPricing.is_active == True,
            ModelPricing.model_name != model,
            (ModelPricing.input_price_per_1k + ModelPricing.output_price_per_1k) < source_total_cost
        )
        
        if same_provider_only:
            query = query.where(ModelPricing.provider == source_provider)
        
        result = await self.db.execute(query)
        cheaper_models = result.scalars().all()
        
        alternatives = []
        
        for alt in cheaper_models:
            if requires_vision and not alt.supports_vision:
                continue
            if requires_function_calling and not alt.supports_function_calling:
                continue
            
            if alt.max_tokens:
                total_tokens = (avg_input_tokens or 0) + (avg_output_tokens or 0)
                if total_tokens > 0 and alt.max_tokens < total_tokens:
                    continue
            
            input_savings = source_pricing["input"] - alt.input_price_per_1k
            output_savings = source_pricing["output"] - alt.output_price_per_1k
            total_savings = input_savings + output_savings
            savings_pct = (total_savings / source_total_cost * 100) if source_total_cost > 0 else 0
            
            # Only learned alternatives can have quality assessments
            # Price-based assumptions are misleading (cheaper ≠ worse quality)
            
            alternatives.append({
                "model": alt.model_name,
                "provider": alt.provider,
                "pricing": {
                    "input_per_1k": round(alt.input_price_per_1k, 6),
                    "output_per_1k": round(alt.output_price_per_1k, 6),
                },
                "savings": {
                    "input_per_1k": round(input_savings, 6),
                    "output_per_1k": round(output_savings, 6),
                    "total_per_1k": round(total_savings, 6),
                    "percentage": round(savings_pct, 2),
                },
                "quality_impact": None,  # Only set for learned alternatives
                "same_provider": alt.provider == source_provider,
                "capabilities": {
                    "vision": alt.supports_vision,
                    "function_calling": alt.supports_function_calling,
                    "max_tokens": alt.max_tokens,
                },
                "source": "dynamic",
                "confidence_score": None,
                "times_implemented": None,
                "times_dismissed": None,
                "savings_accuracy": None,
            })
        
        alternatives.sort(key=lambda x: (x["same_provider"], x["savings"]["percentage"]), reverse=True)
        return alternatives[:max_results]
