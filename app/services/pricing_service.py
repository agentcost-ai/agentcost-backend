"""
AgentCost Backend - Pricing Service
"""

import re

import httpx
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import case, func, or_, select

from ..common import MAX_PRICE_PER_1K
from ..models.db_models import ModelPricing
from ..config import get_settings
from .pricing_math import price_event

# Separators model vendors use between name components ('gpt-4o-mini-2024-07-18',
# 'anthropic/claude-3-5-sonnet', 'gemini-1.5-pro-002').
_TOKEN_SEPARATOR = re.compile(r"[-_/.:@ ]")
# Upper bound on the IN list a fuzzy lookup may build.
_MAX_FUZZY_CANDIDATES = 60


def _per_1k(per_token) -> Optional[float]:
    """Convert an upstream per-token price to per-1k, or None if unpublished.

    Distinguishing None from 0.0 matters: None means "this model has no cache
    rate, bill cached tokens at the full input rate", while 0.0 would mean
    "cached tokens are free" and silently zero out real spend.
    """
    if per_token is None:
        return None
    try:
        value = float(per_token)
    except (TypeError, ValueError):
        return None
    return value * 1000 if value > 0 else None

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
    "xai/": "xai",
    "perplexity/": "perplexity",
    "replicate/": "replicate",
    "cerebras/": "cerebras",
    "deepinfra/": "deepinfra",
    "sambanova/": "sambanova",
    "ai21/": "ai21",
    "novita/": "novita",
}

# Providers whose models use canonical names (strip prefix for dedup).
# Platform/hosting providers keep the prefix to preserve separate pricing.
CANONICAL_PROVIDERS = {
    "openai", "anthropic", "google", "gemini", "groq", "mistral",
    "cohere", "deepseek", "together_ai", "xai", "perplexity", "replicate",
    "ai21", "cerebras", "deepinfra", "fireworks_ai", "novita", "sambanova",
    "moonshot", "minimax", "dashscope", "nlp_cloud",
}

PLATFORM_PROVIDERS = {
    "azure", "azure_ai", "bedrock", "bedrock_converse",
    "vertex_ai", "sagemaker",
}

# Rows deactivated automatically carry this notes prefix; only such rows are
# auto-reactivated when the model reappears upstream.
AUTO_DEACTIVATED_MARKER = "auto-deactivated:"


class PricingService:

    def __init__(self, db: AsyncSession, *, memoize_lookups: bool = False):
        self.db = db
        self._http_client: Optional[httpx.AsyncClient] = None
        # Opt-in, request-scoped memo of get_model_pricing. One optimization
        # request resolves the same names over and over (each group's source
        # model plus every alternative), and a miss costs up to three queries.
        # Off by default: anything longer-lived would serve stale prices.
        self._lookup_memo: Optional[Dict[str, Optional[Dict[str, Any]]]] = (
            {} if memoize_lookups else None
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    @staticmethod
    def _price_payload(model: ModelPricing, match: str) -> Dict[str, Any]:
        """Shape a ModelPricing row for callers, carrying how it was matched."""
        return {
            "input": model.input_price_per_1k,
            "output": model.output_price_per_1k,
            # None where the provider publishes no cache rate; callers fall
            # back to the standard input rate rather than assuming a discount.
            "cached_input": model.cached_input_price_per_1k,
            "cache_write": model.cache_write_price_per_1k,
            "provider": model.provider,
            # "exact" | "fuzzy" — event_service records this as cost_source so
            # an approximated price is distinguishable from a real one.
            "match": match,
            "matched_model": model.model_name,
        }

    async def get_model_pricing(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get pricing for a specific model.

        Falls back to a *deterministic, most-specific* fuzzy match: event_service
        overwrites the SDK's own cost with whatever this returns, so a match that
        depends on planner order would bill a model at unrelated rates.
        """
        if self._lookup_memo is not None and model_name in self._lookup_memo:
            return self._lookup_memo[model_name]

        payload = await self._resolve_model_pricing(model_name)
        if self._lookup_memo is not None:
            self._lookup_memo[model_name] = payload
        return payload

    async def prefetch_pricing(self, model_names) -> None:
        """Warm the memo for many models with a single exact-match IN query.

        Names with no exact row are left out, so they still fall through to the
        fuzzy path on their first real lookup. No-op without the memo enabled.
        """
        if self._lookup_memo is None:
            return

        wanted = {name for name in model_names if name and name not in self._lookup_memo}
        if not wanted:
            return

        query = select(ModelPricing).where(
            ModelPricing.model_name.in_(wanted),
            ModelPricing.is_active == True,  # noqa: E712
        )
        rows = (await self.db.execute(query)).scalars().all()
        self._lookup_memo.update(
            {row.model_name: self._price_payload(row, "exact") for row in rows}
        )

    async def _resolve_model_pricing(self, model_name: str) -> Optional[Dict[str, Any]]:
        query = select(ModelPricing).where(
            ModelPricing.model_name == model_name,
            ModelPricing.is_active == True
        )
        result = await self.db.execute(query)
        model = result.scalar_one_or_none()

        if model:
            return self._price_payload(model, "exact")

        model_lower = model_name.strip().lower()
        if not model_lower:
            return None

        # 1. A known model whose name is contained in the requested one, e.g.
        #    'gpt-4o-mini' for 'gpt-4o-mini-2024-07-18'. Longest wins — the same
        #    "most specific family" rule the SDK uses (cost_calculator.py
        #    _best_substring_match) — so 'gpt-4' can't outbid 'gpt-4o-mini'.
        contained = await self._match_contained_in(model_lower)
        if contained is not None:
            # A row that differs only in case is still an exact hit, not a guess.
            matched_lower = (contained.model_name or "").lower()
            return self._price_payload(
                contained, "exact" if matched_lower == model_lower else "fuzzy"
            )

        # 2. Otherwise a known model that *contains* the requested name, e.g.
        #    'claude-3-5-haiku-20241022' for 'claude-3-5-haiku'. Shortest wins:
        #    the least-decorated superset is the closest variant. Name is the
        #    tie-break so identical-length candidates can't flip between calls.
        escaped = model_lower.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        superset_query = (
            select(ModelPricing)
            .where(
                ModelPricing.is_active == True,
                ModelPricing.model_name.ilike(f"%{escaped}%", escape="\\"),
            )
            .order_by(
                func.length(ModelPricing.model_name).asc(),
                ModelPricing.model_name.asc(),
            )
            .limit(1)
        )
        result = await self.db.execute(superset_query)
        m = result.scalar_one_or_none()

        if m:
            return self._price_payload(m, "fuzzy")

        return None

    @staticmethod
    def _candidate_names(model_name: str) -> List[str]:
        """Every token-boundary slice of *model_name*, longest first.

        'gpt-4o-mini-2024-07-18' yields 'gpt-4o-mini-2024-07', 'gpt-4o-mini',
        '4o-mini', … so a family name stored in the pricing table can be found
        with one indexed IN lookup. Slicing on separators (rather than on every
        character) is what keeps 'gpt-4' from matching 'gpt-4o-mini': billing a
        mini model at gpt-4 rates is a 200x error, not a rounding error.
        """
        lowered = model_name.lower()
        boundaries = list(_TOKEN_SEPARATOR.finditer(lowered))
        starts = [0] + [m.end() for m in boundaries]
        ends = [m.start() for m in boundaries] + [len(lowered)]

        slices = {
            lowered[start:end]
            for start in starts
            for end in ends
            if end > start
        }
        slices.discard("")

        # Longest first: the most specific family is the closest price.
        ordered = sorted(slices, key=lambda s: (-len(s), s))
        return ordered[:_MAX_FUZZY_CANDIDATES]

    async def _match_contained_in(self, model_name: str) -> Optional[ModelPricing]:
        """Most specific active pricing row whose name is part of *model_name*."""
        candidates = self._candidate_names(model_name)
        if not candidates:
            return None

        # Both cases so a differently-cased row still matches, while the
        # equality predicate keeps the model_name index usable.
        lookup = sorted(set(candidates) | {c.upper() for c in candidates})

        rows = (
            await self.db.execute(
                select(ModelPricing).where(
                    ModelPricing.is_active == True,
                    ModelPricing.model_name.in_(lookup),
                )
            )
        ).scalars().all()
        if not rows:
            return None

        return sorted(rows, key=lambda r: (-len(r.model_name or ""), r.model_name or ""))[0]

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
                "cached_input": m.cached_input_price_per_1k,
                "cache_write": m.cache_write_price_per_1k,
                "provider": m.provider,
                "max_tokens": m.max_tokens,
                "supports_vision": m.supports_vision,
                "supports_function_calling": m.supports_function_calling,
                "pricing_source": m.pricing_source,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
        
        return pricing
    
    async def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Calculate cost for a model call."""
        pricing = await self.get_model_pricing(model)
        if pricing is None:
            return 0.0

        return price_event(
            pricing,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    
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
    
    async def sync_from_litellm(
        self,
        track_changes: bool = False,
        bundle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sync pricing from LiteLLM's pricing database.

        ``bundle`` supplies the same JSON directly instead of fetching it, for
        deployments with no egress to GitHub. Parsing and validation are
        identical either way -- an air-gapped catalogue must not be a
        second-class one held to looser rules.
        """
        if bundle is not None:
            pricing_data = bundle
        else:
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
        unchanged_count = 0
        changes = {"new_models": [], "price_changes": [], "capability_changes": []}

        # Load the catalogue once. The per-model SELECT this replaces issued one
        # round trip per entry (~3,500), which is what made a sync take minutes.
        existing_by_name = {
            row.model_name: row
            for row in (await self.db.execute(select(ModelPricing))).scalars()
        }

        # Group source keys by canonical name, then pick one representative per
        # name before writing anything (see _select_representative). Writing
        # colliding keys in sequence meant "last key in the upstream JSON wins",
        # which let wandb's per-million unit error price shared models at up to
        # 540,000x their real rate -- and event_service bills off these rows.
        grouped: Dict[str, List[tuple]] = {}
        collision_count = 0
        rejected_count = 0

        for model_key, model_data in pricing_data.items():
            if not isinstance(model_data, dict):
                skipped_count += 1
                continue

            input_price = model_data.get("input_cost_per_token", 0)
            output_price = model_data.get("output_cost_per_token", 0)

            if input_price == 0 and output_price == 0:
                skipped_count += 1
                continue

            # Use litellm_provider from the data if available, otherwise parse from key
            litellm_provider = model_data.get("litellm_provider")
            if litellm_provider:
                provider = self._normalize_provider(litellm_provider)
                # Canonicalize: strip provider prefix for primary providers to
                # prevent duplicates with OpenRouter (which always strips).
                # Platform providers (azure, bedrock, vertex_ai) keep the prefix
                # because their pricing differs from the canonical provider.
                model_name = self._canonicalize_model_name(model_key, litellm_provider)
            else:
                model_name, provider = self._parse_litellm_model_key(model_key)

            if model_name in grouped:
                collision_count += 1
            grouped.setdefault(model_name, []).append(
                (model_key, model_data, provider, input_price * 1000, output_price * 1000)
            )

        # Names actually written this run. Distinct from grouped: a name whose
        # every listing failed the sanity bound is in grouped but not here, and
        # must be retired below like an absent one -- skipping it would leave a
        # previously-written unit-error row live forever.
        written_names = set()

        for model_name, candidates in grouped.items():
            chosen = self._select_representative(candidates)
            if chosen is None:
                # Every listing is implausible; better no row (event_service
                # falls back to the SDK's own cost) than a unit-error price.
                rejected_count += 1
                skipped_count += 1
                continue
            written_names.add(model_name)

            _key, model_data, provider, input_price_per_1k, output_price_per_1k = chosen

            max_tokens = model_data.get("max_tokens") or model_data.get("max_output_tokens")
            max_input_tokens = model_data.get("max_input_tokens")
            supports_vision = model_data.get("supports_vision", False)
            supports_function_calling = model_data.get("supports_function_calling", False)
            supports_streaming = model_data.get("supports_streaming", True)

            # Prompt-cache rates. LiteLLM quotes per token like the others, and
            # publishes them only for models that actually offer caching -- left
            # as None otherwise so pricing falls back to the full input rate
            # instead of silently discounting.
            cached_input_price_per_1k = _per_1k(
                model_data.get("cache_read_input_token_cost")
            )
            cache_write_price_per_1k = _per_1k(
                model_data.get("cache_creation_input_token_cost")
            )

            existing = existing_by_name.get(model_name)

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
                    if existing.supports_streaming != supports_streaming:
                        changes["capability_changes"].append({
                            "model": model_name, "change": "streaming",
                            "old": existing.supports_streaming, "new": supports_streaming,
                        })
                
                # Assign only what actually differs. Writing every field on every
                # run dirtied all ~3,500 rows each sync, so the UPDATE count
                # reported real churn as if it were price movement and the write
                # cost was paid whether or not anything had changed upstream.
                row_changed = False
                for field, value in (
                    ("input_price_per_1k", input_price_per_1k),
                    ("output_price_per_1k", output_price_per_1k),
                    ("cached_input_price_per_1k", cached_input_price_per_1k),
                    ("cache_write_price_per_1k", cache_write_price_per_1k),
                    ("provider", provider),
                    ("max_tokens", max_tokens),
                    ("max_input_tokens", max_input_tokens),
                    ("supports_vision", supports_vision),
                    ("supports_function_calling", supports_function_calling),
                    ("supports_streaming", supports_streaming),
                    ("pricing_source", "litellm"),
                ):
                    if getattr(existing, field) != value:
                        setattr(existing, field, value)
                        row_changed = True

                if row_changed:
                    now = datetime.now(timezone.utc)
                    existing.source_updated_at = now
                    existing.updated_at = now
                    updated_count += 1
                else:
                    unchanged_count += 1
            else:
                if track_changes:
                    changes["new_models"].append({
                        "model": model_name,
                        "provider": provider,
                        "input_price": round(input_price_per_1k, 6),
                        "output_price": round(output_price_per_1k, 6),
                        "supports_vision": supports_vision,
                        "supports_function_calling": supports_function_calling,
                        "supports_streaming": supports_streaming,
                    })
                
                new_pricing = ModelPricing(
                    model_name=model_name,
                    input_price_per_1k=input_price_per_1k,
                    output_price_per_1k=output_price_per_1k,
                    cached_input_price_per_1k=cached_input_price_per_1k,
                    cache_write_price_per_1k=cache_write_price_per_1k,
                    provider=provider,
                    max_tokens=max_tokens,
                    max_input_tokens=max_input_tokens,
                    supports_vision=supports_vision,
                    supports_function_calling=supports_function_calling,
                    supports_streaming=supports_streaming,
                    pricing_source="litellm",
                    source_updated_at=datetime.now(timezone.utc),
                )
                self.db.add(new_pricing)
                created_count += 1

        # Retire litellm-sourced rows that vanished from the feed; a model
        # nobody prices anymore must stop being recommended as an alternative.
        # Only rows this mechanism (or the repair script) deactivated are ever
        # reactivated on return -- an admin's deliberate disable is not fought.
        deactivated_count = 0
        for name, row in existing_by_name.items():
            if row.pricing_source != "litellm":
                continue
            if name not in written_names and row.is_active:
                row.is_active = False
                row.notes = AUTO_DEACTIVATED_MARKER + (
                    " no plausible listing upstream" if name in grouped
                    else " absent from litellm feed"
                )
                row.updated_at = datetime.now(timezone.utc)
                deactivated_count += 1
            elif (name in written_names and not row.is_active
                  and (row.notes or "").startswith(AUTO_DEACTIVATED_MARKER)):
                row.is_active = True
                row.notes = None
                row.updated_at = datetime.now(timezone.utc)

        await self.db.flush()

        result = {
            "status": "ok",
            "source": "litellm",
            "models_created": created_count,
            "models_updated": updated_count,
            "models_unchanged": unchanged_count,
            "models_skipped": skipped_count,
            # Multi-host listings that collapsed onto an already-claimed name;
            # non-zero is normal.
            "models_deduplicated": collision_count,
            # Names where every listing failed the price sanity bound.
            "models_rejected": rejected_count,
            # Active litellm rows retired because the feed no longer lists them.
            "models_deactivated": deactivated_count,
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
            
            # Check for existing record by canonical (stripped) name
            query = select(ModelPricing).where(ModelPricing.model_name == model_name)
            result = await self.db.execute(query)
            existing = result.scalar_one_or_none()
            
            # Also check if the full model_id exists (from LiteLLM platform providers)
            if not existing and "/" in model_id:
                full_query = select(ModelPricing).where(ModelPricing.model_name == model_id)
                full_result = await self.db.execute(full_query)
                if full_result.scalar_one_or_none():
                    # Model exists under its full platform key; skip to avoid duplicate
                    continue
            
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
    
    @staticmethod
    def _select_representative(candidates: List[tuple]) -> Optional[tuple]:
        """Pick one ``(source_key, model_data, provider, input_1k, output_1k)``
        listing to represent a canonical model name, or None if none is usable.

        Listings above MAX_PRICE_PER_1K are discarded as upstream unit errors
        (no real model costs $10/1k; the admin routes enforce the same bound).
        Of the rest, take the median by price: order-independent, so upstream
        reshuffles cannot rewrite prices, and immune to a single broken listing.
        A whole tuple is chosen so input/output stay one host's coherent pair.
        """
        sane = [
            c for c in candidates
            if c[3] <= MAX_PRICE_PER_1K and c[4] <= MAX_PRICE_PER_1K
        ]
        if not sane:
            return None
        # The source key breaks ties identically on every run.
        sane.sort(key=lambda c: (c[3], c[4], c[0]))
        return sane[len(sane) // 2]

    def _canonicalize_model_name(self, model_key: str, litellm_provider: str) -> str:
        """Canonicalize model name by stripping provider prefix for primary providers.
        
        Primary providers (openai, anthropic, etc.) get their prefix stripped so that
        'openai/gpt-4o-2024-11-20' becomes 'gpt-4o-2024-11-20', matching what SDKs
        report and what OpenRouter stores.
        
        Platform providers (azure, bedrock, vertex_ai) keep the full key because
        their pricing differs from the base provider.
        """
        if "/" not in model_key:
            return model_key
        
        provider_lower = litellm_provider.lower()
        
        # Platform providers: keep the full key for distinct pricing
        for platform in PLATFORM_PROVIDERS:
            if provider_lower.startswith(platform):
                return model_key
        
        # Primary providers: strip the prefix
        _prefix, _, base_name = model_key.partition("/")
        return base_name if base_name else model_key
    
    def _parse_litellm_model_key(self, model_key: str) -> Tuple[str, str]:
        """Parse model key into (name, provider)."""
        for prefix, prov in PROVIDER_PREFIXES.items():
            if model_key.startswith(prefix):
                return (model_key[len(prefix):], prov)
        
        if "/" in model_key:
            parts = model_key.split("/", 1)
            return (parts[1], parts[0])
        
        return (model_key, "unknown")
    
    # Only platform prefixes that remap to a different canonical name.
    # Everything else is derived dynamically from the raw provider string.
    _PLATFORM_REMAP = {
        "vertex_ai": "google",
        "bedrock": "aws",
        "bedrock_converse": "aws",
        "azure": "azure",
        "azure_ai": "azure",
        "sagemaker": "aws",
        "text-completion-openai": "openai",
        "palm": "google",
        "gemini": "google",
        "watsonx": "ibm",
        "oci": "oracle",
    }

    def _normalize_provider(self, litellm_provider: str) -> str:
        """Normalize litellm_provider to a clean, lowercase provider name.
        
        Strategy (no hardcoded map of every provider):
        1. Exact match in _PLATFORM_REMAP → use the remapped name.
        2. Prefix match in _PLATFORM_REMAP (e.g. 'vertex_ai-anthropic_models') → remap.
        3. Otherwise: strip common suffixes (_ai, _chat), replace separators, lowercase.
           This lets any new provider LiteLLM adds flow through automatically.
        """
        raw = litellm_provider.strip()
        if not raw:
            return "unknown"
        
        lower = raw.lower()
        
        # 1. Exact remap
        if lower in self._PLATFORM_REMAP:
            return self._PLATFORM_REMAP[lower]
        
        # 2. Prefix remap (e.g. vertex_ai-anthropic_models, bedrock_converse, azure_ai)
        for prefix, remapped in self._PLATFORM_REMAP.items():
            if lower.startswith(prefix + "-") or lower.startswith(prefix + "_"):
                return remapped
        
        # 3. Dynamic cleanup — derive from the raw string.
        #    'together_ai' → 'together', 'fireworks_ai' → 'fireworks',
        #    'jina_ai' → 'jina', 'lambda_ai' → 'lambda',
        #    'gradient_ai' → 'gradient', etc.
        #    Anything not matching a suffix rule passes through unchanged.
        name = lower.split("-")[0].split("_")  # split on both - and _
        # Remove trailing tokens that are generic qualifiers
        while len(name) > 1 and name[-1] in ("ai", "chat", "models", "gateway"):
            name.pop()
        return "_".join(name)
    
    async def _get_model_record(self, model_name: str) -> Optional[ModelPricing]:
        """Get a ModelPricing record by name."""
        query = select(ModelPricing).where(
            ModelPricing.model_name == model_name,
            ModelPricing.is_active == True,
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
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
        success_rate: Optional[float] = None,
        requires_streaming: bool = False,
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
                success_rate=success_rate,
                requires_streaming=requires_streaming,
            )
            
            if learned_alternatives:
                # Format learned alternatives with confidence data
                formatted = []
                for alt in learned_alternatives:
                    # Get pricing for the alternative
                    alt_pricing = await self.get_model_pricing(alt.alternative_model)
                    if not alt_pricing:
                        continue
                    
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
                            "max_output_tokens": alt.max_output_tokens_threshold,
                            "min_success_rate": alt.min_success_rate_required,
                        },
                        "notes": alt.notes,
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
            requires_streaming=requires_streaming,
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
        success_rate: Optional[float] = None,
        requires_streaming: bool = False,
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
        filtered = []
        for alt in alternatives:
            # Check max_input_tokens_threshold (total context window)
            if alt.max_input_tokens_threshold and alt.max_input_tokens_threshold < total_tokens:
                continue
            # Check max_output_tokens_threshold
            if avg_output_tokens and alt.max_output_tokens_threshold and alt.max_output_tokens_threshold < avg_output_tokens:
                continue
            # Check min_success_rate_required against current agent's success rate
            if success_rate is not None and alt.min_success_rate_required:
                if success_rate < alt.min_success_rate_required:
                    continue
            filtered.append(alt)

        # If streaming is required, verify alternative models support it
        if requires_streaming and filtered:
            verified = []
            for alt in filtered:
                alt_pricing = await self._get_model_record(alt.alternative_model)
                if alt_pricing and not alt_pricing.supports_streaming:
                    continue
                verified.append(alt)
            return verified

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
        requires_streaming: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Dynamic discovery of alternatives based on pricing.

        Filtering and ordering happen in SQL. Doing them in Python meant
        pulling every active model priced below the source into memory — after
        a LiteLLM sync that is thousands of rows — sorting them all, and then
        keeping a handful, once per distinct model in the request.

        The SQL ordering reproduces the previous Python sort exactly: for a
        fixed source model, savings percentage is strictly decreasing in the
        alternative's combined price, so ordering by that price ascending is
        the same ranking.
        """
        total_price = ModelPricing.input_price_per_1k + ModelPricing.output_price_per_1k

        query = select(ModelPricing).where(
            ModelPricing.is_active == True,
            ModelPricing.model_name != model,
            total_price < source_total_cost,
        )

        if same_provider_only:
            query = query.where(ModelPricing.provider == source_provider)
        if requires_vision:
            query = query.where(ModelPricing.supports_vision == True)
        if requires_function_calling:
            query = query.where(ModelPricing.supports_function_calling == True)
        if requires_streaming:
            query = query.where(ModelPricing.supports_streaming == True)

        # Each side of the workload against its own cap; NULL means "unknown"
        # and is accepted. max_tokens is the OUTPUT cap, so comparing it to
        # input+output excluded models whose context easily fits the workload
        # (an 8k-output model was hidden from a 20k-input workload it handles).
        if avg_input_tokens:
            query = query.where(
                or_(
                    ModelPricing.max_input_tokens.is_(None),
                    ModelPricing.max_input_tokens >= avg_input_tokens,
                )
            )
        if avg_output_tokens:
            query = query.where(
                or_(
                    ModelPricing.max_tokens.is_(None),
                    ModelPricing.max_tokens >= avg_output_tokens,
                )
            )

        query = query.order_by(
            case((ModelPricing.provider == source_provider, 1), else_=0).desc(),
            total_price.asc(),
        ).limit(max_results)

        result = await self.db.execute(query)
        cheaper_models = result.scalars().all()

        alternatives = []

        for alt in cheaper_models:
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
                    "streaming": alt.supports_streaming,
                    "max_tokens": alt.max_tokens,
                },
                "source": "dynamic",
                "confidence_score": None,
                "times_implemented": None,
                "times_dismissed": None,
                "savings_accuracy": None,
            })
        
        # Already ordered and capped by the query above.
        return alternatives
