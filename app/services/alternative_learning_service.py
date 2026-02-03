"""
AgentCost Backend - Alternative Learning Service

Auto-generates and learns model alternatives from:
1. Pricing data similarities (1900+ models)
2. Recommendation outcomes (implemented/dismissed)
3. User feedback patterns

This creates a self-improving system where alternatives get better over time.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..models.db_models import ModelPricing, ModelAlternative


class AlternativeLearningService:
    """
    Service for auto-generating and learning model alternatives.
    
    Instead of hardcoding alternatives, this service:
    1. Analyzes model_pricing to find potential alternatives
    2. Tracks user decisions (implement/dismiss) 
    3. Updates confidence scores based on outcomes
    4. Returns alternatives ranked by learned effectiveness
    """
    
    # Quality tier thresholds based on price ratio
    TIER_THRESHOLDS = {
        1: 0.8,   # Premium: 80-100% of source price
        2: 0.5,   # High: 50-80% of source price  
        3: 0.3,   # Medium: 30-50% of source price
        4: 0.1,   # Budget: 10-30% of source price
        5: 0.0,   # Economy: <10% of source price
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def generate_alternatives_from_pricing(
        self,
        max_alternatives_per_model: int = 5,
        min_savings_percent: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Auto-generate alternatives by analyzing model_pricing.
        Groups models by provider + capability, ranks by price.
        
        Returns statistics about alternatives created/updated.
        """
        # Get all active models with pricing
        query = select(ModelPricing).where(ModelPricing.is_active == True)
        result = await self.db.execute(query)
        all_models = result.scalars().all()
        
        if not all_models:
            return {
                "status": "ok",
                "message": "No models found in pricing table",
                "alternatives_created": 0,
                "alternatives_updated": 0,
            }
        
        # Group models by provider
        models_by_provider: Dict[str, List[ModelPricing]] = {}
        for model in all_models:
            provider = model.provider or "unknown"
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model)
        
        created_count = 0
        updated_count = 0
        
        # For each provider, create alternatives between models
        for provider, provider_models in models_by_provider.items():
            # Sort by total price (input + output per 1k)
            sorted_models = sorted(
                provider_models,
                key=lambda m: (m.input_price_per_1k or 0) + (m.output_price_per_1k or 0),
                reverse=True  # Most expensive first
            )
            
            # For each expensive model, find cheaper alternatives
            for i, expensive in enumerate(sorted_models):
                exp_price = (expensive.input_price_per_1k or 0) + (expensive.output_price_per_1k or 0)
                
                if exp_price <= 0:
                    continue
                
                alternatives_for_model = 0
                
                for cheaper in sorted_models[i + 1:]:
                    if alternatives_for_model >= max_alternatives_per_model:
                        break
                    
                    cheap_price = (cheaper.input_price_per_1k or 0) + (cheaper.output_price_per_1k or 0)
                    
                    if cheap_price <= 0:
                        continue
                    
                    # Calculate savings percentage
                    savings_pct = ((exp_price - cheap_price) / exp_price) * 100
                    
                    if savings_pct < min_savings_percent:
                        continue
                    
                    # Check capability compatibility
                    # Don't suggest non-vision model for vision tasks
                    requires_vision = expensive.supports_vision or False
                    requires_function_calling = expensive.supports_function_calling or False
                    
                    if requires_vision and not cheaper.supports_vision:
                        continue
                    if requires_function_calling and not cheaper.supports_function_calling:
                        continue
                    
                    # Calculate price ratio and quality tier
                    price_ratio = cheap_price / exp_price
                    quality_tier = self._calculate_quality_tier(price_ratio)
                    
                    # Create or update alternative
                    is_new = await self._upsert_alternative(
                        source_model=expensive.model_name,
                        alternative_model=cheaper.model_name,
                        source_provider=provider,
                        alternative_provider=cheaper.provider or provider,
                        price_ratio=price_ratio,
                        quality_tier=quality_tier,
                        requires_vision=requires_vision,
                        requires_function_calling=requires_function_calling,
                        max_input_tokens=cheaper.max_tokens,
                    )
                    
                    if is_new:
                        created_count += 1
                    else:
                        updated_count += 1
                    
                    alternatives_for_model += 1
        
        # Also create cross-provider alternatives for similar capability models
        cross_provider_stats = await self._generate_cross_provider_alternatives(
            all_models, max_alternatives_per_model, min_savings_percent
        )
        
        await self.db.commit()
        
        return {
            "status": "ok",
            "total_models_analyzed": len(all_models),
            "providers_analyzed": len(models_by_provider),
            "alternatives_created": created_count + cross_provider_stats["created"],
            "alternatives_updated": updated_count + cross_provider_stats["updated"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    async def _generate_cross_provider_alternatives(
        self,
        all_models: List[ModelPricing],
        max_alternatives: int,
        min_savings: float,
    ) -> Dict[str, int]:
        """Generate alternatives across different providers for similar models."""
        created = 0
        updated = 0
        
        # Sort all models by price
        sorted_models = sorted(
            all_models,
            key=lambda m: (m.input_price_per_1k or 0) + (m.output_price_per_1k or 0),
            reverse=True
        )
        
        # For top 20% most expensive models, find cross-provider alternatives
        top_expensive = sorted_models[:len(sorted_models) // 5]
        
        for expensive in top_expensive:
            exp_price = (expensive.input_price_per_1k or 0) + (expensive.output_price_per_1k or 0)
            
            if exp_price <= 0:
                continue
            
            cross_provider_count = 0
            
            for cheaper in all_models:
                if cross_provider_count >= max_alternatives:
                    break
                
                # Skip same model
                if cheaper.model_name == expensive.model_name:
                    continue
                
                # Skip same provider (already handled above)
                if cheaper.provider == expensive.provider:
                    continue
                
                cheap_price = (cheaper.input_price_per_1k or 0) + (cheaper.output_price_per_1k or 0)
                
                if cheap_price <= 0 or cheap_price >= exp_price:
                    continue
                
                savings_pct = ((exp_price - cheap_price) / exp_price) * 100
                
                if savings_pct < min_savings:
                    continue
                
                # Capability check
                if expensive.supports_vision and not cheaper.supports_vision:
                    continue
                if expensive.supports_function_calling and not cheaper.supports_function_calling:
                    continue
                
                price_ratio = cheap_price / exp_price
                quality_tier = self._calculate_quality_tier(price_ratio)
                
                is_new = await self._upsert_alternative(
                    source_model=expensive.model_name,
                    alternative_model=cheaper.model_name,
                    source_provider=expensive.provider,
                    alternative_provider=cheaper.provider,
                    price_ratio=price_ratio,
                    quality_tier=quality_tier,
                    requires_vision=expensive.supports_vision or False,
                    requires_function_calling=expensive.supports_function_calling or False,
                    max_input_tokens=cheaper.max_tokens,
                )
                
                if is_new:
                    created += 1
                else:
                    updated += 1
                
                cross_provider_count += 1
        
        return {"created": created, "updated": updated}
    
    async def _upsert_alternative(
        self,
        source_model: str,
        alternative_model: str,
        source_provider: str,
        alternative_provider: str,
        price_ratio: float,
        quality_tier: int,
        requires_vision: bool,
        requires_function_calling: bool,
        max_input_tokens: Optional[int] = None,
    ) -> bool:
        """Create or update an alternative. Returns True if new, False if updated."""
        query = select(ModelAlternative).where(
            ModelAlternative.source_model == source_model,
            ModelAlternative.alternative_model == alternative_model,
        )
        result = await self.db.execute(query)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update pricing-related fields, preserve learning data
            existing.price_ratio = price_ratio
            existing.quality_tier = quality_tier
            existing.source_provider = source_provider
            existing.alternative_provider = alternative_provider
            existing.same_provider = (source_provider == alternative_provider)
            existing.requires_vision = requires_vision
            existing.requires_function_calling = requires_function_calling
            if max_input_tokens:
                existing.max_input_tokens_threshold = max_input_tokens
            existing.updated_at = datetime.now(timezone.utc)
            return False
        else:
            # Create new alternative with neutral confidence
            new_alt = ModelAlternative(
                source_model=source_model,
                alternative_model=alternative_model,
                source_provider=source_provider,
                alternative_provider=alternative_provider,
                same_provider=(source_provider == alternative_provider),
                price_ratio=price_ratio,
                quality_tier=quality_tier,
                requires_vision=requires_vision,
                requires_function_calling=requires_function_calling,
                max_input_tokens_threshold=max_input_tokens,
                confidence_score=0.5,  # Neutral start
                source="auto",
            )
            self.db.add(new_alt)
            return True
    
    def _calculate_quality_tier(self, price_ratio: float) -> int:
        """Calculate quality tier from price ratio."""
        for tier, threshold in sorted(self.TIER_THRESHOLDS.items()):
            if price_ratio >= threshold:
                return tier
        return 5  # Economy tier
    
    async def update_from_recommendation_outcome(
        self,
        source_model: str,
        alternative_model: str,
        was_implemented: bool,
        estimated_savings: float = 0.0,
        actual_savings: Optional[float] = None,
        user_feedback: Optional[str] = None,
    ) -> Optional[ModelAlternative]:
        """
        Learn from recommendation outcomes.
        Called when user implements/dismisses a suggestion.
        
        Updates:
        - times_suggested, times_implemented, times_dismissed
        - total_estimated_savings, total_actual_savings
        - avg_accuracy (actual/estimated ratio)
        - confidence_score (recalculated)
        """
        # Get or create the alternative
        query = select(ModelAlternative).where(
            ModelAlternative.source_model == source_model,
            ModelAlternative.alternative_model == alternative_model,
        )
        result = await self.db.execute(query)
        alt = result.scalar_one_or_none()
        
        if not alt:
            # Create minimal entry for tracking
            alt = ModelAlternative(
                source_model=source_model,
                alternative_model=alternative_model,
                confidence_score=0.5,
                source="feedback",
            )
            self.db.add(alt)

        # Normalize nullable counters for legacy rows or new instances
        alt.times_suggested = alt.times_suggested or 0
        alt.times_implemented = alt.times_implemented or 0
        alt.times_dismissed = alt.times_dismissed or 0
        alt.total_estimated_savings = alt.total_estimated_savings or 0.0
        alt.total_actual_savings = alt.total_actual_savings or 0.0
        alt.avg_accuracy = alt.avg_accuracy or 0.0
        
        # Update suggestion count
        alt.times_suggested += 1
        
        if was_implemented:
            alt.times_implemented += 1
            alt.total_estimated_savings += estimated_savings
            
            if actual_savings is not None:
                alt.total_actual_savings += actual_savings
                # Update accuracy ratio
                if alt.total_estimated_savings > 0:
                    alt.avg_accuracy = alt.total_actual_savings / alt.total_estimated_savings
        else:
            alt.times_dismissed += 1
        
        # Recalculate confidence score
        alt.confidence_score = self._calculate_confidence(
            implemented=alt.times_implemented,
            dismissed=alt.times_dismissed,
            accuracy=alt.avg_accuracy,
            user_feedback=user_feedback,
        )
        
        alt.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        
        return alt
    
    def _calculate_confidence(
        self,
        implemented: int,
        dismissed: int,
        accuracy: float,
        user_feedback: Optional[str] = None,
    ) -> float:
        """
        Calculate confidence score based on outcomes.
        
        Formula:
        - Base: implementation_rate (impl / (impl + dismissed))
        - Boost: accuracy bonus if actual savings >= 80% of estimated
        - Penalty: negative feedback keywords
        - Minimum samples: need 3+ decisions to move from neutral
        """
        total = implemented + dismissed
        
        if total < 3:
            # Not enough data, stay near neutral
            # Slightly adjust based on limited data
            if total == 0:
                return 0.5
            impl_rate = implemented / total
            # Blend with neutral (0.5) weighted toward neutral
            return 0.5 * 0.7 + impl_rate * 0.3
        
        # Base score from implementation rate
        impl_rate = implemented / total
        
        # Accuracy bonus: if actual savings >= 80% of estimated
        accuracy_bonus = 0.0
        if accuracy >= 0.8:
            accuracy_bonus = 0.1
        elif accuracy >= 0.5:
            accuracy_bonus = 0.05
        
        # Feedback penalty for negative keywords
        feedback_penalty = 0.0
        if user_feedback:
            negative_words = [
                "bad", "worse", "broken", "failed", "poor", 
                "terrible", "awful", "useless", "wrong", "garbage"
            ]
            feedback_lower = user_feedback.lower()
            if any(word in feedback_lower for word in negative_words):
                feedback_penalty = 0.15
        
        # Calculate final score
        score = impl_rate + accuracy_bonus - feedback_penalty
        
        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, score))
    
    async def get_learned_alternatives(
        self,
        source_model: str,
        requires_vision: bool = False,
        requires_function_calling: bool = False,
        min_confidence: float = 0.3,
        max_results: int = 5,
    ) -> List[ModelAlternative]:
        """
        Get alternatives for a model, ranked by quality and confidence.
        
        Ranking priority:
        1. Same provider alternatives first (safer swaps)
        2. Higher quality tiers (tier 1-2 before tier 4-5)
        3. Higher confidence scores
        4. Better savings (lower price ratio)
        
        Returns alternatives that:
        - Match the source model
        - Meet capability requirements
        - Have confidence >= min_confidence
        - Are active
        """
        query = select(ModelAlternative).where(
            ModelAlternative.source_model == source_model,
            ModelAlternative.is_active == True,
            ModelAlternative.confidence_score >= min_confidence,
        )
        
        if requires_vision:
            query = query.where(ModelAlternative.requires_vision == True)
        if requires_function_calling:
            query = query.where(ModelAlternative.requires_function_calling == True)
        
        # Improved ranking:
        # 1. same_provider DESC (True=1 comes before False=0)
        # 2. quality_tier ASC (tier 1 is best)
        # 3. confidence_score DESC (higher is better)
        # 4. price_ratio ASC (lower = more savings)
        query = query.order_by(
            ModelAlternative.same_provider.desc(),
            ModelAlternative.quality_tier.asc(),
            ModelAlternative.confidence_score.desc(),
            ModelAlternative.price_ratio.asc(),
        ).limit(max_results)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_alternative_stats(self) -> Dict[str, Any]:
        """Get statistics about the alternatives system."""
        # Total alternatives
        total_query = select(func.count(ModelAlternative.id))
        total_result = await self.db.execute(total_query)
        total_raw = total_result.scalar()
        total = int(total_raw) if total_raw is not None else 0
        
        # Active alternatives
        active_query = select(func.count(ModelAlternative.id)).where(
            ModelAlternative.is_active == True
        )
        active_result = await self.db.execute(active_query)
        active_raw = active_result.scalar()
        active = int(active_raw) if active_raw is not None else 0
        
        # Alternatives with learning data (times_suggested > 0)
        learned_query = select(func.count(ModelAlternative.id)).where(
            ModelAlternative.times_suggested > 0
        )
        learned_result = await self.db.execute(learned_query)
        learned_raw = learned_result.scalar()
        learned = int(learned_raw) if learned_raw is not None else 0
        
        # High confidence alternatives (> 0.7)
        high_conf_query = select(func.count(ModelAlternative.id)).where(
            ModelAlternative.confidence_score > 0.7,
            ModelAlternative.times_suggested > 0,
        )
        high_conf_result = await self.db.execute(high_conf_query)
        high_conf_raw = high_conf_result.scalar()
        high_confidence = int(high_conf_raw) if high_conf_raw is not None else 0
        
        # Average confidence for learned alternatives
        avg_conf_query = select(func.avg(ModelAlternative.confidence_score)).where(
            ModelAlternative.times_suggested > 0
        )
        avg_conf_result = await self.db.execute(avg_conf_query)
        avg_conf_raw = avg_conf_result.scalar()
        avg_confidence = float(avg_conf_raw) if avg_conf_raw is not None else 0.5
        
        # Total savings tracked
        savings_query = select(
            func.sum(ModelAlternative.total_estimated_savings),
            func.sum(ModelAlternative.total_actual_savings),
        )
        savings_result = await self.db.execute(savings_query)
        savings_row = savings_result.one()
        
        # Convert Decimal values to float for JSON serialization
        estimated_savings = float(savings_row[0]) if savings_row[0] is not None else 0.0
        actual_savings = float(savings_row[1]) if savings_row[1] is not None else 0.0
        
        return {
            "total_alternatives": total,
            "active_alternatives": active,
            "alternatives_with_feedback": learned,
            "high_confidence_count": high_confidence,
            "average_confidence": round(avg_confidence, 3),
            "total_estimated_savings": round(estimated_savings, 2),
            "total_actual_savings": round(actual_savings, 2),
        }
