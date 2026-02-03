from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from enum import Enum

from ..models.db_models import Event, ModelPricing
from .analytics_service import AnalyticsService
from .pricing_service import PricingService
from .baseline_service import (
    BaselineService, 
    PatternAnalysisService, 
    RecommendationTrackingService,
)


class OptimizationType(str, Enum):
    MODEL_DOWNGRADE = "model_downgrade"
    CACHING = "caching"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    BATCHING = "batching"
    ERROR_REDUCTION = "error_reduction"
    ANOMALY_ALERT = "anomaly_alert"


class OptimizationService:
    
    # Minimum savings to show a suggestion (matches recommendation threshold)
    # Suggestions below this won't have Implement/Dismiss buttons anyway
    MIN_ACTIONABLE_SAVINGS = 1.0
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.pricing_service = PricingService(db)
        self.baseline_service = BaselineService(db)
        self.pattern_service = PatternAnalysisService(db)
        self.tracking_service = RecommendationTrackingService(db)
    
    async def _generate_suggestions(
        self,
        project_id: str,
        days: int = 30,
        include_low_priority: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Core suggestion generation logic without side effects.
        
        This method only analyzes usage and generates suggestions.
        It does NOT create or update recommendation records.
        Used internally by get_suggestions() and get_summary().
        """
        # Ensure baselines exist for anomaly detection (auto-compute if first time)
        await self.baseline_service.ensure_baselines_exist(project_id, days)
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        suggestions = []
        
        model_suggestions = await self._analyze_model_usage(project_id, start_time, end_time, days)
        suggestions.extend(model_suggestions)
        
        caching_suggestions = await self._analyze_caching_opportunities(project_id)
        suggestions.extend(caching_suggestions)
        
        anomaly_suggestions = await self._analyze_anomalies(project_id)
        suggestions.extend(anomaly_suggestions)
        
        error_suggestions = await self._analyze_error_patterns(project_id, start_time, end_time, days)
        suggestions.extend(error_suggestions)
        
        latency_suggestions = await self._analyze_latency_issues(project_id, start_time, end_time)
        suggestions.extend(latency_suggestions)
        
        if not include_low_priority:
            suggestions = [s for s in suggestions if s.get("priority") != "low"]
        
        suggestions.sort(key=lambda x: x.get("estimated_savings_monthly", 0), reverse=True)
        
        return suggestions
    
    async def get_suggestions(
        self,
        project_id: str,
        days: int = 30,
        include_low_priority: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Analyze usage and generate optimization suggestions.
        
        This method generates suggestions AND persists the top 10 as recommendations
        for tracking user actions (implement/dismiss).
        """
        # Generate suggestions without side effects
        suggestions = await self._generate_suggestions(project_id, days, include_low_priority)
        
        # Persist top recommendations for tracking (with de-duplication and cooldowns)
        for suggestion in suggestions[:10]:
            await self.tracking_service.create_recommendation(
                project_id=project_id,
                recommendation_type=suggestion.get("type", "unknown"),
                title=suggestion.get("title", ""),
                description=suggestion.get("description", ""),
                agent_name=suggestion.get("agent_name"),
                model=suggestion.get("model"),
                alternative_model=suggestion.get("alternative_model"),
                estimated_monthly_savings=suggestion.get("estimated_savings_monthly", 0),
                estimated_savings_percent=suggestion.get("estimated_savings_percent", 0),
                metrics_snapshot=suggestion.get("metrics"),
            )
        
        return suggestions
    
    async def _analyze_model_usage(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        days: int,
    ) -> List[Dict[str, Any]]:
        """Analyze model usage and suggest alternatives."""
        suggestions = []
        
        query = select(
            Event.model,
            Event.agent_name,
            func.count(Event.id).label("call_count"),
            func.sum(Event.cost).label("total_cost"),
            func.avg(Event.output_tokens).label("avg_output_tokens"),
            func.avg(Event.input_tokens).label("avg_input_tokens"),
            func.sum(Event.input_tokens).label("total_input_tokens"),
            func.sum(Event.output_tokens).label("total_output_tokens"),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        ).group_by(Event.model, Event.agent_name)
        
        result = await self.db.execute(query)
        
        for row in result:
            model = row.model
            agent = row.agent_name
            # Convert Decimal values to float for arithmetic operations
            cost = float(row.total_cost) if row.total_cost is not None else 0.0
            avg_output = float(row.avg_output_tokens) if row.avg_output_tokens is not None else 0.0
            avg_input = float(row.avg_input_tokens) if row.avg_input_tokens is not None else 0.0
            calls = int(row.call_count) if row.call_count is not None else 0
            total_input = int(row.total_input_tokens) if row.total_input_tokens is not None else 0
            total_output = int(row.total_output_tokens) if row.total_output_tokens is not None else 0
            
            if calls < 10 or cost < 0.01:
                continue
            
            alternatives = await self.pricing_service.discover_alternatives(
                model=model,
                avg_output_tokens=int(avg_output),
                avg_input_tokens=int(avg_input),
                max_results=3,
            )
            
            for alt in alternatives:
                alt_model = alt["model"]
                savings = alt["savings"]
                
                input_savings = (total_input / 1000) * savings["input_per_1k"]
                output_savings = (total_output / 1000) * savings["output_per_1k"]
                period_savings = input_savings + output_savings
                
                if period_savings <= 0:
                    continue
                
                monthly_savings = (period_savings / days) * 30
                if monthly_savings < self.MIN_ACTIONABLE_SAVINGS:
                    continue
                
                monthly_cost = (cost / days) * 30
                savings_percent = (monthly_savings / monthly_cost * 100) if monthly_cost > 0 else 0
                priority = self._calculate_priority(monthly_savings)
                
                action_items = self._build_model_switch_actions(
                    agent=agent,
                    current_model=model,
                    alternative_model=alt_model,
                    monthly_savings=monthly_savings,
                    quality_impact=alt["quality_impact"],
                    calls=calls,
                )
                
                suggestions.append({
                    "type": OptimizationType.MODEL_DOWNGRADE.value,
                    "title": f"Consider {alt_model} for {agent}",
                    "description": (
                        f"Agent '{agent}' uses {model} with average output of "
                        f"{avg_output:.0f} tokens. Switching to {alt_model} could reduce costs."
                    ),
                    "estimated_savings_monthly": round(monthly_savings, 2),
                    "estimated_savings_percent": round(savings_percent, 1),
                    "agent_name": agent,
                    "model": model,
                    "alternative_model": alt_model,
                    "priority": priority,
                    "action_items": action_items,
                    "metrics": {
                        "current_calls": calls,
                        "current_monthly_cost": round(monthly_cost, 2),
                        "avg_output_tokens": round(avg_output, 1),
                        "avg_input_tokens": round(avg_input, 1),
                        "savings_percentage": alt["savings"]["percentage"],
                        "quality_impact": alt["quality_impact"],
                        # Confidence data for "Proven" vs "Suggested" badge
                        "source": alt.get("source"),  # "learned" or "dynamic"
                        "confidence_score": alt.get("confidence_score"),
                        "times_implemented": alt.get("times_implemented"),
                        "savings_accuracy": alt.get("savings_accuracy"),
                    },
                })
                break
        
        return suggestions
    
    async def _analyze_caching_opportunities(self, project_id: str) -> List[Dict[str, Any]]:
        """Identify caching opportunities."""
        suggestions = []
        
        opportunities = await self.pattern_service.analyze_caching_opportunities(
            project_id=project_id,
            min_occurrences=5,
            min_savings=1.0,
        )
        
        for opp in opportunities:
            agent = opp["agent_name"]
            duplicate_rate = opp["duplicate_rate"]
            monthly_savings = opp["estimated_monthly_savings"]
            
            priority = self._calculate_priority(monthly_savings)
            
            action_items = self._build_caching_actions(
                agent=agent,
                duplicate_rate=duplicate_rate,
                unique_patterns=opp["unique_patterns"],
                total_calls=opp["total_calls"],
                duplicate_calls=opp["duplicate_calls"],
            )
            
            suggestions.append({
                "type": OptimizationType.CACHING.value,
                "title": f"Add caching for {agent}",
                "description": (
                    f"Agent '{agent}' has {duplicate_rate}% duplicate queries. "
                    f"Implementing response caching could save approximately "
                    f"${monthly_savings:.2f}/month based on observed patterns."
                ),
                "estimated_savings_monthly": round(monthly_savings, 2),
                "estimated_savings_percent": round(duplicate_rate, 1),
                "agent_name": agent,
                "model": None,
                "priority": priority,
                "action_items": action_items,
                "metrics": {
                    "unique_patterns": opp["unique_patterns"],
                    "total_calls": opp["total_calls"],
                    "duplicate_calls": opp["duplicate_calls"],
                    "duplicate_rate": duplicate_rate,
                },
            })
        
        return suggestions
    
    async def _analyze_anomalies(
        self,
        project_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using statistical baselines.
        """
        suggestions = []
        
        anomalies = await self.baseline_service.detect_anomalies(
            project_id=project_id,
            recent_hours=24,
        )
        
        for anomaly in anomalies:
            if not anomaly.is_anomaly:
                continue
            
            metric_parts = anomaly.metric_name.split("_", 1)
            metric_type = metric_parts[0]
            context = metric_parts[1] if len(metric_parts) > 1 else ""
            
            if metric_type == "cost":
                direction = "higher" if anomaly.z_score > 0 else "lower"
                description = (
                    f"Cost per call is {abs(anomaly.z_score):.1f} standard deviations "
                    f"{direction} than normal for {context}. "
                    f"Current: ${anomaly.current_value:.4f}, "
                    f"Baseline: ${anomaly.baseline_mean:.4f}"
                )
            elif metric_type == "latency":
                direction = "higher" if anomaly.z_score > 0 else "lower"
                description = (
                    f"Latency is {abs(anomaly.z_score):.1f} standard deviations "
                    f"{direction} than normal for {context}. "
                    f"Current: {anomaly.current_value:.0f}ms, "
                    f"Baseline: {anomaly.baseline_mean:.0f}ms"
                )
            elif metric_type == "error":
                description = (
                    f"Error rate is elevated for {context}. "
                    f"Current: {anomaly.current_value*100:.1f}%, "
                    f"Baseline: {anomaly.baseline_mean*100:.1f}%"
                )
            else:
                continue
            
            action_items = self._build_anomaly_actions(
                metric_type=metric_type,
                context=context,
                z_score=anomaly.z_score,
                current_value=anomaly.current_value,
                baseline_mean=anomaly.baseline_mean,
            )
            
            suggestions.append({
                "type": OptimizationType.ANOMALY_ALERT.value,
                "title": f"Anomaly detected: {metric_type} for {context}",
                "description": description,
                "estimated_savings_monthly": 0,
                "estimated_savings_percent": 0,
                "agent_name": None,
                "model": None,
                "priority": anomaly.severity,
                "action_items": action_items,
                "metrics": {
                    "current_value": round(anomaly.current_value, 4),
                    "baseline_mean": round(anomaly.baseline_mean, 4),
                    "baseline_stddev": round(anomaly.baseline_stddev, 4),
                    "z_score": round(anomaly.z_score, 2),
                },
            })
        
        return suggestions
    
    async def _analyze_error_patterns(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        days: int,
    ) -> List[Dict[str, Any]]:
        """
        Analyze error patterns and calculate wasted spend.
        Uses baseline error rates for comparison.
        """
        suggestions = []
        
        # Get error stats per agent
        query = select(
            Event.agent_name,
            Event.model,
            func.count(Event.id).label("total_calls"),
            func.sum(case((Event.success == False, 1), else_=0)).label("error_count"),
            func.sum(case((Event.success == False, Event.cost), else_=0)).label("wasted_cost"),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        ).group_by(
            Event.agent_name, Event.model
        )
        
        result = await self.db.execute(query)
        
        for row in result:
            agent = row.agent_name
            model = row.model
            # Convert Decimal values to float/int for arithmetic operations
            total_calls = int(row.total_calls) if row.total_calls is not None else 0
            errors = int(row.error_count) if row.error_count is not None else 0
            wasted = float(row.wasted_cost) if row.wasted_cost is not None else 0.0
            
            if total_calls < 10 or errors < 3:
                continue
            
            error_rate = errors / total_calls
            
            # Get baseline error rate
            baseline = await self.baseline_service.get_baseline(
                project_id=project_id,
                agent_name=agent,
                model=model,
            )
            
            baseline_error_rate = baseline.avg_error_rate if baseline else 0.02
            
            # Only flag if significantly above baseline
            if error_rate <= baseline_error_rate * 1.5:
                continue
            
            monthly_wasted = (wasted / days) * 30
            
            if monthly_wasted < 0.50:  # Skip if less than $0.50/month wasted
                continue
            
            priority = self._calculate_priority(monthly_wasted)
            
            action_items = self._build_error_actions(
                agent=agent,
                model=model,
                error_rate=error_rate,
                baseline_error_rate=baseline_error_rate,
                error_count=errors,
                total_calls=total_calls,
                monthly_wasted=monthly_wasted,
            )
            
            suggestions.append({
                "type": OptimizationType.ERROR_REDUCTION.value,
                "title": f"Reduce errors in {agent}",
                "description": (
                    f"Agent '{agent}' using {model} has {error_rate*100:.1f}% error rate "
                    f"(baseline: {baseline_error_rate*100:.1f}%), wasting "
                    f"${monthly_wasted:.2f}/month on failed calls."
                ),
                "estimated_savings_monthly": round(monthly_wasted, 2),
                "estimated_savings_percent": round(error_rate * 100, 1),
                "agent_name": agent,
                "model": model,
                "priority": priority,
                "action_items": action_items,
                "metrics": {
                    "total_calls": total_calls,
                    "error_count": errors,
                    "error_rate": round(error_rate * 100, 2),
                    "baseline_error_rate": round(baseline_error_rate * 100, 2),
                    "wasted_cost": round(wasted, 4),
                },
            })
        
        return suggestions
    
    async def _analyze_latency_issues(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Identify latency issues based on baseline deviations.
        """
        suggestions = []
        
        # Get latency stats per agent/model
        query = select(
            Event.agent_name,
            Event.model,
            func.avg(Event.latency_ms).label("avg_latency"),
            func.avg(Event.input_tokens).label("avg_input_tokens"),
            func.count(Event.id).label("call_count"),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        ).group_by(
            Event.agent_name, Event.model
        )
        
        result = await self.db.execute(query)
        
        for row in result:
            agent = row.agent_name
            model = row.model
            # Convert Decimal values to float/int for arithmetic operations
            avg_latency = float(row.avg_latency) if row.avg_latency is not None else 0.0
            avg_input = float(row.avg_input_tokens) if row.avg_input_tokens is not None else 0.0
            calls = int(row.call_count) if row.call_count is not None else 0
            
            if calls < 10:
                continue
            
            # Get baseline
            baseline = await self.baseline_service.get_baseline(
                project_id=project_id,
                agent_name=agent,
                model=model,
            )
            
            if not baseline or baseline.stddev_latency_ms == 0:
                continue
            
            # Calculate z-score
            z_score = (avg_latency - baseline.avg_latency_ms) / baseline.stddev_latency_ms
            
            # Only flag if significantly above baseline (2+ standard deviations)
            if z_score < 2.0:
                continue
            
            action_items = self._build_latency_actions(
                agent=agent,
                model=model,
                avg_latency=avg_latency,
                baseline_latency=baseline.avg_latency_ms,
                avg_input_tokens=avg_input,
                z_score=z_score,
            )
            
            suggestions.append({
                "type": OptimizationType.PROMPT_OPTIMIZATION.value,
                "title": f"Optimize prompts for {agent}",
                "description": (
                    f"Agent '{agent}' has elevated latency "
                    f"({avg_latency:.0f}ms vs {baseline.avg_latency_ms:.0f}ms baseline) "
                    f"with {avg_input:.0f} average input tokens. "
                    f"Consider shortening prompts or using streaming."
                ),
                "estimated_savings_monthly": 0,
                "estimated_savings_percent": 0,
                "agent_name": agent,
                "model": model,
                "priority": "high" if z_score > 3.0 else "medium",
                "action_items": action_items,
                "metrics": {
                    "avg_latency_ms": round(avg_latency, 0),
                    "baseline_latency_ms": round(baseline.avg_latency_ms, 0),
                    "z_score": round(z_score, 2),
                    "avg_input_tokens": round(avg_input, 0),
                },
            })
        
        return suggestions
    
    def _build_model_switch_actions(
        self,
        agent: str,
        current_model: str,
        alternative_model: str,
        monthly_savings: float,
        quality_impact: str,
        calls: int,
    ) -> List[str]:
        actions = []
        
        if quality_impact == "minimal":
            actions.append(
                f"Run A/B test: route 10% of {agent} traffic to {alternative_model} "
                f"and compare output quality scores"
            )
        elif quality_impact == "moderate":
            actions.append(
                f"Evaluate {alternative_model} on your {agent} test suite - "
                f"expect some quality differences"
            )
        else:
            actions.append(
                f"Thoroughly test {alternative_model} - significant capability "
                f"differences expected vs {current_model}"
            )
        
        if calls > 1000:
            actions.append(
                f"With {calls:,} calls/period, implement gradual rollout: "
                f"10% → 25% → 50% → 100% over 2 weeks"
            )
        else:
            actions.append(f"Switch {agent} configuration from {current_model} to {alternative_model}")
        
        actions.append(
            f"Monitor {agent} error rates and user feedback for 48 hours after switch"
        )
        
        if monthly_savings > 100:
            actions.append(f"Expected savings: ${monthly_savings:.2f}/month - prioritize this migration")
        
        return actions
    
    def _build_caching_actions(
        self,
        agent: str,
        duplicate_rate: float,
        unique_patterns: int,
        total_calls: int,
        duplicate_calls: int,
    ) -> List[str]:
        actions = []
        
        cache_size = min(unique_patterns * 2, 10000)
        actions.append(
            f"Implement cache with size {cache_size:,} entries - "
            f"you have {unique_patterns:,} unique query patterns"
        )
        
        if duplicate_rate > 50:
            actions.append(
                f"High duplicate rate ({duplicate_rate:.0f}%) - "
                f"use aggressive caching with 1-hour TTL"
            )
        elif duplicate_rate > 20:
            actions.append(
                f"Moderate duplicates ({duplicate_rate:.0f}%) - "
                f"use 30-minute TTL with LRU eviction"
            )
        else:
            actions.append(f"Use 15-minute TTL for {agent} cache")
        
        if duplicate_calls > 100:
            actions.append(
                f"Add semantic similarity matching - {duplicate_calls:,} "
                f"duplicate calls may have slight variations"
            )
        
        actions.append(f"Log cache hits/misses for {agent} to measure effectiveness")
        
        return actions
    
    def _build_anomaly_actions(
        self,
        metric_type: str,
        context: str,
        z_score: float,
        current_value: float,
        baseline_mean: float,
    ) -> List[str]:
        actions = []
        
        deviation_pct = abs((current_value - baseline_mean) / baseline_mean * 100) if baseline_mean else 0
        
        if metric_type == "cost":
            if z_score > 0:
                actions.append(
                    f"Cost increased {deviation_pct:.0f}% for {context} - "
                    f"check for prompt length changes or model switches"
                )
                actions.append(f"Compare recent {context} token counts to baseline")
            else:
                actions.append(
                    f"Cost decreased {deviation_pct:.0f}% for {context} - "
                    f"verify functionality is not degraded"
                )
        elif metric_type == "latency":
            if z_score > 0:
                actions.append(
                    f"Latency increased {deviation_pct:.0f}% for {context} - "
                    f"check provider status page for incidents"
                )
                actions.append(f"Review recent prompt changes that may have increased token count")
            else:
                actions.append(f"Latency improved for {context} - no action needed")
        elif metric_type == "error":
            actions.append(
                f"Error rate at {current_value*100:.1f}% for {context} - "
                f"check API logs for specific error types"
            )
            actions.append(f"Verify input validation is working for {context}")
        
        if abs(z_score) > 3:
            actions.append(f"Urgent: {abs(z_score):.1f}σ deviation requires immediate investigation")
        
        return actions
    
    def _build_error_actions(
        self,
        agent: str,
        model: str,
        error_rate: float,
        baseline_error_rate: float,
        error_count: int,
        total_calls: int,
        monthly_wasted: float,
    ) -> List[str]:
        actions = []
        
        error_increase = ((error_rate - baseline_error_rate) / baseline_error_rate * 100) if baseline_error_rate else 0
        
        if error_rate > 0.10:
            actions.append(
                f"Critical: {error_rate*100:.1f}% error rate on {agent} - "
                f"query last {error_count} failed requests for common patterns"
            )
        else:
            actions.append(
                f"Error rate {error_increase:.0f}% above baseline - "
                f"review {agent} error logs from past 24 hours"
            )
        
        if error_count > 50:
            actions.append(
                f"Implement retry with exponential backoff for {model} - "
                f"{error_count} failures may be transient"
            )
        
        if monthly_wasted > 10:
            actions.append(
                f"Add pre-call validation for {agent} - "
                f"${monthly_wasted:.2f}/month wasted on failed requests"
            )
        
        actions.append(f"Consider adding fallback model for {agent} when {model} fails")
        
        return actions
    
    def _build_latency_actions(
        self,
        agent: str,
        model: str,
        avg_latency: float,
        baseline_latency: float,
        avg_input_tokens: int,
        z_score: float,
    ) -> List[str]:
        actions = []
        
        latency_increase = avg_latency - baseline_latency
        
        if avg_input_tokens > 2000:
            actions.append(
                f"Reduce prompt size for {agent} - currently {avg_input_tokens:.0f} tokens, "
                f"aim for <2000 tokens"
            )
        
        if latency_increase > 1000:
            actions.append(
                f"Latency increased by {latency_increase:.0f}ms - "
                f"check if {model} is experiencing provider-side delays"
            )
        
        if z_score > 3.0:
            actions.append(
                f"Severe latency issue ({z_score:.1f}σ) - consider switching to faster "
                f"model variant or enabling streaming for {agent}"
            )
        else:
            actions.append(f"Enable response streaming for {agent} to improve perceived latency")
        
        actions.append(f"Profile {agent} prompt construction to identify bottlenecks")
        
        return actions
    
    def _calculate_priority(self, monthly_savings: float) -> str:
        """Calculate priority based on potential monthly savings."""
        if monthly_savings >= 50:
            return "high"
        elif monthly_savings >= 10:
            return "medium"
        else:
            return "low"
    
    async def get_summary(
        self,
        project_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get optimization summary with total potential savings.
        
        Uses _generate_suggestions() to avoid creating recommendation records.
        Includes context for empty state messaging (no data vs no baselines vs optimized).
        """
        # Use _generate_suggestions to avoid side effects
        suggestions = await self._generate_suggestions(project_id, days)
        
        total_savings = sum(s.get("estimated_savings_monthly", 0) for s in suggestions)
        high_priority = [s for s in suggestions if s.get("priority") == "high"]
        
        # Get current spend for context
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        analytics_service = AnalyticsService(self.db)
        overview = await analytics_service.get_overview(project_id, start_time, end_time)
        
        days_in_period = (end_time - start_time).days or 1
        monthly_spend = (overview.total_cost / days_in_period) * 30
        
        savings_percent = (total_savings / monthly_spend * 100) if monthly_spend > 0 else 0
        
        # Get recommendation effectiveness
        effectiveness = await self.tracking_service.get_recommendation_effectiveness(
            project_id
        )
        
        # Group by type
        type_breakdown = {}
        for s in suggestions:
            stype = s.get("type", "other")
            if stype not in type_breakdown:
                type_breakdown[stype] = {"count": 0, "savings": 0}
            type_breakdown[stype]["count"] += 1
            type_breakdown[stype]["savings"] += s.get("estimated_savings_monthly", 0)
        
        # Context for empty state messaging
        # has_data: Are there any events to analyze?
        has_data = overview.total_calls > 0
        # has_baselines: Are there baselines computed (need 10+ calls per agent/model)?
        has_baselines = await self.baseline_service.has_baselines(project_id)
        # Determine the reason for no suggestions
        # "no_data" - No events at all
        # "insufficient_data" - Has events but not enough for meaningful analysis (< 10 per agent)
        # "no_baselines" - Baselines couldn't be computed (each agent/model needs 10+ calls)
        # "optimized" - Has data and baselines, genuinely no opportunities found
        if len(suggestions) == 0:
            if not has_data:
                empty_reason = "no_data"
            elif not has_baselines and overview.total_calls < 10:
                empty_reason = "insufficient_data"
            elif not has_baselines:
                empty_reason = "no_baselines"
            else:
                empty_reason = "optimized"
        else:
            empty_reason = None
        
        return {
            "total_potential_savings_monthly": round(total_savings, 2),
            "total_potential_savings_percent": round(savings_percent, 1),
            "current_monthly_spend": round(monthly_spend, 2),
            "suggestion_count": len(suggestions),
            "high_priority_count": len(high_priority),
            "by_type": type_breakdown,
            "effectiveness": effectiveness,
            "suggestions": suggestions[:5],
            # Empty state context
            "has_data": has_data,
            "has_baselines": has_baselines,
            "event_count": overview.total_calls,
            "empty_reason": empty_reason,
        }
    
    async def refresh_baselines(self, project_id: str, days: int = 30) -> Dict[str, Any]:
        """Manually trigger baseline recalculation."""
        return await self.baseline_service.compute_baselines(project_id, days)
