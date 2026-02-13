# AgentCost Backend - Baseline Analysis Service

# Computes statistical baselines for anomaly detection and pattern analysis.
# Uses standard deviation and percentile-based thresholds instead of hardcoded values.

import hashlib
from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, text


from ..models.db_models import (
    Event, 
    ProjectBaseline, 
    InputPatternCache,
    OptimizationRecommendation,
)


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a metric."""
    metric_name: str
    current_value: float
    baseline_mean: float
    baseline_stddev: float
    z_score: float
    is_anomaly: bool
    severity: str  # low, medium, high


def _json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-serializable types into JSON-safe values.
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


class BaselineService:
    """Service for computing and managing statistical baselines."""
    
    ANOMALY_THRESHOLD_MEDIUM = 2.0  # 2 standard deviations
    ANOMALY_THRESHOLD_HIGH = 3.0    # 3 standard deviations
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def compute_baselines(
        self,
        project_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Compute or update statistical baselines for a project.
        Calculates mean and standard deviation for key metrics per agent/model.
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Compute baselines per agent/model combination
        query = select(
            Event.agent_name,
            Event.model,
            func.count(Event.id).label('call_count'),
            func.avg(Event.cost).label('avg_cost'),
            func.avg(Event.input_tokens).label('avg_input'),
            func.avg(Event.output_tokens).label('avg_output'),
            func.avg(Event.latency_ms).label('avg_latency'),
            # Standard deviations using SQL
            func.coalesce(
                func.sqrt(
                    func.avg(Event.cost * Event.cost) - 
                    func.avg(Event.cost) * func.avg(Event.cost)
                ), 0
            ).label('stddev_cost'),
            func.coalesce(
                func.sqrt(
                    func.avg(Event.input_tokens * Event.input_tokens) - 
                    func.avg(Event.input_tokens) * func.avg(Event.input_tokens)
                ), 0
            ).label('stddev_input'),
            func.coalesce(
                func.sqrt(
                    func.avg(Event.output_tokens * Event.output_tokens) - 
                    func.avg(Event.output_tokens) * func.avg(Event.output_tokens)
                ), 0
            ).label('stddev_output'),
            func.coalesce(
                func.sqrt(
                    func.avg(Event.latency_ms * Event.latency_ms) - 
                    func.avg(Event.latency_ms) * func.avg(Event.latency_ms)
                ), 0
            ).label('stddev_latency'),
            # Error rate
            func.sum(case((Event.success == False, 1), else_=0)).label('error_count'),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        ).group_by(
            Event.agent_name, Event.model
        )
        
        result = await self.db.execute(query)
        rows = result.all()
        
        baselines_updated = 0
        
        for row in rows:
            call_count = int(row.call_count) if row.call_count is not None else 0
            if call_count < 10:  # Need minimum samples for meaningful baseline
                continue
            
            # Convert Decimal values to float for storage
            avg_cost = float(row.avg_cost) if row.avg_cost is not None else 0.0
            stddev_cost = float(row.stddev_cost) if row.stddev_cost is not None else 0.0
            avg_input = float(row.avg_input) if row.avg_input is not None else 0.0
            stddev_input = float(row.stddev_input) if row.stddev_input is not None else 0.0
            avg_output = float(row.avg_output) if row.avg_output is not None else 0.0
            stddev_output = float(row.stddev_output) if row.stddev_output is not None else 0.0
            avg_latency = float(row.avg_latency) if row.avg_latency is not None else 0.0
            stddev_latency = float(row.stddev_latency) if row.stddev_latency is not None else 0.0
            error_count = int(row.error_count) if row.error_count is not None else 0
            
            error_rate = error_count / call_count if call_count > 0 else 0.0
            daily_calls = call_count / days
            
            # Check if baseline exists
            existing_query = select(ProjectBaseline).where(
                ProjectBaseline.project_id == project_id,
                ProjectBaseline.agent_name == row.agent_name,
                ProjectBaseline.model == row.model,
            )
            existing_result = await self.db.execute(existing_query)
            baseline = existing_result.scalar_one_or_none()
            
            if baseline:
                baseline.avg_cost_per_call = avg_cost
                baseline.stddev_cost_per_call = stddev_cost
                baseline.avg_input_tokens = avg_input
                baseline.stddev_input_tokens = stddev_input
                baseline.avg_output_tokens = avg_output
                baseline.stddev_output_tokens = stddev_output
                baseline.avg_latency_ms = avg_latency
                baseline.stddev_latency_ms = stddev_latency
                baseline.avg_daily_calls = daily_calls
                baseline.avg_error_rate = error_rate
                baseline.sample_count = call_count
                baseline.sample_days = days
                baseline.last_calculated_at = datetime.now(timezone.utc)
            else:
                baseline = ProjectBaseline(
                    project_id=project_id,
                    agent_name=row.agent_name,
                    model=row.model,
                    avg_cost_per_call=avg_cost,
                    stddev_cost_per_call=stddev_cost,
                    avg_input_tokens=avg_input,
                    stddev_input_tokens=stddev_input,
                    avg_output_tokens=avg_output,
                    stddev_output_tokens=stddev_output,
                    avg_latency_ms=avg_latency,
                    stddev_latency_ms=stddev_latency,
                    avg_daily_calls=daily_calls,
                    avg_error_rate=error_rate,
                    sample_count=call_count,
                    sample_days=days,
                )
                self.db.add(baseline)
            
            baselines_updated += 1
        
        await self.db.flush()
        
        return {
            "status": "ok",
            "baselines_updated": baselines_updated,
            "analysis_period_days": days,
        }
    
    async def has_baselines(self, project_id: str) -> bool:
        """
        Check if baselines exist for a project.
        Used to trigger auto-computation on first optimization request.
        """
        query = select(func.count(ProjectBaseline.id)).where(
            ProjectBaseline.project_id == project_id
        )
        result = await self.db.execute(query)
        count = result.scalar()
        return count > 0
    
    async def ensure_baselines_exist(self, project_id: str, days: int = 30) -> bool:
        """
        Ensure baselines exist for a project, computing them if necessary.
        
        Returns True if baselines were computed, False if they already existed.
        """
        if await self.has_baselines(project_id):
            return False
        
        await self.compute_baselines(project_id, days)
        return True
    
    async def get_baseline(
        self,
        project_id: str,
        agent_name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[ProjectBaseline]:
        """
        Get baseline for a specific agent/model combination.
        
        If both agent_name and model are provided, returns the specific baseline.
        If only one is provided, or neither, may return multiple matches - 
        in that case returns the most recently calculated one.
        """
        query = select(ProjectBaseline).where(
            ProjectBaseline.project_id == project_id
        )
        
        if agent_name:
            query = query.where(ProjectBaseline.agent_name == agent_name)
        if model:
            query = query.where(ProjectBaseline.model == model)
        
        # Order by most recently calculated to get the freshest baseline
        query = query.order_by(ProjectBaseline.last_calculated_at.desc())
        
        result = await self.db.execute(query)
        # Use .first() to handle cases where multiple baselines match
        return result.scalars().first()
    
    async def detect_anomalies(
        self,
        project_id: str,
        recent_hours: int = 24,
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in recent usage compared to baselines.
        Uses z-score analysis to find significant deviations.
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=recent_hours)
        
        # Get baselines
        baseline_query = select(ProjectBaseline).where(
            ProjectBaseline.project_id == project_id
        )
        baseline_result = await self.db.execute(baseline_query)
        baselines = {
            (b.agent_name, b.model): b 
            for b in baseline_result.scalars().all()
        }
        
        if not baselines:
            return []
        
        # Get recent stats
        recent_query = select(
            Event.agent_name,
            Event.model,
            func.count(Event.id).label('call_count'),
            func.avg(Event.cost).label('avg_cost'),
            func.avg(Event.latency_ms).label('avg_latency'),
            func.avg(Event.input_tokens).label('avg_input'),
            func.avg(Event.output_tokens).label('avg_output'),
            func.sum(case((Event.success == False, 1), else_=0)).label('error_count'),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        ).group_by(
            Event.agent_name, Event.model
        )
        
        recent_result = await self.db.execute(recent_query)
        anomalies = []
        
        for row in recent_result:
            baseline = baselines.get((row.agent_name, row.model))
            if not baseline or baseline.sample_count < 10:
                continue
            
            # Convert Decimal values to float for arithmetic operations
            avg_cost = float(row.avg_cost) if row.avg_cost is not None else 0.0
            avg_latency = float(row.avg_latency) if row.avg_latency is not None else 0.0
            call_count = int(row.call_count) if row.call_count is not None else 0
            error_count = int(row.error_count) if row.error_count is not None else 0
            
            # Check cost anomaly
            if baseline.stddev_cost_per_call > 0:
                cost_z = (
                    (avg_cost - baseline.avg_cost_per_call) / 
                    baseline.stddev_cost_per_call
                )
                if abs(cost_z) > self.ANOMALY_THRESHOLD_MEDIUM:
                    anomalies.append(AnomalyResult(
                        metric_name=f"cost_{row.agent_name}_{row.model}",
                        current_value=avg_cost,
                        baseline_mean=baseline.avg_cost_per_call,
                        baseline_stddev=baseline.stddev_cost_per_call,
                        z_score=cost_z,
                        is_anomaly=True,
                        severity="high" if abs(cost_z) > self.ANOMALY_THRESHOLD_HIGH else "medium",
                    ))
            
            # Check latency anomaly
            if baseline.stddev_latency_ms > 0:
                latency_z = (
                    (avg_latency - baseline.avg_latency_ms) / 
                    baseline.stddev_latency_ms
                )
                if abs(latency_z) > self.ANOMALY_THRESHOLD_MEDIUM:
                    anomalies.append(AnomalyResult(
                        metric_name=f"latency_{row.agent_name}_{row.model}",
                        current_value=avg_latency,
                        baseline_mean=baseline.avg_latency_ms,
                        baseline_stddev=baseline.stddev_latency_ms,
                        z_score=latency_z,
                        is_anomaly=True,
                        severity="high" if abs(latency_z) > self.ANOMALY_THRESHOLD_HIGH else "medium",
                    ))
            
            # Check error rate anomaly
            current_error_rate = error_count / call_count if call_count > 0 else 0
            if current_error_rate > baseline.avg_error_rate * 2 and current_error_rate > 0.05:
                anomalies.append(AnomalyResult(
                    metric_name=f"error_rate_{row.agent_name}_{row.model}",
                    current_value=current_error_rate,
                    baseline_mean=baseline.avg_error_rate,
                    baseline_stddev=0,
                    z_score=0,
                    is_anomaly=True,
                    severity="high" if current_error_rate > 0.2 else "medium",
                ))
        
        return anomalies


class PatternAnalysisService:
    """Service for analyzing input patterns and detecting caching opportunities."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    def _hash_input(self, input_text: str) -> str:
        """Create a normalized hash of input for pattern matching."""
        # Normalize: lowercase, strip whitespace, remove common variable parts
        normalized = input_text.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    async def record_pattern(
        self,
        project_id: str,
        agent_name: str,
        cost: float,
        input_text: Optional[str] = None,
        input_hash: Optional[str] = None,
    ) -> None:
        """
        Record an input pattern occurrence.
        
        Args:
            project_id: Project ID
            agent_name: Agent name
            cost: Cost of the call
            input_text: Raw input text (will be hashed)
            input_hash: Pre-computed hash (used if input_text not provided)
        """
        if input_text:
            pattern_hash = self._hash_input(input_text)
        elif input_hash:
            pattern_hash = input_hash
        else:
            # No input data to record
            return
        
        query = select(InputPatternCache).where(
            InputPatternCache.project_id == project_id,
            InputPatternCache.agent_name == agent_name,
            InputPatternCache.input_hash == pattern_hash,
        )
        result = await self.db.execute(query)
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.occurrence_count += 1
            existing.last_seen_at = datetime.now(timezone.utc)
            existing.total_cost_for_pattern += cost
            existing.avg_cost_per_occurrence = (
                existing.total_cost_for_pattern / existing.occurrence_count
            )
        else:
            pattern = InputPatternCache(
                project_id=project_id,
                agent_name=agent_name,
                input_hash=pattern_hash,
                occurrence_count=1,
                total_cost_for_pattern=cost,
                avg_cost_per_occurrence=cost,
            )
            self.db.add(pattern)
        
        await self.db.flush()
    
    async def analyze_caching_opportunities(
        self,
        project_id: str,
        min_occurrences: int = 5,
        min_savings: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Analyze patterns to find caching opportunities.
        Returns agents with high duplicate query rates.
        """
        # Get patterns with multiple occurrences
        query = select(
            InputPatternCache.agent_name,
            func.count(InputPatternCache.id).label('unique_patterns'),
            func.sum(InputPatternCache.occurrence_count).label('total_calls'),
            func.sum(
                case(
                    (InputPatternCache.occurrence_count > 1, 
                     InputPatternCache.occurrence_count - 1),
                    else_=0
                )
            ).label('duplicate_calls'),
            func.sum(InputPatternCache.total_cost_for_pattern).label('total_cost'),
            func.sum(
                case(
                    (InputPatternCache.occurrence_count > 1,
                     InputPatternCache.avg_cost_per_occurrence * 
                     (InputPatternCache.occurrence_count - 1)),
                    else_=0
                )
            ).label('potential_savings'),
            func.min(InputPatternCache.first_seen_at).label('first_seen'),
            func.max(InputPatternCache.last_seen_at).label('last_seen'),
        ).where(
            InputPatternCache.project_id == project_id
        ).group_by(
            InputPatternCache.agent_name
        ).having(
            func.sum(
                case(
                    (InputPatternCache.occurrence_count > 1, 
                     InputPatternCache.occurrence_count - 1),
                    else_=0
                )
            ) >= min_occurrences
        )
        
        result = await self.db.execute(query)
        opportunities = []
        
        for row in result:
            # Convert Decimal values to int/float for arithmetic operations
            total_calls = int(row.total_calls) if row.total_calls is not None else 0
            duplicate_calls = int(row.duplicate_calls) if row.duplicate_calls is not None else 0
            unique_patterns = int(row.unique_patterns) if row.unique_patterns is not None else 0
            total_cost = float(row.total_cost) if row.total_cost is not None else 0.0
            potential_savings = float(row.potential_savings) if row.potential_savings is not None else 0.0
            first_seen = row.first_seen
            last_seen = row.last_seen
            
            if total_calls == 0:
                continue
            
            duplicate_rate = duplicate_calls / total_calls
            monthly_savings = None
            savings_estimated = False
            coverage_days = None

            if first_seen and last_seen:
                coverage_days = max(1, (last_seen - first_seen).days + 1)
                if coverage_days >= 7:
                    monthly_savings = (potential_savings / coverage_days) * 30
                else:
                    savings_estimated = True
            else:
                savings_estimated = True

            if monthly_savings is not None and monthly_savings < min_savings:
                continue
            
            opportunities.append({
                "agent_name": row.agent_name,
                "unique_patterns": unique_patterns,
                "total_calls": total_calls,
                "duplicate_calls": duplicate_calls,
                "duplicate_rate": round(duplicate_rate * 100, 1),
                "total_cost": round(total_cost, 4),
                "potential_savings": round(potential_savings, 4),
                "estimated_monthly_savings": round(monthly_savings, 2) if monthly_savings is not None else None,
                "savings_estimated": savings_estimated,
                "coverage_days": coverage_days,
            })
        
        opportunities.sort(key=lambda x: x["estimated_monthly_savings"] or 0, reverse=True)
        return opportunities
    
    async def get_top_duplicate_patterns(
        self,
        project_id: str,
        agent_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get the most frequently duplicated patterns."""
        query = select(InputPatternCache).where(
            InputPatternCache.project_id == project_id,
            InputPatternCache.occurrence_count > 1,
        )
        
        if agent_name:
            query = query.where(InputPatternCache.agent_name == agent_name)
        
        query = query.order_by(
            InputPatternCache.occurrence_count.desc()
        ).limit(limit)
        
        result = await self.db.execute(query)
        patterns = result.scalars().all()
        
        return [
            {
                "agent_name": p.agent_name,
                "input_hash": p.input_hash[:16] + "...",
                "occurrence_count": p.occurrence_count,
                "total_cost": round(p.total_cost_for_pattern, 4),
                "avg_cost": round(p.avg_cost_per_occurrence, 6),
                "first_seen": p.first_seen_at.isoformat() if p.first_seen_at else None,
                "last_seen": p.last_seen_at.isoformat() if p.last_seen_at else None,
            }
            for p in patterns
        ]


class RecommendationTrackingService:
    """Service for tracking optimization recommendations and outcomes."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # Cooldown periods to prevent recommendation spam
    DISMISSED_COOLDOWN_DAYS = 7  # Don't recreate dismissed recommendations for 7 days
    IMPLEMENTED_COOLDOWN_DAYS = 30  # Don't recreate implemented recommendations for 30 days
    MINIMUM_SAVINGS_THRESHOLD = 1.0  # Only persist recommendations >= $1.00/month
    
    async def create_recommendation(
        self,
        project_id: str,
        recommendation_type: str,
        title: str,
        description: str,
        agent_name: Optional[str] = None,
        model: Optional[str] = None,
        alternative_model: Optional[str] = None,
        estimated_monthly_savings: float = 0.0,
        estimated_savings_percent: float = 0.0,
        metrics_snapshot: Optional[Dict] = None,
        expires_days: int = 30,
    ) -> Optional[OptimizationRecommendation]:
        """
        Create and store a new recommendation with intelligent de-duplication.
        
        Returns None if:
        - Similar recommendation was dismissed within DISMISSED_COOLDOWN_DAYS
        - Similar recommendation was implemented within IMPLEMENTED_COOLDOWN_DAYS
        - Estimated savings is below MINIMUM_SAVINGS_THRESHOLD
        """
        # Skip low-value recommendations
        if estimated_monthly_savings < self.MINIMUM_SAVINGS_THRESHOLD:
            return None

        safe_metrics = _json_safe(metrics_snapshot) if metrics_snapshot is not None else None
        
        # Check for existing pending recommendation - update it if found
        pending_query = select(OptimizationRecommendation).where(
            OptimizationRecommendation.project_id == project_id,
            OptimizationRecommendation.recommendation_type == recommendation_type,
            OptimizationRecommendation.agent_name == agent_name,
            OptimizationRecommendation.model == model,
            OptimizationRecommendation.alternative_model == alternative_model,
            OptimizationRecommendation.status == "pending",
        ).order_by(OptimizationRecommendation.created_at.desc())
        pending_result = await self.db.execute(pending_query)
        existing_pending = pending_result.scalars().first()
        
        if existing_pending:
            # Update existing pending recommendation with fresh data
            existing_pending.title = title
            existing_pending.description = description
            existing_pending.estimated_monthly_savings = estimated_monthly_savings
            existing_pending.estimated_savings_percent = estimated_savings_percent
            existing_pending.metrics_snapshot = safe_metrics
            existing_pending.expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
            await self.db.flush()
            return existing_pending
        
        # Check for recently dismissed recommendation - don't recreate within cooldown
        dismissed_cutoff = datetime.now(timezone.utc) - timedelta(days=self.DISMISSED_COOLDOWN_DAYS)
        dismissed_query = select(OptimizationRecommendation).where(
            OptimizationRecommendation.project_id == project_id,
            OptimizationRecommendation.recommendation_type == recommendation_type,
            OptimizationRecommendation.agent_name == agent_name,
            OptimizationRecommendation.model == model,
            OptimizationRecommendation.alternative_model == alternative_model,
            OptimizationRecommendation.status == "dismissed",
            OptimizationRecommendation.dismissed_at >= dismissed_cutoff,
        )
        dismissed_result = await self.db.execute(dismissed_query)
        if dismissed_result.scalars().first():
            # Recently dismissed - respect user's decision
            return None
        
        # Check for recently implemented recommendation - don't recreate within cooldown
        implemented_cutoff = datetime.now(timezone.utc) - timedelta(days=self.IMPLEMENTED_COOLDOWN_DAYS)
        implemented_query = select(OptimizationRecommendation).where(
            OptimizationRecommendation.project_id == project_id,
            OptimizationRecommendation.recommendation_type == recommendation_type,
            OptimizationRecommendation.agent_name == agent_name,
            OptimizationRecommendation.model == model,
            OptimizationRecommendation.alternative_model == alternative_model,
            OptimizationRecommendation.status == "implemented",
            OptimizationRecommendation.implemented_at >= implemented_cutoff,
        )
        implemented_result = await self.db.execute(implemented_query)
        if implemented_result.scalars().first():
            # Recently implemented - no need to suggest again
            return None
        
        # Create new recommendation
        recommendation = OptimizationRecommendation(
            project_id=project_id,
            recommendation_type=recommendation_type,
            title=title,
            description=description,
            agent_name=agent_name,
            model=model,
            alternative_model=alternative_model,
            estimated_monthly_savings=estimated_monthly_savings,
            estimated_savings_percent=estimated_savings_percent,
            metrics_snapshot=safe_metrics,
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_days),
        )
        
        self.db.add(recommendation)
        await self.db.flush()
        await self.db.refresh(recommendation)
        
        return recommendation
    
    async def mark_implemented(
        self,
        recommendation_id: str,
        project_id: str,
    ) -> Optional[OptimizationRecommendation]:
        """Mark a recommendation as implemented and trigger learning."""
        query = select(OptimizationRecommendation).where(
            OptimizationRecommendation.id == recommendation_id,
            OptimizationRecommendation.project_id == project_id,
        )
        result = await self.db.execute(query)
        recommendation = result.scalar_one_or_none()
        
        if recommendation:
            recommendation.status = "implemented"
            recommendation.implemented_at = datetime.now(timezone.utc)
            await self.db.flush()
            
            # Trigger learning for model alternatives
            if recommendation.model and recommendation.alternative_model:
                from .alternative_learning_service import AlternativeLearningService
                learning_service = AlternativeLearningService(self.db)
                await learning_service.update_from_recommendation_outcome(
                    source_model=recommendation.model,
                    alternative_model=recommendation.alternative_model,
                    was_implemented=True,
                    estimated_savings=recommendation.estimated_monthly_savings or 0.0,
                )
        
        return recommendation
    
    async def mark_dismissed(
        self,
        recommendation_id: str,
        project_id: str,
        feedback: Optional[str] = None,
    ) -> Optional[OptimizationRecommendation]:
        """Mark a recommendation as dismissed with optional feedback and trigger learning."""
        query = select(OptimizationRecommendation).where(
            OptimizationRecommendation.id == recommendation_id,
            OptimizationRecommendation.project_id == project_id,
        )
        result = await self.db.execute(query)
        recommendation = result.scalar_one_or_none()
        
        if recommendation:
            recommendation.status = "dismissed"
            recommendation.dismissed_at = datetime.now(timezone.utc)
            recommendation.user_feedback = feedback
            await self.db.flush()
            
            # Trigger learning for model alternatives (dismissed)
            if recommendation.model and recommendation.alternative_model:
                from .alternative_learning_service import AlternativeLearningService
                learning_service = AlternativeLearningService(self.db)
                await learning_service.update_from_recommendation_outcome(
                    source_model=recommendation.model,
                    alternative_model=recommendation.alternative_model,
                    was_implemented=False,
                    estimated_savings=recommendation.estimated_monthly_savings or 0.0,
                    user_feedback=feedback,
                )
        
        return recommendation
    
    async def record_outcome(
        self,
        recommendation_id: str,
        project_id: str,
        actual_savings: float,
    ) -> Optional[OptimizationRecommendation]:
        """Record the actual savings after implementation and update learning accuracy."""
        query = select(OptimizationRecommendation).where(
            OptimizationRecommendation.id == recommendation_id,
            OptimizationRecommendation.project_id == project_id,
        )
        result = await self.db.execute(query)
        recommendation = result.scalar_one_or_none()
        
        if recommendation:
            recommendation.actual_savings = actual_savings
            recommendation.outcome_measured_at = datetime.now(timezone.utc)
            await self.db.flush()
            
            # Update learning with actual outcome for accuracy tracking
            if recommendation.model and recommendation.alternative_model:
                from .alternative_learning_service import AlternativeLearningService
                learning_service = AlternativeLearningService(self.db)
                await learning_service.update_from_recommendation_outcome(
                    source_model=recommendation.model,
                    alternative_model=recommendation.alternative_model,
                    was_implemented=True,
                    estimated_savings=recommendation.estimated_monthly_savings or 0.0,
                    actual_savings=actual_savings,
                )
        
        return recommendation
    
    async def get_pending_recommendations(
        self,
        project_id: str,
    ) -> List[OptimizationRecommendation]:
        """Get all pending recommendations for a project."""
        query = select(OptimizationRecommendation).where(
            OptimizationRecommendation.project_id == project_id,
            OptimizationRecommendation.status == "pending",
        ).order_by(
            OptimizationRecommendation.estimated_monthly_savings.desc()
        )
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_recommendation_effectiveness(
        self,
        project_id: str,
    ) -> Dict[str, Any]:
        """Calculate effectiveness metrics for recommendations."""
        query = select(
            func.count(OptimizationRecommendation.id).label('total'),
            func.sum(case(
                (OptimizationRecommendation.status == "implemented", 1),
                else_=0
            )).label('implemented'),
            func.sum(case(
                (OptimizationRecommendation.status == "dismissed", 1),
                else_=0
            )).label('dismissed'),
            func.sum(OptimizationRecommendation.estimated_monthly_savings).label('estimated_total'),
            func.sum(
                case(
                    (OptimizationRecommendation.status == "implemented",
                     OptimizationRecommendation.actual_savings),
                    else_=0
                )
            ).label('actual_total'),
        ).where(
            OptimizationRecommendation.project_id == project_id
        )
        
        result = await self.db.execute(query)
        row = result.one()
        
        implementation_rate = (
            (row.implemented or 0) / row.total * 100 
            if row.total > 0 else 0
        )
        
        accuracy = 0
        if row.estimated_total and row.actual_total:
            accuracy = min(100, (row.actual_total / row.estimated_total) * 100)
        
        return {
            "total_recommendations": row.total or 0,
            "implemented": row.implemented or 0,
            "dismissed": row.dismissed or 0,
            "implementation_rate": round(implementation_rate, 1),
            "estimated_savings_total": round(row.estimated_total or 0, 2),
            "actual_savings_total": round(row.actual_total or 0, 2),
            "accuracy_percent": round(accuracy, 1),
        }
