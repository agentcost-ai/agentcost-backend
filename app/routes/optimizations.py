"""
AgentCost Backend - Optimization API Routes

Endpoints for cost optimization suggestions with recommendation tracking.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ..database import get_db
from ..models.db_models import Project
from ..services.optimization_service import OptimizationService
from ..services.baseline_service import (
    BaselineService,
    PatternAnalysisService,
    RecommendationTrackingService,
)
from ..utils.auth import validate_api_key

router = APIRouter(prefix="/v1/optimizations", tags=["Optimizations"])


class RecommendationFeedback(BaseModel):
    feedback: Optional[str] = None


@router.get("", response_model=List[Dict[str, Any]])
async def get_optimizations(
    days: int = Query(30, ge=1, le=90, description="Days of history to analyze"),
    include_low_priority: bool = Query(True, description="Include low priority suggestions"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get cost optimization suggestions for your project.
    
    Analyzes usage patterns and suggests ways to reduce costs:
    - Model downgrades based on actual usage and dynamic pricing
    - Caching opportunities from detected input patterns
    - Anomaly alerts from statistical baseline deviations
    - Error pattern fixes with calculated wasted spend
    
    All suggestions use real pricing data and statistical analysis.
    """
    optimization_service = OptimizationService(db)
    return await optimization_service.get_suggestions(
        project.id,
        days,
        include_low_priority=include_low_priority,
        persist_recommendations=False,
    )


@router.post("/recommendations/generate", response_model=List[Dict[str, Any]])
async def generate_recommendations(
    days: int = Query(30, ge=1, le=90, description="Days of history to analyze"),
    include_low_priority: bool = Query(True, description="Include low priority suggestions"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Generate optimization suggestions and persist top recommendations.
    
    This endpoint intentionally has side effects and should be called
    explicitly by clients when they want new recommendations recorded.
    """
    optimization_service = OptimizationService(db)
    return await optimization_service.get_suggestions(
        project.id,
        days,
        include_low_priority=include_low_priority,
        persist_recommendations=True,
    )


@router.get("/summary")
async def get_optimization_summary(
    days: int = Query(30, ge=1, le=90, description="Days of history to analyze"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get optimization summary with total potential savings.
    
    Returns:
    - Total potential monthly savings and percentage
    - Current monthly spend for context
    - Suggestion count and breakdown by type
    - Recommendation effectiveness metrics
    - Top 5 recommendations
    """
    optimization_service = OptimizationService(db)
    return await optimization_service.get_summary(project.id, days)


@router.post("/baselines/refresh")
async def refresh_baselines(
    days: int = Query(30, ge=7, le=90, description="Days of history for baseline calculation"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Refresh statistical baselines for the project.
    
    Baselines are used for anomaly detection and threshold calculations.
    They should be recalculated periodically (e.g., weekly) or after
    significant changes to your agent configurations.
    """
    baseline_service = BaselineService(db)
    return await baseline_service.compute_baselines(project.id, days)


@router.get("/baselines")
async def get_baselines(
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    model: Optional[str] = Query(None, description="Filter by model"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get current statistical baselines for the project.
    
    Baselines include mean and standard deviation for:
    - Cost per call
    - Token usage (input and output)
    - Latency
    - Daily call volume
    - Error rate
    """
    from sqlalchemy import select
    from ..models.db_models import ProjectBaseline
    
    query = select(ProjectBaseline).where(
        ProjectBaseline.project_id == project.id
    )
    
    if agent_name:
        query = query.where(ProjectBaseline.agent_name == agent_name)
    if model:
        query = query.where(ProjectBaseline.model == model)
    
    result = await db.execute(query)
    baselines = result.scalars().all()
    
    return [
        {
            "agent_name": b.agent_name,
            "model": b.model,
            "avg_cost_per_call": round(b.avg_cost_per_call, 6),
            "stddev_cost_per_call": round(b.stddev_cost_per_call, 6),
            "avg_input_tokens": round(b.avg_input_tokens, 1),
            "avg_output_tokens": round(b.avg_output_tokens, 1),
            "avg_latency_ms": round(b.avg_latency_ms, 1),
            "stddev_latency_ms": round(b.stddev_latency_ms, 1),
            "avg_daily_calls": round(b.avg_daily_calls, 1),
            "avg_error_rate": round(b.avg_error_rate, 4),
            "sample_count": b.sample_count,
            "last_calculated_at": b.last_calculated_at.isoformat() if b.last_calculated_at else None,
        }
        for b in baselines
    ]


@router.get("/caching-opportunities")
async def get_caching_opportunities(
    min_occurrences: int = Query(5, ge=2, description="Minimum duplicate occurrences"),
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get detailed caching opportunity analysis.
    
    Returns agents with detected duplicate input patterns and
    calculated potential savings from implementing caching.
    """
    pattern_service = PatternAnalysisService(db)
    opportunities = await pattern_service.analyze_caching_opportunities(
        project_id=project.id,
        min_occurrences=min_occurrences,
    )
    
    return {
        "opportunities": opportunities,
        "total_potential_monthly_savings": sum(
            (o.get("estimated_monthly_savings") or 0) for o in opportunities
        ),
    }


@router.get("/recommendations")
async def get_pending_recommendations(
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get pending optimization recommendations.
    
    These are recommendations that have been generated but not yet
    implemented or dismissed.
    """
    tracking_service = RecommendationTrackingService(db)
    recommendations = await tracking_service.get_pending_recommendations(project.id)
    
    return [
        {
            "id": r.id,
            "type": r.recommendation_type,
            "title": r.title,
            "description": r.description,
            "agent_name": r.agent_name,
            "model": r.model,
            "alternative_model": r.alternative_model,
            "estimated_monthly_savings": r.estimated_monthly_savings,
            "estimated_savings_percent": r.estimated_savings_percent,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "expires_at": r.expires_at.isoformat() if r.expires_at else None,
        }
        for r in recommendations
    ]


@router.post("/recommendations/{recommendation_id}/implement")
async def mark_recommendation_implemented(
    recommendation_id: str,
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Mark a recommendation as implemented.
    
    This helps track which recommendations users adopt and measure
    their actual effectiveness over time.
    """
    tracking_service = RecommendationTrackingService(db)
    recommendation = await tracking_service.mark_implemented(
        recommendation_id=recommendation_id,
        project_id=project.id,
    )
    
    if not recommendation:
        raise HTTPException(
            status_code=404, 
            detail="This recommendation is no longer available. It may have expired or already been actioned."
        )
    
    return {
        "status": "ok",
        "recommendation_id": recommendation_id,
        "implemented_at": recommendation.implemented_at.isoformat(),
        "message": "Recommendation marked as implemented. We'll track the results.",
    }


@router.post("/recommendations/{recommendation_id}/dismiss")
async def dismiss_recommendation(
    recommendation_id: str,
    feedback: RecommendationFeedback = None,
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Dismiss a recommendation with optional feedback.
    
    Feedback helps improve future recommendations.
    """
    tracking_service = RecommendationTrackingService(db)
    recommendation = await tracking_service.mark_dismissed(
        recommendation_id=recommendation_id,
        project_id=project.id,
        feedback=feedback.feedback if feedback else None,
    )
    
    if not recommendation:
        raise HTTPException(
            status_code=404, 
            detail="This recommendation is no longer available. It may have expired or already been actioned."
        )
    
    return {
        "status": "ok",
        "recommendation_id": recommendation_id,
        "dismissed_at": recommendation.dismissed_at.isoformat(),
        "message": "Recommendation dismissed. Thank you for your feedback.",
    }


@router.get("/recommendations/effectiveness")
async def get_recommendation_effectiveness(
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    Get effectiveness metrics for recommendations.
    
    Shows how many recommendations were implemented, dismissed,
    and the accuracy of estimated vs actual savings.
    """
    tracking_service = RecommendationTrackingService(db)
    return await tracking_service.get_recommendation_effectiveness(project.id)
