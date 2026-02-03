"""
AgentCost Backend - Analytics Service

Business logic for analytics queries.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from ..models.db_models import Event
from ..models.schemas import (
    AnalyticsOverview,
    AgentStats,
    ModelStats,
    TimeSeriesPoint,
    AnalyticsResponse,
)


class AnalyticsService:
    """Service for analytics queries"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_overview(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> AnalyticsOverview:
        """
        Get overview metrics for a time period.
        
        Args:
            project_id: Project ID
            start_time: Period start
            end_time: Period end
            
        Returns:
            AnalyticsOverview with aggregated metrics
        """
        query = select(
            func.count(Event.id).label('total_calls'),
            func.sum(Event.total_tokens).label('total_tokens'),
            func.sum(Event.input_tokens).label('total_input_tokens'),
            func.sum(Event.output_tokens).label('total_output_tokens'),
            func.sum(Event.cost).label('total_cost'),
            func.avg(Event.latency_ms).label('avg_latency'),
            func.sum(case((Event.success == True, 1), else_=0)).label('success_count'),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        )
        
        result = await self.db.execute(query)
        row = result.one()
        
        # Convert Decimal values to int/float for arithmetic operations
        total_calls = int(row.total_calls) if row.total_calls is not None else 0
        total_tokens = int(row.total_tokens) if row.total_tokens is not None else 0
        total_input_tokens = int(row.total_input_tokens) if row.total_input_tokens is not None else 0
        total_output_tokens = int(row.total_output_tokens) if row.total_output_tokens is not None else 0
        total_cost = float(row.total_cost) if row.total_cost is not None else 0.0
        avg_latency = float(row.avg_latency) if row.avg_latency is not None else 0.0
        success_count = int(row.success_count) if row.success_count is not None else 0
        
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else 0.0
        avg_tokens_per_call = total_tokens / total_calls if total_calls > 0 else 0.0
        success_rate = (success_count / total_calls * 100) if total_calls > 0 else 100.0
        
        return AnalyticsOverview(
            total_cost=round(total_cost, 6),
            total_calls=total_calls,
            total_tokens=total_tokens,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            avg_cost_per_call=round(avg_cost_per_call, 6),
            avg_tokens_per_call=round(avg_tokens_per_call, 2),
            avg_latency_ms=round(avg_latency, 2),
            success_rate=round(success_rate, 2),
            period_start=start_time,
            period_end=end_time,
        )
    
    async def get_agent_stats(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10,
    ) -> List[AgentStats]:
        """
        Get per-agent statistics.
        
        Args:
            project_id: Project ID
            start_time: Period start
            end_time: Period end
            limit: Max agents to return
            
        Returns:
            List of AgentStats
        """
        query = select(
            Event.agent_name,
            func.count(Event.id).label('total_calls'),
            func.sum(Event.total_tokens).label('total_tokens'),
            func.sum(Event.cost).label('total_cost'),
            func.avg(Event.latency_ms).label('avg_latency'),
            func.sum(case((Event.success == True, 1), else_=0)).label('success_count'),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        ).group_by(
            Event.agent_name
        ).order_by(
            func.sum(Event.cost).desc()
        ).limit(limit)
        
        result = await self.db.execute(query)
        
        agents = []
        for row in result:
            # Convert Decimal values to int/float for arithmetic operations
            total_calls = int(row.total_calls) if row.total_calls is not None else 0
            total_tokens = int(row.total_tokens) if row.total_tokens is not None else 0
            total_cost = float(row.total_cost) if row.total_cost is not None else 0.0
            avg_latency = float(row.avg_latency) if row.avg_latency is not None else 0.0
            success_count = int(row.success_count) if row.success_count is not None else 0
            success_rate = (success_count / total_calls * 100) if total_calls > 0 else 100.0
            
            agents.append(AgentStats(
                agent_name=row.agent_name,
                total_calls=total_calls,
                total_tokens=total_tokens,
                total_cost=round(total_cost, 6),
                avg_latency_ms=round(avg_latency, 2),
                success_rate=round(success_rate, 2),
            ))
        
        return agents
    
    async def get_model_stats(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10,
    ) -> List[ModelStats]:
        """
        Get per-model statistics.
        
        Args:
            project_id: Project ID
            start_time: Period start
            end_time: Period end
            limit: Max models to return
            
        Returns:
            List of ModelStats
        """
        query = select(
            Event.model,
            func.count(Event.id).label('total_calls'),
            func.sum(Event.total_tokens).label('total_tokens'),
            func.sum(Event.input_tokens).label('input_tokens'),
            func.sum(Event.output_tokens).label('output_tokens'),
            func.sum(Event.cost).label('total_cost'),
            func.avg(Event.latency_ms).label('avg_latency'),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        ).group_by(
            Event.model
        ).order_by(
            func.sum(Event.cost).desc()
        ).limit(limit)
        
        result = await self.db.execute(query)
        
        # First pass: collect all models and calculate total cost
        rows_data = []
        total_cost_all = 0.0
        for row in result:
            rows_data.append(row)
            row_cost = float(row.total_cost) if row.total_cost is not None else 0.0
            total_cost_all += row_cost
        
        models = []
        for row in rows_data:
            # Convert Decimal values to int/float for arithmetic operations
            total_calls = int(row.total_calls) if row.total_calls is not None else 0
            total_tokens = int(row.total_tokens) if row.total_tokens is not None else 0
            input_tokens = int(row.input_tokens) if row.input_tokens is not None else 0
            output_tokens = int(row.output_tokens) if row.output_tokens is not None else 0
            model_cost = float(row.total_cost) if row.total_cost is not None else 0.0
            avg_latency = float(row.avg_latency) if row.avg_latency is not None else 0.0
            cost_share = (model_cost / total_cost_all * 100) if total_cost_all > 0 else 0.0
            
            models.append(ModelStats(
                model=row.model,
                total_calls=total_calls,
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost=round(model_cost, 6),
                avg_latency_ms=round(avg_latency, 2),
                cost_share=round(cost_share, 1),
            ))
        
        return models
    
    async def get_timeseries(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        granularity: str = "hour",  # hour, day
    ) -> List[TimeSeriesPoint]:
        """
        Get time series data.
        
        Args:
            project_id: Project ID
            start_time: Period start
            end_time: Period end
            granularity: Time bucket size (hour or day)
            
        Returns:
            List of TimeSeriesPoint
        """
        # For SQLite, we use strftime. For PostgreSQL, use date_trunc
        if granularity == "day":
            time_bucket = func.date(Event.timestamp)
        else:
            # Hour granularity - use strftime for SQLite compatibility
            time_bucket = func.strftime('%Y-%m-%d %H:00:00', Event.timestamp)
        
        query = select(
            time_bucket.label('time_bucket'),
            func.count(Event.id).label('calls'),
            func.sum(Event.total_tokens).label('tokens'),
            func.sum(Event.cost).label('cost'),
            func.avg(Event.latency_ms).label('avg_latency'),
        ).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        ).group_by(
            time_bucket
        ).order_by(
            time_bucket
        )
        
        result = await self.db.execute(query)
        
        timeseries = []
        for row in result:
            # Parse timestamp
            if isinstance(row.time_bucket, str):
                ts = datetime.fromisoformat(row.time_bucket)
            else:
                ts = row.time_bucket
            
            # Convert Decimal values to int/float for arithmetic operations
            calls = int(row.calls) if row.calls is not None else 0
            tokens = int(row.tokens) if row.tokens is not None else 0
            cost = float(row.cost) if row.cost is not None else 0.0
            avg_latency = float(row.avg_latency) if row.avg_latency is not None else 0.0
            
            timeseries.append(TimeSeriesPoint(
                timestamp=ts,
                calls=calls,
                tokens=tokens,
                cost=round(cost, 6),
                avg_latency_ms=round(avg_latency, 2),
            ))
        
        return timeseries
    
    async def get_full_analytics(
        self,
        project_id: str,
        days: int = 7,
    ) -> AnalyticsResponse:
        """
        Get full analytics response.
        
        Args:
            project_id: Project ID
            days: Number of days to analyze
            
        Returns:
            Complete AnalyticsResponse
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Determine granularity based on time range
        granularity = "day" if days > 2 else "hour"
        
        # Get all analytics in parallel would be better, but for simplicity:
        overview = await self.get_overview(project_id, start_time, end_time)
        agents = await self.get_agent_stats(project_id, start_time, end_time)
        models = await self.get_model_stats(project_id, start_time, end_time)
        timeseries = await self.get_timeseries(project_id, start_time, end_time, granularity)
        
        return AnalyticsResponse(
            overview=overview,
            agents=agents,
            models=models,
            timeseries=timeseries,
        )
