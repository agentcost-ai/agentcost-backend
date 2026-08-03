"""
AgentCost Backend - Analytics Service

Business logic for analytics queries.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ..models.db_models import Event
from ..utils.sql_dialect import as_utc_datetime, dialect_name, utc_timestamp
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
        self._dialect = dialect_name(db)

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
        # Empty window -> 100: nothing failed. Reporting 0 made the dashboard
        # and PDF render an idle project as "100% error rate" in red.
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
            # Same empty-window convention as get_overview.
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

    async def get_distinct_model_count(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """
        How many distinct models ran in the window, including the top-N tail.

        Quoting the length of a top-N list instead makes "3 of 10 models drive
        80% of spend" appear for a project that actually runs 40.
        """
        query = select(func.count(func.distinct(Event.model))).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        )
        return int((await self.db.execute(query)).scalar() or 0)

    async def _window_cost(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> float:
        """Total spend over the whole filtered window."""
        query = select(func.sum(Event.cost)).where(
            Event.project_id == project_id,
            Event.timestamp >= start_time,
            Event.timestamp <= end_time,
        )
        return float((await self.db.execute(query)).scalar() or 0.0)

    async def get_model_stats(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10,
        window_cost: Optional[float] = None,
    ) -> List[ModelStats]:
        """
        Get per-model statistics.

        Args:
            project_id: Project ID
            start_time: Period start
            end_time: Period end
            limit: Max models to return
            window_cost: Spend over the whole window, used as the cost_share
                denominator. Pass it when the caller has already summed the same
                window; omit it and this queries for it.

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
        rows_data = list(result)

        # Share is against every model in the window, including the tail the
        # limit above truncated: renormalizing over the slice would make the
        # listed shares add to 100% however much spend the tail holds.
        total_cost_all = (
            window_cost
            if window_cost is not None
            else await self._window_cost(project_id, start_time, end_time)
        )

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
        # Bucket in UTC, matching the UTC window bounds above.
        ts = utc_timestamp(self._dialect)
        if granularity == "day":
            time_bucket = func.date(ts)
        else:
            # Hour granularity
            if self._dialect == "sqlite":
                time_bucket = func.strftime('%Y-%m-%d %H:00:00', ts)
            else:
                time_bucket = func.date_trunc('hour', ts)

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
            # Buckets come back as text (SQLite), date or naive datetime (PG);
            # all of them are UTC, so label them as such.
            ts = as_utc_datetime(row.time_bucket)

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
