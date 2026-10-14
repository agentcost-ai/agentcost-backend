"""
AgentCost Backend - The caller's clock

A day on a chart should be the viewer's day. Bucketing in UTC put everything a
user in India did before 5:30 am on the previous day, and a "Last 7 days" window
measured back to the minute began part-way through a day, so that first day
shrank and the total moved every time the page was refreshed.

The dashboard sends ``X-Timezone-Offset`` -- minutes east of UTC, e.g. 330 for
India -- and analytics read it from here. Without the header everything stays
in UTC, so API clients see exactly what they saw before.
"""

from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Header

# UTC-12:00 to UTC+14:00, the widest offsets in use.
_LIMIT_MINUTES = 14 * 60

_offset_minutes: ContextVar[int] = ContextVar("client_utc_offset_minutes", default=0)


async def use_client_timezone(
    x_timezone_offset: Optional[str] = Header(
        None,
        description="Minutes east of UTC (330 for India). Days and hours are bucketed in this zone.",
    ),
) -> None:
    """Router dependency. Async on purpose: an async dependency runs in the
    request's own task, so the value is visible to the endpoint that follows."""
    try:
        minutes = int(x_timezone_offset) if x_timezone_offset else 0
    except ValueError:
        minutes = 0
    _offset_minutes.set(max(-_LIMIT_MINUTES, min(_LIMIT_MINUTES, minutes)))


def offset_minutes() -> int:
    return _offset_minutes.get()


def offset() -> timedelta:
    return timedelta(minutes=offset_minutes())


def local_midnight(moment: datetime) -> datetime:
    """The start of ``moment``'s day on the caller's clock, as a UTC instant."""
    local = moment.astimezone(timezone.utc) + offset()
    return local.replace(hour=0, minute=0, second=0, microsecond=0) - offset()


def local_hour(moment: datetime) -> datetime:
    """The start of ``moment``'s hour on the caller's clock, as a UTC instant."""
    local = moment.astimezone(timezone.utc) + offset()
    return local.replace(minute=0, second=0, microsecond=0) - offset()


def local_month_start(moment: datetime) -> datetime:
    """The first of ``moment``'s month on the caller's clock, as a UTC instant."""
    local = moment.astimezone(timezone.utc) + offset()
    return local.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - offset()
