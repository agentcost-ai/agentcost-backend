"""
AgentCost Backend - SQL Dialect Helpers

Primitives for rendering the same query correctly on PostgreSQL and SQLite.
Services import these instead of sniffing settings.database_url.
"""

from datetime import datetime, timezone

from sqlalchemy import func

from ..config import get_settings
from ..models.db_models import Event

settings = get_settings()


def dialect_name(session) -> str:
    """
    Resolve the dialect of the engine a session is actually bound to.

    Services must not sniff ``settings.database_url`` for this: the tests bind
    an in-memory SQLite engine while DATABASE_URL still points at PostgreSQL,
    so URL-sniffing picks the wrong SQL for the connection in hand.
    """
    try:
        bind = session.get_bind()
        return bind.dialect.name
    except Exception:  # noqa: BLE001 - fall back to the configured URL
        return "sqlite" if "sqlite" in settings.database_url else "postgresql"


def utc_timestamp(dialect: str):
    """Event.timestamp in UTC: PG's date_trunc/extract otherwise bucket it in the session TimeZone."""
    if dialect == "postgresql":
        return func.timezone("UTC", Event.timestamp)
    return Event.timestamp


def stddev_pop(column, dialect: str):
    """Population standard deviation of ``column``.

    The hand-rolled ``sqrt(avg(x*x) - avg(x)*avg(x))`` form is unsafe on
    PostgreSQL in two separate ways, and safe on SQLite in both, which is why
    the test suite never caught it:

    * Catastrophic cancellation. For low-variance data the two averages agree
      to ~16 significant digits, so their float8 difference lands on a tiny
      negative (-2.1e-22 for three rows of 0.00123). PG's ``sqrt`` then raises
      InvalidArgumentForPowerFunction; SQLite's returns NULL. ``coalesce``
      cannot rescue the PG case because the error happens before there is a
      NULL to coalesce.
    * int4 overflow. ``x * x`` on an Integer column stays int4 on PG and
      overflows, while SQLite promotes to a 64-bit int and absorbs it.

    PG's own aggregate accumulates in numeric and has neither problem, and it
    computes the same population variance the manual form intended. SQLite has
    no stddev aggregate, so it keeps the manual form with the radicand clamped
    at zero.
    """
    if dialect == "postgresql":
        return func.stddev_pop(column)
    variance = func.avg(column * column) - func.avg(column) * func.avg(column)
    return func.sqrt(func.max(variance, 0))


def as_utc_datetime(value) -> datetime:
    """Normalize a DB time bucket (str / date / naive datetime) to aware UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    elif not isinstance(value, datetime) and hasattr(value, "year"):
        # date -> midnight UTC
        value = datetime(value.year, value.month, value.day)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
