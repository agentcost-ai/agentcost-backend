"""
Days are the viewer's days, and a past day's total does not move.

Ranges used to be measured back to the minute and bucketed at UTC midnight:
the oldest day of "Last 7 days" was a fraction that shrank on every refresh,
and for a viewer in India anything before 5:30 am landed on the previous day.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.routes.analytics import parse_time_range
from app.utils import client_time

IST = 330  # minutes east of UTC


def _event(at: datetime, cost: float):
    return {
        "agent_name": "claude-code",
        "model": "unpriced-model",
        "input_tokens": 100,
        "output_tokens": 10,
        "cost": cost,
        "timestamp": at.isoformat(),
    }


async def _ingest(client, project, *events):
    response = await client.post(
        "/v1/events/batch", json={"project_id": project.id, "events": list(events)}
    )
    assert response.status_code == 200, response.text


@pytest.fixture
def ist():
    token = client_time._offset_minutes.set(IST)
    yield
    client_time._offset_minutes.reset(token)


def test_a_week_is_today_and_six_whole_days_before_it(ist):
    start, end = parse_time_range("7d")

    local_start = start + timedelta(minutes=IST)
    assert (local_start.hour, local_start.minute) == (0, 0)
    now_local = datetime.now(timezone.utc) + timedelta(minutes=IST)
    assert (now_local.date() - local_start.date()).days == 6
    assert end > datetime.now(timezone.utc)


def test_the_start_does_not_move_between_refreshes(ist):
    assert parse_time_range("30d")[0] == parse_time_range("30d")[0]


def test_24h_starts_on_the_hour(ist):
    start, _ = parse_time_range("24h")
    local = start + timedelta(minutes=IST)
    assert (local.minute, local.second) == (0, 0)


def test_without_the_header_days_are_utc_days():
    start, _ = parse_time_range("7d")
    assert (start.hour, start.minute) == (0, 0)


async def test_late_night_work_belongs_to_the_night_it_was_done(client, test_project):
    """01:00 in India on a day is still the previous day in UTC."""
    yesterday = (datetime.now(timezone.utc) + timedelta(minutes=IST)).date() - timedelta(days=1)
    one_am_ist = datetime(yesterday.year, yesterday.month, yesterday.day, 1, 0,
                          tzinfo=timezone.utc) - timedelta(minutes=IST)
    await _ingest(client, test_project, _event(one_am_ist, 2.0))

    in_india = await client.get(
        "/v1/analytics/timeseries?range=7d&granularity=day",
        headers={"X-Timezone-Offset": str(IST)},
    )
    in_utc = await client.get("/v1/analytics/timeseries?range=7d&granularity=day")

    def days(response):
        return [
            (datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")) + timedelta(minutes=IST)).date()
            for p in response.json()
        ]

    assert days(in_india) == [yesterday]  # the bucket starts at local midnight
    utc_bucket = datetime.fromisoformat(in_utc.json()[0]["timestamp"].replace("Z", "+00:00"))
    assert utc_bucket.date() == yesterday - timedelta(days=1)


async def test_a_past_day_and_the_total_stay_put(client, test_project):
    headers = {"X-Timezone-Offset": str(IST)}
    now = datetime.now(timezone.utc)
    await _ingest(
        client,
        test_project,
        _event(now - timedelta(days=6, hours=1), 1.0),
        _event(now - timedelta(days=3), 2.0),
        _event(now - timedelta(minutes=5), 3.0),
    )

    first = (await client.get("/v1/analytics/overview?range=7d", headers=headers)).json()
    again = (await client.get("/v1/analytics/overview?range=7d", headers=headers)).json()
    series = (
        await client.get("/v1/analytics/timeseries?range=7d&granularity=day", headers=headers)
    ).json()

    assert first["total_cost"] == again["total_cost"]
    assert round(sum(point["cost"] for point in series), 6) == first["total_cost"]


async def test_a_nonsense_offset_is_ignored_not_an_error(client, test_project):
    for value in ("abc", "99999", "-99999", ""):
        response = await client.get(
            "/v1/analytics/overview?range=7d", headers={"X-Timezone-Offset": value}
        )
        assert response.status_code == 200, value


async def test_the_browser_may_send_the_header(client):
    response = await client.options(
        "/v1/analytics/overview",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-timezone-offset",
        },
    )
    assert "x-timezone-offset" in response.headers.get("access-control-allow-headers", "").lower()
