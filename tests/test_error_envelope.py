"""Every error carries a machine-readable envelope, and `detail` stays a string.

Before this, HTTPException produced {"detail": "..."} and validation failures
produced {"detail": [ ...objects... ]} -- two shapes, neither with a code an
agent could branch on. The dashboard's AuthContext does `data.detail || "..."`,
so `detail` must remain a string in every case.
"""

import pytest
from httpx import AsyncClient

from app.utils.errors import code_for_status, error_body, summarize_validation_errors


def _assert_envelope(body: dict, status: int, code: str):
    assert "error" in body, f"no error object in {body}"
    error = body["error"]
    assert error["code"] == code
    assert error["status"] == status
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["hint"], str) and error["hint"]
    assert error["docs"].startswith("https://")
    # Legacy field the dashboard and SDK read directly.
    assert isinstance(body["detail"], str) and body["detail"]


@pytest.mark.asyncio
async def test_unknown_path_returns_envelope(client: AsyncClient):
    response = await client.get("/v1/definitely-not-a-real-path")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    _assert_envelope(response.json(), 404, "not_found")


@pytest.mark.asyncio
async def test_wrong_method_returns_envelope(client: AsyncClient):
    response = await client.post("/v1/health")
    assert response.status_code == 405
    _assert_envelope(response.json(), 405, "method_not_allowed")


@pytest.mark.asyncio
async def test_missing_credentials_returns_envelope(client: AsyncClient):
    # The shared client pre-sets an API key; drop it to hit the auth path.
    response = await client.get("/v1/events", headers={"Authorization": ""})
    assert response.status_code == 401
    _assert_envelope(response.json(), 401, "unauthorized")
    # The 401 challenge header survives the handler.
    assert response.headers.get("www-authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_validation_error_detail_is_a_string_not_a_list(client: AsyncClient):
    """422 used to return detail as a list, which clients rendered as [object Object]."""
    response = await client.post("/v1/auth/login", json={})
    assert response.status_code == 422
    body = response.json()
    _assert_envelope(body, 422, "validation_error")
    assert isinstance(body["detail"], str)
    # The structured form moves to error.fields.
    assert body["error"]["fields"], "validation fields missing"
    assert all({"field", "message", "type"} <= set(f) for f in body["error"]["fields"])


@pytest.mark.asyncio
async def test_successful_requests_are_untouched(client: AsyncClient):
    response = await client.get("/v1/health")
    assert response.status_code == 200
    assert "error" not in response.json()


def test_code_for_status_covers_the_common_statuses():
    assert code_for_status(404) == "not_found"
    assert code_for_status(429) == "rate_limited"
    # Anything unmapped still gets a usable code rather than a KeyError.
    assert code_for_status(418) == "error"


def test_error_body_shape():
    body = error_body(503, "Upstream asleep.", hint="Retry in 30s.")
    assert body["detail"] == "Upstream asleep."
    assert body["error"]["hint"] == "Retry in 30s."
    assert body["error"]["code"] == "service_unavailable"
    assert "fields" not in body["error"]


def test_summarize_validation_errors_joins_locations():
    summary, fields = summarize_validation_errors(
        [
            {"loc": ["body", "email"], "msg": "field required", "type": "missing"},
            {"loc": ["body", "password"], "msg": "too short", "type": "value_error"},
        ]
    )
    assert "body.email: field required" in summary
    assert "body.password: too short" in summary
    assert [f["field"] for f in fields] == ["body.email", "body.password"]


def test_summarize_validation_errors_truncates_long_lists():
    errors = [{"loc": ["body", f"f{i}"], "msg": "bad", "type": "x"} for i in range(8)]
    summary, fields = summarize_validation_errors(errors)
    assert "(+3 more)" in summary
    assert len(fields) == 8
