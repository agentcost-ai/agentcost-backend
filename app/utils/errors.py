"""Structured JSON error envelope.

Every error the API returns carries a machine-readable `error` object -- a
stable `code`, a human `message`, and a `hint` saying what to do next -- so an
agent can branch on the failure without parsing prose.

`detail` is kept alongside it, and is always a STRING. Existing clients
(dashboard AuthContext, SDK http_client) read `detail` directly, so dropping it
would break them. The one shape change is 422: FastAPI's default handler puts a
list of validation objects in `detail`, which those clients render as
"[object Object]"; here the list moves to `error.fields` and `detail` becomes
the readable summary.
"""

import logging
from typing import Any, Dict, List, Optional

from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

DOCS_URL = "https://agentcost.tech/docs/api"

# Stable code per status. Codes are part of the contract -- rename one and every
# agent branching on it breaks -- so they are spelled out rather than derived.
_CODE_BY_STATUS: Dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    410: "gone",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "service_unavailable",
}

_HINT_BY_STATUS: Dict[int, str] = {
    400: "Check the request parameters against the OpenAPI spec at /openapi.json.",
    401: (
        "Send an SDK API key as 'Authorization: Bearer sk_...', or a session token "
        "plus a project_id query parameter. The public pricing endpoints under "
        "/v1/pricing need no credentials at all."
    ),
    403: "The credential is valid but lacks permission for this project or resource.",
    404: (
        "Check the path against /openapi.json. The public catalogue lives at "
        "/v1/pricing and /v1/pricing/{model_name}."
    ),
    405: "Check the allowed methods for this path in /openapi.json.",
    409: "The resource already exists or is in a state that blocks this operation.",
    410: "This resource has been retired and will not come back.",
    413: "Split the payload into smaller batches and retry.",
    415: "Send 'Content-Type: application/json'.",
    422: "Fix the fields listed in error.fields and resend.",
    429: "Back off for the number of seconds in the Retry-After header, then retry.",
    500: "This is a bug on our side. Retry once; if it persists, report it at https://agentcost.tech/contact.",
    502: "An upstream dependency failed. Retry with backoff.",
    503: "The service is temporarily unavailable. Retry after the Retry-After interval.",
}

_DEFAULT_CODE = "error"
_DEFAULT_HINT = "See https://agentcost.tech/docs/api for the full API reference."


def code_for_status(status: int) -> str:
    """Stable machine-readable code for an HTTP status."""
    return _CODE_BY_STATUS.get(status, _DEFAULT_CODE)


def hint_for_status(status: int) -> str:
    return _HINT_BY_STATUS.get(status, _DEFAULT_HINT)


def error_body(
    status: int,
    message: str,
    *,
    code: Optional[str] = None,
    hint: Optional[str] = None,
    fields: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the response body for an error.

    Returns both the structured `error` object and the legacy string `detail`.
    """
    error: Dict[str, Any] = {
        "code": code or code_for_status(status),
        "message": message,
        "hint": hint or hint_for_status(status),
        "status": status,
        "docs": DOCS_URL,
    }
    if fields:
        error["fields"] = fields
    return {"error": error, "detail": message}


def summarize_validation_errors(errors: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Turn FastAPI's validation error list into (summary string, field list).

    `loc` is joined with dots and the leading "body"/"query"/"path" scope is kept,
    so a caller can see which part of the request to fix.
    """
    fields: List[Dict[str, Any]] = []
    parts: List[str] = []
    for err in errors:
        loc = ".".join(str(piece) for piece in err.get("loc", ()))
        msg = str(err.get("msg", "invalid value"))
        fields.append({"field": loc, "message": msg, "type": str(err.get("type", ""))})
        parts.append(f"{loc}: {msg}" if loc else msg)

    if not parts:
        return "Request validation failed.", fields
    summary = "Request validation failed - " + "; ".join(parts[:5])
    if len(parts) > 5:
        summary += f"; (+{len(parts) - 5} more)"
    return summary, fields


class ErrorEnvelopeMiddleware:
    """Turn an unhandled exception into the JSON envelope, inside the CORS layer.

    Starlette's own ServerErrorMiddleware is the OUTERMOST layer, so the 500s it
    produces are plain text and carry no Access-Control-Allow-Origin -- a browser
    sees an opaque network failure instead of a status. Registering this first
    makes it the innermost middleware, so its response still travels back out
    through CORSMiddleware.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_wrapper(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "Unhandled error on %s %s",
                scope.get("method", "?"),
                scope.get("path", "?"),
            )
            # Headers are already on the wire -- there is no envelope to send, and
            # swallowing it here would hide the failure from the server logs.
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content=error_body(500, "Internal server error."),
            )
            await response(scope, receive, send)
