"""
AgentCost Backend - Outbound Webhooks

Push egress for consumers that cannot poll, e.g. an enforcement point that
holds cached budget state and needs to know when to refresh it.

Delivery is best-effort by design: a slow or down endpoint must never delay
event ingestion or roll back the work that triggered it, so failures are
logged and dropped rather than retried into the request that produced them.
"""

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

# Short: a hanging webhook must not hold resources open behind ingest.
WEBHOOK_TIMEOUT_SECONDS = 5.0

SIGNATURE_HEADER = "X-AgentCost-Signature"
TIMESTAMP_HEADER = "X-AgentCost-Timestamp"
EVENT_HEADER = "X-AgentCost-Event"

# Strong references to in-flight deliveries. The event loop only keeps weak
# references to tasks, so without this a delivery can be garbage-collected
# mid-flight and silently never sent.
_inflight: set = set()


@dataclass
class DeliveryResult:
    """Outcome of one delivery attempt, safe to return to a project editor."""

    delivered: bool
    status_code: Optional[int] = None
    error: Optional[str] = None


def sign_payload(secret: str, timestamp: str, body: str) -> str:
    """HMAC-SHA256 over ``timestamp.body``, hex encoded.

    The timestamp is inside the signed string so a captured delivery cannot be
    replayed later with a fresh header. Receivers should reject a stale
    timestamp before comparing the digest.
    """
    message = f"{timestamp}.{body}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


async def _destination_blocked(url: str) -> Optional[str]:
    """Reason this URL must not be delivered to, or None if it is allowed.

    Resolves the host at delivery time and refuses private, loopback,
    link-local and otherwise non-public addresses, so a webhook URL cannot be
    used to probe the network the backend runs in. ``webhook_allow_private_urls``
    disables the check for local development and self-hosted installs.
    """
    if get_settings().webhook_allow_private_urls:
        return None

    host = urlsplit(url).hostname
    if not host:
        return "URL has no host."

    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError:
        return f"Host {host!r} does not resolve."

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            return f"Host {host!r} resolves to non-public address {address}."
    return None


async def deliver(
    url: str,
    event_type: str,
    payload: dict[str, Any],
    *,
    secret: Optional[str] = None,
) -> DeliveryResult:
    """POST one event to a webhook URL.

    Never raises: every caller is on a path where the useful work is already
    done and an exception would undo it.
    """
    try:
        blocked = await _destination_blocked(url)
    except Exception as exc:  # noqa: BLE001 - never raise past this point
        blocked = f"Destination check failed: {exc}"
    if blocked:
        logger.warning("Webhook to %s refused: %s", url, blocked)
        return DeliveryResult(delivered=False, error=blocked)

    body = json.dumps(
        {
            "event": event_type,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        },
        separators=(",", ":"),
        default=str,
    )

    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AgentCost-Webhook/1",
        EVENT_HEADER: event_type,
        TIMESTAMP_HEADER: timestamp,
    }
    if secret:
        headers[SIGNATURE_HEADER] = sign_payload(secret, timestamp, body)

    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
            response = await client.post(url, content=body, headers=headers)
        # Redirects are not followed, so only a 2xx proves the payload landed.
        if not (200 <= response.status_code < 300):
            logger.warning(
                "Webhook %s returned HTTP %s for event %s",
                url,
                response.status_code,
                event_type,
            )
            return DeliveryResult(
                delivered=False,
                status_code=response.status_code,
                error=f"Endpoint returned HTTP {response.status_code}.",
            )
        return DeliveryResult(delivered=True, status_code=response.status_code)
    except Exception as exc:  # noqa: BLE001 - see module docstring
        logger.warning("Webhook delivery to %s failed for %s: %s", url, event_type, exc)
        return DeliveryResult(delivered=False, error=str(exc))


def dispatch(
    url: Optional[str],
    event_type: str,
    payload: dict[str, Any],
    *,
    secret: Optional[str] = None,
) -> None:
    """Schedule a delivery without waiting for it.

    Detached from the caller's task so ingest latency never includes a
    third-party HTTP round trip.
    """
    if not url:
        return
    try:
        task = asyncio.create_task(deliver(url, event_type, payload, secret=secret))
    except RuntimeError:
        # No running loop (sync context or shutdown). Dropping is correct:
        # the alternative is blocking a request thread.
        logger.debug("No event loop available to dispatch webhook %s", event_type)
        return
    _inflight.add(task)
    task.add_done_callback(_inflight.discard)
