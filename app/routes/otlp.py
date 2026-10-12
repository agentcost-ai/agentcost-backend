"""
AgentCost Backend - OTLP receiver

Lets Claude Code and Codex report usage through the OpenTelemetry export they
already have, with nothing installed on the machine: an organisation sets a
few environment variables in managed settings and every session reports in,
including ones that run in the cloud and leave no local log to collect.

    OTEL_EXPORTER_OTLP_ENDPOINT=https://api.agentcost.tech/v1/otlp
    OTEL_EXPORTER_OTLP_PROTOCOL=http/json
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer <project API key>

Only the per-call usage records in the logs signal are read. Metrics and
traces are accepted and discarded, so one endpoint setting covers all three
without the exporter logging errors.
"""

import json
import logging
import zlib

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_db
from ..models.db_models import Project
from ..models.schemas import EventBatchRequest
from ..services.otlp_service import events_from_logs
from ..utils.auth import validate_api_key
from .events import ingest_events_batch

router = APIRouter(prefix="/v1/otlp", tags=["OTLP"])

logger = logging.getLogger(__name__)


async def _json_body(request: Request) -> dict:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail=(
                "Only OTLP over HTTP with JSON is accepted. Set "
                "OTEL_EXPORTER_OTLP_PROTOCOL=http/json (Claude Code) or "
                'protocol = "json" (Codex).'
            ),
        )

    body = await request.body()
    if request.headers.get("content-encoding", "").lower() == "gzip":
        # Bounded: the size middleware only saw the compressed length.
        limit = get_settings().max_request_size_mb * 1024 * 1024
        inflater = zlib.decompressobj(wbits=31)
        try:
            body = inflater.decompress(body, limit + 1)
        except zlib.error as exc:
            raise HTTPException(status_code=400, detail="Body is not valid gzip.") from exc
        if len(body) > limit or inflater.unconsumed_tail:
            raise HTTPException(status_code=413, detail="Request body too large.")

    try:
        payload = json.loads(body or b"{}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Body is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")
    return payload


@router.post("/v1/logs")
async def receive_logs(
    request: Request,
    db: AsyncSession = Depends(get_db),
    project: Project = Depends(validate_api_key),
):
    """
    OTLP/HTTP logs from Claude Code or Codex.

    Each model call in the export becomes one event and goes through the same
    pricing, deduplication, budget and guardrail path as `/v1/events/batch`.
    Records that are not model calls are ignored. A project over its hard
    budget cap gets the same 429 it would on the batch endpoint.
    """
    events = events_from_logs(await _json_body(request))

    stored = 0
    size = get_settings().max_batch_size
    for start in range(0, len(events), size):
        batch = EventBatchRequest.model_validate(
            {"project_id": project.id, "events": events[start : start + size]}
        )
        stored += (await ingest_events_batch(batch, db=db, project=project)).events_stored

    if events:
        logger.debug("OTLP: %d call(s) for project %s, %d stored", len(events), project.id, stored)
    # An empty object is OTLP's "all accepted".
    return {}


@router.post("/v1/metrics", include_in_schema=False)
@router.post("/v1/traces", include_in_schema=False)
async def discard_signal(project: Project = Depends(validate_api_key)):
    """Accepted so a single endpoint setting works; nothing is read or kept."""
    return {}
