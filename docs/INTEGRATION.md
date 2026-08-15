# Integrating with AgentCost


How to get cost data in and out of AgentCost from a system that is not a Python
application. For how the platform works internally, see ARCHITECTURE.md

---

## Choosing a method

| Method | Use when | Gives you |
|---|---|---|
| [Direct HTTP](#1-direct-http) | Sidecars, proxies, gateways, any non-Python runtime | Full cost, model, latency and outcome analytics |
| [Python SDK](#2-python-sdk) | Instrumenting the agent process itself | The above, plus automatic run structure |
| [Run correlation](#3-correlating-with-an-external-control-plane) | Another system already owns a run identity | A joined view of one run across both systems |
| [Pulling cost signals](#4-consuming-cost-signals) | Acting on budget or optimization data | Budget state, webhooks, Prometheus, routing rules |
| [Local mode / CLI](#5-offline-and-pre-deployment) | Air-gapped evaluation, CI gating | Cost projection with no network calls |

---

## 1. Direct HTTP

One endpoint, no SDK, no language constraint.

```bash
curl -X POST https://api.agentcost.tech/v1/events/batch \
  -H "Authorization: Bearer $AGENTCOST_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "3f2b1c8e-...",
    "events": [{
      "agent_name":    "code-agent",
      "model":         "claude-sonnet-4",
      "input_tokens":  18240,
      "output_tokens": 612,
      "cached_tokens": 16000,
      "latency_ms":    2310,
      "timestamp":     "2026-08-13T09:12:44Z",
      "success":       true
    }]
  }'
```

Required per event: `model`, `input_tokens`, `output_tokens`, `timestamp`. Everything
else is optional. The server reprices against its own catalogue, so an accurate `cost`
is not required from you — only accurate token counts.

**Batching.** Up to 100 events per request, 10 MB body, 100 requests/minute per key. Send
whatever cadence suits you; there is no ordering requirement.

**Partial validation.** Malformed events are dropped individually and echoed back in
`rejected[]`; the batch still returns 200. One bad event never blocks your queue.

**Idempotency.** Set `event_id` (unique per project) if your delivery can retry. A replay
returns 200 with `events_duplicate` incremented and stores nothing.

```json
{"status":"ok","events_stored":0,"events_duplicate":1,"outcomes_recorded":0}
```

### Cached tokens

If your runtime sees the raw provider response, forward the cache counts — they change
the cost materially on agentic coding workloads, which are heavily cached.

| Provider | Read from | Send as |
|---|---|---|
| OpenAI | `usage.prompt_tokens_details.cached_tokens` | `cached_tokens` (already a subset of `input_tokens`) |
| Anthropic | `usage.cache_read_input_tokens` | `cached_tokens`, **and add it to** `input_tokens` — Anthropic's `input_tokens` excludes it |
| Anthropic | `usage.cache_creation_input_tokens` | `cache_write_tokens` (billed at a premium, kept separate) |

What that earns is readable back as a number — hit rate and dollar savings against the
full-input-rate baseline, priced per model:

```bash
curl -H "Authorization: Bearer $KEY" \
  "https://api.agentcost.tech/v1/analytics/cache?range=30d"
```

```json
{"cache_hit_rate": 71.4, "read_savings": 212.40, "write_premium": 18.05, "net_savings": 194.35}
```

### Grouping dimensions

`user_id` and `session_id` in metadata are promoted to indexed columns and become
groupable:

```json
"metadata": { "user_id": "alice@example.com", "session_id": "run-7f3a" }
```

```bash
curl -H "Authorization: Bearer $KEY" \
  "https://api.agentcost.tech/v1/analytics/by/user?range=30d"
```

This is what answers *what is each developer costing us*. Without it, cost is only
attributable to agents and models.

### Capability hints

If you can see the request, forward what it needed. This is what lets the optimizer
recommend a cheaper model without breaking the workload:

```json
"metadata": { "_ac_caps": { "vision": true, "tools": true, "tool_count": 3 } }
```

Booleans and counts only. When absent, requirements are treated as *unknown* and the
optimizer assumes they are required — safe, but it withholds the larger savings.

---

## 2. Python SDK

```bash
pip install agentcost
```

```python
from agentcost import track_costs

track_costs.init(api_key="sk_...", project_id="<project uuid>")
```

Existing OpenAI, Anthropic, Gemini and LangChain calls are tracked with no further
changes. The SDK adds run structure that HTTP emitters have to supply themselves:

```python
with track_costs.workflow("refactor-run"):
    with track_costs.step("plan"):
        llm.invoke(...)
    with track_costs.tool("edit_file"):
        llm.invoke(...)
    track_costs.outcome(success=False, label="denied")
```

---

## 3. Correlating with an external control plane

For a system that already owns a run identity — a policy layer, an orchestrator, a CI
job — and wants its records joined to AgentCost's cost data.

### The join is at the run, not the event

The two event streams are different shapes. A control plane typically records one entry
per action (hundreds to thousands per run); AgentCost records one per inference (tens).
Joining event-to-event is not meaningful. **The run id is the only shared contract.**

Mint the id upstream — whatever starts the run should own it. Do not mirror a decision
stream into AgentCost; send only what carries tokens.

### Step 1 — propagate the run id

If you wrap the agent process, export one variable and every event inherits it:

```bash
AGENTCOST_TRACE_ID=0532f9c4-a022-4e98-a543-d8e17c5b90a6
AGENTCOST_WORKFLOW=refactor-run   # optional
```

That is the entire integration for the common case: the SDK the agent already runs
picks both up from the environment — with or without `workflow()` in the code, and an
active `workflow()` always wins. No code change in the agent.

Otherwise set `trace_id` explicitly, in-process or on each posted event:

```python
with track_costs.workflow("refactor-run", trace_id=external_run_id):
    ...
```

Trace ids accept up to 64 characters — UUIDs and ULIDs fit. A longer id is rejected
rather than truncated, since truncating would silently merge two distinct runs.

### Step 2 — report the outcome

When a run ends for a reason AgentCost cannot see — a denial, a timeout, an abort — post
the outcome. **No events are required**:

```bash
curl -X POST https://api.agentcost.tech/v1/events/batch \
  -H "Authorization: Bearer $AGENTCOST_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "3f2b1c8e-...",
    "events": [],
    "outcomes": [{
      "trace_id": "0532f9c4-a022-4e98-a543-d8e17c5b90a6",
      "workflow": "refactor-run",
      "success":  false,
      "label":    "denied:postgres.query"
    }]
  }'
```

Outcomes upsert on `trace_id`; a later write replaces an earlier optimistic one. This is
what makes cost-per-success and spend-on-failed-runs meaningful.

### Step 3 — read the joined run

```bash
curl -H "Authorization: Bearer $KEY" \
  https://api.agentcost.tech/v1/analytics/traces/0532f9c4-a022-4e98-a543-d8e17c5b90a6
```

Returns every call in the run, its position in the tree, its cost, and the outcome.

### Ordering caveat

Two systems timestamping independently have no shared clock. AgentCost's `step_index`
orders siblings within its own trace; it says nothing about where an external record
falls between two steps. Present a joined timeline as approximate, or stamp the current
step name on your own records when you write them.

---

## 4. Consuming cost signals

### Budget state, for polling

```bash
curl -H "Authorization: Bearer $KEY" \
  https://api.agentcost.tech/v1/projects/$PROJECT_ID/budget-state
```

```json
{
  "enabled": true, "mode": "hard_cap", "currency": "USD",
  "budget": 5000.0, "spend_mtd": 4820.15, "remaining": 179.85,
  "utilization_percent": 96.4, "exhausted": false,
  "period_ends_at": "2026-09-01T00:00:00+00:00",
  "as_of": "2026-08-13T09:14:02+00:00"
}
```

Side-effect free and cheap. Poll every 15–60s and hold the result as cached state.

> **Do not call this inside a latency-sensitive decision path.** If you need to gate
> actions on remaining budget in sub-millisecond time, poll and cache. `as_of` is
> returned so you can reason about how stale your copy is, and `period_ends_at` so
> "nearly out with three weeks left" is distinguishable from "nearly out on the last day".

### Webhooks, for reacting

```bash
curl -X PUT https://api.agentcost.tech/v1/projects/$PROJECT_ID/webhook \
  -H "Authorization: Bearer $USER_JWT" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-endpoint.example/agentcost", "secret": "<shared secret>"}'
```

Requires project-edit permission (not merely view) and an HTTPS URL. The secret is
write-only — `GET` on the same path returns the URL and whether a secret is set, never
the secret. `{"url": null}` disables the hook and clears the secret; rotating a secret
restates the URL (a `secret` without a `url` is rejected).

Verify the wiring immediately with a test delivery — same payload shape and signature
as the real thing, event type `webhook.test`:

```bash
curl -X POST https://api.agentcost.tech/v1/projects/$PROJECT_ID/webhook/test \
  -H "Authorization: Bearer $USER_JWT"
```

Threshold crossings are POSTed as
they happen, signed with `HMAC-SHA256(secret, "{timestamp}.{body}")` in
`X-AgentCost-Signature`. Verify the timestamp is within your tolerance window *before*
comparing the digest.

Delivery rules: only a 2xx counts as delivered — redirects are not followed. The
destination host is resolved at send time and non-public addresses are refused
(self-hosted installs posting to internal listeners set
`WEBHOOK_ALLOW_PRIVATE_URLS=true`). Best-effort, not retried — a slow endpoint must
never delay ingestion; poll `budget-state` as the reliable channel.

### Prometheus

```yaml
scrape_configs:
  - job_name: agentcost
    metrics_path: /v1/metrics
    authorization:
      credentials: <project_api_key>
    static_configs:
      - targets: ['api.agentcost.tech']
```

Windowed gauges, not monotonic counters — use `max_over_time` rather than `rate()`.

### Optimization recommendations as rules

`GET /v1/optimizations` returns model-downgrade suggestions already shaped as rules —
current model, proposed model, projected saving, and the capability check behind it:

```json
{
  "type": "model_downgrade",
  "agent_name": "code-agent",
  "model": "<current>",
  "alternative_model": "<proposed>",
  "estimated_savings_monthly": 412.80,
  "metrics": {
    "capability_requirements": { "requires_vision": "false", "requires_function_calling": "true" },
    "capabilities_verified": true,
    "confidence_score": 0.82
  }
}
```

A system that routes model traffic can consume these directly. **Gate on
`capabilities_verified`** — `false` means the workload's requirements could not be
observed and the candidate was filtered against an assumed superset rather than a
measured one. Applying an unverified suggestion automatically risks routing a workload
to a model that cannot serve it.

---

## 5. Offline and pre-deployment

**Local mode** — events held in memory, no network calls at all. For air-gapped
evaluation and tests:

```python
track_costs.init(local_mode=True)
events = track_costs.get_local_events()
```

**CI gating** — project cost before deploying and fail the build on a regression:

```bash
agentcost analyze ./agent --model gpt-4o --runs-per-day 2000 --fail-on high
```

Runs entirely locally and transmits nothing.

**Air-gapped catalogue** — the sync fetches LiteLLM pricing from GitHub. Where there is
no egress, carry the bundle in:

```bash
curl -X POST https://agentcost.internal/v1/pricing/import \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  --data-binary @model_prices_and_context_window.json
```

Same parsing, sanity bounds and change tracking as the network sync.

---

## Errors

| Status | Meaning | Action |
|---|---|---|
| `200` with `events_rejected > 0` | Some events failed validation; the rest stored | Inspect `rejected[]` — it names the field and reason |
| `200` with `events_duplicate > 0` | Replay of an already-stored `event_id` | None; this is a successful no-op |
| `401` | API key rejected | Check the key |
| `403` | Key not valid for this `project_id` | `project_id` must be the project **UUID**, not its name |
| `422` | Batch malformed, or empty of both events and outcomes | Check the payload shape |
| `429` | Rate limited, or the project's budget hard cap was reached | Back off; if persistent, check budget settings |
| `5xx` | Transient | Retry with backoff; use `event_id` to stay idempotent |

---

## Support

- SDK reference: [agentcost.tech/docs/sdk](https://agentcost.tech/docs/sdk)
- Issues: [github.com/agentcost-ai/agentcost-backend/issues](https://github.com/agentcost-ai/agentcost-backend/issues)
