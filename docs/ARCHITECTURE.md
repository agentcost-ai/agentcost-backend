# AgentCost Observability Architecture


## 1. Scope

AgentCost answers one question: **what did this run cost, and which part of it was
expensive.** It instruments LLM calls after they complete, prices them against a synced
model catalogue, and attributes each one to a position in the workflow that produced it.

It is not an inference proxy. It does not sit in the request path, does not modify
requests or responses, and cannot block a call. Instrumentation is passive and
out-of-band: if AgentCost is unreachable the host application is unaffected, and events
are queued, retried, then dropped with a warning.

**Stack:** Python 3.9+ SDK · FastAPI + SQLAlchemy backend · PostgreSQL or SQLite ·
MIT licensed · self-hostable.

---

## 2. Data path

```
 in-process                                        │ network │  server
 ─────────────────────────────────────────────────────────────────────────────────
 ① Interceptor  →  ② Trace funnel  →  ③ Batcher  →  ④ POST  →  ⑤ Reprice  →  ⑥ Persist
   wraps the        stamps trace       size- or     bearer     authoritative   + baselines
   provider         position from      time-        auth,      cost from the   + budget
   client           contextvars        triggered    per-event  catalogue       evaluation
                                                    validation
```

Stages ① – ③ run inside the host process and never block it. Stage ④ is the only
network boundary and the only integration surface a non-Python client needs.

---

## 3. Collection

The SDK patches the provider clients it finds installed at `init()`. Each interceptor
wraps the call, lets it run untouched, and reads token usage off the returned object.
**The provider's own reported usage is authoritative**; `tiktoken` is a fallback only
when the SDK returns none. Streaming responses are wrapped so usage is captured when the
stream is consumed, not when it is created.

| Surface | Covers | Streaming |
|---|---|---|
| `openai` | `chat.completions` and the Responses API, sync + async | Yes |
| `anthropic` | `messages`, sync + async, plus the `stream()` helper | Yes |
| `google-genai` | `generate_content`, via `usage_metadata` | Yes |
| `langchain-core` | Callback-based; covers LangGraph and CrewAI when routed through LangChain | Yes |

### Prompt-cache accounting

Cached input is a **subset** of the prompt billed at a lower rate. The two major
providers report it differently and the SDK normalises both to one convention —
`input_tokens` is the whole prompt, `cached_tokens` is the part read from cache:

| Provider | Reports | Normalisation |
|---|---|---|
| OpenAI | `usage.prompt_tokens_details.cached_tokens`, already a subset of `prompt_tokens` | Passed through |
| Anthropic | `cache_read_input_tokens` **alongside** `input_tokens`, which holds only the uncached remainder | Summed into `input_tokens`; read count kept separately |

Anthropic cache **writes** are tracked as `cache_write_tokens` and priced separately —
they are billed at a premium over standard input, not a discount, so folding them in
with cache reads would have the wrong sign.

### Capability fingerprint

Each call records what capabilities it exercised, so the optimizer can tell whether a
cheaper model would still work:

```json
{"vision": true, "tools": true, "tool_count": 3, "structured_output": true}
```

Booleans and counts only — never tool definitions, prompt text or image data. Written
under the reserved metadata key `_ac_caps` so caller metadata cannot collide with it.
Omitted entirely when a call used none of these, so the common case adds no bytes.

---

## 4. What leaves the process

The part most relevant to a security review. **Prompt and completion text never leave
the process.** Input text is extracted only to compute a SHA-256 hash of its normalised
form, used to detect repeated identical calls; the source text is discarded.

| Field | Type | Notes | Sent |
|---|---|---|---|
| `agent_name` | string | Caller-assigned label | Always |
| `model` | string | As reported by the provider | Always |
| `input_tokens`, `output_tokens` | int | Provider-reported | Always |
| `cached_tokens`, `cache_write_tokens` | int | Prompt-cache accounting | When non-zero |
| `cost` | float | Client estimate; server recomputes and overrides | Always |
| `latency_ms` | int | Wall clock around the call | Always |
| `success`, `error` | bool, string | Exception type and message on failure | Always |
| `streaming` | bool | Whether the response was streamed | When true |
| `input_hash` | sha256 | Of normalised input text. One-way. | Always |
| `metadata` | object | Only what the caller supplies, plus `_ac_caps` | Caller-controlled |
| trace fields | string, int | See [§8](#8-trace-model) | Inside a `workflow()` |
| **prompt text** | — | Hashed in-process, never transmitted | **Never** |
| **completion text** | — | Accumulated only to count stream tokens, then discarded | **Never** |
| **API keys, headers** | — | Not read, not stored | **Never** |

---

## 5. Delivery semantics

Events go to a hybrid batcher that flushes on whichever comes first: **batch size**
(default 10; server caps a request at 100) or **flush interval** (default 5s). Each
flush is a POST on its own daemon thread, so the calling thread is never blocked.

| Property | Behaviour |
|---|---|
| Delivery guarantee | At-most-once. Buffered in memory; a hard process kill loses the buffer. |
| Transport retry | 3 attempts, backoff 1s / 2s / 4s, on 429 / 500 / 502 / 503 / 504 |
| Failed-batch queue | Up to 100 batches held in memory, retried each flush interval |
| Client rate limit | 10 requests/second, enforced SDK-side before the POST |
| Shutdown | `atexit` flush with a 5s grace budget, then a `RuntimeWarning` naming the dropped count |
| Failure isolation | Every capture path is guarded; an instrumentation fault never propagates into the caller's LLM call |
| Backend unreachable | Warns once via `logging` and `warnings`, then keeps retrying quietly |

> **Design note.** Telemetry is never written to disk. Losing a batch is treated as
> preferable to adding a local write path and its failure modes. Where cost data is a
> compliance artefact rather than an operational signal, that trade-off is wrong and a
> durable spool is needed — see [§13](#13-known-limits).

---

## 6. Ingest contract

One endpoint. Nothing about it assumes the Python SDK.

```http
POST /v1/events/batch
Authorization: Bearer <project_api_key>
Content-Type: application/json
```

```json
{
  "project_id": "3f2b1c8e-...",
  "events": [
    {
      "agent_name":    "code-agent",
      "model":         "claude-sonnet-4",
      "input_tokens":  18240,
      "output_tokens": 612,
      "cached_tokens": 16000,
      "latency_ms":    2310,
      "timestamp":     "2026-08-13T09:12:44Z",
      "success":       true,
      "event_id":      "delivery-7f3a-01",
      "input_hash":    "<sha256 hex>",
      "metadata":      { "user_id": "alice", "repo": "billing-svc" },
      "trace_id":      "0532f9c4-a022-4e98-a543-d8e17c5b90a6",
      "workflow":      "refactor-run",
      "step_name":     "generate-patch",
      "step_index":    3,
      "tool_name":     "edit_file"
    }
  ],
  "outcomes": [
    { "trace_id": "0532f9c4-...", "success": false, "label": "denied:postgres.query" }
  ]
}
```

Required per event: `model`, `input_tokens`, `output_tokens`, `timestamp` (ISO 8601).
Everything else is optional.

**Response**

```json
{
  "status": "ok",
  "events_stored": 1,
  "events_received": 1,
  "events_rejected": 0,
  "events_duplicate": 0,
  "outcomes_recorded": 1,
  "rejected": [],
  "timestamp": "2026-08-13T09:12:45Z"
}
```

| Property | Value |
|---|---|
| Auth | Project API key, SHA-256 hashed at rest. Key must match `project_id` or the request 403s. |
| Max events per request | 100 (configurable; 1000 hard ceiling) |
| Max request body | 10 MB |
| Server rate limit | 100 req/min per key by default; separate buckets per project |
| Partial validation | Malformed events are dropped **individually** and echoed in `rejected[]`; the batch still returns 200. One bad event never blocks a queue. |
| Idempotency | Optional `event_id`, unique per project. A replay returns 200 with `events_duplicate` incremented and stores nothing. |
| Outcome-only batches | `events` may be empty or omitted when `outcomes` is present. Outcomes upsert on `trace_id`; last write wins. |
| Backpressure | 429 when a project's monthly budget hard cap is reached |

---

## 7. Server-side processing

Ingest is split into *price it* and *write it*, so budget evaluation runs against the
cost the server will actually store rather than the figure a client claimed.

### Repricing

Pricing for every distinct model in the batch is fetched once, not per event. Cost is
recomputed server-side and the result overrides whatever the client sent:

```
cost = (input_tokens − cached_tokens)/1000 × input_rate
     + cached_tokens/1000            × cached_rate
     + cache_write_tokens/1000       × cache_write_rate
     + output_tokens/1000            × output_rate
```

Where a provider publishes no cache rate, cached tokens fall back to the **standard
input rate**. A discount is never assumed — the distinction between "no published rate"
and "zero" is preserved as `NULL` rather than `0.0` throughout.

Each row records how it was priced, in `cost_source`:

- `database-exact` — matched the catalogue by name
- `database-fuzzy` — matched by longest-substring, so a versioned or provider-prefixed
  name still prices. Longest match wins specifically so `gpt-4` cannot capture
  `gpt-4o-mini`.
- `client-sdk` — no catalogue entry; the client's estimate was kept

### Model catalogue

Synced from LiteLLM's pricing dataset on a 24-hour cron, with OpenRouter as a secondary
source and a manual import path for disconnected deployments. Per model: input, output,
cache-read and cache-write price per 1K; provider; context limits; and capability flags
`supports_vision`, `supports_function_calling`, `supports_streaming`. Sync runs are
logged with a per-model diff.

### Also at ingest

- **Dimension promotion** — `user_id` and `session_id` are copied out of metadata into
  indexed columns so analytics can group by them
- **Repeat-pattern folding** — `(agent_name, input_hash)` occurrences and summed cost,
  accumulated in one pass, feeding cache-opportunity detection
- **Budget evaluation** — month-to-date spend against budget, threshold crossings
  deduplicated per period, fanned out to webhook, in-app and email. Skipped entirely for
  projects with no budget set.
- **Timestamp normalisation** — naive timestamps pinned to UTC, future-dated ones clamped
  to now, so a clock-skewed client cannot write rows invisible to every time-bounded query

### Derived analysis, computed on read

| Layer | Produces |
|---|---|
| **Baselines** | Per `(agent, model)` mean and standard deviation of cost/call, tokens, latency, daily volume, error rate. Anomaly thresholds derived from the project's own distribution, not hardcoded. |
| **Optimizer** | Six suggestion classes: model downgrade, caching, prompt, batching, error reduction, latency. Plus non-LLM candidates — classifier-shaped workloads that need no model, detected on *maximum* output tokens rather than average, since an average hides the tenth of calls that write prose. |
| **Alternative learning** | Candidate swaps ranked by price similarity, re-ranked by whether users implemented or dismissed them. Each carries `source` (learned vs dynamic) and a confidence score. |
| **Trace analytics** | Per workflow, step and tool cost; repeated-work detection within a run; run cost distribution; outcome stats including cost-per-success and spend-on-failures. |
| **Executive report** | KPIs with period deltas, latency percentiles, Pareto cost concentration, token efficiency, error breakdown, run-rate projection, budget status. |

### Downgrade safety

A model-downgrade suggestion is only produced when the candidate is a capability
superset of the current model. Requirements come from the `_ac_caps` fingerprint;
**unknown requirements are treated as required**, which narrows candidates rather than
widening them. Each suggestion reports `capabilities_verified` so a consumer applying
them automatically can gate on measured rather than assumed requirements.

---

## 8. Trace model

A **trace** is one run of a workflow; **spans** nest inside it. Every event carries its
own span id and its parent's, so the server rebuilds the tree without the client ever
sending one.

```python
with track_costs.workflow("refactor-run") as trace_id:
    with track_costs.step("plan"):
        llm.invoke(...)                  # step_index 0, depth 0
    with track_costs.tool("edit_file"):
        with track_costs.step("generate-patch"):
            llm.invoke(...)              # depth 1, tool_name=edit_file
    track_costs.outcome(False, label="denied")
```

Implementation notes that matter for a foreign integration:

- Built on `contextvars`, so concurrent tasks and threads each keep their own trace with
  nothing threaded through call signatures.
- A nested `workflow()` joins the enclosing trace rather than starting a new one — the
  cost question is always about the outermost run.
- `step_index` is allocated from the trace, not the span, so it stays comparable across
  sibling branches that ran concurrently. Timestamps cannot order those.
- Outside a `workflow()`, `step()` and `tool()` are no-ops. Instrumenting a helper never
  depends on how it is called.
- `outcome()` is sent once when the workflow closes, so a run that later fails overwrites
  an earlier optimistic result.
- Untraced events are excluded from workflow queries rather than folded in, which would
  overstate a workflow's totals. They remain visible in agent and model analytics.

### Joining a run minted elsewhere

Three ways to supply a foreign run id, in order of least effort:

**1. Environment variable.** A process that wraps the agent exports one variable and
every event inherits it, with no code change in the agent:

```bash
export AGENTCOST_TRACE_ID=0532f9c4-a022-4e98-a543-d8e17c5b90a6
```

**2. Explicit argument.** `track_costs.workflow("name", trace_id=...)` — always wins over
the environment.

**3. Over HTTP.** Set `trace_id` on each event posted to `/v1/events/batch`.

Trace ids accept up to 64 characters, so canonical UUIDs and ULIDs fit. A longer id is
rejected rather than truncated — truncating would silently merge two distinct runs.

---

## 9. Storage

One row per LLM call in `events`. Indexes:

```
(project_id, timestamp)              (project_id, trace_id)
(project_id, agent_name, timestamp)  (project_id, workflow, timestamp)
(project_id, model, timestamp)       (project_id, user_id, timestamp)
(project_id, input_hash)             (project_id, session_id, timestamp)
(project_id, event_id)
```

Trace fields are nullable throughout — an untraced call is a legitimate row, not a broken
one. Run outcomes live in `trace_outcomes`, unique on `(project_id, trace_id)`.

---

## 10. Query surface

Read-only unless noted. Same project API key as ingest.

### Analytics

| Endpoint | Returns |
|---|---|
| `GET /v1/analytics/overview` | Totals, deltas, headline KPIs |
| `GET /v1/analytics/agents` | Per-agent cost, calls, tokens, latency, error rate |
| `GET /v1/analytics/models` | Same, per model |
| `GET /v1/analytics/by/{dimension}` | Grouped by `user`, `session`, `workflow`, `tool`, `model` or `agent` |
| `GET /v1/analytics/cache` | Prompt-cache hit rate and savings, priced per model against the full-input-rate baseline |
| `GET /v1/analytics/timeseries` | Bucketed cost and volume |
| `GET /v1/analytics/report` | Assembled executive report |

### Workflow and trace

| Endpoint | Returns |
|---|---|
| `GET /v1/analytics/workflows` | Cost per run, calls per run, success rate, max/avg spread |
| `GET /v1/analytics/workflows/steps` | Per-step cost within workflows |
| `GET /v1/analytics/workflows/tools` | Cost attributable to each tool |
| `GET /v1/analytics/workflows/repeated-work` | Duplicate calls within a run and their avoidable cost |
| `GET /v1/analytics/workflows/outcomes` | Cost per success, spend on failed runs |
| `GET /v1/analytics/workflows/distribution` | Run cost distribution |
| `GET /v1/analytics/traces` | Trace list |
| `GET /v1/analytics/traces/{trace_id}` | Full tree for one run: every call, span position, cost, outcome |

### Optimization

| Endpoint | Returns |
|---|---|
| `GET /v1/optimizations` | Suggestions with savings estimates and action items |
| `GET /v1/optimizations/baselines` | Statistical baselines per agent and model |
| `GET /v1/optimizations/recommendations` | Persisted pending recommendations |
| `POST /v1/optimizations/recommendations/{id}/implement` | Records adoption; feeds the learning loop |

### Pricing and budget

| Endpoint | Returns |
|---|---|
| `GET /v1/pricing` | Full model catalogue with prices and capability flags |
| `POST /v1/pricing/sync/litellm` | Force a catalogue sync |
| `POST /v1/pricing/import` | Load a catalogue from an uploaded bundle (air-gapped deployments) |
| `GET /v1/projects/{id}/budget-state` | Compact budget position for polling — see [§11](#11-egress) |
| `GET` / `PUT /v1/projects/{id}/webhook` | Read or configure signed push egress |
| `GET /v1/metrics` | Prometheus exposition — see [§11](#11-egress) |
| `POST /v1/integrations/openai/costs` | Retroactive import from OpenAI org billing. Admin key used once, never persisted. |
| `POST /v1/integrations/anthropic/costs` | Same for Anthropic's cost report |

### Recommendation payload

Relevant if a downstream system acts on a suggestion rather than displaying it. A
model-downgrade recommendation is already structured as a rule:

```json
{
  "type": "model_downgrade",
  "agent_name": "code-agent",
  "model": "<current>",
  "alternative_model": "<proposed>",
  "estimated_savings_monthly": 412.80,
  "estimated_savings_percent": 38.4,
  "priority": "high",
  "metrics": {
    "current_calls": 8140,
    "quality_impact": "<tier>",
    "capability_requirements": {
      "requires_vision":           "true | false | unknown",
      "requires_function_calling": "true | false | unknown",
      "requires_json_mode":        "true | false | unknown"
    },
    "capabilities_verified": true,
    "source": "learned | dynamic",
    "confidence_score": 0.0
  }
}
```

---

## 11. Egress

### Budget state (pull)

```http
GET /v1/projects/{id}/budget-state
Authorization: Bearer <project_api_key>
```

```json
{
  "enabled": true, "mode": "hard_cap", "currency": "USD",
  "budget": 5000.0, "spend_mtd": 4820.15, "remaining": 179.85,
  "utilization_percent": 96.4, "thresholds_crossed": [50, 75, 90],
  "exhausted": false,
  "period_key": "2026-08", "period_ends_at": "2026-09-01T00:00:00+00:00",
  "as_of": "2026-08-13T09:14:02+00:00"
}
```

Designed for polling every 15–60s and holding the result as cached state. Side-effect
free — it never records threshold alerts and never counts an in-flight cost. `as_of` is
returned so a consumer can reason about staleness.

> Do not call this inside a latency-sensitive decision path. A consumer that must gate
> on budget in sub-millisecond time should poll and cache, not query inline.

### Webhooks (push)

Configure with `PUT /v1/projects/{id}/webhook` (requires project-edit permission — a
webhook URL is an exfiltration path for spend data, so view access is not enough):

```json
{ "url": "https://your-endpoint.example/agentcost", "secret": "<shared secret>" }
```

`GET /v1/projects/{id}/webhook` returns the configured URL and whether a secret is set;
the secret itself is never returned by any endpoint. Send `{"url": null}` to disable,
which also clears the stored secret. HTTPS is required (localhost excepted, for local
development).

Threshold crossings are then POSTed as they happen.

```json
{
  "event": "budget.threshold_crossed",
  "sent_at": "2026-08-13T09:14:02+00:00",
  "data": {
    "project_id": "3f2b1c8e-...", "period_key": "2026-08",
    "thresholds_crossed": [90], "spend_mtd": 4820.15,
    "budget": 5000.0, "utilization_percent": 96.4,
    "currency": "USD", "mode": "hard_cap"
  }
}
```

| Header | Meaning |
|---|---|
| `X-AgentCost-Event` | Event type |
| `X-AgentCost-Timestamp` | Unix seconds |
| `X-AgentCost-Signature` | `HMAC-SHA256(secret, "{timestamp}.{body}")`, hex |

The timestamp is inside the signed string, so a captured delivery cannot be replayed
with a fresh header. Reject a timestamp outside your tolerance window before comparing
the digest. Delivery is best-effort and not retried — a slow endpoint must never delay
event ingestion.

### Prometheus

```http
GET /v1/metrics
Authorization: Bearer <project_api_key>
```

```yaml
scrape_configs:
  - job_name: agentcost
    metrics_path: /v1/metrics
    authorization:
      credentials: <project_api_key>
    static_configs:
      - targets: ['api.agentcost.tech']
```

Series: `agentcost_calls`, `agentcost_cost_usd`, `agentcost_tokens`,
`agentcost_cached_tokens`, `agentcost_latency_ms_avg`, `agentcost_errors`,
`agentcost_cost_usd_by_model`, `agentcost_cost_usd_by_agent`, plus
`agentcost_budget_utilization_percent` and `agentcost_budget_remaining` when a budget is
set. None carry the `_total` suffix, which Prometheus reserves for monotonic counters.

Values are **windowed gauges**, not monotonic counters — `window_hours` sets the lookback
(default 24). Use `max_over_time` style queries rather than `rate()`. Per-model and
per-agent series are capped at the 50 costliest to bound cardinality.

---

## 12. Deployment

| Mode | Notes |
|---|---|
| Hosted | `https://api.agentcost.tech` |
| Self-hosted | Docker Compose, MIT licensed. Set `SECRET_KEY`, `DATABASE_URL` (PostgreSQL), `CORS_ORIGINS`. |
| Air-gapped | Supported. The catalogue sync reaches GitHub for LiteLLM pricing; use `POST /v1/pricing/import` with a bundle carried in instead. |

**Multi-worker:** set `RATE_LIMIT_BACKEND=redis` and `REDIS_URL`. The default in-memory
limiter keeps counters per process, so N workers give each client an effective N× limit.

---

## 13. Known limits

Stated plainly, because they are easier to design around than to discover.

| Limit | Detail |
|---|---|
| **Python-only tracer** | The ingest endpoint is language-agnostic, but automatic trace structure exists only in the Python SDK. A foreign emitter gets cost without run shape unless it populates the span fields itself. |
| **At-most-once delivery** | Telemetry buffers in memory only. A hard process kill loses it. No durable spool yet. |
| **Analytics read raw events** | A daily rollup table exists in the schema but is not yet populated; queries scan the events table. Fine into the low millions of rows, not beyond. |
| **Budget cap blocks ingest, not spend** | Hitting the hard cap returns 429 on *telemetry*. AgentCost is not in the inference path and cannot stop the underlying calls. Enforcement has to live somewhere that is. |
| **No OTLP export** | Prometheus and webhooks are supported; OpenTelemetry traces are not exported yet. |
| **Structured-output workloads are not rehomed** | The catalogue has no per-model JSON-mode flag, so a workload known to require structured output gets no downgrade suggestions. |
| **Anthropic 1-hour cache writes are under-priced** | LiteLLM publishes only the 5-minute-TTL write rate; 1-hour-TTL writes bill at 2× standard input upstream but are priced here at the 5-minute rate. Splitting by TTL needs a per-TTL breakdown end to end. |
| **Token totals exclude Anthropic cache writes** | `input_tokens` is normalised to prompt = uncached + cache reads; cache-write tokens are priced but not counted in `total_tokens`, so token analytics run slightly low on write-heavy calls. Cost is unaffected. |
| **Webhook delivery refuses private addresses** | The delivery-time SSRF guard resolves the host and refuses non-public addresses. Local listeners (including the `http://localhost` development exception) require `WEBHOOK_ALLOW_PRIVATE_URLS=true` in the server environment. |

---

## Reference

- Source: [agentcost-sdk](https://github.com/agentcost-ai/agentcost-sdk) · [agentcost-backend](https://github.com/agentcost-ai/agentcost-backend)
- SDK reference: [agentcost.tech/docs/sdk](https://agentcost.tech/docs/sdk)
