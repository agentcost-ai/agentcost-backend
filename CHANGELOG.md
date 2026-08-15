# Changelog

All notable changes to the AgentCost backend.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Fixed — cost accuracy

- **Prompt-cache tokens were discarded at ingest, overstating cost.**
  The SDK read `cached_tokens` off the provider response, but `EventCreate` did not
  declare the field and had no `model_config`, so Pydantic's default `extra="ignore"`
  dropped it silently. The server then repriced the full prompt at the standard input
  rate. Cost was overstated on every cache-heavy workload — measured at 13.00 vs 4.90
  on a 90%-cached call. Agentic coding assistants cache aggressively, so this affected
  them worst.
  `cached_tokens`, `cache_write_tokens` and `streaming` are now accepted, stored and
  priced. Cache rates come from LiteLLM's `cache_read_input_token_cost` and
  `cache_creation_input_token_cost`. Where a provider publishes no cache rate, cached
  tokens bill at the standard input rate — a discount is never assumed, and `NULL` is
  kept distinct from `0.0` throughout so a missing rate cannot zero out real spend.

- **Model-downgrade suggestions could break the workload they were optimising.**
  Capability inference read event metadata for `tools` / `response_format` / image
  parts, but nothing populated those keys, so it resolved to `unknown` for effectively
  every project. The requirement check then read `state == "true"`, making `unknown`
  evaluate as *not required* — so the optimizer would propose a text-only model for a
  workload sending images, or a model without tool support for an agent that calls tools.
  Unknown now fails closed for vision and tool calling: candidates are restricted to a
  capability superset of the current model, which withholds a suggestion at worst.
  Structured output still only blocks on a positively known requirement, because the
  catalogue has no per-model JSON-mode flag and treating unknown as required would
  suppress every suggestion for every existing project. Suggestions now carry
  `capabilities_verified` so a consumer applying them automatically can gate on
  measured rather than assumed requirements.

- **Run outcomes were discarded when a batch carried no valid events.**
  `EventBatchRequest` required at least one event, and the ingest route returned before
  persisting outcomes when the event list came back empty from per-event validation. A
  run that ended for a reason outside AgentCost's view could not be reported at all.

### Fixed — schema migration

- **New columns were unreachable on existing deployments.**
  `create_all()` only creates tables; a column added to an existing table reaches
  production solely through `_DESIRED_COLUMNS` in `app/database.py`. Tests build a fresh
  schema, so a missing entry passes every test and causes a total ingest outage on
  deploy. All columns added in this release are now registered, along with the index and
  column-width changes.

- **A duplicate `model_pricing` key in `_DESIRED_COLUMNS` silently discarded entries.**
  Python keeps the last key in a dict literal, so the earlier block's columns were never
  migrated. Merged, and `test_database_migrations.py` now parses the literal's AST and
  fails on any repeated table.

### Added — external run correlation

- **Trace identifier columns widened from 32 to 64 characters**
  (`events.trace_id`, `span_id`, `parent_span_id`, `trace_outcomes.trace_id`).
  A canonical UUID is 36 characters, so a run id minted by another system was rejected
  by validation — silently, from the sender's perspective. Longer ids are still
  rejected rather than truncated, since truncating would merge two distinct runs.

- **Outcome-only batches.** `events` may now be empty or omitted when `outcomes` is
  present, so a system that ends a run without making an LLM call can still report how
  it ended. `outcomes_recorded` was added to the ingest response.

- **Idempotency.** Optional per-event `event_id`, unique per project. A replay returns
  200 with `events_duplicate` incremented and stores nothing. Deduplication covers
  repeats within a single batch, across deliveries, and across *concurrent* deliveries:
  a partial unique index on `(project_id, event_id) WHERE event_id IS NOT NULL` backs
  the ingest path's lookup, and a conflict from a mid-flight race is retried once with
  the duplicates dropped rather than surfaced as an error.

### Added — cost dimensions

- **`user_id` and `session_id` promoted from metadata to indexed columns**, and
  `GET /v1/analytics/by/{dimension}` for grouping by `user`, `session`, `workflow`,
  `tool`, `model` or `agent`. This is what answers *what is each developer costing us*;
  previously cost was only attributable to agents and models. Values are coerced to
  strings so an integer id does not split one person across two buckets. Events with no
  value are excluded rather than bucketed under a placeholder. Historical rows are
  backfilled from stored metadata when the columns are first added, with the same
  coercion rules, so the dimension starts populated rather than empty.

- **`GET /v1/analytics/cache`** — prompt-cache hit rate and savings for a window,
  priced per model against the full-input-rate baseline. What caching actually earned,
  as a number.

### Added — egress

- **`GET /v1/projects/{id}/budget-state`** — compact, side-effect-free budget position
  authenticated with the project API key. Intended for polling and caching by a consumer
  that must not add a network call to a latency-sensitive path. Returns `as_of` and
  `period_ends_at` so staleness and time-remaining are both reasonable about.

- **Signed webhooks.** `GET` / `PUT /v1/projects/{id}/webhook` configure push delivery of
  budget threshold crossings, and `POST …/webhook/test` sends a signed sample so the
  wiring is verified at configuration time rather than at the first real crossing.
  Payloads are signed `HMAC-SHA256(secret, "{timestamp}.{body}")` in
  `X-AgentCost-Signature`, with the timestamp inside the signed string so a captured
  delivery cannot be replayed with a fresh header. HTTPS required; requires project-edit
  permission, since a webhook URL is an exfiltration path for spend data; a secret sent
  without a URL is rejected so a rotation can never silently disable the hook. At send
  time the destination host is resolved and non-public addresses are refused (SSRF
  guard; `WEBHOOK_ALLOW_PRIVATE_URLS=true` opts out for local listeners), redirects are
  not followed, and only a 2xx counts as delivered. Delivery is best-effort and never
  delays ingestion.

- **`GET /v1/metrics`** — Prometheus exposition of calls, cost, tokens, cached tokens,
  latency, errors, per-model and per-agent cost, plus budget utilisation and remaining.
  Windowed gauges rather than monotonic counters — and named accordingly, without the
  `_total` suffix Prometheus reserves for counters. Per-dimension series capped at the
  50 costliest to bound cardinality.

- **`POST /v1/pricing/import`** — load a pricing catalogue from an uploaded LiteLLM
  bundle instead of fetching it, for air-gapped and egress-restricted deployments. Same
  parsing, sanity bounds and change tracking as the network sync.

### Changed

- `PricingService.calculate_cost` and the ingest path now share one cache-aware pricing
  function, `pricing_math.price_event`, so the two cannot drift and report different
  costs for the same call. (`event_service.price_event` remains as an import alias.)

### Testing

- 74 new tests: cache pricing, capability guard, external correlation (including the
  concurrent-replay race), dimensions and egress, webhook configuration and delivery
  guards, cache analytics, the air-gapped pricing import, and migration rehearsals on
  both SQLite and PostgreSQL.
- **Fixed a session-wide test isolation bug.** The rate limiter is a module-level
  singleton with a 100 req/min window shared across the entire pytest session, so adding
  tests eventually caused *unrelated earlier* tests to fail with 429s, pointing nowhere
  near the cause. The suite had a hard ceiling on its own size. `conftest.py` now clears
  the limiter between tests.
- `test_database_migrations.py` gained a rehearsal that builds a pre-upgrade `events`
  table, runs the bootstrap over it, and writes a row using the new columns — the test
  that would have caught the migration gap above.

### Known gaps

Not addressed in this release; see `docs/ARCHITECTURE.md` §13.

- Capability fingerprints are forward-only: existing downgrade suggestions report
  `capabilities_verified: false` until new traffic accumulates.
- Anthropic 1-hour-TTL cache writes are priced at the 5-minute rate (the only one
  LiteLLM publishes); heavy 1-hour usage is slightly under-billed.
- `daily_aggregates` remains unpopulated; analytics still scan raw events.
- Webhooks have no retry and no delivery log; `budget-state` polling is the reliable
  channel.
- No dashboard UI for budget-state, dimension analytics or webhook configuration.
- No durable telemetry spool, no OTLP export.

### Verification

- The PostgreSQL-only migration path (`_WIDEN_COLUMNS`, the partial unique index, the
  dimension backfill) is rehearsed end-to-end by
  `scripts/rehearse_migrations_pg.py`, which upgrades a pre-release schema inside a
  disposable `postgres:16` container and exercises the widened ids, the backfill and
  the uniqueness enforcement. Run it before any deploy that touches the migration maps.
