# ADR-0002: Persist agent run traces

Status: Proposed
Date: 2026-07-28

## Context

An ACP prompt turn already produces a complete, ordered account of itself.
`rp-acp`'s turn loop emits `session/update` notifications for every
assistant message chunk, thought chunk, tool call, tool-call result, plan
revision, and usage report — and then throws all of it away. The editor
renders the stream live; nothing survives the process.

That is fine for an editor, which has the transcript on screen. It is not
enough for two things operators keep asking for:

- **"What did the agent actually do?"** — after the fact, for a run that
  edited files or ran commands. Today the only evidence is whatever
  scrolled past in the editor, plus `tracing` output on stderr.
- **"Show a user their own run."** — any product built on top of this
  needs a durable, replayable record to render.

What exists today doesn't cover it. `GET /v1/usage` is cumulative counters
per `provider/model`. `GET /v1/generation?id=` returns one request's
token/cost breakdown from `GenerationCache` — 1000 entries, in memory,
insertion-order eviction, explicitly documented as "a recent-history
cache, not a durable audit log." Prometheus metrics are aggregates. None
of them records *steps*.

The forcing function is that the data is already being produced and
already flows past a natural persistence seam. `[persistence]` exists,
with SQLite and Postgres backends and a fire-and-forget write path the
`Router` already uses on every completed request.

## Decision

Add an opt-in `[trace]` subsystem that records agent runs as durable,
ordered event rows through the existing `Persistence` layer, and exposes
them over an admin-gated read API.

Four constraints shape the whole design:

1. **Coalesced, not raw.** Trace rows are written where the turn already
   accumulates state, not where it notifies. Persisting the notification
   stream verbatim would mean a row per streamed token.
2. **Never in the request path.** Same contract as usage persistence: a
   trace write that fails, or a backend that's gone, is logged and
   dropped. It must never fail, slow, or reorder the turn it describes.
3. **Structure by default, content by opt-in.** Prompts and model output
   are the most sensitive thing this process handles. The default records
   what happened; recording what was said is a deliberate switch.
4. **Bounded by default.** An append-only table with no retention policy
   is an outage waiting to happen on the operator's disk.

### Data model

Two tables, added to the shared `SCHEMA_SQL` constant in
`rp-router::persistence`:

```sql
CREATE TABLE IF NOT EXISTS trace_runs (
    run_id      TEXT PRIMARY KEY,
    source      TEXT NOT NULL,           -- 'acp' today
    session_id  TEXT,                    -- ACP sessionId; null for other sources
    model       TEXT NOT NULL,           -- the configured model or route alias
    client_name TEXT,                    -- [[clients]] attribution, when set
    cwd         TEXT,
    started_at  BIGINT NOT NULL,         -- unix millis
    ended_at    BIGINT,                  -- null while in flight
    stop_reason TEXT,                    -- ACP stop reason, once known
    turns       BIGINT NOT NULL DEFAULT 0,
    prompt_tokens     BIGINT NOT NULL DEFAULT 0,
    completion_tokens BIGINT NOT NULL DEFAULT 0,
    cost_usd    DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS trace_events (
    run_id     TEXT NOT NULL,
    seq        BIGINT NOT NULL,          -- monotonic within a run
    at         BIGINT NOT NULL,          -- unix millis
    kind       TEXT NOT NULL,            -- see below
    tool_call_id TEXT,                   -- promoted for lookup; null off tool events
    tool_name  TEXT,
    status     TEXT,                     -- completed | failed, on tool events
    payload    TEXT NOT NULL,            -- JSON, shape depends on kind
    PRIMARY KEY (run_id, seq)
);
```

`payload` is `TEXT` holding JSON in **both** backends, not Postgres
`JSONB`. `SCHEMA_SQL` is one string executed against both, and the module
is built around that; a type that only exists on one backend would force
it to split. Structured querying inside payloads is not a goal — the
columns worth filtering on are promoted out.

`(run_id, seq)` as the primary key is load-bearing, not decoration.
`PostgresBackend::record` does `tokio::spawn` per write, so two writes
issued in order can land out of order. An explicit sequence number
assigned at emit time means the reader can always reconstruct the true
order regardless of arrival.

Event kinds, one row each:

| `kind` | Written when | Payload |
| --- | --- | --- |
| `user_prompt` | A turn starts | Content block kinds; text only under `record_content` |
| `assistant_message` | A model request finishes | Token counts; text only under `record_content` |
| `assistant_thought` | A model request finishes with reasoning | Same, gated additionally on `record_thoughts` |
| `tool_call` | A tool call resolves | Tool name, title, kind, arguments, resulting status |
| `plan` | `update_plan` runs | The full entry list — it's small and already structured |
| `model_request` | Each loop iteration completes | Resolved `provider/model`, finish reason, usage, cost |
| `permission` | A permission prompt resolves | Tool name and the outcome |
| `stop` | The turn ends | Stop reason |

Note what's absent: no per-chunk rows. One `assistant_message` row per
model request, written from the accumulated `Completion` the loop already
builds, is the whole point of constraint 1.

### Write path

Extend `Persistence` with two fire-and-forget methods mirroring `record`:

```rust
pub fn begin_run(&self, run: &TraceRun);
pub fn record_trace_event(&self, event: &TraceEvent);
pub fn finish_run(&self, run_id: &str, stop_reason: &str, totals: &RunTotals);
```

SQLite routes through the existing background writer thread via new
`SqliteWrite` variants. One change is needed there: the writer channel is
currently an unbounded `std::sync::mpsc::channel`, which is fine at usage
volume (one write per completed request) but is an unbounded memory sink
at trace volume if the writer falls behind. Trace writes should go through
a **bounded** `sync_channel` and drop on full, counted in metrics. Usage
writes keep the existing unbounded channel — dropping a billing event and
dropping a trace row are not the same severity.

Postgres keeps the spawn-per-write shape. Ordering is already handled by
`seq`.

In `rp-acp`, `Session` gains an `AtomicU64` sequence counter and a
`run_id`; `turn::run` emits at the points it already has the data —
after the prompt is appended, after each `stream_completion` returns,
after each `run_tool_call`, and on every `return` path for the `stop`
event. The emit helper takes `Option<&Persistence>` and is a no-op when
tracing is off, so the turn loop carries no branching of its own.

### Config

```toml
[trace]
retention_days = 30      # optional, default 30; 0 disables pruning entirely
max_runs = 10_000        # optional, default 10_000; oldest pruned first
record_content = false   # optional, default false — prompt and message text
record_thoughts = false  # optional, default false — reasoning traces
```

Absent means tracing is off, matching every other optional subsystem.
Present but with `[persistence]` absent is a config error at startup, not
a silent no-op: asking for durable traces with nothing durable behind them
is a mistake worth surfacing loudly.

`record_content = false` still records that a message happened, its token
counts, and every tool call with its arguments — enough to answer "what
did the agent do," which is the primary question. Tool arguments are
recorded either way, since a `write_file` call is meaningless without its
path.

Pruning runs on the writer thread, on a low-frequency tick rather than
per-write: delete `trace_runs` older than `retention_days` or beyond
`max_runs`, then their events.

### Read API

```
GET /v1/traces?limit=&before=&session=&client=
GET /v1/traces/{run_id}
```

Both gated behind `server.admin_key_env`, not the ordinary API key. Traces
can contain prompt text and file paths; they belong with the admin surface
that already manages clients and spend, not with the OpenAI-compatible
routes.

The list endpoint returns `trace_runs` rows newest-first with a cursor;
the detail endpoint returns the run plus its events ordered by `seq`.

### Metrics

`rp_trace_events_recorded_total` and `rp_trace_events_dropped_total`, so a
writer falling behind is visible rather than silent.

## Alternatives considered

**OpenTelemetry spans instead of tables.** Better for dashboards,
alerting, and correlating with the rest of an operator's infrastructure,
and it avoids owning a storage format. It loses on the use case that
motivated this: rendering one run back to the person who triggered it.
Querying a tracing backend to paint a product UI is the wrong shape, and
it puts a second, heavier dependency in the path. These aren't exclusive —
an OTel exporter can be added later over the same emit points.

**Extend `GenerationCache` instead.** Cheapest option, and wrong. It is
in-memory, capacity-bounded, and per-request; a trace is durable, ordered,
and per-run. Widening it would mean rewriting it into this anyway, minus
the schema.

**Persist the raw `session/update` stream.** The most faithful record, and
the most expensive — a row per streamed token, and a replay format welded
to ACP's wire types. The coalesced model loses per-token timing, which
nothing has asked for.

**Write traces from `rp-server` for HTTP requests too.** Deliberately out
of scope for now. A single chat completion has no steps to trace; its
interesting data is already in `/v1/usage` and `/v1/generation`. The
schema leaves room (`source`, nullable `session_id`) to add it without
migration if that changes.

## Consequences

- The `Persistence` layer stops being purely aggregate. Every table so far
  upserts counters; `trace_events` is the first append-only, unbounded-
  by-nature table, which is what makes retention a required feature rather
  than a nicety.
- Operators enabling `record_content` are choosing to put prompt and
  response text in their database. That should be said plainly in
  `config.example.toml` and `SECURITY.md`, next to the existing note about
  keys never living in config.
- The write path grows a bounded channel and a drop policy, so under
  sustained load traces become lossy before anything else degrades. That
  is the intended failure mode and the reason for the dropped-events
  counter.
- `rp-acp` gains a dependency on a `Persistence` handle it currently
  doesn't hold. It already holds an `Arc<Router>`; exposing the handle
  through the router keeps the crate boundary as it is.
- Once runs are durable, "resume a session" stops being obviously
  impossible — the conversation would be reconstructible. This ADR does
  not propose that, and `loadSession` stays unadvertised, but it is the
  natural next question and worth noting as a thing this opens rather than
  forecloses.

## Phasing

1. Schema, `Persistence` methods, `[trace]` config, emit points in
   `rp-acp`. Tests: SQLite round-trip, `TEST_POSTGRES_URL`-gated Postgres
   round-trip, and an ACP integration test asserting a scripted turn
   produces the expected ordered rows.
2. Read API and admin gating.
3. Retention and pruning, with the metrics counters.

Phase 1 alone is useful — it makes runs inspectable with `sqlite3`. Phase
3 must land before anyone runs it unattended.
