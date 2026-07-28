# Architecture

## Overview

rusty_provider exposes one OpenAI-compatible HTTP API
(`/v1/chat/completions` and friends) in front of several upstream LLM
providers (OpenAI, Anthropic, Gemini, and any OpenAI-compatible backend —
Groq, Together, Fireworks). It resolves a request's `model` string to a
provider (or a config-defined fallback chain), applies policy in front of
dispatch (guardrails, moderation, web search, budgets, rate limits), and
forwards to whichever adapter that provider needs. It is not a model
host — it holds no weights and does no inference itself, only routing,
policy, and protocol translation. It is not multi-tenant SaaS — there's
no signup flow or per-tenant database; "clients" are config-defined API
keys sharing one process.

There are two binaries over the same `Router`: `rp-server` (the HTTP API)
and `rusty-provider-acp` (an Agent Client Protocol coding agent an editor
spawns over stdio). They are alternative front ends, not layers — neither
calls the other, and all policy lives below both.

## Boundaries

The core seam is `Provider` (`rp-core::provider`) — every upstream
backend is reached only through this trait, so `rp-router`'s dispatch and
fallback logic is written once and never branches on provider identity.

| Port | Adapter(s) | Notes |
| ---- | ---------- | ----- |
| `Provider` (`rp-core`) | `AnthropicProvider`, `GeminiProvider`, `OpenAiCompatibleProvider` (`rp-providers`) | `OpenAiCompatibleProvider` covers OpenAI, Groq, Together, and Fireworks — same wire format, different `base_url`/key, so one adapter serves all four. `chat`/`chat_stream` both take an optional per-request `api_key_override` for BYOK. |
| Usage/budget persistence (`rp-router::persistence`) | in-memory only, SQLite (`rusqlite`), Postgres (`tokio-postgres`, optional TLS) | Selected by `[persistence].backend` in config. A misconfigured or unreachable backend is a soft failure — the router still starts and runs in-memory-only, logged as a warning, same as a misconfigured provider. |
| Auxiliary HTTP backends (moderation, web search, budget webhook) | `ModerationClient` (OpenAI `/moderations`-shaped), `WebSearchClient` (Brave-shaped), `WebhookNotifier` | Not behind a shared trait — each is a thin, independently swappable `reqwest`-based client, since there's exactly one implementation of each today. All three fail open: their own unavailability never blocks or fails the request that triggered them. |
| Workspace access for the ACP agent (`rp-acp::tools`) | the connected ACP client, via `fs/*` and `terminal/*` | Inverted on purpose: the agent has no filesystem or process adapter of its own, so the *editor* is the implementation. Which tools exist at all is derived from the capabilities that editor advertised at `initialize`. |

## Structure

A 5-crate Cargo workspace, layered so each crate only depends on the ones
before it:

- `rp-core` — the shared request/response types (OpenAI chat-completions
  shape), the `Provider` trait, and error types. No I/O, no `reqwest` in
  its own logic beyond the types adapters serialize to/from.
- `rp-providers` — one adapter per upstream API, implementing `Provider`.
  Each owns its own wire-format translation (message shapes, tool-calling,
  streaming SSE parsing, reasoning/thinking-token handling) and its own
  `reqwest::Client`.
- `rp-router` — the `Router`: resolves a model string or route alias to a
  provider chain, applies fallback/retry on retryable errors, and hosts
  every cross-cutting policy (pricing/cost tracking, rate limiting,
  budgets, guardrails, moderation, web search, presets, auto-routing,
  an opt-in response cache, persistence). This is the largest and most
  stateful crate — it holds the process's in-memory routing/uptime/spend
  state alongside whatever persistence backend is configured.
- `rp-server` — the axum HTTP layer: route registration, request
  extraction/auth, and translating `Router` results to HTTP responses.
  Deliberately thin — almost no policy logic lives here, so the same
  `Router` could in principle be driven by a different transport.
- `rp-acp` — the second transport that "in principle" anticipated: an
  Agent Client Protocol agent over JSON-RPC on stdio. Owns the ACP wire
  types, the transport, session state, and the tool-calling loop; owns no
  policy, and reaches upstream models only through the same `Router`
  methods `rp-server` calls. A sibling of `rp-server`, not a layer above
  it — they never interact.

This is a modular monolith by design, not a stepping stone to
microservices — one process, one deploy artifact. The crate boundaries
exist for compile-time separation and testability (each crate's tests
run independently), not for independent deployment or scaling.

## Data flow

A `POST /v1/chat/completions` request, in order:

1. `rp-server::routes::chat_completions` authenticates the caller
   (bearer token against `[[clients]]` or `server.api_key_env`) and
   checks its inbound rate-limit bucket.
2. `Router::apply_preset` — if `preset` is set, merges in that preset's
   saved model/provider-prefs/system-prompt/sampling-params.
3. `Router::apply_web_search` — if requested, runs a live search and
   prepends the results to the last user message as plain-text context.
4. Guardrails (`rp-router::guardrails::apply`) — regex block/redact over
   the (now web-search-augmented) message text.
5. `Router::apply_moderation` — an external classifier checks the
   (guardrail-redacted) text; a flagged request is rejected with `400`.
6. Budget check (`Router::check_client_budget`) — rejects with `402` if
   the caller's tracked spend already crosses its configured budget.
7. `Router::dispatch`/`dispatch_stream` — for a non-streaming request
   with `[cache]` configured, checks the response cache first; a hit
   returns the stored response immediately, skipping everything below
   (chain resolution, dispatch, and usage/cost recording — that
   bookkeeping already ran once when the response was first computed).
   On a miss, resolves `model` to a provider chain and tries each
   candidate in order, via that provider's `Provider` implementation,
   falling back on a retryable error (rate limit, timeout, 5xx). Usage/
   cost is recorded and persisted after a successful attempt (and the
   response cached, if configured); a budget-crossing event fires the
   configured webhook.

An ACP `session/prompt` turn, in order:

1. `rp-acp::agent` resolves the session (created earlier by `session/new`,
   which fixed the workspace `cwd` and captured the client's capabilities
   at `initialize`) and checks the configured client's budget once.
2. `rp-acp::turn` appends the prompt's content blocks to the session's
   conversation and enters a loop bounded by `[acp].max_turn_requests`.
3. Each iteration builds a `ChatRequest` — the tool list derived from the
   client's capabilities — and runs steps 2–5 of the HTTP flow above
   (preset, web search, guardrails, moderation) against the same `Router`
   methods, then `Router::dispatch_stream`.
4. Streamed text and reasoning go out as `session/update` notifications
   while tool-call fragments accumulate. A guardrail or moderation block
   ends the turn as `refusal` rather than an error.
5. Each requested tool call is reported, permission-gated if it mutates
   anything, executed by calling *back* to the client (`fs/*`,
   `terminal/*`), reported again with its result, and appended to the
   conversation. Then back to step 3.
6. The loop ends when the model stops calling tools (`end_turn`), the cap
   is hit (`max_turn_requests`), or `session/cancel` arrives —
   which is observed mid-stream and mid-command, not just between steps.

## Key decisions
See [docs/adr/](./docs/adr/) for the record of individual decisions and their tradeoffs.

## Non-goals

- **Not a model host.** No inference, no weights — pure routing and
  protocol translation in front of upstream provider APIs.
- **Not a full LLM gateway UI/analytics product.** No dashboard; the only
  operator surfaces are the admin HTTP API, `GET /v1/usage`, and
  `GET /metrics` (Prometheus).
- **Not multi-tenant SaaS.** `[[clients]]` are config-defined, not
  self-serve; there's no signup flow, billing integration, or per-tenant
  database — everything lives in one process's config and one shared
  (optionally persistent) usage store.
- **Not a complete ACP implementation.** `rp-acp` implements protocol
  version 1's agent side and advertises only what it actually does.
  Session persistence (`session/load`), client-supplied MCP servers,
  session modes, elicitation, and authentication are deliberately absent
  rather than stubbed — sessions are process-scoped, and provider
  credentials come from the environment the agent was launched in.
- **Not a semantic/fuzzy cache.** `[cache]` (opt-in) is exact-match only —
  a hash of the entire request — with a TTL and fixed-capacity eviction;
  there's no embedding-based or near-duplicate matching, and streaming
  requests always bypass it.
