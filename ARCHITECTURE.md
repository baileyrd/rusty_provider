# Architecture

## Overview

rusty_provider is a single Rust binary (`rp-server`) that exposes one
OpenAI-compatible HTTP API (`/v1/chat/completions` and friends) in front
of several upstream LLM providers (OpenAI, Anthropic, Gemini, and any
OpenAI-compatible backend — Groq, Together, Fireworks). It resolves a
request's `model` string to a provider (or a config-defined fallback
chain), applies policy in front of dispatch (guardrails, moderation, web
search, budgets, rate limits), and forwards to whichever adapter that
provider needs. It is not a model host — it holds no weights and does no
inference itself, only routing, policy, and protocol translation. It is
not multi-tenant SaaS — there's no signup flow or per-tenant database;
"clients" are config-defined API keys sharing one process.

## Boundaries

The core seam is `Provider` (`rp-core::provider`) — every upstream
backend is reached only through this trait, so `rp-router`'s dispatch and
fallback logic is written once and never branches on provider identity.

| Port | Adapter(s) | Notes |
| ---- | ---------- | ----- |
| `Provider` (`rp-core`) | `AnthropicProvider`, `GeminiProvider`, `OpenAiCompatibleProvider` (`rp-providers`) | `OpenAiCompatibleProvider` covers OpenAI, Groq, Together, and Fireworks — same wire format, different `base_url`/key, so one adapter serves all four. `chat`/`chat_stream` both take an optional per-request `api_key_override` for BYOK. |
| Usage/budget persistence (`rp-router::persistence`) | in-memory only, SQLite (`rusqlite`), Postgres (`tokio-postgres`, optional TLS) | Selected by `[persistence].backend` in config. A misconfigured or unreachable backend is a soft failure — the router still starts and runs in-memory-only, logged as a warning, same as a misconfigured provider. |
| Auxiliary HTTP backends (moderation, web search, budget webhook) | `ModerationClient` (OpenAI `/moderations`-shaped), `WebSearchClient` (Brave-shaped), `WebhookNotifier` | Not behind a shared trait — each is a thin, independently swappable `reqwest`-based client, since there's exactly one implementation of each today. All three fail open: their own unavailability never blocks or fails the request that triggered them. |

## Structure

A 6-crate Cargo workspace, layered so each crate only depends on ones
earlier in this list. `rp-mcp` and `rp-cli` are independent leaves that
both sit on top of `rp-router`, not on each other; `rp-server` depends on
`rp-core`/`rp-providers`/`rp-router`/`rp-mcp` (mounting its MCP handler
alongside the HTTP API) but not on `rp-cli`, which is a separate binary
entirely, never linked into `rp-server`.

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
- `rp-mcp` — optional MCP (Model Context Protocol) support, both
  directions: exposes `Router::dispatch`/`embeddings` as MCP tools
  (`chat_completion`/`list_models`/`embeddings`), and a gateway proxying
  configured `[[mcp.upstreams]]` (stdio subprocess or Streamable HTTP)
  under `"{upstream}/{tool}"` names into the same merged `tools/list`.
  Depends only on `rp-core`/`rp-router` — no `rp-providers` dependency,
  since it talks to `Router`, never to an upstream LLM provider directly.
  See [docs/MCP.md](./docs/MCP.md).
- `rp-server` — the axum HTTP layer: route registration, request
  extraction/auth, mounting `rp-mcp`'s handler behind the same auth as
  every other route, and translating `Router` results to HTTP responses.
  Deliberately thin — almost no policy logic lives here, so the same
  `Router` could in principle be driven by a different transport.
- `rp-cli` — a small, synchronous, read-only operator tool (`config
  check`/`providers list`/`keys check`, plus `setup` for rewriting a
  known third-party CLI tool's own config file) built on
  `rp-router::Config` directly, so it can never drift from the schema
  `rp-server` actually loads. Not part of the request-serving path at
  all, and not built into the Docker image.

This is a modular monolith by design, not a stepping stone to
microservices — one process, one deploy artifact. The crate boundaries
exist for compile-time separation and testability (each crate's tests
run independently), not for independent deployment or scaling.

## Data flow

A `POST /v1/chat/completions` request, in order:

1. `rp-server::routes::chat_completions` authenticates the caller
   (bearer token against `[[clients]]`, `server.api_key_env`, or, if
   `[jwt]` is configured, a verified JWT -- see `rp-server::jwt`) and
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

## Key decisions
See [docs/adr/](./docs/adr/) for the record of individual decisions and their tradeoffs.

## Non-goals

- **Not a model host.** No inference, no weights — pure routing and
  protocol translation in front of upstream provider APIs.
- **Not a GUI/desktop product.** No Electron app, no PWA, no traffic
  interception (no MITM proxy, no CA/trust-store changes). `rp-cli setup`
  (see [ADR-0004](./docs/adr/0004-cli-target-config-rewriting.md)) rewrites
  a known third-party CLI's own config file to point its endpoint at
  rusty_provider -- static, opt-in, and data-driven (`cli_targets.toml`) --
  which is a different, much smaller thing than sitting in the middle of
  that tool's traffic. The JSON API (the admin HTTP API,
  `GET /v1/usage`, `GET /v1/free-tiers`, `GET /metrics`) stays the
  canonical operator surface either way; `GET /dashboard` is a
  read-mostly, client-side-rendered view over that same API (one static
  HTML file, no build step, no JS framework), not a packaged product. See
  [ADR-0003](./docs/adr/0003-minimal-static-dashboard.md) (superseding
  [ADR-0002](./docs/adr/0002-reporting-surface-is-json-only.md)) for why a
  minimal dashboard was added without pulling in a frontend toolchain.
  The dashboard does carry an i18n framework (a `t()`-keyed string
  dictionary and language switcher, English-only today) -- that's just
  the switching mechanism, not a translation project, and it applies
  only to the dashboard's own UI chrome; server-generated JSON error
  messages stay English.
- **Not multi-tenant SaaS.** `[[clients]]` are config-defined, not
  self-serve; there's no signup flow or billing integration. Persistence
  (SQLite/Postgres) lets multiple *trusted* processes share one usage/spend
  store, but that's operator-run shared infrastructure, not a hosted
  product with per-tenant isolation.
- **No streaming response cache.** `[cache]` (opt-in) covers only
  non-streaming `/v1/chat/completions` responses, in either of two
  mutually exclusive modes: exact-match (a hash of the entire request) or
  `mode = "semantic"` (embedding-cosine-similarity matching on message
  text via this router's own `/v1/embeddings` dispatch, every other field
  still matched exactly). Both share the same TTL/fixed-capacity eviction
  shape. Streaming requests always bypass caching entirely — replaying a
  cached SSE chunk sequence is left for a future version, not a design
  decision to never support it.
