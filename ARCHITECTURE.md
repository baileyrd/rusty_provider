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

A 4-crate Cargo workspace, layered so each crate only depends on the ones
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
  persistence). This is the largest and most stateful crate — it holds
  the process's in-memory routing/uptime/spend state alongside whatever
  persistence backend is configured.
- `rp-server` — the axum HTTP layer: route registration, request
  extraction/auth, and translating `Router` results to HTTP responses.
  Deliberately thin — almost no policy logic lives here, so the same
  `Router` could in principle be driven by a different transport.

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
7. `Router::dispatch`/`dispatch_stream` — resolves `model` to a
   provider chain and tries each candidate in order, via that
   provider's `Provider` implementation, falling back on a retryable
   error (rate limit, timeout, 5xx). Usage/cost is recorded and
   persisted after a successful attempt; a budget-crossing event fires
   the configured webhook.

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
- **Not a caching layer.** Every request that isn't rejected upstream of
  dispatch goes to the provider; there's no response cache today.
