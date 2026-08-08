# gap-analysis.md — rusty_provider vs. OmniRoute

Run scope settled 2026-08-06 (see chat): OmniRoute (diegosouzapw/OmniRoute) is a
full TypeScript product (Electron desktop app, PWA, 43-language i18n,
dashboard, 291 providers) built around aggregating documented free-tier LLM
quotas behind one OpenAI-compatible endpoint. rusty_provider is a headless
Rust HTTP router; its `ARCHITECTURE.md` currently states "not a full LLM
gateway UI/analytics product, no dashboard" and "not multi-tenant SaaS" as
explicit non-goals.

User's scope decision for this run:
- Parity target: revise those two non-goals so a JSON-only reporting surface
  and richer routing/config surface are in scope — **not** literal product
  parity (no Electron app, no PWA, no 43-language i18n, no MITM-based
  coding-tool config injection). Those stay explicitly out of scope.
- Provider breadth: don't chase OmniRoute's raw 291-provider count. Confirm
  the existing `[providers.X]` config is already provider-count-agnostic
  (`kind = "openai"` covers any OpenAI-wire-compatible backend today — Groq/
  Together/Fireworks already prove this), then add a curated batch of
  documented free/high-value OpenAI-compatible endpoints as ready-to-uncomment
  config presets + a reference doc, rather than one adapter per provider.
- In-scope differentiators (user-selected): token/output compression,
  free-tier tracking endpoint, more routing strategies, an operator CLI.

Source for all rows below: **spec** (read directly from OmniRoute's own docs —
no `cargo public-api`-diffable surface exists between a Rust workspace and a
TypeScript monorepo, and rusty_provider has no pre-existing ROADMAP.md to
audit against).

## Explicitly out of scope this round

Noted so a later run doesn't rediscover these as "missing": Electron desktop
app, PWA/service worker, 43-language i18n, MITM-based third-party CLI config
injection (`omniroute setup-*`), ACP agent spawning, account-rotation/"combo"
multi-key-per-provider pooling (OmniRoute's reset-aware/quota-share routing
depends on this and rusty_provider has no multi-account-per-provider concept
today), literal 291-provider adapter count, marketing site/Discord/sponsor
infrastructure. Also explicitly declined: reproducing OmniRoute's free-tier
*aggregation* model as-is — many of its 220+ tracked providers' own ToS
explicitly prohibit proxy/resale use (see `docs/reference/FREE_TIERS.md`'s ToS
table in OmniRoute); rusty_provider's version is scoped as an **operator
self-declared** budget report (matching the existing `zdr`/`no_training`
trust model), not a service that aggregates or launders other providers'
free tiers on the operator's behalf.

## Gaps

| Symbol | Category | Source | Platforms | Reference | Breaking? | Est. size | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ARCHITECTURE.md` non-goals | docs | spec | n/a | user scope decision | no | S | Soften "no dashboard"/"not multi-tenant SaaS" to reflect the new reporting endpoint below; record as an ADR. Prerequisite for the rest — do first. |
| Curated provider presets | docs/config | spec | n/a | `docs/reference/PROVIDER_REFERENCE.md`, `public/providers/*` | no | M | `config.example.toml` already supports any OpenAI-wire backend via `kind = "openai"`. Add ~20-25 commented-out presets (Mistral, Cerebras, SambaNova, DeepSeek, OpenRouter, Cloudflare Workers AI, HuggingFace router, NVIDIA NIM, Novita, DeepInfra, Nebius, Moonshot/Kimi, Zhipu/GLM, Alibaba Qwen/DashScope, 01.AI/Yi, xAI/Grok, Perplexity) with base_url + known free-tier notes, plus a `docs/PROVIDERS.md` reference table. |
| `[[free_tiers]]` config + `GET /v1/free-tiers` | fn (new) | spec | n/a | `docs/reference/FREE_TIERS.md` | no | M | Operator-declared monthly free-token budget per provider/model (self-declared like `zdr`, never verified against upstream). New endpoint reports configured budget vs. this process's tracked usage (reuses existing usage accounting) — remaining budget, not live scraped quota. |
| `sort: "quality"` | fn (existing, new variant) | spec | n/a | OmniRoute `docs/routing/*` general routing-strategy set | no | S | New operator-declared `quality_score` field on `[[pricing]]`; sorts candidates descending. Purely additive arm in `Router`'s existing `sort` match. |
| `sort: "random"` | fn (existing, new variant) | spec | n/a | — | no | S | Weighted-random ordering across the resolved chain, for simple load distribution instead of deterministic ranking. Additive arm. |
| `sort: "free_tier_remaining"` | fn (existing, new variant) | spec | n/a | OmniRoute reset-aware routing (docs/guides/FEATURES.md) | no | S | Depends on the `[[free_tiers]]` gap above. Prefers the candidate with the most configured-budget headroom left this period. |
| `transforms: ["rtk"]` tool-output compression | fn (new) | spec | n/a | `docs/compression/RTK_COMPRESSION.md` | no | L | New opt-in transform (alongside existing `middle-out`) that runs a built-in filter catalog over `role: "tool"` message content before dispatch — strip ANSI, collapse duplicate lines, condense git/test/build/package-manager/docker output. Mirrors the existing context-compression opt-in pattern; MVP covers 5 filter categories, not OmniRoute's full 49. |
| `rp-cli` operator CLI | fn (new crate) | spec | n/a | `docs/reference/CLI-TOOLS.md` (scoped down) | no | M | New `rp-cli` binary: `config check` (validate `config.toml` parses + report which providers/clients are active), `providers list` (resolved providers + skip reasons), `keys check` (which `api_key_env` vars are set, no values printed). Not OmniRoute's MITM/ACP-spawn CLI — pure config/ops tooling. |

Total: 8 issues, all additive (no `breaking-change` label needed), no new
third-party dependencies anticipated beyond what's already in the workspace
(the `rtk` transform and `rp-cli` are pure Rust/std + existing crates).

## Additional reference: agentgateway.dev

Cross-checked against [agentgateway](https://agentgateway.dev/) — also
Rust-based, a closer architectural peer than OmniRoute (unified gateway for
HTTP/gRPC/MCP/A2A traffic, not just OpenAI-shaped chat completions).

**Already covered, no new gap:** model/cost/latency-aware routing
(`provider.sort`), token budgets (`[[clients]].budget_usd`), per-request
cost calculation (`cost_usd`), team/user cost attribution
(`organization`/`workspace` on `[[clients]]`, `GET /v1/admin/organizations`
— agentgateway's "virtual scoped keys" equivalent), prompt
redaction/blocking (`[[guardrails]]` — regex-based, not agentgateway's
NER-style "PII-shield," but same slot), OpenTelemetry-adjacent observability
(`GET /metrics` Prometheus, per-provider stats).

**Identified but initially not filed as parity-gap issues** — both crossed
the skill's own stop-and-ask line (new protocol surface / new third-party
dependency), so neither was auto-implemented in the original run:

- **JWT/OIDC authentication — done.** User approved this as an explicit
  follow-up on 2026-08-06. Shipped in #109/#110 (merged): `[jwt]` config,
  `hs256_secret_env` (shared secret) or `jwks_url` (RS256, cached by `kid`),
  optional `issuer`/`audience` validation, additive alongside
  `server.api_key_env`/`[[clients]]`, fails closed on any verification
  failure. New dependency `jsonwebtoken`, approved at the time. Verified via
  a live smoke-test (#111 fixed a follow-on gap: `rp-cli` hadn't been
  updated to know about `jwt.hs256_secret_env`). The "no JWT-claims-to-
  `[[clients]]`-identity mapping in this pass" scope cut from #109/#110 was
  itself closed by #125: opt-in `[jwt].client_claim` maps a verified
  token's claim to a `[[clients]].name` for budget/rate-limit/usage
  purposes (`/v1/admin/*` stays untouched by design).
- **MCP (Model Context Protocol) support — done.** User approved this as an
  explicit follow-up on 2026-08-06, explicitly asking for both directions
  ("expose rusty_provider as an MCP server" and "proxy other MCP servers"),
  reusing [`baileyrd/rusty_mcp`](https://github.com/baileyrd/rusty_mcp) as
  the scaffold rather than hand-rolling MCP plumbing. New `rp-mcp` crate:
  `chat_completion`/`list_models`/`embeddings` tools wrapping the router's
  own dispatch (server direction), plus a gateway proxying configured
  `[[mcp.upstreams]]` (stdio subprocess or Streamable HTTP) under
  `"{upstream}/{tool}"` names (gateway direction), merged into one
  `tools/list`. Mounted inside rp-server's existing app/port, reusing the
  same `server.api_key_env`/`[[clients]]`/`[jwt]` auth rather than
  `rusty_mcp`'s own OAuth 2.1 — see `docs/MCP.md` for the full design
  rationale. New dependencies `rusty-mcp` (git) and `rmcp`, approved at the
  time as part of the same instruction. Verified via `crates/mcp/tests/`
  and `crates/server/tests/http_endpoints.rs`'s `mcp_endpoint_*` tests, both
  driving the merged handler with a real `rmcp` client over an in-process
  transport. The one item this left explicitly deferred -- a dropped
  upstream connection just failed its calls until restart, no reconnect --
  was itself closed by a follow-up: `[mcp]` upstreams now get
  reconnect-with-backoff (`reconnect_backoff_secs`/
  `reconnect_backoff_max_secs`/`max_reconnect_attempts`), a background
  supervisor task per upstream that redials with exponential backoff once
  a *previously connected* upstream drops. Verified via
  `crates/mcp/src/gateway.rs`'s backoff-policy unit tests plus a live
  smoke test (a real subprocess repeatedly killed and observed
  reconnecting through several cycles).

One additional gap **was** filed since it's additive and dependency-free
(the router can already call an embeddings provider itself):

| Symbol | Category | Source | Platforms | Reference | Breaking? | Est. size | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic response cache | fn (existing, new mode) | spec | n/a | agentgateway.dev "semantic caching" | no | M | Opt-in alongside the existing exact-match `[cache]`: embed the request via the already-configured embeddings provider, cosine-similarity match against cached entries above a configurable threshold. No new dependency — reuses the router's own `/v1/embeddings` dispatch path. |

## Follow-up pass — 2026-08-07

Since the 2026-08-06 baseline above, rusty_provider also shipped JWT/OIDC
auth, MCP server+gateway with reconnect-backoff, the semantic response
cache, and `rp-cli setup` (non-MITM CLI config-file rewriting, its own
ADR-0004 — narrows "no MITM-based CLI config injection" to "no traffic
interception" in `ARCHITECTURE.md`'s non-goals). This section re-runs the
comparison against OmniRoute's *current* docs (cloned at
`diegosouzapw/omniroute`, commit `6c22f8d` / 2026-08-07) rather than
re-deriving scope from scratch — the "explicitly out of scope" list above
stands unchanged unless noted otherwise below.

OmniRoute's own doc tree has grown substantially since the baseline pass
(agent-protocol frameworks, a plugin marketplace, a memory subsystem, an
eval framework, more routing strategies) — almost all of it is product/UI
surface that was already ruled out by the standing non-goals. The rows
below are the subset that's genuinely new *and* server-side/router-level
enough to plausibly belong in a headless HTTP router.

| Symbol | Category | Source | Breaking? | Est. size | Notes |
| --- | --- | --- | --- | --- | --- |
| Configurable CORS allowlist | fn (existing, hardening) | spec (`docs/security/CORS.md`) | no | S | `crates/server/src/lib.rs` currently hardcodes `CorsLayer::permissive()` — any browser origin can read any response. New `[server].cors_allowed_origins`, default unchanged (permissive) to avoid a silent breaking change for existing deployments. No new dependency (`tower_http::cors` already in use). |
| Webhook HMAC signing + retry-with-backoff | fn (existing, hardening) | spec (`docs/frameworks/WEBHOOKS.md`) | no | S/M | `[webhook]` is currently fire-and-forget, unsigned, single-attempt. Add HMAC-SHA256 body signing (`ring::hmac`, already a workspace dependency) plus retry-with-backoff on 5xx/network errors, mirroring the same backoff shape MCP reconnect already uses. |
| Budget warning threshold | fn (existing, new field) | spec (`docs/reference/API_REFERENCE.md`) | no | S | New optional `[[clients]].budget_warning_threshold` (e.g. 0.8) alongside the existing hard `budget_usd` limit; fires a new `budget_warning` `[webhook]` event before the hard cutoff. Additive field, no schema break. |
| Reasoning replay cache | fn (new) | spec (`docs/routing/REASONING_REPLAY.md`) | no | M | Some reasoning-capable models (DeepSeek-reasoner, Kimi-K-series, QwQ, GLM-thinking) hard-reject a follow-up turn missing prior `reasoning_content` — most client SDKs strip it themselves. Cache it server-side keyed by `tool_call_id`, re-inject transparently on the next turn. In-memory by default, optionally `[persistence]`-backed like the existing response cache. No new dependency. |
| `strategy = "fusion"` routing | fn (new route strategy) | spec (`docs/routing/AUTO-COMBO.md`) | no | M/L | Fan a request to every model in a panel in parallel, then a configured judge model synthesizes one answer from anonymized outputs. Does **not** depend on the declined multi-account-pooling ("combo") concept — one key per provider, same as every other route. Same abstraction level as the existing `sort:` strategies. |
| Per-request budget cap (`max_request_price_usd` + fallback policy) | fn (existing, new field) | spec (`docs/routing/AUTO-COMBO.md`) | no | S | Complements the existing per-candidate `provider.max_price` ceiling with a per-request total-cost cap, estimated from `max_tokens` × pricing before dispatch; `provider.budget_fallback: "strict"\|"cheapest"` chooses hard-402 vs. serve-via-cheapest when every candidate exceeds it. |
| Routing-decision trace headers | fn (new) | spec (`docs/reference/API_REFERENCE.md`) | no | S | New response headers (`X-RP-Decision: strategy=...;provider=...;latency_ms=...`, `X-RP-Fallback-Attempts`) so a caller can see which concrete provider/model actually served an alias/chain request without a separate `GET /v1/generation?id=` round trip. Headers only, no schema change. |
| Opt-in external pricing sync | fn (new) | spec (`docs/guides/COST_TRACKING.md`) | no | M | Optionally sync `[[pricing]]` rates from LiteLLM's public `model_prices_and_context_window.json` on an interval; explicit config-set entries always take precedence over synced ones. No new Cargo dependency (`reqwest` already present) — **but** this is the one candidate that adds a new *runtime* dependency on a third-party URL outside the operator's own configured providers, which is a different kind of new surface than a crate. Flagged for an explicit decision rather than filed automatically. |

**Borderline — flagged, not auto-included, needs a scope call:**

- **Memory system** (`docs/frameworks/MEMORY.md`/`MEMORY_BACKEND.md`) —
  persistent per-API-key conversational memory with regex fact extraction
  and FTS5/vector hybrid retrieval, injected into requests server-side.
  The single biggest doc-volume gap found, but it pushes rusty_provider
  from "stateless router" toward "stateful AI platform" — arguably
  adjacent to the standing "not multi-tenant SaaS" non-goal rather than a
  router capability. Would need its own ADR either way given the size.
- **Eval framework** (`docs/frameworks/EVALS.md`) — a suite-runner
  (built-in + custom test cases, scoring rubrics, A/B comparison) for
  regression-testing routing/model changes over the existing
  chat-completions path. Server-side and API-testable, but a materially
  new subsystem (persistence tables, runner, scoring engine) — closer to
  "product feature" than "router capability." Est. **L**.
- **Reasoning routing rules** (`docs/routing/REASONING_ROUTING.md`) — a
  priority-ordered rule engine (scope: key/combo/model/global) forcing or
  defaulting reasoning effort/budget, plus known-incompatible-model
  rejection. Meaningfully overlaps what `[[presets]]` + the providers'
  existing reasoning translation already do; the one piece with clear
  marginal value (reject a known-incompatible model before dispatch) may
  not justify a whole new rule-engine surface on its own.
- **3-state circuit breaker** (CLOSED/OPEN/HALF_OPEN + probe requests) —
  cited explicitly in OmniRoute's own comparison table. rusty_provider's
  existing health-based deprioritization (issue #75: EWMA success-rate
  stable-partition) reaches a similar practical outcome via a simpler
  mechanism. Possibly an internal-implementation nuance rather than a
  user-facing gap.
- **Auto-Combo category×tier suffix composition**
  (`auto/coding:fast`, `auto/reasoning:pro`, etc.) — rusty_provider's
  existing `auto_routing` (3-tier complexity classifier +
  `auto_bias: cost|quality`) already covers similar ground with a
  simpler model; the suffix-composition scheme may be more surface area
  than value added.

**User decision (2026-08-07):** skip all 5 borderline items and the opt-in
pricing sync. File issues for the 7 remaining candidates (CORS allowlist,
webhook HMAC signing + retry, budget warning threshold, reasoning replay
cache, `strategy = "fusion"` routing, per-request budget cap, routing-
decision trace headers) — filed as #135-#141, `parity-gap` labeled.

**Re-confirmed out of scope, not re-flagged:** free-tier *aggregation*
(Quota Sharing Engine, "stack these providers" — both depend on the
declined multi-account-pooling concept), MITM/traffic-interception docs,
Relay Backend Strategy / Provider Plugin Manifest (OmniRoute's own
TS-core→native-sidecar split — not applicable, rusty_provider already
*is* the native router), Route Guard Tiers / Ban Detection / Public Creds
(Electron-spawn-capable-route and OAuth-CLI-scraping concerns with no
rusty_provider analog), the 43-language i18n rows in OmniRoute's own
comparison table.

## Follow-up pass — 2026-08-08

All 7 issues from the 2026-08-07 pass shipped and merged (#135-#141:
CORS allowlist, webhook HMAC signing + retry, budget warning threshold,
reasoning replay cache, `strategy = "fusion"` routing, per-request budget
cap, routing-decision trace headers). Re-ran the comparison against
OmniRoute's current docs (`diegosouzapw/omniroute`, `release/v3.8.50`,
commit `caf768e`, up from `6c22f8d` on 2026-08-07).

Doc delta since the last pass is small — 91 lines across 4 files
(`docs/compression/COMPRESSION_ENGINES.md`,
`docs/compression/COMPRESSION_GUIDE.md`, `docs/guides/TROUBLESHOOTING.md`,
`docs/reference/ENVIRONMENT.md`) — and almost entirely not applicable:
a Next.js standalone-build packaging fix for the LLMLingua compression
worker (build tooling, no router-level analog), clarifying prose about
RTK/Caveman's real-world savings range (docs-only, no behavior change),
and a cron-driven Claude-OAuth warmup scheduler (`OMNIROUTE_WARMUP_*` —
depends on the declined multi-account/OAuth-connection model, no
rusty_provider analog).

One item is a plausible small gap, flagged rather than auto-filed since
it's arguably already covered by existing issue #64 (global
`server.max_concurrent_requests`) under a different mechanism, not
obviously missing:

- **Heavyweight-request admission control** (`docs/guides/TROUBLESHOOTING.md`
  "chat_admission_busy") — OmniRoute reserves a small, separately-capped
  pool of *heavyweight* request slots (`OMNIROUTE_CHAT_MAX_HEAVY_IN_FLIGHT`,
  default `1`), where "heavyweight" is judged by structural size (≥200
  messages, ≥64 tools, ≥32k estimated tokens, or exhausting bounded
  structure-estimation) rather than raw byte count, and rejects with a
  retryable `503`/`chat_admission_busy` + `Retry-After` when the pool is
  full — protecting host memory from a handful of oversized requests
  specifically, distinct from #64's overall in-flight cap (which caps
  *every* request the same way regardless of size). Not filed
  automatically pending a scope call on whether this distinction is worth
  a second admission-control axis.

No other new gaps found. Nothing else in this pass rises to "worth a new
parity-gap issue" — the current batch (#135-#141) is complete.
