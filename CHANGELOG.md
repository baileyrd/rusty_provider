# Changelog

All notable changes to this repo are documented here.
Format: Added / Changed / Deprecated / Removed / Fixed / Security, newest first.

## [Unreleased]
### Added
- `X-RP-Decision` (`strategy=...; provider=...; model=...; latency_ms=...`)
  and `X-RP-Fallback-Attempts` response headers on `/v1/chat/completions`
  (streaming and non-streaming) — which concrete candidate actually served
  an alias/chain request, and how many candidates it took, without a
  separate `GET /v1/generation?id=` round trip.
- `provider.max_request_price_usd` + `provider.budget_fallback` — caps a
  single request's estimated cost (`max_tokens * completion_per_million`
  per candidate). `budget_fallback: "strict"` (default `"cheapest"`)
  narrows the chain to only candidates under the cap, `402`-ing if none
  fit; `"cheapest"` always serves the request, routing to the cheapest
  fitting candidate or, failing that, the overall cheapest one anyway.
- `strategy = "fusion"` on a `[[routes]]` alias — dispatches the alias's
  `chain` (the "panel") in parallel instead of sequentially, then
  synthesizes one final answer via a designated `judge` model from
  whichever candidates responded within `fusion_timeout_secs` (each
  independently timed out, so the total wait doesn't scale with panel
  size). Panel answers reach the judge under an anonymized label, not by
  provider/model. A tool-calling or streaming request bypasses fusion
  entirely and falls back to ordinary sequential-chain dispatch, as does a
  fusion alias with no `judge` configured (a startup warning, not a hard
  failure). Usage/cost accounting covers every contributing panel member
  plus the judge, not just the judge's own call.
- Reasoning replay for tool-continuation turns — some OpenAI-compatible
  reasoning models (DeepSeek-reasoner, Kimi-K-series, QwQ, GLM-thinking)
  reject a tool-answering turn missing the `reasoning_content` behind the
  tool call. A non-streaming response's reasoning is now cached in memory
  by `tool_calls[].id` and transparently re-injected into the matching
  assistant message on the next request, even when the calling client
  stripped it (most do).
- `[[clients]].budget_warning_threshold` — a new `budget_warning`
  `[webhook]` event fires once a client's spend crosses this fraction of
  `budget_usd`, ahead of the hard `budget_exceeded` cutoff. Config-only
  for now, not yet settable via the admin API.
- `[webhook]` HMAC-SHA256 signing (`signing_secret_env`) and
  retry-with-backoff (`retry_backoff_secs`/`retry_backoff_max_secs`/
  `max_retries`) — a `5xx`/network-error delivery now retries instead of
  failing after one attempt; a `4xx` is still treated as permanent.
- `server.cors_allowed_origins` — restricts CORS to an explicit browser-
  origin allowlist. Unset preserves the existing any-origin behavior.
- `rp-cli setup` (`list`/`show`/`apply`) — rewrites a known third-party CLI
  tool's own config file (opencode, Crush) to point its endpoint at
  rusty_provider. Data-driven target list (`crates/cli/cli_targets.toml`,
  extensible via `--targets`), dry-run by default, `--yes`-gated writes
  with an automatic backup, never writes a literal API key (an env-var
  reference naming `--api-key-env` when the target format supports one).
  Static file rewriting only — no proxy, no traffic interception (ADR-0004).
- `[[mcp.upstreams]]` reconnect-with-backoff — a previously established
  upstream connection that drops (subprocess crash, HTTP network blip) is
  now redialed automatically with exponential backoff
  (`reconnect_backoff_secs`/`reconnect_backoff_max_secs`/
  `max_reconnect_attempts`) instead of staying dead until `rp-server`
  restarts. A startup connection failure is still a soft warning.
- Dashboard i18n framework — UI text goes through a `t()`-keyed
  translation dictionary and a language switcher, persisted in
  `localStorage`. Framework only — only `en` is populated today.
- `[jwt].client_claim` — a verified JWT's claim value matched against a
  configured `[[clients]].name` resolves that client's identity for the
  rest of the request (budget enforcement, per-subject rate limiting,
  usage/spend tracking), the same as a static per-client API key already
  gets. No match falls back to the prior unscoped behavior unchanged.
  `/v1/admin/*` is untouched by design.
- `GET /dashboard` — one self-contained HTML file, no build step, no JS
  framework, compiled into `rp-server` via `include_str!`. Renders
  entirely client-side against the existing JSON endpoints
  (`/v1/models`, `/v1/usage`, `/v1/providers/stats`, `/v1/free-tiers`,
  `/v1/admin/clients` + usage-history sparkline + reset-spend button),
  subject to the same auth those endpoints already enforce. The page
  itself is served unauthenticated (it carries no secrets).
- `GET /v1/admin/clients/{name}/usage-history?days=N` — day-bucketed
  `requests`/`prompt_tokens`/`completion_tokens`/`cost_usd` for a client,
  oldest first, over the last `N` days (default 30, capped at 90). New
  `client_daily_usage` table in both persistence backends. Empty
  (`data: []`) without `[persistence]` configured.
- Same-candidate retry-with-backoff (fixed 200ms, one retry) on a
  genuinely transient error (timeout, network error, `5xx`) before
  falling through to the next chain entry, across `dispatch`/
  `dispatch_stream`/`embeddings`. Excludes rate limits and unsupported-
  content/feature mismatches, since retrying the same candidate can't fix
  either.
- Structured `tracing::info!` "admin action" audit event (identity,
  organization, action, target) on every successful admin client
  create/update/delete/reset-spend mutation.
- `Router::from_config` now warns at startup when a `[[routes]]` alias's
  chain references a provider name with no matching `[[providers]]`
  entry, instead of only surfacing the typo implicitly at request time.
- `server.max_concurrent_requests` — a server-wide in-flight request
  ceiling (`Semaphore`-based); once saturated, the next request gets
  `503` immediately rather than queuing. Distinct from per-caller rate
  limiting, which bounds rate, not total in-flight count. Unset by
  default (no cap).
- `[mcp]` — opt-in MCP (Model Context Protocol) support, both directions
  at once, built on [`rusty_mcp`](https://github.com/baileyrd/rusty_mcp):
  rusty_provider's own routing exposed as MCP tools
  (`chat_completion`/`list_models`/`embeddings`), plus a gateway proxying
  configured `[[mcp.upstreams]]` (stdio subprocess or Streamable HTTP)
  under `"{upstream}/{tool}"` names, merged into one `tools/list`.
  Mounted inside the existing app/port under the same auth every other
  route already uses. `MCP_STDIO=1` serves the same handler over stdio
  for desktop clients. See `docs/MCP.md`.
- `docs/PROVIDERS.md` + curated commented-out provider presets in
  `config.example.toml` (~20 more OpenAI-wire-compatible backends).
- `[[free_tiers]]` config + `GET /v1/free-tiers` — operator-declared,
  self-tracked free-token budget reporting per "provider/model".
- Three new `provider.sort` strategies: `"quality"`, `"random"`,
  `"free_tier_remaining"`.
- `transforms: ["rtk"]` — built-in tool-output compression (git/test/
  build/package/generic categories), composable with `"middle-out"`.
- `rp-cli` — new 5th workspace crate, a read-only operator CLI (`config
  check`/`providers list`/`keys check`).
- `[cache].mode = "semantic"` — embedding-cosine-similarity response
  caching, opt-in alongside the existing exact-match mode.
- `[jwt]` — JWT/OIDC bearer-token authentication (HS256 shared-secret or
  JWKS/RS256), additive alongside `server.api_key_env`/`[[clients]]`.
  Fails closed on any verification failure.
- `.github/workflows/audit.yml` runs `cargo audit` against `Cargo.lock`
  on every push/PR touching a manifest, plus daily on a schedule.
- MIT `LICENSE` file — `Cargo.toml` had declared `license = "MIT"` since
  the workspace's first commit, but the license text itself was never
  reproduced anywhere in the repo.
- Multi-stage `Dockerfile` (+ `.dockerignore`) producing a slim
  `debian:bookworm-slim` runtime image for `rp-server`, built via
  `cargo-chef` from `rust:1-bookworm`. Runs as non-root; ships a
  `HEALTHCHECK` against `/health`. Nothing secret baked in — config and
  API keys are supplied at `docker run` time. `docker build` CI job.
- `GET /ready`, distinct from `GET /health`. `/health` is a cheap,
  unauthenticated liveness check that never touches anything external.
  `/ready` confirms the router can actually serve traffic — when
  `[persistence]` is configured, a round trip against its database,
  `503` with a reason on failure; always `200` otherwise.
- `server.max_body_bytes` (default 20 MiB), applied as a request-wide
  body-size limit; rejected requests get `413` before a handler parses
  the body.
- `[cache]` — opt-in, in-memory, exact-match response cache for
  non-streaming `/v1/chat/completions`, keyed by a hash of the whole
  incoming request. Entries expire after `ttl_secs` (default 300), evicts
  oldest past `max_entries` (default 1000). Streaming always bypasses it.
  New `rusty_provider_cache_lookups_total` Prometheus counter.
- `POST /v1/embeddings`, OpenAI-compatible request/response shape.
  Implemented by the OpenAI-compatible adapter and Gemini
  (`batchEmbedContents`); Anthropic has no embeddings API and always
  falls through. Reuses `dispatch`'s chain-resolution/retry logic.
- Standard governance/docs scaffold: `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `ARCHITECTURE.md`, this file, `RELEASE_NOTES.md`,
  PR/issue templates, and an ADR log seed.
### Changed
- `ARCHITECTURE.md` non-goals: "no dashboard"/"not multi-tenant SaaS"
  softened to "no UI" (ADR-0002), later to "no Electron/PWA/desktop
  product" once the dashboard shipped (ADR-0003, superseding ADR-0002);
  "no i18n" dropped once the dashboard's i18n framework shipped; "no
  MITM-based third-party CLI config injection" narrowed to "no traffic
  interception" now that `rp-cli setup` covers the static
  config-file-rewriting case (ADR-0004).
- When `model: "auto"` resolves to a `[[routes]]` alias spanning multiple
  candidates, dispatch now defaults `provider.sort` to `"price"` among
  them, unless the request already set its own explicit `sort` (which
  always wins unchanged).
- Chain resolution now stably deprioritizes (not re-ranks) any candidate
  with an observed EWMA success rate below `0.5` by default, not only
  when a request explicitly opts in with `sort: "uptime"`.
### Fixed
- `rp-cli keys check`/`config check` didn't report `jwt.hs256_secret_env`
  status (added after `rp-cli` itself was written) — both now mirror the
  existing admin-API status line.
- `ARCHITECTURE.md` (added by an earlier governance-scaffold pass)
  predated the opt-in response cache and still listed "no response cache
  today" as a non-goal; now documents the cache and narrows the non-goal
  to what's still true (exact-match only, no semantic/fuzzy matching at
  the time it was written — since superseded by `[cache].mode =
  "semantic"` above).
- axum's `Json`/`Bytes` extractors enforced an implicit, non-configurable
  ~2 MB body limit tight enough to reject a legitimate multimodal request
  (inline base64 image/audio/PDF); replaced by explicit
  `server.max_body_bytes` above rather than left silently in place.
### Security
- [RUSTSEC-2024-0437](https://rustsec.org/advisories/RUSTSEC-2024-0437)
  (uncontrolled recursion, crash) in `protobuf` 2.28.0, pulled in
  transitively via `prometheus`'s default feature though only its
  text-exposition encoder is ever used here — fixed by building
  `prometheus` with `default-features = false`, dropping the dependency
  (and the advisory) entirely.

<!-- ## [0.1.0] - YYYY-MM-DD
### Added
- Initial release -->
