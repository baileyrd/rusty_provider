# Release Notes

<!--
Two variants, pick the one that fits this repo's actual unit of change:

1. No version tags yet (pre-1.0, nothing published) — track by PR instead, same way
   AISF does it: one entry per merged PR against main, reverse chronological, each
   linking to its PR and (where one exists) to the doc that covers the change in full
   detail. Use "## PR #N — <summary>" headers.

2. Actual version tags exist — use "## vX.Y.Z - YYYY-MM-DD" headers instead, each
   linking to the PRs it shipped and a compare link to the previous tag. Add an
   "### Upgrade notes" subsection under any entry with a breaking change.

Either way, keep the tone AISF's file uses: bolded category tags inline in the
bullet (**Added:** / **Changed:** / **Fixed:**), not separate subheaders per
category — and state known limitations or deliberate scope cuts plainly instead of
leaving them implied.
-->

User-facing and operator-facing changes to rusty_provider, one entry per
merged PR against `main`, newest first. No version tags exist yet, so
entries are tracked by PR rather than by release.

---

## PR #98 — Update ARCHITECTURE.md's stale caching claims
**2026-07-22** · [#98](https://github.com/baileyrd/rusty_provider/pull/98)

- **Fixed:** `ARCHITECTURE.md` (added by an earlier repo-config pass)
  predated the opt-in response cache from #65/#86 and still listed *"no
  response cache today"* as a non-goal. Now documents the cache in the
  `rp-router` structure bullet and the dispatch data-flow step (a hit
  is checked first and skips chain resolution/dispatch/usage-recording
  entirely), and narrows the non-goal to what's still true: exact-match
  only, no semantic/fuzzy matching.

---

## PR #96 — Add cargo-audit CI job; drop prometheus's unused protobuf feature
**2026-07-22** · [#96](https://github.com/baileyrd/rusty_provider/pull/96)

- **Added:** `.github/workflows/audit.yml` runs `cargo audit` against
  `Cargo.lock` on every push/PR touching a `Cargo.toml`/`Cargo.lock`,
  plus daily on a schedule — a newly published advisory against an
  already-pinned dependency can't go unnoticed between pushes.
- **Fixed:** a real, live advisory —
  [RUSTSEC-2024-0437](https://rustsec.org/advisories/RUSTSEC-2024-0437)
  (uncontrolled recursion, crash) in `protobuf` 2.28.0, pulled in
  transitively via `prometheus`'s default `protobuf` feature. Only
  `TextEncoder` (Prometheus text-exposition format) is ever used here,
  never the protobuf wire format, so `prometheus` now builds with
  `default-features = false` — drops the dependency (and the advisory)
  entirely, no functional change.

---

## PR #94 — Add MIT LICENSE file
**2026-07-22** · [#94](https://github.com/baileyrd/rusty_provider/pull/94)

- **Fixed:** `Cargo.toml` has declared `license = "MIT"` since the
  workspace's first commit, but the license text itself was never
  reproduced anywhere in the repo — a real compliance gap for anyone
  consuming the crate or repo. Adds a standard MIT `LICENSE` file and a
  README "License" section linking to it.

---

## PR #92 — Add multi-stage Dockerfile for container deployment
**2026-07-22** · [#92](https://github.com/baileyrd/rusty_provider/pull/92)

- **Added:** a multi-stage `Dockerfile` (+ `.dockerignore`) producing a
  slim `debian:bookworm-slim` runtime image for `rp-server`. Uses
  `cargo-chef`, built from the official `rust:1-bookworm` image, to split
  dependency compilation from the workspace's own source, so a
  source-only edit doesn't force `ring`/`rusqlite`/`tokio-postgres` and
  the rest of the dependency graph to recompile.
- Runtime installs `ca-certificates` explicitly, since
  `rustls-native-certs` (outbound provider TLS, and an optional
  TLS-enabled `[persistence]` Postgres connection) reads the OS trust
  store at runtime, not just at build time. Runs as a non-root user;
  ships a `HEALTHCHECK` against `/health`.
- Nothing secret is baked in — `config.toml` and provider API keys are
  supplied at `docker run` time (bind-mount + env vars), documented in a
  new README "Docker" section.
- Added a `docker build` CI job for ongoing verification.

---

## PR #90 — Add GET /ready readiness check, distinct from /health
**2026-07-22** · [#90](https://github.com/baileyrd/rusty_provider/pull/90)

- **Added:** `GET /ready`, distinct from the existing `GET /health`.
  `/health` stays a cheap, unauthenticated liveness check that never
  touches anything external. `/ready` actually confirms the router can
  serve traffic: when `[persistence]` is configured, a trivial round
  trip against its database, returning `503` with a reason if that
  fails. Without `[persistence]` there's nothing external to check, so
  `/ready` is always `200`, same as `/health`.
- Point an orchestrator's readiness probe at `/ready` and its liveness
  probe at `/health` — a `503` from `/ready` should pull an instance out
  of rotation without restarting it, since the process itself is fine.
- No new config knobs; reuses the existing `[persistence]` section.

---

## PR #88 — Add configurable request body size limit
**2026-07-22** · [#88](https://github.com/baileyrd/rusty_provider/pull/88)

- **Added:** `server.max_body_bytes`, applied as a `DefaultBodyLimit`
  layer over the whole router, defaulting to 20 MiB. Rejected requests
  get `413 Payload Too Large` before a handler ever parses the body.
- **Fixed:** axum's `Json`/`Bytes` extractors already enforced an
  implicit 2 MB body limit even without this config, but that ceiling
  was neither explicit nor operator-configurable, and was tight enough
  to reject a legitimate multimodal request — an inline
  base64-encoded image, audio clip, or PDF adds ~33% overhead over the
  original file's size. `max_body_bytes` replaces that implicit
  ceiling rather than adding a second one on top of it.
- Applies globally, not only to `/v1/chat/completions`.

---

## PR #86 — Add opt-in response cache for identical requests
**2026-07-22** · [#86](https://github.com/baileyrd/rusty_provider/pull/86)

- **Added:** `[cache]`, an opt-in, in-memory, exact-match cache of
  non-streaming `/v1/chat/completions` responses, keyed by a hash of the
  entire incoming request. Fully off (no overhead) unless `[cache]` is
  configured. Entries expire after `ttl_secs` (default 300) and the
  cache holds at most `max_entries` (default 1000), evicting the
  oldest entry once over capacity — the same eviction strategy
  `GET /v1/generation?id=` already uses.
- **Known limitation:** exact-match only, no semantic/fuzzy matching —
  any difference in the request (model, messages, sampling parameters,
  provider preferences) is a cache miss.
- **Known limitation:** streaming requests always bypass the cache in
  both directions; caching a replayed SSE chunk sequence is left for a
  future version.
- A cache hit skips dispatch to the provider and skips re-recording
  usage/cost/latency/throughput/generation-cache bookkeeping for that
  request, since it already ran once when the response was first
  computed — this keeps `/v1/usage` and `/metrics` from double-counting
  a single real generation. New `rusty_provider_cache_lookups_total`
  Prometheus counter, labeled `hit`/`miss`.
- Not the same thing as `cache_read_per_million`/`cache_write_per_million`
  or `cache_control` (see [Prompt caching](README.md#prompt-caching)),
  which price a provider's own prompt-cache discount rather than a
  router-side response cache.
- 18 new unit tests across `rp-router` (`cache.rs`, `config.rs`,
  `metrics.rs`, and `dispatch`-level cache hit/miss/bypass behavior).

---

## PR #84 — Add POST /v1/embeddings endpoint
**2026-07-21** · [#84](https://github.com/baileyrd/rusty_provider/pull/84)

- **Added:** `POST /v1/embeddings`, OpenAI-compatible request/response
  shape. Implemented by the OpenAI-compatible adapter (direct
  passthrough) and Gemini (via `batchEmbedContents`, used even for a
  single input to avoid a second wire shape). `Router::embeddings`
  reuses `dispatch`'s chain-resolution and retryable-error fallback.
- **Known limitation:** Anthropic has no embeddings API at all, so it
  always returns a retryable `UnsupportedFeature` error — a chain
  naming it alongside a real embeddings provider falls through rather
  than failing, but it can never itself serve an embeddings request.
- **Known limitation:** none of `[[presets]]`, `[[guardrails]]`,
  `[moderation]`, `[web_search]`, or spend budgets apply to this
  endpoint yet — only auth and inbound rate-limiting, same as
  `/v1/chat/completions`'s auth layer. Cost/latency/throughput
  tracking also don't apply, since there's no established pricing
  shape for a prompt-only, no-completion-tokens request yet.
- 20 new/updated unit and integration tests across `rp-core`,
  `rp-providers`, `rp-router`, and `rp-server`; full suite passing.

## PR #83 — Add standard governance/docs scaffold
**2026-07-21** · [#83](https://github.com/baileyrd/rusty_provider/pull/83)

- **Added:** `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `ARCHITECTURE.md`, `CHANGELOG.md`, this file, PR/issue templates, and
  an ADR log seed (`docs/adr/0001-template.md`), via the repo-config
  skill. `ARCHITECTURE.md`'s boundary table and structure sections are
  filled in for real (the `Provider` trait, the 3 adapters, the
  persistence backend port, the request-dispatch data flow), not left
  as scaffold.
- **Known limitation:** the skill's default CI template
  (`ci-rust.yml`) was dropped rather than added, since this repo
  already has a working `.github/workflows/ci.yml` (fmt/clippy/test
  plus a Postgres service for the `[persistence]` backend's tests) —
  adding a second, less-tailored "CI" workflow would have run
  redundant, weaker checks on every push.
