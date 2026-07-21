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
