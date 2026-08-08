# rusty_provider

[![CI](https://github.com/baileyrd/rusty_provider/actions/workflows/ci.yml/badge.svg)](https://github.com/baileyrd/rusty_provider/actions/workflows/ci.yml)

A Rust AI provider router: one OpenAI-compatible HTTP API in front of
OpenAI, Anthropic, Gemini, Groq, Together AI, and Fireworks, with
config-driven fallback chains across providers. Point any existing OpenAI
SDK/client at it.

## Layout

- `crates/core` (`rp-core`) — unified request/response types (OpenAI chat
  completions shape) and the `Provider` trait every adapter implements.
- `crates/providers` (`rp-providers`) — adapters:
  - `OpenAiCompatibleProvider` — OpenAI, Groq, Together, Fireworks (same
    `/chat/completions` wire format, different base URL/key).
  - `AnthropicProvider` — Messages API (`/v1/messages`).
  - `GeminiProvider` — `generateContent` / `streamGenerateContent`.
- `crates/router` (`rp-router`) — TOML config loading and the `Router`
  that resolves a model string to a provider (or a named fallback chain)
  and dispatches, retrying the next candidate on rate limits, timeouts,
  and 5xx errors. A momentary blip (timeout, network error, `5xx`) gets
  one same-provider retry with a short fixed backoff before the chain
  moves on — a rate limit or a structural mismatch (unsupported content/
  feature) moves on immediately instead, since retrying the same
  candidate can't fix either.
- `crates/server` (`rp-server`) — the axum HTTP server exposing the
  OpenAI-compatible API.
- `crates/mcp` (`rp-mcp`) — MCP (Model Context Protocol) support, built on
  [`rusty_mcp`](https://github.com/baileyrd/rusty_mcp): rusty_provider's own
  routing exposed as MCP tools, plus a gateway proxying other MCP servers'
  tools through the same endpoint. See [MCP](#mcp-model-context-protocol)
  below.

## Running

```sh
cp config.example.toml config.toml   # gitignored — edit routes/providers here
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
# any provider whose env var isn't set is skipped at startup (with a warning)

cargo run -p rp-server
```

The server listens on `server.host:server.port` from `config.toml`
(default `0.0.0.0:8080`). Set `server.api_key_env` in the config to require
clients to send `Authorization: Bearer <token>`.

### Docker

```sh
docker build -t rusty_provider .
docker run -p 8080:8080 \
  -v "$(pwd)/config.toml:/app/config.toml:ro" \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  rusty_provider
```

The image is a multi-stage build (`cargo-chef` for dependency-layer
caching, so a source-only change doesn't force every dependency to
recompile) producing a slim `debian:bookworm-slim` runtime image running
as a non-root user. Nothing secret is baked in — mount your own
`config.toml` (or point `CONFIG_PATH` elsewhere) and pass provider API
keys as env vars, same as running it directly with `cargo run`. Point an
orchestrator's liveness/readiness probes at [`/health` and
`/ready`](#get-health) respectively; a `HEALTHCHECK` hitting `/health` is
also baked into the image for Compose/Fly/Railway-style deployments that
check container health directly rather than via an external prober.

### Operator CLI

`rp-cli` is a small, synchronous companion binary for checking a config
before you deploy it and, optionally, pointing a third-party CLI tool at
your running instance. Its `config`/`providers`/`keys` commands never make
a network call and never print a resolved secret's value, only whether its
env var is set:

```sh
cargo run -p rp-cli -- config check --path config.toml
cargo run -p rp-cli -- providers list --path config.toml
cargo run -p rp-cli -- keys check --path config.toml
```

- **`config check`** — parses `config.toml` (the exact same `Config` type
  and TOML schema `rp-server` loads at startup, so there's nothing for
  this to drift out of sync with), then reports provider/route/client
  counts, which providers will actually activate vs. get skipped (and
  why), any invalid `[[guardrails]]` regex, and whether persistence/the
  admin API are configured.
- **`providers list`** — every `[providers.*]` entry with its resolved
  status (`active` or `skipped (X not set)`).
- **`keys check`** — every `*_env` field this config references (provider/
  client keys, `server.api_key_env`/`admin_key_env`, and any configured
  `[persistence]`/`[webhook]`/`[moderation]`/`[web_search]` credential),
  each marked `set`/`NOT SET` — never the actual value.

`--path` defaults to `config.toml` in the current directory. Not built
into the Docker image described above (`rp-server` only) — run it from a
checkout, or `cargo install --path crates/cli` for a standalone binary.

#### `setup` — point a third-party CLI tool at rusty_provider

Several terminal AI coding agents ([opencode](https://opencode.ai),
[Crush](https://charm.land/crush), and others) already read their
provider/endpoint settings from a local JSON or TOML file. `rp-cli setup`
rewrites just that field, in place, to point the tool at a running
rusty_provider instance — see
[ADR-0004](./docs/adr/0004-cli-target-config-rewriting.md) for why this is
scoped to static file rewriting (no proxy, no traffic interception, no
trust-store changes):

```sh
cargo run -p rp-cli -- setup list
cargo run -p rp-cli -- setup show opencode --api-key-env RUSTY_PROVIDER_KEY
cargo run -p rp-cli -- setup apply opencode --api-key-env RUSTY_PROVIDER_KEY --yes
```

- **`setup list`** — every known target with its config file path and
  whether that file currently exists.
- **`setup show <name>`** — a dry run: prints the file that would be
  written (merged with whatever's already there), without touching disk.
  Always run this before `apply`.
- **`setup apply <name> --yes`** — writes it for real. Requires `--yes`.
  Backs up the previous file to `<path>.bak` first (skipped only if the
  file didn't exist). Merges into the existing file rather than
  overwriting it — unrelated keys are kept, and it refuses rather than
  clobbers if an existing value at that path isn't an object/table.
- `--base-url` defaults to `http://localhost:8080/v1` (rp-server's own
  documented default `server.host:server.port`); `--config-path` overrides
  a target's default config file location; `--targets <path>` replaces the
  built-in target list (`crates/cli/cli_targets.toml`) with your own —
  useful for a tool not listed here, or a config schema that's drifted
  from what's shipped.
- **Never writes a literal API key.** A target whose config format
  supports an env-var-reference syntax (opencode's `{env:VAR}`, Crush's
  `$VAR`) gets that syntax naming whatever variable `--api-key-env` names
  — `rp-cli` never reads the variable's actual value to do this. Omit
  `--api-key-env` and that field is skipped entirely (reported in the
  output) rather than left half-configured.

## API

### `POST /v1/chat/completions`

Same request/response shape as OpenAI's chat completions endpoint.
`model` is either:

- `"provider/model"` to address one provider directly, e.g.
  `"anthropic/claude-sonnet-5"`, `"openai/gpt-4o"`, `"groq/llama-3.3-70b-versatile"`.
- a route alias defined under `[[routes]]` in the config, e.g. `"smart"` —
  the router tries each entry in that chain in order and falls back on
  retryable errors.

A request can also send its own ad-hoc fallback list with `models` (each
entry a `"provider/model"` string), à la OpenRouter's `models` field,
instead of relying on an operator-predefined `[[routes]]` alias:

```jsonc
{
  "model": "anthropic/claude-sonnet-5",
  "models": ["openai/gpt-4o", "groq/llama-3.3-70b-versatile"],
  "messages": [{"role": "user", "content": "Say hi in one word."}]
}
```

`model` is tried first, then each of `models` in order on a retryable
error — same fallback behavior as a configured route alias, just
assembled by the client for this one request. A non-empty `models`
entirely bypasses `[[routes]]` alias lookup, so `model` must itself be a
direct `"provider/model"` here, not an alias.

Every response (streaming and non-streaming alike) carries two headers
identifying what actually happened, without a separate
`GET /v1/generation?id=` round trip:

- `X-RP-Decision: strategy=<direct|fallback|fusion>; provider=<name>;
  model=<name>; latency_ms=<n>` — `strategy` is `"direct"` for a literal
  `"provider/model"` request, `"fallback"` for a route alias or the
  request's own `models` list, `"fusion"` when `strategy = "fusion"`
  actually engaged (see [Fusion routing](#fusion-routing)). `provider`/
  `model` reflect the candidate that actually served the request, not
  necessarily the first (or only) one named — a route alias or fallback
  chain that fell through to its second entry reports that entry, not the
  alias itself. `latency_ms` covers this whole call, including any
  same-provider retries and fallen-through candidates.
- `X-RP-Fallback-Attempts` — how many chain candidates the router had to
  move through before landing on the one that served the request (`1` if
  the first one tried succeeded outright). For `strategy = "fusion"` this
  is the panel size instead, since every candidate is dispatched
  concurrently rather than tried in sequence. `0` on a response cache hit
  — nothing was actually dispatched that time.

For a streaming request, both headers are set on the initial HTTP
response (the winning candidate is already known before the first chunk
is produced), not as SSE trailers.

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart",
    "messages": [{"role": "user", "content": "Say hi in one word."}]
  }'
```

Set `"stream": true` for a server-sent-events stream of OpenAI-style
chunks (`data: {...}\n\n`, terminated by `data: [DONE]\n\n`). Fallback
happens before the first byte is streamed to the client; once a provider's
stream has started, a mid-stream failure ends the SSE connection rather
than silently switching providers.

Tool/function calling is supported: pass `tools` (OpenAI's function-calling
shape) and optionally `tool_choice` in the request; the router translates
them into each provider's own tool-use convention (Anthropic's `tool_use`/
`tool_result` content blocks, Gemini's `functionCall`/`functionResponse`
parts) and translates `tool_calls` back into the OpenAI shape in the
response — both streamed and non-streamed.

A message's `content` can be either a plain string or an array of typed
parts, matching OpenAI's multimodal shape, so a user turn can attach one
or more images or audio clips alongside text:

```jsonc
{
  "model": "smart",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image, what's said in this clip, and what does this document say?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
      // or a base64-encoded image inline:
      // {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}}
      {"type": "input_audio", "input_audio": {"data": "UklGRi4...", "format": "wav"}},
      {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0...", "filename": "report.pdf"}}
      // or a remote PDF, same as image_url:
      // {"type": "file", "file": {"file_data": "https://example.com/report.pdf"}}
    ]
  }]
}
```

The router translates these into each provider's own format for
`Role::User` messages:

- `image_url`: Anthropic's `image` content block, Gemini's `inlineData`/
  `fileData` parts. A `data:<mime>;base64,<data>` URI is passed through as
  inline base64; a plain `https://` URL is passed through as a remote
  reference (Gemini additionally needs a MIME type for this case, which
  is guessed from the URL's extension, defaulting to `image/jpeg`).
- `input_audio`: Gemini's `inlineData` (its MIME type is `audio/<format>`,
  e.g. `audio/wav` or `audio/mp3` — Gemini's accepted audio types happen
  to match the `format` string directly, so no guessing is needed the way
  image URLs require). **Anthropic's Messages API has no audio-input
  support at all**, so a user message containing `input_audio` sent to
  Anthropic fails with a retryable error instead of silently dropping the
  audio — if it's part of a `[[routes]]` fallback chain, the router moves
  on to the next candidate rather than failing the whole request; if
  Anthropic is the only (or last) candidate, the request fails with `400`.
- `file` (PDF ingestion): Anthropic's native `document` content block,
  Gemini's `inlineData`/`fileData` parts — the exact same
  base64-vs-remote-reference split as `image_url` (Gemini's MIME guess
  here defaults to `application/pdf` instead of an image type). `filename`
  is carried through to OpenAI-compatible verbatim but has no equivalent
  on Anthropic/Gemini's native document/file parts, so it's dropped in
  translation there. There's no parsing-engine selection (native-text vs.
  OCR) — every provider uses whatever its own API defaults to; this
  router has no PDF-processing pipeline of its own to pick one.

System, assistant, and tool messages only ever send their plain text to a
provider — image, audio, and file parts in a non-user role are silently
dropped rather than translated, since none of the three providers accept
any of those modalities there. `OpenAiCompatibleProvider` needs no
translation for any of the three content types — all pass straight
through, since this router's wire shape already matches OpenAI's.

A request can constrain the model's output shape with `response_format`,
matching the OpenAI convention:

```jsonc
{
  "model": "smart",
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather_report",
      "schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "temperature_f": {"type": "number"}
        },
        "required": ["city", "temperature_f"]
      },
      "strict": true
    }
  },
  "messages": [{"role": "user", "content": "What's the weather in Boston?"}]
}
```

`"type"` is one of:

- `"text"` (the default) — unconstrained free-form output.
- `"json_object"` — loose JSON mode: the model must emit syntactically
  valid JSON, with no particular shape enforced.
- `"json_schema"` — strict schema-constrained JSON, validated against
  `json_schema.schema`.

Per-provider support:

- **OpenAI-compatible** needs no translation — `response_format` matches
  the wire shape already and passes straight through.
- **Gemini** has native support for both variants via
  `generationConfig.responseMimeType`/`responseSchema`; Gemini's schema
  dialect is a subset of OpenAPI 3.0 Schema, close enough to plain JSON
  Schema for typical use but not a perfect match for every keyword.
- **Anthropic** has no native `response_format`. `"json_schema"` is
  emulated by defining a single synthetic tool from `json_schema.schema`,
  forcing the model to call it (`tool_choice`), and unwrapping that tool
  call back into plain JSON content in the response — transparent to the
  client either way, streamed or not. `"json_object"` has no equivalent
  trick (there's no schema to build a tool from, and nothing in the API
  reliably constrains output to "valid JSON, any shape"), so it fails with
  a retryable error instead: a `[[routes]]` fallback chain moves on to a
  provider that actually supports it, and a direct `"anthropic/..."`
  request fails with `400`.

A request can ask a reasoning-capable model to think before answering with
`reasoning`:

```jsonc
{
  "model": "smart",
  "reasoning": {
    "effort": "high",     // "low" / "medium" / "high" -- how much to think
    "max_tokens": 8000,   // or an explicit thinking-token budget instead of effort
    "exclude": false      // true: still think, but don't return the reasoning text
  },
  "messages": [{"role": "user", "content": "..."}]
}
```

Both `effort` and `max_tokens` are optional and mutually exclusive in
effect — `max_tokens` wins if both are set. With neither set, requesting
`reasoning` at all still turns thinking on, using `medium`'s effort
mapping. The response's `message.reasoning` (or, streamed, each chunk's
`delta.reasoning`) carries the model's reasoning as plain text, separate
from the answer in `content` — `None`/absent when there's nothing to show
(no `reasoning` requested, `exclude: true`, or the model returned none).
This is a plain-text summary, not full fidelity: providers with richer
structure (e.g. Anthropic's signed, replayable thinking blocks) don't
round-trip that structure back into a follow-up request the way their own
native SDKs would.

Per-provider translation:

- **Gemini** has native support via `generationConfig.thinkingConfig`
  (`thinkingBudget` / `includeThoughts`). Response parts Gemini marks
  `thought: true` are collected into `reasoning` instead of `content`.
- **Anthropic** has native support via extended thinking
  (`"thinking": {"type": "enabled", "budget_tokens": N}`). Anthropic
  requires `budget_tokens >= 1024` and `max_tokens > budget_tokens`; both
  are enforced automatically (the budget is floored to 1024, and
  `max_tokens` is raised if needed) so a low-effort or unset-`max_tokens`
  request never gets rejected by the upstream API. Anthropic has no
  server-side way to suppress `thinking` blocks the way Gemini's
  `includeThoughts` does, so `exclude: true` is enforced client-side —
  the model still thinks (and is still billed for it), the text is just
  dropped before it reaches the response.
- **OpenAI-compatible** sends the widely-adopted `reasoning_effort` field
  and parses `message.reasoning_content` / `delta.reasoning_content` from
  the response — the convention used across DeepSeek, Groq, and most other
  OpenAI-compatible reasoning models. `effort` maps straight through;
  `max_tokens` has no equivalent on this wire format and is ignored.

#### Reasoning replay for tool-continuation turns

Some OpenAI-compatible reasoning models (DeepSeek-reasoner, Kimi-K-series,
QwQ, GLM-thinking, and similar) reject a follow-up turn that answers a
tool call if that turn's assistant message is missing the
`reasoning_content` behind the decision to call it — but most client
SDKs strip `reasoning` before sending the next request, since it's meant
to be read, not replayed. This router closes that gap itself: whenever a
non-streaming response comes back with both `tool_calls` and `reasoning`,
it caches the reasoning trace in memory against every `tool_calls[].id`
in that turn. The next request answering one of those tool calls — even
with `reasoning` stripped, exactly as a typical client sends it — gets
the cached trace transparently re-injected into the matching assistant
message before dispatch, and the OpenAI-compatible adapter sends it under
the `reasoning_content` key these models expect. A message that already
carries its own `reasoning` is left alone; a tool call with no cached
entry (never seen, evicted, or from a provider that never returned
reasoning) is a no-op, same as today. The cache is bounded (most recent
1000 tool calls, oldest evicted first) and populated only from
non-streaming responses — a streaming response's `tool_calls`/`reasoning`
arrive as incremental deltas this router doesn't reassemble elsewhere
either, so this doesn't (yet) populate the cache; replay itself still
applies to a streaming request if an earlier non-streaming turn already
populated the entry it needs.

### Prompt caching

A message can mark itself as the end of a cacheable prefix with
`cache_control`, matching Anthropic's own breakpoint shape:

```jsonc
{
  "model": "smart",
  "messages": [
    {"role": "system", "content": "... a long, reused system prompt ...", "cache_control": {"type": "ephemeral"}},
    {"role": "user", "content": "What's the weather in Boston?"}
  ]
}
```

- **Anthropic** is the only provider with an explicit cache-breakpoint API,
  so this is a direct, mostly-untranslated passthrough: the marked
  message's last content block gets Anthropic's
  `"cache_control": {"type": "ephemeral"}`, and a system message with
  `cache_control` set switches `system` from a plain string to a block
  array (only the block form can carry a breakpoint) — every other request
  keeps the plain-string `system` shape exactly as before. Anthropic's
  response usage separately reports `cache_creation_input_tokens` (tokens
  newly written to the cache, billed at a premium) and
  `cache_read_input_tokens` (tokens served from it, billed at a steep
  discount) on top of its already-non-cached `input_tokens` — this router
  folds all three into a single cache-inclusive `usage.prompt_tokens`,
  matching how OpenAI and Gemini already report theirs, and surfaces the
  breakdown separately.
- **OpenAI-compatible** and **Gemini** cache automatically server-side, with
  no request-side marker — `cache_control` is silently a no-op there rather
  than an error, since it's an optimization hint, not a correctness
  requirement, and both still answer correctly without it.

Every response's `usage` may include `cached_tokens` (prompt tokens served
from a cache) and `cache_creation_tokens` (prompt tokens newly written to
one, Anthropic only) — both a breakdown of `prompt_tokens`, not additive on
top of it, and both absent (not `0`) when the provider reports no cache
accounting or nothing was cached. `[[pricing]]` entries can price these
separately with `cache_read_per_million`/`cache_write_per_million`
(defaulting to `prompt_per_million`, i.e. no assumed discount, when unset)
so `cost_usd` reflects the actual cache economics instead of pricing every
prompt token at the full rate. The cumulative totals at `GET /v1/usage` and
`GET /metrics`, and the SQLite/Postgres persistence layer, still only track
`prompt_tokens`/`completion_tokens`/`cost_usd` — the cache breakdown is
per-response only, not accumulated.

If `[[pricing]]` has an entry for the model that actually served the
request, the response (and, for streaming, whichever chunk carries the
final `usage`) includes an extra `cost_usd` field — the request's
estimated dollar cost, computed from `usage.prompt_tokens` /
`usage.completion_tokens` (split into fresh/cached/cache-write portions
when the response reports any caching) against that pricing entry. It's
not part of the OpenAI schema, so existing OpenAI SDKs/clients just ignore
it; it's simply absent (not `0`/`null`) when the model has no configured
pricing, so don't read a missing field as "this was free." Every request
also adds to a running per-model total queryable at `GET /v1/usage`
(below), whether or not pricing is configured for it.

### Context compression

By default, a request whose messages don't fit the target model's context
window just fails at the provider. `transforms: ["middle-out"]` opts into
automatic truncation instead:

```jsonc
{
  "model": "smart",
  "transforms": ["middle-out"],
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, "..."]
}
```

If the resolved candidate has a `context_length` set in its `[[pricing]]`
entry (see `GET /v1/models`) and the request's messages are estimated to
exceed it, messages are dropped from the middle of the conversation
(oldest-first among the middle) until it's estimated to fit — the first
message (typically `system`) and the most recent one are always kept
intact, since both ends carry the most load-bearing context. The budget
reserves room for the response using `max_tokens` (or a default of `4096`
when unset), and "estimated to fit" is a crude, tokenizer-free heuristic
(`chars / 4`) rather than each provider's actual tokenizer — good enough
for a rough "will this fit" call, not an exact accounting. Truncation is
evaluated per fallback-chain candidate (since different models can have
different `context_length`s), so a request might get truncated for one
candidate but sent unmodified to another. Without a `context_length` on
record for the candidate, or without `transforms` set at all, the request
goes out unmodified — same as today.

`transforms: ["rtk"]` opts into a different kind of compression, aimed at
tool-call-heavy coding-agent sessions rather than raw message count:

```jsonc
{
  "model": "smart",
  "transforms": ["rtk"],
  "messages": [
    {"role": "user", "content": "run the tests"},
    {"role": "assistant", "tool_calls": ["..."]},
    {"role": "tool", "tool_call_id": "...", "content": "running 500 tests\ntest a::b ... ok\n... (huge)"}
  ]
}
```

Every `role: "tool"` message's text is stripped of ANSI escape codes and
run through a built-in filter, chosen by sniffing the content (not the
originating command, which this router never sees): `git` (status/diff
output — collapses long runs of repeated file-status lines), `test`
(cargo/pytest/jest-shaped output — collapses consecutive passing-test
lines while always keeping failures and the summary line), `build`
(compiler output — collapses repeated `Compiling`-style lines while always
keeping `warning:`/`error:` lines), `package` (npm/pip-shaped install
output — keeps only summary lines like `added N packages`), and a generic
fallback (deduplicates repeated lines, then keeps the first/last 40 lines
of anything still very long). Every category leaves short/already-compact
output untouched. System/user/assistant messages are never touched — only
`role: "tool"` content.

`"rtk"` and `"middle-out"` compose when both are set: `"rtk"` runs first
(so tool messages are already shrunk before `"middle-out"`'s token-budget
estimate runs against them), then `"middle-out"` drops whole messages if
the request is still over budget after that. This is a fixed built-in
5-category catalog, not OmniRoute's TOML-configurable, 49-filter one — see
`crates/router/src/rtk.rs` if you need a category it doesn't cover yet.

### Sampling parameters

Beyond `temperature`, `top_p`, `max_tokens`, and `stop`, a request can set a
fuller sampling-parameter surface:

```jsonc
{
  "model": "smart",
  "top_k": 40,
  "min_p": 0.05,
  "top_a": 0.2,
  "frequency_penalty": 0.3,
  "presence_penalty": 0.4,
  "repetition_penalty": 1.1,
  "logit_bias": {"1234": -100},
  "seed": 42,
  "messages": [{"role": "user", "content": "..."}]
}
```

Each field is native to some providers and absent from others' own APIs; an
unsupported field is silently dropped rather than erroring, on the same
reasoning as `cache_control` above — these are sampling hints, not a
structural contract like `response_format`, so the request still produces a
valid response either way:

| Field | Anthropic | Gemini | OpenAI-compatible |
| --- | --- | --- | --- |
| `top_k` | native | native | passthrough¹ |
| `min_p` | ignored | ignored | passthrough¹ |
| `top_a` | ignored | ignored | passthrough¹ |
| `frequency_penalty` | ignored | native | native |
| `presence_penalty` | ignored | native | native |
| `repetition_penalty` | ignored | ignored | passthrough¹ |
| `logit_bias` | ignored | ignored | native |
| `seed` | ignored | native | native |

¹ Not part of OpenAI's own API, but common on OpenAI-compatible inference
servers (Groq, Together, Fireworks, vLLM, etc.), so the OpenAI-compatible
adapter passes these through unconditionally rather than guessing which
backend supports them.

A request can also constrain and order the resolved fallback chain with a
`provider` field, independent of whether `model` was a direct
`"provider/model"` or a route alias:

```jsonc
{
  "model": "smart",
  "provider": {
    "only": ["anthropic", "openai"],   // drop every other candidate in the chain
    "ignore": ["openai"],              // and then drop these too
    "zdr": true,                       // then drop any provider not marked zdr in config
    "data_collection": true,           // then drop any provider not marked no_training in config
    "max_price": 5.0,                  // then drop anything pricier than $5/M prompt tokens
    "require_parameters": true,        // then drop anything that can't honor every field set below
    "sort": "price"                    // or "latency" / "throughput" — sort what's left
  },
  "messages": [{"role": "user", "content": "..."}]
}
```

- `only` / `ignore` take provider names matching your `[providers.*]` config
  keys (e.g. `"anthropic"`, `"groq"`) — `only` is applied first, then
  `ignore`. If nothing survives, the request fails fast with `400` rather
  than silently falling through to an unfiltered chain.
- `zdr: true` drops any provider not marked `zdr = true` in
  `[providers.*]` config. That flag is self-declared by the operator —
  the router trusts it and never verifies it against the provider, so it's
  only as accurate as your own config.
- `data_collection: true` drops any provider not marked `no_training = true`
  in `[providers.*]` config. This is a separate axis from `zdr`, not an
  alias for it: `zdr` is about data *retention* (does the provider keep
  your data at all), `data_collection` is about *training* (if they keep
  it, do they learn from it). A provider can satisfy one without the
  other — set either, both, or neither depending on what your compliance
  requirements actually need. Same self-declared, unverified trust model
  as `zdr`.
- `max_price` drops any candidate priced above it, in USD per million
  prompt tokens — the same `prompt_per_million` figure `sort: "price"`
  reads from `[[pricing]]`. Unlike `sort: "price"`, this is a hard
  ceiling enforced *before* dispatch, not an after-the-fact ranking, and a
  candidate with no configured price is dropped along with everything
  above the ceiling — with a cap in effect, an unpriced entry can't be
  trusted to be under it.
- `require_parameters: true` drops any candidate whose provider adapter
  can't actually honor every field *this specific request* sets —
  `tools`, `response_format`, `top_k`, a message's `cache_control`, and
  so on (see `GET /v1/models`'s `supported_params` for the exact
  per-provider list). `temperature`/`top_p`/`max_tokens`/`stop` never
  disqualify a candidate, since every provider kind supports all four.
  Without this, an unsupported field is either silently dropped (most
  sampling params) or rejected only after a wasted round trip
  (`response_format`'s `"json_object"` on Anthropic); this filters those
  candidates out before dispatch instead of finding out the hard way.
- `sort: "price"` stable-sorts the remaining candidates ascending by the
  prompt-token price configured in `[[pricing]]` (see `config.example.toml`)
  — entries with no configured price sort last, keeping their relative
  order. This is a static, operator-maintained price table, not a live feed.
- `sort: "latency"` stable-sorts ascending by a running average (EWMA) of
  this router's own observed response time per "provider/model", measured
  from request-sent to response-received (time-to-first-byte for streaming
  requests, full round-trip for non-streaming). This needs no config —
  it's built up automatically from real traffic — but it's in-memory only
  (resets on restart) and per-process, not a shared/global feed; a
  "provider/model" this router hasn't successfully called yet sorts last.
- `sort: "throughput"` sorts descending (fastest generation first) by a
  running average (EWMA) of observed completion tokens/sec. For streaming
  requests this is measured from when the request was sent to whichever
  chunk carries the final `usage.completion_tokens` — the router
  instruments the stream in flight rather than reading it itself, since it
  hands streamed responses straight to the HTTP layer. Same caveats as
  `"latency"`: no config needed, in-memory only, per-process; an
  unobserved "provider/model" sorts last.
- `sort: "uptime"` sorts descending (most reliable first) by a running
  average (EWMA) of this router's own observed success rate per
  "provider/model" — `1.0` recorded for a successful attempt, `0.0` for a
  failed one (retryable or fatal), sampled only on an actual dispatch
  attempt against that provider, not a candidate skipped locally (e.g. by
  this router's own outbound rate limit). Same caveats as `"latency"`/
  `"throughput"`: no config needed, in-memory only, per-process; an
  unobserved "provider/model" sorts last rather than being assumed
  healthy. This is a deterministic ranking, not weighted-random load
  balancing across "healthy" candidates — every request still tries the
  sorted chain in order with fallback, the same as any other `sort` value.

  Automatic, always on regardless of `sort` (except `sort: "uptime"` itself,
  which already covers this more thoroughly): a candidate whose observed
  EWMA success rate has dropped below `0.5` is moved after every other
  candidate as a final pass, on top of whatever ordering (chain config
  order, or another `sort`) already applied. This is a stable partition,
  not a ranking — deprioritized candidates keep their relative order among
  themselves, and everyone else keeps theirs. An unobserved candidate is
  never deprioritized this way (optimistic until this router has actually
  seen it fail), unlike `sort: "uptime"`'s own convention of sorting an
  unobserved entry last too.
- `sort: "quality"` sorts descending by an operator-declared
  `quality_score` on `[[pricing]]` — an arbitrary scale you define
  yourself (nothing here measures model quality), unranked entries sort
  last, same convention as `"price"`.
- `sort: "random"` isn't a ranking at all — it shuffles the resolved chain,
  for simple load distribution across candidates with no meaningful
  ordering between them (e.g. same price, no observed latency yet).
- `sort: "free_tier_remaining"` sorts descending by remaining budget from
  [Free tiers](#free-tiers) — a "provider/model" with a `[[free_tiers]]`
  entry and headroom left sorts first, one that's exhausted (`0` left)
  sorts after every candidate with headroom, and one with no
  `[[free_tiers]]` entry at all sorts last of all.
- `max_request_price_usd` caps this one request's estimated cost, in USD
  — estimated per candidate as `max_tokens * completion_per_million`
  (from `[[pricing]]`), since `max_tokens` is the one lever a caller
  actually controls over a response's worst-case size; `max_price` above
  is a different axis (a per-million-token ceiling on individual
  candidates), not a per-request total. Only takes effect when the
  request also sets `max_tokens` — with nothing bounding completion
  length there's no worst case to estimate, so this field is silently a
  no-op otherwise. A candidate with no configured price can't be judged
  against the cap either, dropped for the same reason an unpriced
  candidate is under `max_price`.
- `budget_fallback` controls what happens once `max_request_price_usd`
  excludes at least one candidate: `"strict"` narrows the chain to just
  the candidates that fit, failing the request with `402` if none do;
  `"cheapest"` (the default) always serves the request instead — routing
  to the cheapest candidate that fits, or, if none fit, the overall
  cheapest candidate anyway rather than refusing outright. Ignored
  without `max_request_price_usd` also set.

```jsonc
{
  "model": "smart",
  "max_tokens": 1000,
  "messages": [{"role": "user", "content": "..."}],
  "provider": {
    "max_request_price_usd": 0.01,
    "budget_fallback": "strict"
  }
}
```

### Logprobs

```jsonc
{
  "model": "openai/gpt-4o-mini",
  "logprobs": true,
  "top_logprobs": 2,
  "messages": [{"role": "user", "content": "..."}]
}
```

`logprobs: true` asks the provider to return the log-probability of each
generated token alongside the response; `top_logprobs` (0-20) additionally
asks for the N most likely alternative tokens at each position. When
present, the response carries a `logprobs` field on each choice:

```jsonc
{
  "choices": [{
    "message": {"role": "assistant", "content": "..."},
    "logprobs": {
      "content": [
        {"token": "Hi", "logprob": -0.02, "bytes": [72, 105], "top_logprobs": [
          {"token": "Hi", "logprob": -0.02, "bytes": [72, 105]},
          {"token": "Hello", "logprob": -4.1, "bytes": [72, 101, 108, 108, 111]}
        ]}
      ]
    }
  }]
}
```

This is a diagnostic/eval feature, not a structural contract, so support is
native-or-nothing rather than translated:

| Provider | Behavior |
| --- | --- |
| OpenAI-compatible | native passthrough — `logprobs`/`top_logprobs` forwarded verbatim, response `logprobs` parsed straight from the wire shape |
| Anthropic | ignored — the Messages API has no logprobs equivalent; response `logprobs` is always `null` |
| Gemini | ignored — same as Anthropic; response `logprobs` is always `null` |

Mostly useful for evals and fine-tuning tooling rather than general chat
traffic, so there's no `require_parameters` exemption here: a request that
sets `logprobs` and also sets `provider.require_parameters: true` will
correctly drop Anthropic/Gemini candidates from the chain, same as any
other field they don't support.

### `POST /v1/embeddings`

Same request/response shape as OpenAI's embeddings endpoint — `model` is
"provider/model" or a `[[routes]]` alias, exactly like
`/v1/chat/completions`, and `input` accepts either a single string or a
batch (`string[]`):

```sh
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/text-embedding-3-small",
    "input": "Say hi in one word."
  }'
```

Only OpenAI-compatible backends and Gemini (via `batchEmbedContents`)
support embeddings — Anthropic has no embeddings API at all, so an
Anthropic candidate in a fallback chain always fails over to the next
entry rather than erroring the whole request, the same
`ProviderError::UnsupportedFeature` pattern used elsewhere for a
provider that can't represent part of a request. Unlike
`/v1/chat/completions`, this endpoint dispatches straight to the
resolved provider chain with plain auth and inbound rate-limiting —
none of `[[presets]]`, `[[guardrails]]`, `[moderation]`, `[web_search]`,
or spend budgets apply here, since none of those have an established
meaning for a prompt-only, no-completion-tokens request yet. `usage` is
`null` for Gemini, which reports no token count for embeddings calls at
all.

### `GET /v1/models`

Lists configured route aliases, `provider/*` for every provider with a
resolved API key, and rich metadata for every concrete "provider/model"
with a `[[pricing]]` entry:

```json
{
  "object": "list",
  "data": [
    {"id": "smart", "object": "model", "owned_by": "router-alias"},
    {"id": "anthropic/*", "object": "model", "owned_by": "anthropic"},
    {
      "id": "anthropic/claude-sonnet-5",
      "object": "model",
      "owned_by": "anthropic",
      "context_length": 200000,
      "pricing": {
        "prompt": 3.0,
        "completion": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
        "quality_score": 0.9
      },
      "supported_params": ["temperature", "top_p", "max_tokens", "..."]
    }
  ]
}
```

`context_length`/`pricing`/`supported_params` are only known for a concrete
"provider/model" with a `[[pricing]]` entry — a route alias (which can span
models with different context windows and pricing) and a `"{provider}/*"`
wildcard omit all three. `pricing` mirrors the entry's
`prompt_per_million`/`completion_per_million`/`cache_read_per_million`/
`cache_write_per_million` (cache rates already defaulted to `prompt` when
left unset in config, same as `cost_usd` computation uses), plus
`quality_score` when the entry sets one (omitted, not `null`, when unset —
see `sort: "quality"` above).
`context_length` is purely informational — not enforced against actual
request size. `supported_params` lists which `ChatRequest` fields that
model's provider adapter gives an actual effect to (native support or,
for OpenAI-compatible, unconditional passthrough) — see the
[Sampling parameters](#sampling-parameters), `response_format`, `reasoning`,
and prompt-caching sections above for what's native per provider versus a
silent no-op.

### `GET /v1/usage`

Cumulative request/token/cost totals per "provider/model", accumulated
since the process started:

```json
{
  "object": "list",
  "data": [
    {
      "model": "anthropic/claude-sonnet-5",
      "requests": 42,
      "prompt_tokens": 8190,
      "completion_tokens": 3110,
      "cost_usd": 0.071
    }
  ]
}
```

Like the latency/throughput metrics, this is in-memory only by default —
it resets on restart and isn't shared across processes, unless you
configure `[persistence]` (see below), in which case it survives restarts
and reflects every process sharing the same database. `cost_usd` only
accumulates for models with a `[[pricing]]` entry; it stays `0.0` for
everything else (which means "unpriced," not "free" — `requests` and
`*_tokens` still count normally regardless of pricing).

### `GET /v1/generation?id=`

`GET /v1/usage` is aggregate-only; this looks up one specific completed
request's own token/cost breakdown by the `id` from its
`/v1/chat/completions` response (or, for a streamed request, the `id` on
its chunks):

```json
{
  "id": "chatcmpl-abc123",
  "model": "anthropic/claude-sonnet-5",
  "created": 1700000000,
  "prompt_tokens": 8190,
  "completion_tokens": 3110,
  "total_tokens": 11300,
  "cost_usd": 0.071
}
```

`404` if `id` doesn't match a request this process has actually served.
This is a recent-history lookup, not a durable audit log: it's backed by
an in-memory, per-process cache of the last 1000 requests (oldest evicted
first once full), with no `[persistence]` backing regardless of whether
`[persistence]` is configured for `GET /v1/usage` — unlike that endpoint,
this one always resets on restart and never reflects another process's
traffic. `cost_usd` is omitted (not `0.0`) for a request to a model with
no `[[pricing]]` entry, same unpriced-means-absent convention as
`ChatResponse.cost_usd`.

### `GET /v1/providers/stats`

The same per-"provider/model" EWMA figures `sort: "latency"`/
`"throughput"`/`"uptime"` consult internally when ranking a fallback
chain, surfaced directly instead of staying a purely internal signal:

```json
{
  "object": "list",
  "data": [
    {
      "model": "anthropic/claude-sonnet-5",
      "latency_ms": 812.4,
      "throughput_tokens_per_sec": 46.2,
      "uptime": 1.0
    }
  ]
}
```

Only "provider/model" pairs this process has actually dispatched to at
least once are listed — one this process has never tried isn't included
at all, rather than included with every figure absent. Each of
`latency_ms`/`throughput_tokens_per_sec`/`uptime` is independently
omitted if this process hasn't observed that particular figure yet (e.g.
every attempt so far failed before a latency sample could be taken, but
did count toward `uptime`). Same caveats as the sorts that consume this
data: in-memory only, resets on restart, and reflects only this process's
own traffic, not a shared or global feed — behind a load balancer, each
process reports its own view.

### `GET /metrics`

The same underlying data as above, in Prometheus text exposition format
for scraping:

- `rusty_provider_dispatch_attempts_total{provider,model,outcome}` —
  counter, one increment per candidate tried in a fallback chain.
  `outcome` is `success`, `retryable_error` (fell through to the next
  candidate), `error` (fatal, chain aborted), `not_configured` (candidate
  skipped, no resolved API key), or `rate_limited` (candidate skipped,
  this router's own outbound self-throttle — see
  `[providers.X].requests_per_minute`).
- `rusty_provider_prompt_tokens_total{provider,model}` /
  `rusty_provider_completion_tokens_total{provider,model}` — counters.
- `rusty_provider_cost_usd_total{provider,model}` — counter; same
  unpriced-means-zero caveat as `GET /v1/usage`.
- `rusty_provider_response_latency_seconds{provider,model}` — histogram;
  full round-trip for non-streaming requests, time-to-first-byte for
  streaming ones.
- `rusty_provider_throughput_tokens_per_second{provider,model}` —
  histogram of observed completion-token generation rate per response.
- `rusty_provider_provider_configured{provider}` — gauge, `1`/`0`, set
  once at startup per `[providers.*]` entry.

Subject to the same `server.api_key_env` auth as every other endpoint —
if you've enabled it, point Prometheus's scrape config at it with a
bearer token:

```yaml
scrape_configs:
  - job_name: rusty_provider
    bearer_token: "your-token-here"
    static_configs:
      - targets: ["localhost:8080"]
```

### `GET /health`

Liveness check — "the process is up," nothing more. Always `200 "ok"`,
unauthenticated, and never touches a database or provider. Distinct from
`/ready` below, which actually checks whether the router can serve
traffic right now.

### `GET /ready`

Readiness check. `200 {"status": "ready"}` when this router can actually
serve traffic; `503 {"status": "not ready", "reason": "..."}` when it
can't. Today the only thing checked is `[persistence]`, if configured —
a trivial round trip confirming the database is actually reachable right
now, not just that it was reachable at startup. Without `[persistence]`
there's nothing external to check, so `/ready` and `/health` behave
identically (both always `200`).

Point an orchestrator's readiness probe (e.g. Kubernetes) at `/ready` and
its liveness probe at `/health` — a `503` from `/ready` should pull this
instance out of a load balancer's rotation without restarting it (the
process itself is fine, it just can't serve traffic right now, most
likely because `[persistence]`'s database is down); a failing `/health`
means the process itself needs restarting.

## JWT/OIDC authentication

`[jwt]` is an additional way to satisfy authentication, alongside (never
instead of) `server.api_key_env` and `[[clients]]` keys — a presented
bearer token that doesn't match a known static key is tried as a JWT
before being rejected:

```toml
[jwt]
# Simplest setup: a shared secret, no network call ever needed.
hs256_secret_env = "JWT_SECRET"

# Or, real OIDC provider integration (Auth0, Okta, Keycloak, ...):
# jwks_url = "https://your-idp.example.com/.well-known/jwks.json"
# jwks_cache_secs = 300   # optional, this is the default

issuer = "https://your-idp.example.com/"   # optional -- verified against `iss` if set
audience = "rusty-provider"                # optional -- verified against `aud` if set
client_claim = "sub"                       # optional -- maps a claim to a [[clients]] name
```

If both `hs256_secret_env` (resolved) and `jwks_url` are set, HS256 wins
(no network call needed). `jwks_url` fetches an RS256 JWKS document,
caching keys by `kid` for `jwks_cache_secs` — a token whose `kid` isn't in
the current cache triggers an immediate re-fetch rather than waiting out
the rest of the TTL, so key rotation doesn't require a restart. Neither
mode resolving (`hs256_secret_env` unset/unresolvable and no `jwks_url`)
disables JWT auth entirely at startup with a warning, the same
soft-failure pattern a misconfigured provider or moderation backend
already gets.

This is an **authentication** check, not a best-effort content check like
[Moderation](#moderation) or [Web search](#web-search) — it fails
**closed**: an expired token, a bad signature, a wrong issuer/audience, or
an unreachable JWKS endpoint all mean "not authenticated," never "let it
through anyway." The algorithm used to validate a token is always chosen
by *this router's own configured mode* (`HS256` or `RS256`), never by
trusting the token's own `alg` header — closing the classic JWT
algorithm-confusion hole.

A request with `[jwt]` configured but no `server.api_key_env`/
`[[clients]]` at all still requires *some* credential — `[jwt]` alone is
enough to turn authentication on, the same as either of those would be by
themselves. Without `client_claim` set, a JWT-authenticated caller gets
the same access a valid `server.api_key_env` token would: no per-subject
budget/spend tracking, and rate limiting falls back to the same source-IP
bucket an unmatched caller gets (see [Rate limiting](#rate-limiting)
below).

Setting `client_claim` (e.g. `"sub"`) maps a verified token's claim value
to a configured `[[clients]].name` — a match resolves the exact same
identity a static per-client API key would for the rest of that request:
the client's own budget is enforced, its usage/spend is tracked under its
name (visible via `/v1/usage`, `/v1/admin/clients/{name}/usage-history`),
and it's rate-limited under its own `client:{name}` bucket instead of the
IP fallback. No match (the claim is absent from the token, or no
`[[clients]]` entry has that name) falls back to the same unmapped
behavior above, not an error — a JWT that already passed verification
stays authenticated either way; `client_claim` only affects which
identity, if any, it resolves to. `/v1/admin/*` is unaffected by `[jwt]`
entirely, `client_claim` included — it stays `server.admin_key_env`/
admin-role `[[clients]]`' own API keys only, never a JWT, even one that
maps to an admin-role client.

## MCP (Model Context Protocol)

`[mcp]` gives rusty_provider a Model Context Protocol surface, in both
directions at once:

- **Server**: its own routing exposed as MCP tools (`chat_completion`,
  `list_models`, `embeddings`) — any MCP client can call it directly.
- **Gateway**: other MCP servers' tools proxied through the same endpoint,
  namespaced `"{upstream}/{tool}"` — one client connection point instead of
  many.

```toml
[mcp]
enabled = true
path = "/mcp"   # optional, this is the default

[[mcp.upstreams]]
name = "filesystem"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

[[mcp.upstreams]]
name = "example"
transport = "http"
url = "https://mcp.example.com/mcp"
bearer_token_env = "EXAMPLE_MCP_TOKEN"   # optional
```

The endpoint is mounted inside this same app and port, guarded by the exact
same `server.api_key_env`/`[[clients]]`/`[jwt]` auth every other route
already uses — there's no separate MCP auth model to configure. An upstream
that fails to connect at startup is logged and simply absent from the tool
list, the same soft-failure pattern `[jwt]`/`[webhook]`/`[persistence]`
already use, until `rp-server` restarts. An upstream that connects and
*later* drops is different: it's reconnected automatically with
exponential backoff (`reconnect_backoff_secs`/`reconnect_backoff_max_secs`/
`max_reconnect_attempts` under `[mcp]`) — see [docs/MCP.md](docs/MCP.md)
for the details.

For a desktop client that spawns its MCP server as a stdio subprocess
instead of talking HTTP, set `MCP_STDIO=1` when starting `rp-server` — it
serves the same combined tool set over stdio instead of binding a port.

Full walkthrough (connecting a real client, testing upstreams, the wire
protocol details) in [`docs/MCP.md`](docs/MCP.md).

## Rate limiting

Both directions are entirely opt-in — with no `[[clients]]`,
`server.default_rate_limit_rpm`, or per-provider `requests_per_minute`
configured, nothing is limited.

**Inbound** (protecting this router from its own callers): define
`[[clients]]` in config, each with its own API key and requests-per-minute
limit. Presenting a client's key both authenticates the request (in
addition to `server.api_key_env`, if set) and buckets its rate limit under
that client's name. A caller with no matching client key falls back to a
bucket keyed by source IP, limited by `server.default_rate_limit_rpm` if
set (otherwise uncapped). `POST /v1/chat/completions` and
`POST /v1/embeddings` are both rate limited this way — metadata endpoints
(`/v1/models`, `/v1/usage`, `/metrics`) aren't. Rejections return `429`
with a `Retry-After` header.

Every rate-limit-checked response — success or `429` — also carries
`X-RateLimit-Limit` (the bucket's requests-per-minute capacity),
`X-RateLimit-Remaining` (tokens left after this request; `0` on a `429`),
and `X-RateLimit-Reset` (seconds from now until the bucket is back to
full capacity, same "seconds from now" convention as `Retry-After` rather
than a Unix timestamp, since this is a continuously-refilling token
bucket, not a fixed window with a natural epoch boundary). A caller with
no matching client and no `default_rate_limit_rpm` configured gets none
of these headers at all, same as it gets no `429` — there's no bucket to
report on.

The source IP is the raw TCP peer address. Behind a reverse proxy this is
the proxy's address, not the real client's — this router doesn't parse
`X-Forwarded-For`, since trusting it without a configured list of trusted
proxies would let any caller spoof their bucket. If you're behind a proxy
and need real per-IP limits, terminate TLS/proxying somewhere that
preserves the original connection, or rely on named `[[clients]]` keys
instead (unaffected by this, since they're identified by API key).

**Outbound** (protecting each provider's own limits from this router):
set `requests_per_minute` on a `[providers.*]` entry to self-throttle
calls to it. When that provider's bucket is empty, the router treats it
exactly like a retryable provider error (429) and falls back to the next
candidate in the chain — it does not queue or wait. If every candidate in
a chain is outbound-throttled, the client gets a `429` with `Retry-After`
for the shortest wait among them.

Like the pricing table, none of this is a live feed — it's config you set
based on limits you already know (a provider's published rate limit, or
how much traffic you want to allow a given caller). Both directions show
up in `GET /metrics` (`rusty_provider_dispatch_attempts_total` with
`outcome="rate_limited"` for outbound,
`rusty_provider_inbound_rate_limit_rejections_total` for inbound) and use
the same in-memory, per-process, resets-on-restart token buckets as
everything else this router tracks itself.

### Concurrency cap

`server.max_concurrent_requests` bounds requests being handled *at once*,
server-wide, across every caller and route — a different axis from the
rate limiting above, which bounds one caller's request *rate* but not the
total number in flight. A burst spread across many clients (or a single
client under a generous/no rate limit) can still exhaust upstream provider
rate limits or local resources with no backpressure; this closes that gap:

```toml
[server]
max_concurrent_requests = 200
```

Once that many requests are in flight, the next one gets `503` immediately
rather than queuing — a caller waiting behind a long queue at an already-
saturated server is worse than one told plainly to retry. Unset (the
default) means no cap, same as before this existed.

## CORS

By default every route allows any browser origin (`Access-Control-Allow-Origin: *`) — the same behavior this router has always had. Set `server.cors_allowed_origins` to restrict that to an explicit allowlist instead:

```toml
[server]
cors_allowed_origins = ["https://my-dashboard.example", "https://app.example.com"]
```

An origin not on the list gets no `Access-Control-Allow-Origin` header at all (the browser blocks the response from being read by page JS), same as any other CORS-restricted API. Only origin is restricted — methods and headers stay wildcard either way, since there's no cookie-based/credentialed auth here for that to be unsafe with (bearer tokens go in `Authorization`, not a cookie jar). An entry that doesn't parse as a valid `Origin` header value is skipped with a startup warning rather than failing the whole list, logged the same way an invalid `[[guardrails]]` pattern is.

## Request body size limit

`server.max_body_bytes` caps an inbound request body, in bytes, rejected
with `413 Payload Too Large` before a handler ever parses it:

```toml
[server]
max_body_bytes = 20971520  # 20 MiB, this is the default
```

axum's `Json`/`Bytes` extractors already refuse anything over 2 MB even
without this section present — `max_body_bytes` replaces that built-in
ceiling rather than adding a second one on top of it. The default is
raised well past 2 MB because a legitimate multimodal request (an image,
audio clip, or PDF passed inline) is base64-encoded, which alone adds
~33% over the original file's size — 2 MB is tight enough to reject
ordinary attachments, not just abuse. Applies to every route, not only
`/v1/chat/completions`.

## Spend budgets

Rate limits cap how *often* a client can call this router; `budget_usd` on
a `[[clients]]` entry caps how much they can *spend*:

```toml
[[clients]]
name = "hermes"
api_key_env = "CLIENT_HERMES_API_KEY"
requests_per_minute = 60
budget_usd = 50.0
budget_period = "monthly"   # or "total" (default) / "daily" / "weekly"
budget_warning_threshold = 0.8   # optional -- see "Webhook notifications" below
```

Spend is tracked from the same `cost_usd` this router already computes for
`GET /v1/usage`, so it only ever counts requests to a model with a
`[[pricing]]` entry — an unpriced request never counts against a budget,
the same way it never adds to `cost_usd` there. Once a client's tracked
spend for the current period reaches `budget_usd`, further requests from
that client get `402` until the period resets (or forever, for the
default `"total"` period — there's no automatic reset, so raising the
config value, or restarting the process, is the only way a `"total"`
client keeps going). A request already in flight when a client crosses
its budget is still allowed to complete — the check happens before
dispatch, using spend as of the *start* of the request, not a mid-flight
cutoff — so the client's actual spend can end up somewhat over
`budget_usd` by the time it's cut off, not capped exactly at it.

`"daily"` resets at each UTC calendar-day boundary (midnight UTC).
`"weekly"` resets every 7 days counted from the Unix epoch
(1970-01-01T00:00:00Z, a Thursday) — a fixed 7-day cadence, not aligned to
any particular weekday like a calendar Monday-start or Sunday-start week.
`"monthly"` resets at each UTC calendar-month boundary, same as before.

This only applies to named `[[clients]]`, the same as the rate-limiting
client bucket — there's no budget for the IP-bucketed fallback used by
unmatched callers, since there's no stable identity to track spend
against. Like the rest of this router's own tracking, spend is in-memory
and per-process by default (resets on restart, not shared across
processes) unless `[persistence]` is configured, in which case it's
backed by the same SQLite file or Postgres database as `GET /v1/usage` —
see Persistence below — and every process/host sharing that backend
enforces the same budget consistently instead of each tracking its own
slice of a client's traffic. Rejections show up in `GET /metrics` as
`rusty_provider_client_budget_rejections_total`, labeled by client name.

### Webhook notifications

Without more, a client crossing its budget only surfaces as the `402`
above and that Prometheus counter — nothing an operator can act on
proactively. `[webhook]` adds a push notification on top:

```toml
[webhook]
url = "https://hooks.example.com/rusty-provider"
auth_header_env = "WEBHOOK_AUTH_HEADER"   # optional; e.g. "Bearer <token>"
timeout_secs = 10                         # optional, this is the default
signing_secret_env = "WEBHOOK_SIGNING_SECRET"  # optional
retry_backoff_secs = 1                    # optional, these three are the defaults
retry_backoff_max_secs = 30
max_retries = 3
```

This router POSTs a JSON body to `url` on three events:

```jsonc
// A client's tracked spend just crossed budget_warning_threshold * budget_usd (if configured).
{"event": "budget_warning", "client": "hermes", "spent_usd": 41.00, "budget_usd": 50.0, "warning_threshold": 0.8, "period": "monthly"}
// A client's tracked spend just reached or passed its budget.
{"event": "budget_exceeded", "client": "hermes", "spent_usd": 51.20, "budget_usd": 50.0, "period": "monthly"}
// An operator manually reset a client's spend via the admin API.
{"event": "budget_reset", "client": "hermes", "budget_usd": 50.0, "period": "monthly"}
```

`budget_warning` and `budget_exceeded` both fire on the specific request
that pushes tracked spend from under the threshold to at-or-over it, not
on every subsequent request past that point — `budget_warning` is a
heads-up before the cutoff, not a second limit, and doesn't affect
whether a request is allowed through (only `budget_usd` itself does
that). The request that crossed either threshold is still charged and let
through before the event fires, same as the `402` only starting on the
*next* request once `budget_usd` itself is crossed. Under `[persistence]`,
"just crossed" is a best-effort, eventually-consistent read-back rather
than an atomic check-and-set, so two concurrent requests to the same
client right at a boundary could both fire (or, rarely, neither) — same
class of caveat the tracked spend total itself already carries.
`budget_warning_threshold` isn't settable via the admin API yet — only
through `[[clients]]` in config.toml. `auth_header_env` names an env var
holding the exact value to
send as this POST's `Authorization` header (e.g. `"Bearer <token>"`), so
the receiver can verify the request came from this router; leaving it
unset sends no `Authorization` header at all. `signing_secret_env` names
an env var holding an HMAC-SHA256 secret; when set, every delivery
carries an `X-RP-Signature: sha256=<hex>` header computed over the exact
JSON body sent, so the receiver can verify the request actually came from
this router rather than trusting `auth_header_env` alone — compute the
same HMAC over the raw request body on your end and compare. Delivery is
fire-and-forget — a slow or unreachable receiver never adds latency to
the request that triggered the event. A failed delivery (network error,
or a `5xx` response) retries with exponential backoff
(`retry_backoff_secs`, doubling up to `retry_backoff_max_secs`, up to
`max_retries` retries); a `4xx` response is treated as permanent and not
retried. Giving up after the retry budget is exhausted is only logged,
never surfaced to the client that triggered the event.

## Free tiers

`[[clients]].budget_usd` caps what a *caller* spends; `[[free_tiers]]`
tracks a different thing — how much of a *provider's* free allowance
you've used, per "provider/model":

```toml
[[free_tiers]]
model = "groq/llama-3.3-70b-versatile"
monthly_free_tokens = 117000000
# period = "monthly"   # optional, this is the default -- "total" / "daily" / "weekly" / "monthly"
```

This is self-declared, like `[providers.*]`'s `zdr`/`no_training` flags —
you tell it what you believe the provider's free tier grants, and it never
verifies that number against the provider's own systems. It's the honest,
scoped-down version of what a free-tier *aggregator* product would try to
do for you automatically: aggregating other providers' quotas on your
behalf raises real ToS problems (several providers' terms restrict
proxy/resale use of a free-tier key), so this router only ever reports
against numbers you configured yourself, for your own account.

**`GET /v1/free-tiers`** reports every configured entry's budget, this
period's tracked prompt+completion token usage, and what's left:

```json
{
  "object": "list",
  "data": [
    {
      "model": "groq/llama-3.3-70b-versatile",
      "monthly_free_tokens": 117000000,
      "tokens_used": 250000,
      "tokens_remaining": 116750000,
      "period": "monthly"
    }
  ]
}
```

Tracked usage is the same `prompt_tokens + completion_tokens` this router
already counts for [`GET /v1/usage`](#get-v1usage) — a request to a
"provider/model" with no `[[free_tiers]]` entry is simply never counted
here, the same way an unpriced model never counts against `cost_usd`.
`tokens_remaining` saturates at `0` rather than going negative once usage
exceeds the configured budget — this endpoint is reporting-only, unlike
`[[clients]].budget_usd`; nothing here blocks a request or returns `402`,
since a *provider's* free-tier exhaustion is something that provider's own
API would reject on its own, not something this router can pre-empt
without a live reading of your actual remaining quota (which no provider
here exposes).

`period` uses the same reset cadence as `[[clients]].budget_period` —
`"total"` (never resets), `"daily"`/`"weekly"`/`"monthly"` (UTC calendar
boundaries, or a fixed 7-day cadence from the Unix epoch for `"weekly"`).
Tracking is in-memory only, per-process — it resets on restart and isn't
shared across processes, with no `[persistence]` backing (unlike
`GET /v1/usage`), so a load-balanced deployment's `/v1/free-tiers` reflects
only the process answering that particular request.

## Guardrails

`[[guardrails]]` entries check every request's message text — before
it's ever dispatched to a provider — against a regex, and either block
or redact what matches:

```toml
[[guardrails]]
name = "no-ssn"
pattern = '\d{3}-\d{2}-\d{4}'
action = "block"

[[guardrails]]
name = "no-email"
pattern = '\S+@\S+'
action = "redact"
replacement = "<email>"   # optional, defaults to "[redacted]"
```

`action = "block"` rejects the request with `400` (and the guardrail's
`name` in the error message) the moment `pattern` matches anywhere in
its message text. `action = "redact"` replaces every match with
`replacement` and lets the (now-redacted) request continue — the
provider never sees the original text. Multiple `[[guardrails]]` apply
in config order, and a later guardrail sees whatever an earlier
`"redact"` already rewrote (so a `"block"` guardrail placed after a
`"redact"` one can catch the redaction marker itself, if that's useful,
or just check for its own independent pattern). An invalid regex
`pattern` is skipped at startup with a warning, same as a misconfigured
provider or client, rather than refusing to start the router over one
bad pattern.

Only plain text is scanned — a message's own text content, or the text
parts of a multimodal message; image/audio/file parts are untouched,
since a regex has nothing meaningful to check there. This is scoped
globally (every request, regardless of which client sent it), since
rusty has no workspace/org concept to scope guardrails to individually
the way OpenRouter's org-level guardrails can be.

## Moderation

`[moderation]` checks every request's message text against an external
moderation endpoint before it's ever dispatched to a provider, blocking
anything flagged:

```toml
[moderation]
api_key_env = "OPENAI_API_KEY"          # can reuse [providers.openai]'s key
base_url = "https://api.openai.com/v1"  # optional, this is the default
model = "omni-moderation-latest"        # optional, this is the default
timeout_secs = 10                       # optional, this is the default
```

This is a different axis from [Guardrails](#guardrails) above: a guardrail
is a regex pattern the operator writes and fully controls (a specific SSN
format, a specific banned word); moderation defers the actual judgment
call — hate, violence, self-harm, and whatever other categories the
backend classifies — to a third-party classifier the operator doesn't
have to enumerate patterns for. Only OpenAI's `/moderations` endpoint (or
a compatible one — `base_url` is configurable) is supported; Anthropic
and Gemini don't expose a public moderation API of their own. Moderation
runs *after* guardrails, so a guardrail's own redaction is what gets
checked, not the raw input.

A flagged request is rejected with `400` and the triggering category
names (also recorded in the `rusty_provider_moderation_blocked_total`
Prometheus counter, labeled by category). Only plain text is checked —
the same message-text/text-parts scope as guardrails; a request with no
text content at all (image-only) skips the check entirely rather than
making an empty call.

If the moderation backend itself can't be reached, or returns something
this router can't parse, the request is let through — moderation fails
*open*, not closed. This is deliberate: an unreachable/misbehaving
moderation backend is treated the same as this router's other auxiliary
systems (an unreachable webhook, an unreachable persistence backend, an
invalid guardrail regex at startup) — logged, not something that should
take down chat completions entirely. `moderation.api_key_env` set but
unresolvable at startup disables moderation the same way a misconfigured
provider is skipped, with a warning.

## Presets

`[[presets]]` saves a named `(model, provider prefs, system prompt,
sampling params)` bundle, referenced from a request by slug:

```toml
[[presets]]
name = "support-bot"
model = "smart"                                          # can be a [[routes]] alias
system_prompt = "You are a helpful, concise support agent."
temperature = 0.2
max_tokens = 500

[presets.provider]
only = ["anthropic", "openai"]
```

```jsonc
{
  "preset": "support-bot",
  "messages": [{"role": "user", "content": "..."}]
}
```

Every field a preset supplies is a per-field *default* — whatever the
request itself already sets always wins — with one exception: `model`
overrides the request's `model` outright when set, since centralizing
model selection is the point of a preset. That means the wire schema's
`model` field is still required (the OpenAI-compatible schema doesn't
make it optional), but a preset's `model` takes over regardless of what
value the request sent — a client using a preset doesn't need to think
about `model` at all, and if it does set one, a preset with its own
`model` still wins. `system_prompt` is prepended as a new `role:
"system"` message, but only if the request has no system message of its
own — never appended alongside or merged with one the caller already
provided. `provider`, if the request's own `provider` is unset, becomes
the request's provider preferences wholesale (no per-field merge between
the two — the request's own `provider`, if set at all, wins entirely,
same all-or-nothing rule route `only`/`ignore`/etc. already follow
elsewhere). Every sampling-param field (`temperature`, `top_p`,
`max_tokens`, `stop`, and the fuller set from
[Sampling parameters](#sampling-parameters)) fills in only where the
request left that specific field unset. An unknown `preset` name is a
`400`, same as any other invalid request field. Presets apply before
[Guardrails](#guardrails), so a preset's own `system_prompt` is still
scanned by whatever guardrails are configured.

## Auto-routing

`model: "auto"` in a request routes it to one of three tiers, picked by a
heuristic (not ML) complexity score — roughly OpenRouter's
`openrouter/auto`:

```toml
[auto_routing]
simple_model = "openai/gpt-4o-mini"                      # can be a [[routes]] alias
medium_model = "smart"
complex_model = "anthropic/claude-opus-4-8"
simple_max_score = 200                                    # optional, defaults shown
medium_max_score = 800
```

```jsonc
{
  "model": "auto",
  "messages": [{"role": "user", "content": "..."}]
}
```

Each tier's model is a `"provider/model"` string or a `[[routes]]` alias,
exactly like `model` anywhere else, so a tier can point at a whole
fallback chain rather than one fixed model. The complexity score is an
estimated prompt-token count (the same tokenizer-free `chars / 4`
estimate used elsewhere in this router, e.g. for
[context compression](#context-compression)) summed across every
message, plus flat bonuses for signals that tend to mean a harder task:
multi-turn context, code in the conversation, tool use, requested
reasoning, or a JSON-schema output constraint. The score has no fixed
unit or universal threshold — `simple_max_score`/`medium_max_score` are
something to tune against your own traffic: a request scoring at or
below `simple_max_score` goes to `simple_model`, at or below
`medium_max_score` goes to `medium_model`, and anything higher goes to
`complex_model`.

A request can set `"provider": {"auto_bias": "cost"}` or `"quality"` to
shift both thresholds for just that request, without touching the
operator's configured defaults: `"cost"` doubles both thresholds (a
request has to score higher before escalating into a pricier tier, so it
stays on the cheaper tiers longer), `"quality"` halves them (escalating
into a pricier tier sooner). Unset, or any other value, is `"balanced"`
— the thresholds apply as configured. `auto_bias` only has any effect
when `model` is `"auto"`.

Without `[auto_routing]` configured, `"auto"` isn't special-cased at all
— it resolves like any other unrecognized alias, a `400`.

When a tier's model is a `[[routes]]` alias spanning multiple candidates,
an auto-routed request defaults to `sort: "price"` among them — cheapest
candidate first — unless the request already set its own `provider.sort`
explicitly, which always wins unchanged. This is what actually connects
the complexity classifier to the pricing system: picking a tier is still
purely complexity-based, but *within* that tier, cost now breaks the tie
by default instead of the classifier and pricing staying two disconnected
mechanisms.

## Fusion routing

`strategy = "fusion"` on a `[[routes]]` alias replaces its default
sequential-fallback behavior with a parallel panel + judge-synthesis
dispatch instead:

```toml
[[routes]]
alias = "panel"
chain = ["anthropic/claude-sonnet-5", "openai/gpt-4o", "gemini/gemini-2.5-pro"]
strategy = "fusion"
judge = "anthropic/claude-opus-4-8"     # required for fusion to actually engage
fusion_timeout_secs = 30                 # optional, default shown
```

`chain` doubles as the fusion "panel" here — every entry is dispatched
concurrently rather than tried one at a time, and `judge` synthesizes a
single final answer from whichever candidates responded. Each panel member
is independently bounded by `fusion_timeout_secs`, so the total wait is
capped at that duration regardless of panel size — a candidate that's
still slow when its own timeout expires, or that errors outright, is
simply absent from what reaches the judge rather than blocking everyone
else; the request only fails if *every* candidate does. The judge sees
each surviving answer under an anonymized `"Candidate 1"`/`"Candidate 2"`
label, not the provider/model that produced it, so synthesis goes by the
answers' merits rather than any name the judge might otherwise recognize
and favor.

A tool-calling request (`tools` set) bypasses fusion entirely and falls
through to ordinary sequential-chain dispatch instead — a judge doing
plain-text synthesis can't meaningfully merge structured `tool_calls` from
multiple candidates. The same is true of a streaming request, since
synthesizing one answer from a panel is inherently a whole-response
operation with no incremental form. `[[routes]]` alias with
`strategy = "fusion"` but no `judge` set is a soft misconfiguration, not a
startup failure: it logs a warning and dispatches that alias exactly like
an ordinary `strategy = "fallback"` chain instead.

`GET /v1/usage`/`GET /metrics`/`GET /v1/generation?id=` all reflect the
full cost of a fusion request — every panel member that actually
responded plus the judge each get their own usage/cost entry, and the
final response returned to the caller carries the summed usage/cost
across all of them, not just the judge's own call.

## BYOK (bring your own key)

A request can supply its own API key for a configured provider, used for
that request's own calls instead of the operator's `api_key_env`-resolved
one:

```jsonc
{
  "model": "openai/gpt-4o-mini",
  "messages": [{"role": "user", "content": "..."}],
  "provider": {
    "byok": {
      "openai": "sk-the-callers-own-openai-key"
    }
  }
}
```

`byok` maps a provider name to the key to use for it, for this request
only — never written to config, logged, or echoed back in any response.
The operator still needs a `[providers.X]` entry for that provider (its
`kind`/`base_url` are what the router needs to know how to call it at
all); `byok` only swaps the credential, not the endpoint. A chain that
spans multiple providers can mix and match — a provider name present in
`byok` uses that key, any other candidate in the same chain falls back to
its own configured key as usual. A provider name in `byok` that doesn't
match any candidate actually tried is simply never used, not an error.

This is a credential swap only, not a separate billing mode: `cost_usd`
and every other cost/budget/usage figure this router tracks are computed
and recorded exactly the same regardless of whose key served the request
— rusty_provider has no visibility into (and makes no attempt to reduce)
what the provider itself bills the caller's own account for a BYOK
request. The outbound self-throttle (`[providers.X].requests_per_minute`)
still applies too, since that protects this router process's own call
pattern to the provider's endpoint, independent of which key is paying.

## Web search

`[web_search]` lets a request trigger a live web search whose results get
woven into the conversation before dispatch, so the model can ground its
answer in information beyond its training data:

```toml
[web_search]
api_key_env = "BRAVE_SEARCH_API_KEY"
base_url = "https://api.search.brave.com/res/v1/web/search"  # optional, this is the default
max_results = 5                                                # optional, this is the default
timeout_secs = 10                                              # optional, this is the default
```

```jsonc
{
  "model": "smart",
  "messages": [{"role": "user", "content": "what's new in Rust"}],
  "web_search": true
}
```

Loosely mirrors OpenRouter's `:online` model suffix / `web` plugin,
scoped down considerably: only [Brave Search's API](https://brave.com/search/api/)
(or a compatible one — `base_url` is configurable) is supported, and
results are woven in as plain text rather than surfaced as a structured
citations/annotations response field. When `web_search: true` and
`[web_search]` is configured, the router searches using the latest
`user`-role message's own text as the query, then prepends a numbered
block of results (title, snippet, URL) onto that same message before
[Guardrails](#guardrails) and [Moderation](#moderation) run — so both see
whatever the search actually returned, not just the original question.
`web_search` is silently a no-op — the request goes out completely
unmodified — when it's unset/`false`, `[web_search]` isn't configured,
there's no user-message text to search with (an image-only turn), or the
search comes back with zero results.

A search-backend failure (network error, non-2xx, an unparseable body)
never blocks or errors the request either — it's logged and the request
proceeds unmodified, the same fail-open resilience
[Moderation](#moderation) gives an unreachable classifier. Every outcome
(`results`, `no_results`, `error`) increments the
`rusty_provider_web_search_total` Prometheus counter, labeled by outcome,
so an operator can tell a quiet backend from a genuinely idle feature.
`web_search.api_key_env` set but unresolvable at startup disables web
search the same way a misconfigured provider is skipped, with a warning.

## Response cache

`[cache]` turns on an in-memory, exact-match cache of non-streaming chat
completions, so identical requests within a short window are served
without ever reaching a provider:

```toml
[cache]
ttl_secs = 300        # optional, this is the default
max_entries = 1000    # optional, this is the default
```

Not to be confused with [Prompt caching](#prompt-caching) above —
`cache_control` and `cache_read_per_million`/`cache_write_per_million`
price a *provider's own* prompt-cache discount (Anthropic, OpenAI, Gemini),
whereas `[cache]` is a router-side cache that skips the provider entirely
on a hit.

The cache key is a hash of the entire incoming request — model, messages,
every sampling parameter, `provider` preference, and so on — so this is
exact-match only, not semantic/fuzzy matching: any difference at all is a
miss. Only [non-streaming](#post-v1chatcompletions) requests are
cacheable; a request with `"stream": true` always bypasses the cache in
both directions (it's neither served from it nor written to it). Entries
expire after `ttl_secs` (checked lazily, on lookup) and the cache holds at
most `max_entries`, evicting the oldest entry once over capacity — the
same fixed-capacity, insertion-order eviction [`GET
/v1/generation?id=`](#get-v1generationid) already uses.

A cache hit returns the stored response as-is and skips *all* of that
request's usual bookkeeping — usage/cost accounting, latency and
throughput histograms, the `/v1/generation?id=` record — since all of it
was already recorded once, when the response was first computed. Re-running
it on every hit would inflate [`/v1/usage`](#get-v1usage) with generations
that never actually happened. Every lookup increments the
`rusty_provider_cache_lookups_total` Prometheus counter, labeled `hit` or
`miss`. `[cache]` absent leaves caching fully off, with no overhead.

### Semantic mode

`[cache].mode = "semantic"` swaps exact-match for embedding-cosine-
similarity matching on message content, while keeping every other field
exact-match — so a differently-worded-but-equivalent request can still
hit:

```toml
[cache]
mode = "semantic"
similarity_threshold = 0.85   # optional, this is the default
embedding_model = "openai/text-embedding-3-small"
```

`embedding_model` is a `"provider/model"` string, exactly like `model`
elsewhere — this router embeds the request's messages by calling it
through its own [`POST /v1/embeddings`](#post-v1embeddings) dispatch path
(fallback/retry included, since `embedding_model` can itself be a
`[[routes]]` alias). `similarity_threshold` (0.0-1.0, higher is stricter)
is the minimum cosine similarity for a lookup to count as a hit.
`model`, every sampling parameter, `tools`, and `provider` preferences
still have to match *exactly* — "semantic" only ever fuzzes the message
text, nothing else. An embedding-call failure (network error, the
provider down) fails open: that one request just skips the cache in both
directions rather than failing, the same resilience pattern
[Moderation](#moderation)'s own backend failures already get. A request
with `"stream": true` still bypasses caching entirely in either mode,
same as exact.

Without `embedding_model` set, or if it names a provider this process
didn't actually configure, `"semantic"` mode falls back to `"exact"` at
startup with a warning — same soft-failure pattern a misconfigured
provider or moderation backend already gets, rather than refusing to
start the router. `mode` defaults to `"exact"`, so an existing `[cache]`
section with no `mode` set keeps its current exact-match behavior
unchanged.

## Admin API

Setting `server.admin_key_env` unlocks a small admin API for inspecting
and managing configured clients' spend budgets:

```toml
[server]
admin_key_env = "RUSTY_PROVIDER_ADMIN_KEY"
```

- **`GET /v1/admin/clients`** — every `[[clients]]` entry the caller can
  see (see [Organizations, workspaces & roles](#organizations-workspaces--roles)
  below for what that means for a scoped caller), its `organization`/
  `workspace`/`role`, `requests_per_minute`, and (for clients with
  `budget_usd` set) its current-period `spent_usd` and `budget_period`. A
  client with no configured budget still appears, with `budget_usd`/
  `budget_period`/`spent_usd` all `null`.
- **`POST /v1/admin/clients/{name}/reset-spend`** — zeroes that client's
  tracked spend for the current period, immediately un-blocking a client
  that's hit `402`. `404` for a client name that doesn't exist or has no
  configured budget.
- **`GET /v1/admin/clients/{name}/usage-history?days=N`** — day-bucketed
  `requests`/`prompt_tokens`/`completion_tokens`/`cost_usd` for `name`,
  oldest first, over the last `N` days (default `30`, capped at `90`).
  Unlike every other endpoint above, this isn't limited to clients with a
  configured `budget_usd` — any client visible to the caller is queryable,
  since it's a usage rollup, not a budget/spend concern. Requires
  `[persistence]` to be configured (there's no in-memory equivalent —
  history needs to survive a restart to mean anything): responds `200`
  with an empty `data` array rather than an error when it isn't, or for a
  day with nothing recorded. `404` for an unknown client name (or one
  outside a scoped caller's organization).
- **`POST /v1/admin/clients`** — provisions a new client at runtime, no
  config-file edit or restart needed. Body:
  ```jsonc
  {
    "name": "acme",
    "requests_per_minute": 60,
    "budget_usd": 10.0,       // optional, omit for unrestricted
    "budget_period": "monthly", // optional, "total" (default) / "daily" / "weekly" / "monthly"
    "api_key": "...",         // optional -- omit to have the server generate one
    "organization": "acme-corp", // optional, see Organizations below
    "workspace": "prod",      // optional
    "role": "member"          // optional, "member" (default) / "admin"
  }
  ```
  Responds `201` with the same shape plus `api_key` — the server-generated
  key (if you didn't supply one) is only ever shown in this response, the
  same hygiene as GitHub/Stripe-style API keys, so save it immediately.
  `400` for an empty `name`, a `requests_per_minute` of `0`, or a negative
  `budget_usd`; `409` if `name` or `api_key` collides with an existing
  client. A caller authenticated as a scoped (organization-admin) client
  always has `organization` pinned to its own, regardless of what's sent —
  see [Organizations, workspaces & roles](#organizations-workspaces--roles).
- **`PATCH /v1/admin/clients/{name}`** — updates an existing client
  (config-defined or runtime-provisioned). Every field is optional and
  independent: omit a field to leave it unchanged, send `"budget_usd":
  null` to explicitly clear a configured budget (as opposed to omitting
  `budget_usd` entirely, which leaves it as-is), and set
  `"rotate_api_key": true` to revoke the client's current key and issue a
  new one, returned in the response the same one-time way creation does.
  `404` for an unknown client, `400` for an invalid `requests_per_minute`/
  `budget_usd`. `organization`/`workspace`/`role` are set at creation only
  — not updatable through this endpoint.
- **`DELETE /v1/admin/clients/{name}`** — removes a client entirely,
  immediately revoking its key and dropping its budget/spend tracking.
  `404` for an unknown client.

Requests to every route above need `Authorization: Bearer <token>`, from
one of two sources: `admin_key_env`'s resolved value (unscoped — sees and
manages every client, in every organization), or an admin-role client's
own key (scoped to its own `organization` — see
[Organizations, workspaces & roles](#organizations-workspaces--roles)).
**Not** `server.api_key_env` or a plain (`role = "member"`) `[[clients]]`
key, which authenticate chat completions but deliberately don't also grant
admin access. Leaving `admin_key_env` unset, with no admin-role client
configured either, disables the admin API entirely: every route `404`s,
as if it didn't exist, rather than silently falling open once *any* auth
is configured elsewhere.

Runtime-provisioned clients (created/updated/deleted via this API) are
**in-memory only** — they don't survive a restart, and aren't written to
`[persistence]`'s database even when one is configured (unlike usage/cost
tracking and spend, which are). Only `[[clients]]` entries defined in
`config.toml` come back after a restart; treat the admin API as a way to
provision short-lived or emergency access without a deploy, not a
permanent client registry. A config-defined client can still be updated or
deleted at runtime through this API — the change just doesn't get written
back to `config.toml`, so a later restart reverts it to what the file
says.

Every successful mutation (`admin_create_client`, `admin_update_client`,
`admin_delete_client`, `admin_reset_client_spend`) emits a structured
`tracing::info!` audit line — `identity` (`"global"` or `"scoped"`),
`organization` (empty for global or an unscoped caller), `action`, and
`target` (the client name acted on) — so "who changed this client's
budget, and when" is answerable from logs. Rejected/no-op requests (`404`,
`409`, a validation error) aren't logged here, since nothing changed for
there to be anything to record.

## Dashboard

`GET /dashboard` serves a small read-only web UI over the JSON endpoints
above — models, provider health, cumulative usage, free-tier budgets, and
(admin-authenticated) clients with per-client spend and a 30-day usage
history sparkline, with a "reset spend" action.

It's a single static HTML file with vanilla JS — no build step, no npm,
no JS framework, no CDN dependency — compiled directly into the `rp-server`
binary. The page itself carries no secrets and needs no auth of its own to
load (same reasoning as `/health`); it prompts for a bearer token in the
browser and attaches it to every `fetch()` call, so it's subject to
exactly the same `check_auth`/`check_admin_auth` rules those endpoints
already enforce elsewhere in this file. Use an admin-role client's own key
(see [Organizations, workspaces & roles](#organizations-workspaces--roles)
below) to unlock every panel with one token; a plain client key or
`server.api_key_env` only unlocks the non-admin panels (models, provider
stats, usage, free tiers), and the global `admin_key_env` alone only
unlocks the clients panel. The token is kept in the browser tab's session
storage only — never sent anywhere but this server, never written to
disk.

The dashboard's own UI text (panel titles, column headers, button
labels) goes through a small `t()`-keyed translation dictionary and a
language switcher in the header, persisted in `localStorage`. This is a
switching *mechanism*, not a translation project — only `en` is
populated today; a new locale is added by extending the dictionary in
`dashboard.html`, not by this router guessing a translation nobody
asked for. Server-generated JSON error messages (the ones every
endpoint already returns) are unaffected and stay English.

## Organizations, workspaces & roles

`[[clients]]` entries (config-defined or admin-API-provisioned) can carry
three extra fields:

```toml
[[clients]]
name = "acme-admin"
api_key_env = "CLIENT_ACME_ADMIN_API_KEY"
requests_per_minute = 60
organization = "acme-corp"
workspace = "prod"
role = "admin"
```

`organization` and `workspace` are labels — a client's `organization`
groups it (and only it) for `GET /v1/admin/organizations` below;
`workspace` sub-groups it within that rollup. Neither has any effect on
chat completions: two clients in different organizations are otherwise
completely ordinary, independent `[[clients]]` entries. `role` is the one
with teeth: `"admin"` (the default is `"member"`, i.e. no admin access at
all) lets that client's own API key *also* authenticate to `/v1/admin/*`,
in addition to `server.admin_key_env` — but only ever scoped to clients
sharing its own `organization`. An admin-role client with no
`organization` set is scoped to the shared "no organization" bucket
(every other org-less client), not to every client everywhere — that
breadth is reserved for `admin_key_env`.

Concretely, an org-scoped admin's key:

- `GET /v1/admin/clients` / `GET /v1/admin/organizations` only return
  clients in its own organization.
- `POST /v1/admin/clients` always creates the new client in its own
  organization, regardless of what `organization` the request body sends.
- `PATCH`/`DELETE /v1/admin/clients/{name}` and
  `POST /v1/admin/clients/{name}/reset-spend` `404` (not `403`, so a
  scoped admin can't distinguish "wrong organization" from "doesn't
  exist") for any client outside its own organization.

**`GET /v1/admin/organizations`** rolls up every client the caller can see
into `(organization, [clients])` groups — one group per distinct
`organization` value (`server.admin_key_env` sees every group; a
scoped admin only ever sees its own single group):

```jsonc
{
  "object": "list",
  "data": [
    {
      "organization": "acme-corp",
      "clients": [
        {
          "name": "acme-admin",
          "workspace": "prod",
          "role": "admin",
          "requests_per_minute": 60,
          "budget_usd": null,
          "spent_usd": null
        }
      ]
    }
  ]
}
```

This is deliberately not a full multi-tenant identity system — there's no
separate "organization" or "workspace" entity with its own settings,
membership list, or invitations, just a label on `[[clients]]` and a
same-organization check on top of the existing admin API. It's enough to
let an operator delegate "manage your own team's clients" without handing
out the unscoped `admin_key_env`, which is the actual problem this scopes
down to.

## Persistence

By default, cumulative usage/cost stats (`GET /v1/usage`) and each
client's `budget_usd` spend tracking (see Spend budgets above) both live
only in memory — they reset on restart and each process only knows about
its own traffic. Setting `[persistence]` in config switches both to a
durable, shared backend — either a single SQLite file, or a networked
Postgres database:

```toml
# Option 1: a single SQLite file.
[persistence]
backend = "sqlite"
sqlite_path = "usage.db"

# Option 2: a shared Postgres database.
[persistence]
backend = "postgres"
postgres_url_env = "DATABASE_URL"
postgres_tls = "require"  # or "disable" (the default) for a plaintext connection
```

Either way, the schema (a `usage_stats` table and a `client_spend` table)
is created automatically on first use if it doesn't exist. Every completed request/streamed response persists its usage delta
(and, for budgeted clients, its spend delta) to the backend, and both
`GET /v1/usage` and `check_client_budget` read fresh from it rather than
an in-memory cache — so restarting a process doesn't lose history, and
every `rusty_provider` process pointed at the same backend reports a
consistent combined total and enforces the same budget, rather than each
only seeing its own slice of traffic.

**SQLite** is a single file, not a distributed database — it works well
for multiple processes on one host or a shared local volume, but isn't
meant for processes spread across different machines over a network
filesystem. **Postgres** is the way to get that: any number of
`rusty_provider` processes, on any number of hosts, pointed at the same
database, see a consistent combined total and enforce budgets
consistently across the whole fleet. Connections are unencrypted by
default (`postgres_tls = "disable"`); set `postgres_tls = "require"` to
encrypt them, verified against the host's native root certificate store —
the same trust store `reqwest` already uses for outbound provider calls,
so there's no separate CA bundle to manage. `"require"` refuses to fall
back to plaintext even if the server doesn't support TLS. Either way, the
connection string comes from the environment variable named by
`postgres_url_env`, the same way provider/client API keys are kept out of
the config file.

Persisting is best-effort and asynchronous: if the database becomes
briefly unavailable, requests still succeed, `GET /v1/usage` falls back
to that process's in-memory view rather than erroring, and a client
budget check treats an unreadable backend as "unspent" for that one
check rather than blocking the request. An invalid/unreachable backend at
startup (e.g. `sqlite_path`'s parent directory doesn't exist, or
`postgres_url_env` names an unset env var or an unreachable database) is
a startup warning, not a hard failure — the router falls back to
in-memory-only tracking rather than refusing to start.

`GET /metrics` (Prometheus) is unaffected by this setting and always
stays per-process — Prometheus aggregates across processes at scrape
time via its own query layer, not here.

## Config

See `config.example.toml`. Provider API keys are always read from
environment variables (named by `api_key_env`) — never stored in the
config file itself. `[[pricing]]` entries are optional and only affect
requests that opt into `"provider": {"sort": "price"}`; a provider's `zdr`
flag is optional and only affects requests that opt into
`"provider": {"zdr": true}`.

`config.example.toml` also ships a commented-out block of curated presets
for more OpenAI-wire-compatible backends beyond the six enabled by
default — see [docs/PROVIDERS.md](docs/PROVIDERS.md) for the fuller
reference table (what each one is, free-tier/ToS notes). Adding one is a
config change only: `kind = "openai"` already covers any backend speaking
OpenAI's `/chat/completions` shape, the same way Groq/Together/Fireworks
already share one adapter.

## Using with local agent tools (Hermes, OpenClaw, etc.)

Any local coding-agent tool that lets you point it at a custom
OpenAI-compatible endpoint can use rusty_provider as its model backend —
this covers tools like Hermes and OpenClaw, whose own model-provider
settings just need:

- **Base URL**: `http://localhost:8080/v1` (or wherever `rp-server` is
  running/reachable).
- **API key**: the value of `RUSTY_PROVIDER_API_KEY` (or whatever env var
  `server.api_key_env` points at) if you've enabled auth; otherwise any
  non-empty placeholder string, since most clients require *something* in
  the field even when the server doesn't check it.
- **Model**: a `"provider/model"` string or a configured route alias (see
  `config.example.toml`) — whichever the tool lets you type in as the model
  name.

Since these tools drive actions (editing files, running commands) through
function/tool calling, make sure the underlying model you route to
actually supports tool use, and that your `[[routes]]` fallback chain (if
you use one) only includes models that do — a chain that silently falls
back to a model without tool support will make the agent behave oddly
rather than fail loudly.

## License

MIT — see [LICENSE](LICENSE).
