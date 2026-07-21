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
  and 5xx errors.
- `crates/server` (`rp-server`) — the axum HTTP server exposing the
  OpenAI-compatible API.

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
        "cache_write": 3.75
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
left unset in config, same as `cost_usd` computation uses).
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
  candidate), `error` (fatal, chain aborted), or `not_configured`
  (candidate skipped, no resolved API key).
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

Liveness check.

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
set (otherwise uncapped). Only `POST /v1/chat/completions` is rate
limited — metadata endpoints (`/v1/models`, `/v1/usage`, `/metrics`)
aren't. Rejections return `429` with a `Retry-After` header.

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
```

This router POSTs a JSON body to `url` on two events:

```jsonc
// A client's tracked spend just reached or passed its budget.
{"event": "budget_exceeded", "client": "hermes", "spent_usd": 51.20, "budget_usd": 50.0, "period": "monthly"}
// An operator manually reset a client's spend via the admin API.
{"event": "budget_reset", "client": "hermes", "budget_usd": 50.0, "period": "monthly"}
```

`budget_exceeded` fires on the specific request that pushes tracked spend
from under budget to at-or-over it, not on every subsequent over-budget
request — the request that crossed it is still charged and let through
before this fires, same as the `402` only starting on the *next* request.
Under `[persistence]`, "just crossed" is a best-effort, eventually-consistent
read-back rather than an atomic check-and-set, so two concurrent requests
to the same client right at the boundary could both fire (or, rarely,
neither) — same class of caveat the tracked spend total itself already
carries. `auth_header_env` names an env var holding the exact value to
send as this POST's `Authorization` header (e.g. `"Bearer <token>"`), so
the receiver can verify the request came from this router; leaving it
unset sends no `Authorization` header at all. Delivery is fire-and-forget
— a slow or unreachable receiver never adds latency to the request that
triggered the event, and a delivery failure is only logged, never
surfaced to the client.

## Admin API

Setting `server.admin_key_env` unlocks a small admin API for inspecting
and managing configured clients' spend budgets:

```toml
[server]
admin_key_env = "RUSTY_PROVIDER_ADMIN_KEY"
```

- **`GET /v1/admin/clients`** — every configured `[[clients]]` entry, its
  `requests_per_minute`, and (for clients with `budget_usd` set) its
  current-period `spent_usd` and `budget_period`. A client with no
  configured budget still appears, with `budget_usd`/`budget_period`/
  `spent_usd` all `null`.
- **`POST /v1/admin/clients/{name}/reset-spend`** — zeroes that client's
  tracked spend for the current period, immediately un-blocking a client
  that's hit `402`. `404` for a client name that doesn't exist or has no
  configured budget.
- **`POST /v1/admin/clients`** — provisions a new client at runtime, no
  config-file edit or restart needed. Body:
  ```jsonc
  {
    "name": "acme",
    "requests_per_minute": 60,
    "budget_usd": 10.0,       // optional, omit for unrestricted
    "budget_period": "monthly", // optional, "total" (default) / "daily" / "weekly" / "monthly"
    "api_key": "..."          // optional -- omit to have the server generate one
  }
  ```
  Responds `201` with the same shape plus `api_key` — the server-generated
  key (if you didn't supply one) is only ever shown in this response, the
  same hygiene as GitHub/Stripe-style API keys, so save it immediately.
  `400` for an empty `name`, a `requests_per_minute` of `0`, or a negative
  `budget_usd`; `409` if `name` or `api_key` collides with an existing
  client.
- **`PATCH /v1/admin/clients/{name}`** — updates an existing client
  (config-defined or runtime-provisioned). Every field is optional and
  independent: omit a field to leave it unchanged, send `"budget_usd":
  null` to explicitly clear a configured budget (as opposed to omitting
  `budget_usd` entirely, which leaves it as-is), and set
  `"rotate_api_key": true` to revoke the client's current key and issue a
  new one, returned in the response the same one-time way creation does.
  `404` for an unknown client, `400` for an invalid `requests_per_minute`/
  `budget_usd`.
- **`DELETE /v1/admin/clients/{name}`** — removes a client entirely,
  immediately revoking its key and dropping its budget/spend tracking.
  `404` for an unknown client.

Requests to every route above need `Authorization: Bearer <token>` matching
`admin_key_env`'s resolved value — **not** `server.api_key_env` or any
`[[clients]]` key, which authenticate chat completions but deliberately
don't also grant access to every other client's spend data or the ability
to provision/reset/delete clients. Leaving `admin_key_env` unset disables
the admin API entirely: every route `404`s, as if it didn't exist, rather
than silently falling open once *any* auth is configured elsewhere.

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
