# rusty_provider

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

If `[[pricing]]` has an entry for the model that actually served the
request, the response (and, for streaming, whichever chunk carries the
final `usage`) includes an extra `cost_usd` field — the request's
estimated dollar cost, computed from `usage.prompt_tokens` /
`usage.completion_tokens` against that pricing entry. It's not part of the
OpenAI schema, so existing OpenAI SDKs/clients just ignore it; it's simply
absent (not `0`/`null`) when the model has no configured pricing, so don't
read a missing field as "this was free." Every request also adds to a
running per-model total queryable at `GET /v1/usage` (below), whether or
not pricing is configured for it.

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

### `GET /v1/models`

Lists configured route aliases and `provider/*` for every provider with a
resolved API key.

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

Like the latency/throughput metrics, this is in-memory only — it resets on
restart and isn't persisted or shared across processes. `cost_usd` only
accumulates for models with a `[[pricing]]` entry; it stays `0.0` for
everything else (which means "unpriced," not "free" — `requests` and
`*_tokens` still count normally regardless of pricing).

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

## Not yet implemented

- Billing / spend-based limits (cost estimation, cumulative usage
  tracking, and requests-per-minute rate limiting are supported — see
  `cost_usd`, `GET /v1/usage`, and Rate limiting above — but there's no
  persistence, multi-process aggregation, or spend-based enforcement, e.g.
  cutting a client off once they've cost you $X rather than once they've
  sent N requests/minute)
- Multi-turn image or audio content
