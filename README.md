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

### `GET /health`

Liveness check.

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

- Usage metering / billing
- Multi-turn image or audio content
