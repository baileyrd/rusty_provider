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

### `GET /v1/models`

Lists configured route aliases and `provider/*` for every provider with a
resolved API key.

### `GET /health`

Liveness check.

## Config

See `config.example.toml`. Provider API keys are always read from
environment variables (named by `api_key_env`) — never stored in the
config file itself.

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

- Per-request cost/latency-based routing (only manual model selection and
  ordered fallback chains today)
- Usage metering / billing
- Multi-turn image or audio content
