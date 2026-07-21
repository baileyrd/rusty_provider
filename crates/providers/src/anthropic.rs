//! Adapter for the Anthropic Messages API (`/v1/messages`). Differs from
//! the OpenAI shape in several ways this module has to bridge: system
//! prompts are a top-level field rather than a message role, `max_tokens`
//! is required rather than optional, message content is a list of typed
//! blocks rather than a plain string once tool use is involved, and
//! streaming is a sequence of typed SSE events rather than uniform delta
//! chunks.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use rp_core::{
    CacheControl, ChatChunk, ChatMessage, ChatMessageDelta, ChatRequest, ChatResponse, ChatStream,
    Choice, ChunkChoice, ContentPart, FunctionCallDelta, MessageContent, Provider, ProviderError,
    ReasoningConfig, ResponseFormat, Role, Tool, ToolCall, ToolCallDelta, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::http::{map_error_response, map_reqwest_error};
use crate::util::{effort_thinking_budget, gen_id, now};

const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic requires `max_tokens`; used when the client doesn't specify one.
const DEFAULT_MAX_TOKENS: u32 = 4096;
/// Anthropic rejects `budget_tokens` below this.
const MIN_THINKING_BUDGET: u32 = 1024;

pub struct AnthropicProvider {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }
}

#[derive(Debug, Serialize)]
struct WireMessage {
    role: &'static str,
    content: Vec<Value>,
}

#[derive(Debug, Serialize)]
struct WireTool<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    input_schema: Value,
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<&'a [String]>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<WireTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Value>,
    /// Set when `response_format` asked for schema-constrained output: the
    /// name of the synthetic tool `tools`/`tool_choice` above were built
    /// from, so `chat`/`chat_stream` can recognize that tool's `tool_use`
    /// block in the response and unwrap it into plain content instead of
    /// surfacing it as a real tool call. Never sent on the wire.
    #[serde(skip)]
    forced_output_tool_name: Option<String>,
    /// Mirrors `reasoning.exclude` -- Anthropic has no server-side toggle to
    /// suppress `thinking` blocks the way Gemini's `includeThoughts` does,
    /// so `chat`/`chat_stream` drop them client-side when this is set.
    /// Never sent on the wire.
    #[serde(skip)]
    exclude_reasoning: bool,
}

/// Stamps a `cache_control` breakpoint (Anthropic's `{"type":"ephemeral"}`
/// block, serialized straight from `CacheControl` with no translation)
/// onto the last block in `blocks`, if `cache_control` is set. Anthropic
/// treats a breakpoint as covering everything up to and including the
/// block it's on, so the last block is always the right place for one
/// message's marker.
fn apply_cache_control(blocks: &mut [Value], cache_control: Option<&CacheControl>) {
    let Some(cache_control) = cache_control else {
        return;
    };
    if let Some(Value::Object(last)) = blocks.last_mut() {
        last.insert(
            "cache_control".to_string(),
            serde_json::to_value(cache_control).unwrap_or(Value::Null),
        );
    }
}

/// Split messages into Anthropic's shape: a top-level `system` (a plain
/// string when no system message requests caching, matching prior
/// behavior exactly; an array of blocks, one per `Role::System` message,
/// when at least one does -- only the block form can carry
/// `cache_control`) plus a list of user/assistant turns, each with content
/// as typed blocks — text, `tool_use` (an assistant message's
/// `tool_calls`), or `tool_result` (a `Role::Tool` message answering one).
///
/// Errs with `ProviderError::UnsupportedContent` if any user message
/// contains audio -- Anthropic's Messages API has no audio-input support,
/// unlike image content -- so a fallback chain can move on to a candidate
/// that might support it instead of silently dropping it.
fn build_messages(
    messages: &[ChatMessage],
) -> Result<(Option<Value>, Vec<WireMessage>), ProviderError> {
    let mut system_blocks: Vec<Value> = Vec::new();
    let mut turns = Vec::new();

    for m in messages {
        match m.role {
            Role::System => {
                if let Some(c) = &m.content {
                    let mut block = json!({"type": "text", "text": c.as_plain_text()});
                    apply_cache_control(std::slice::from_mut(&mut block), m.cache_control.as_ref());
                    system_blocks.push(block);
                }
            }
            Role::Tool => {
                let mut content = vec![json!({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                    "content": m.content.as_ref().map(MessageContent::as_plain_text).unwrap_or_default(),
                })];
                apply_cache_control(&mut content, m.cache_control.as_ref());
                turns.push(WireMessage {
                    role: "user",
                    content,
                });
            }
            Role::User => {
                let mut content = match &m.content {
                    Some(content) => content_to_blocks(content)?,
                    None => vec![json!({"type": "text", "text": ""})],
                };
                apply_cache_control(&mut content, m.cache_control.as_ref());
                turns.push(WireMessage {
                    role: "user",
                    content,
                });
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                if let Some(content) = &m.content {
                    let text = content.as_plain_text();
                    if !text.is_empty() {
                        blocks.push(json!({"type": "text", "text": text}));
                    }
                }
                for tc in m.tool_calls.iter().flatten() {
                    let input: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": input,
                    }));
                }
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": ""}));
                }
                apply_cache_control(&mut blocks, m.cache_control.as_ref());
                turns.push(WireMessage {
                    role: "assistant",
                    content: blocks,
                });
            }
        }
    }

    let system = if system_blocks.is_empty() {
        None
    } else if system_blocks
        .iter()
        .all(|b| b.get("cache_control").is_none())
    {
        // No caching requested anywhere in system -- keep the plain-string
        // shape every existing caller/test already expects.
        let text = system_blocks
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        Some(Value::String(text))
    } else {
        Some(Value::Array(system_blocks))
    };
    Ok((system, turns))
}

/// Translates a message's content into Anthropic content blocks, turning
/// `image_url` parts into `image` blocks and `file` parts into `document`
/// blocks: a `data:<mime>;base64,<data>` URI becomes a `base64` source,
/// anything else (an `https://` URL) a `url` source. Errs on `input_audio`
/// parts -- Anthropic's Messages API has no audio-input support to
/// translate them into.
fn content_to_blocks(content: &MessageContent) -> Result<Vec<Value>, ProviderError> {
    match content {
        MessageContent::Text(text) => Ok(vec![json!({"type": "text", "text": text})]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(json!({"type": "text", "text": text})),
                ContentPart::ImageUrl { image_url } => Ok(image_block(&image_url.url)),
                ContentPart::File { file } => Ok(document_block(&file.file_data)),
                ContentPart::InputAudio { .. } => Err(ProviderError::UnsupportedContent(
                    "Anthropic's Messages API does not support audio input content".to_string(),
                )),
            })
            .collect(),
    }
}

fn image_block(url: &str) -> Value {
    match parse_data_uri(url) {
        Some((media_type, data)) => json!({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }),
        None => json!({
            "type": "image",
            "source": {"type": "url", "url": url},
        }),
    }
}

/// Anthropic's native PDF support: a `document` block, using the same
/// `base64`-vs-`url` source split as `image_block`.
fn document_block(file_data: &str) -> Value {
    match parse_data_uri(file_data) {
        Some((media_type, data)) => json!({
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }),
        None => json!({
            "type": "document",
            "source": {"type": "url", "url": file_data},
        }),
    }
}

/// Parses a `data:<mime>;base64,<data>` URI into its `(mime, data)` parts.
/// Returns `None` for anything else (e.g. a plain `https://` URL).
fn parse_data_uri(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media_type = meta.strip_suffix(";base64")?;
    Some((media_type, data))
}

fn to_wire_tools(tools: &[Tool]) -> Vec<WireTool<'_>> {
    tools
        .iter()
        .map(|t| WireTool {
            name: &t.function.name,
            description: t.function.description.as_deref(),
            input_schema: t
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        })
        .collect()
}

/// Map OpenAI's `tool_choice` vocabulary (`"auto"` / `"none"` /
/// `"required"` / `{"type":"function","function":{"name":...}}`) onto
/// Anthropic's (`{"type":"auto"|"any"|"none"|"tool", "name"?:...}`).
fn to_wire_tool_choice(choice: &Value) -> Value {
    match choice {
        Value::String(s) if s == "none" => json!({"type": "none"}),
        Value::String(s) if s == "required" => json!({"type": "any"}),
        Value::Object(_) => match choice.pointer("/function/name").and_then(Value::as_str) {
            Some(name) => json!({"type": "tool", "name": name}),
            None => json!({"type": "auto"}),
        },
        _ => json!({"type": "auto"}),
    }
}

/// Anthropic's Messages API has no native `response_format` -- so
/// schema-constrained output (`ResponseFormat::JsonSchema`) is faked by
/// defining a single synthetic tool whose `input_schema` is the requested
/// schema, then forcing the model to call it (`tool_choice: {"type":"tool",
/// "name": ...}}`). The caller unwraps that forced tool_use block back into
/// plain JSON content instead of surfacing it as a real tool call -- see
/// `chat`/`chat_stream`.
///
/// `ResponseFormat::JsonObject` (loose, schema-less JSON mode) has no
/// equivalent trick: there's no schema to build a tool from, and nothing in
/// Anthropic's API reliably constrains output to "valid JSON, any shape".
/// That's rejected with `UnsupportedFeature` so a fallback chain can move on
/// to a provider (OpenAI-compatible, Gemini) that actually supports it,
/// rather than silently ignoring the request.
fn forced_structured_output_tool(req: &ChatRequest) -> Result<Option<WireTool<'_>>, ProviderError> {
    match &req.response_format {
        None | Some(ResponseFormat::Text) => Ok(None),
        Some(ResponseFormat::JsonObject) => Err(ProviderError::UnsupportedFeature(
            "Anthropic's Messages API has no schema-less JSON response mode".to_string(),
        )),
        Some(ResponseFormat::JsonSchema { json_schema }) => Ok(Some(WireTool {
            name: &json_schema.name,
            description: json_schema.description.as_deref(),
            input_schema: json_schema.schema.clone(),
        })),
    }
}

/// Builds Anthropic's `thinking: {"type": "enabled", "budget_tokens": N}`
/// request field, picking `budget_tokens` the same way Gemini's
/// `thinkingBudget` is picked (explicit `reasoning.max_tokens` if set, else
/// a fraction of the request's own `max_tokens`, else a flat default) but
/// clamped to Anthropic's minimum of `MIN_THINKING_BUDGET`.
fn thinking_config(reasoning: &ReasoningConfig, req_max_tokens: Option<u32>) -> Value {
    let budget = reasoning
        .max_tokens
        .unwrap_or_else(|| effort_thinking_budget(reasoning.effort.as_deref(), req_max_tokens))
        .max(MIN_THINKING_BUDGET);
    json!({"type": "enabled", "budget_tokens": budget})
}

/// Builds a `Usage` from Anthropic's wire usage object, folding
/// `input_tokens` (non-cached) and the two cache counters into a single
/// cache-inclusive `prompt_tokens` total, matching how OpenAI/Gemini
/// already report it -- so cost accounting and clients can treat
/// `prompt_tokens` uniformly across providers regardless of whether any of
/// it came from a cache.
fn usage_from_wire(wire: &WireUsage) -> Usage {
    let cached = wire.cache_read_input_tokens;
    let cache_creation = wire.cache_creation_input_tokens;
    let prompt_tokens = wire.input_tokens + cached + cache_creation;
    Usage {
        prompt_tokens,
        completion_tokens: wire.output_tokens,
        total_tokens: prompt_tokens + wire.output_tokens,
        cached_tokens: (cached > 0).then_some(cached),
        cache_creation_tokens: (cache_creation > 0).then_some(cache_creation),
    }
}

fn map_stop_reason(reason: &str) -> &'static str {
    match reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        _ => "stop",
    }
}

impl<'a> WireRequest<'a> {
    fn from_core(
        req: &'a ChatRequest,
        model: &'a str,
        stream: bool,
    ) -> Result<Self, ProviderError> {
        let (system, messages) = build_messages(&req.messages)?;

        let (tools, tool_choice, forced_output_tool_name) =
            match forced_structured_output_tool(req)? {
                Some(tool) => {
                    let name = tool.name.to_string();
                    (
                        Some(vec![tool]),
                        Some(json!({"type": "tool", "name": name})),
                        Some(name),
                    )
                }
                None => (
                    req.tools.as_deref().map(to_wire_tools),
                    req.tool_choice.as_ref().map(to_wire_tool_choice),
                    None,
                ),
            };

        let thinking = req
            .reasoning
            .as_ref()
            .map(|r| thinking_config(r, req.max_tokens));
        let budget_tokens = thinking
            .as_ref()
            .and_then(|t| t.get("budget_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as u32);
        let mut max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        if let Some(budget) = budget_tokens {
            // Anthropic requires `max_tokens > budget_tokens`.
            if max_tokens <= budget {
                max_tokens = budget + DEFAULT_MAX_TOKENS;
            }
        }
        let exclude_reasoning = req
            .reasoning
            .as_ref()
            .and_then(|r| r.exclude)
            .unwrap_or(false);

        Ok(Self {
            model,
            max_tokens,
            messages,
            system,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            stop_sequences: req.stop.as_deref(),
            stream,
            tools,
            tool_choice,
            thinking,
            forced_output_tool_name,
            exclude_reasoning,
        })
    }
}

#[derive(Deserialize)]
struct WireContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
    /// Set on `"thinking"` blocks. `"redacted_thinking"` blocks carry an
    /// encrypted `data` field instead, which has no readable text to
    /// surface -- those are counted but otherwise ignored.
    #[serde(default)]
    thinking: String,
}

#[derive(Deserialize)]
struct WireUsage {
    /// Non-cached input tokens only -- `prompt_tokens` in our `Usage` adds
    /// this to the two cache fields below to get the true total, matching
    /// how OpenAI/Gemini already report a cache-inclusive total.
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

#[derive(Deserialize)]
struct WireResponse {
    content: Vec<WireContentBlock>,
    stop_reason: Option<String>,
    usage: WireUsage,
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn chat(&self, req: &ChatRequest, model: &str) -> Result<ChatResponse, ProviderError> {
        let body = WireRequest::from_core(req, model, false)?;
        let forced_output_tool_name = body.forced_output_tool_name.clone();
        let exclude_reasoning = body.exclude_reasoning;

        let resp = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            return Err(map_error_response(resp).await);
        }

        let wire: WireResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Decode(e.to_string()))?;

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();
        for b in &wire.content {
            match b.kind.as_str() {
                "text" => text.push_str(&b.text),
                "thinking" => reasoning.push_str(&b.thinking),
                "tool_use" => {
                    let id = b.id.clone().unwrap_or_default();
                    let tool_name = b.name.clone().unwrap_or_default();
                    let arguments = b.input.clone().unwrap_or_else(|| json!({})).to_string();
                    // The forced structured-output tool's call is the
                    // client's actual JSON answer, not a real tool call --
                    // fold its arguments into `text` instead of surfacing
                    // it as a `tool_calls` entry the client would have to
                    // answer with a follow-up `role: "tool"` message.
                    if forced_output_tool_name.as_deref() == Some(tool_name.as_str()) {
                        text.push_str(&arguments);
                    } else {
                        tool_calls.push(ToolCall::function(id, tool_name, arguments));
                    }
                }
                _ => {}
            }
        }
        let reasoning = if exclude_reasoning || reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        };

        let finish_reason = if forced_output_tool_name.is_some() {
            // Anthropic reports `stop_reason: "tool_use"` for the forced
            // call, which `map_stop_reason` would turn into `"tool_calls"`
            // -- wrong here, since from the client's perspective this
            // completed normally with a JSON answer, not a tool call it
            // needs to act on.
            Some("stop".to_string())
        } else {
            wire.stop_reason
                .as_deref()
                .map(map_stop_reason)
                .map(str::to_string)
        };

        Ok(ChatResponse {
            id: gen_id("chatcmpl"),
            object: "chat.completion",
            created: now(),
            model: format!("anthropic/{model}"),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: if text.is_empty() {
                        None
                    } else {
                        Some(MessageContent::Text(text))
                    },
                    name: None,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                    reasoning,
                    cache_control: None,
                },
                finish_reason,
            }],
            usage: Some(usage_from_wire(&wire.usage)),
            cost_usd: None,
        })
    }

    async fn chat_stream(
        &self,
        req: &ChatRequest,
        model: &str,
    ) -> Result<ChatStream, ProviderError> {
        let body = WireRequest::from_core(req, model, true)?;
        let is_forced_structured_output = body.forced_output_tool_name.is_some();
        let exclude_reasoning = body.exclude_reasoning;

        let resp = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            return Err(map_error_response(resp).await);
        }

        let full_model = format!("anthropic/{model}");
        let mut input_tokens: u32 = 0;
        let mut cache_creation_tokens: u32 = 0;
        let mut cached_tokens: u32 = 0;

        let stream = resp.bytes_stream().eventsource().filter_map(move |ev| {
            let full_model = full_model.clone();
            let result = (|| -> Option<Result<ChatChunk, ProviderError>> {
                let ev = match ev {
                    Ok(ev) => ev,
                    Err(e) => return Some(Err(ProviderError::Decode(e.to_string()))),
                };
                let value: Value = match serde_json::from_str(&ev.data) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(ProviderError::Decode(e.to_string()))),
                };
                let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
                let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;

                match kind {
                    "message_start" => {
                        input_tokens = value
                            .pointer("/message/usage/input_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as u32;
                        cache_creation_tokens = value
                            .pointer("/message/usage/cache_creation_input_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as u32;
                        cached_tokens = value
                            .pointer("/message/usage/cache_read_input_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as u32;
                        Some(Ok(empty_chunk(
                            &full_model,
                            ChatMessageDelta {
                                role: Some(Role::Assistant),
                                ..Default::default()
                            },
                            None,
                        )))
                    }
                    "content_block_start" => {
                        if value.pointer("/content_block/type").and_then(Value::as_str)
                            != Some("tool_use")
                        {
                            return None;
                        }
                        // The forced structured-output tool's call streams
                        // as plain content (see `content_block_delta`
                        // below), not a tool call the client has to
                        // recognize the start of.
                        if is_forced_structured_output {
                            return None;
                        }
                        let id = value
                            .pointer("/content_block/id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let tool_name = value
                            .pointer("/content_block/name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        Some(Ok(empty_chunk(
                            &full_model,
                            ChatMessageDelta {
                                tool_calls: Some(vec![ToolCallDelta {
                                    index,
                                    id: Some(id),
                                    kind: Some("function".to_string()),
                                    function: Some(FunctionCallDelta {
                                        name: Some(tool_name),
                                        arguments: Some(String::new()),
                                    }),
                                }]),
                                ..Default::default()
                            },
                            None,
                        )))
                    }
                    "content_block_delta" => {
                        match value.pointer("/delta/type").and_then(Value::as_str) {
                            Some("text_delta") => {
                                let text = value
                                    .pointer("/delta/text")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();
                                Some(Ok(empty_chunk(
                                    &full_model,
                                    ChatMessageDelta {
                                        content: Some(text),
                                        ..Default::default()
                                    },
                                    None,
                                )))
                            }
                            Some("input_json_delta") => {
                                let partial = value
                                    .pointer("/delta/partial_json")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();
                                // The forced structured-output tool's
                                // streamed input *is* the JSON the client
                                // asked for -- deliver it as accumulating
                                // `content`, the same shape every other
                                // provider streams a JSON-mode answer in,
                                // rather than a tool-call argument delta.
                                let delta = if is_forced_structured_output {
                                    ChatMessageDelta {
                                        content: Some(partial),
                                        ..Default::default()
                                    }
                                } else {
                                    ChatMessageDelta {
                                        tool_calls: Some(vec![ToolCallDelta {
                                            index,
                                            id: None,
                                            kind: None,
                                            function: Some(FunctionCallDelta {
                                                name: None,
                                                arguments: Some(partial),
                                            }),
                                        }]),
                                        ..Default::default()
                                    }
                                };
                                Some(Ok(empty_chunk(&full_model, delta, None)))
                            }
                            Some("thinking_delta") => {
                                if exclude_reasoning {
                                    return None;
                                }
                                let text = value
                                    .pointer("/delta/thinking")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();
                                if text.is_empty() {
                                    return None;
                                }
                                Some(Ok(empty_chunk(
                                    &full_model,
                                    ChatMessageDelta {
                                        reasoning: Some(text),
                                        ..Default::default()
                                    },
                                    None,
                                )))
                            }
                            _ => None,
                        }
                    }
                    "message_delta" => {
                        let stop_reason = if is_forced_structured_output {
                            // See the `chat` (non-streaming) method for why
                            // this is always "stop" rather than whatever
                            // `map_stop_reason` would make of Anthropic's
                            // "tool_use" here.
                            Some("stop".to_string())
                        } else {
                            value
                                .pointer("/delta/stop_reason")
                                .and_then(Value::as_str)
                                .map(map_stop_reason)
                                .map(str::to_string)
                        };
                        let output_tokens = value
                            .pointer("/usage/output_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as u32;
                        let mut chunk =
                            empty_chunk(&full_model, ChatMessageDelta::default(), stop_reason);
                        let prompt_tokens = input_tokens + cache_creation_tokens + cached_tokens;
                        chunk.usage = Some(Usage {
                            prompt_tokens,
                            completion_tokens: output_tokens,
                            total_tokens: prompt_tokens + output_tokens,
                            cached_tokens: (cached_tokens > 0).then_some(cached_tokens),
                            cache_creation_tokens: (cache_creation_tokens > 0)
                                .then_some(cache_creation_tokens),
                        });
                        Some(Ok(chunk))
                    }
                    "message_stop" | "content_block_stop" | "ping" => None,
                    "error" => {
                        let message = value
                            .pointer("/error/message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown error")
                            .to_string();
                        Some(Err(ProviderError::Upstream {
                            status: 500,
                            message,
                        }))
                    }
                    _ => None,
                }
            })();
            async move { result }
        });

        Ok(Box::pin(stream))
    }
}

fn empty_chunk(model: &str, delta: ChatMessageDelta, finish_reason: Option<String>) -> ChatChunk {
    ChatChunk {
        id: gen_id("chatcmpl"),
        object: "chat.completion.chunk",
        created: now(),
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage: None,
        cost_usd: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rp_core::FunctionDef;

    fn tool(name: &str, description: Option<&str>, parameters: Option<Value>) -> Tool {
        Tool {
            kind: "function".to_string(),
            function: FunctionDef {
                name: name.to_string(),
                description: description.map(str::to_string),
                parameters,
            },
        }
    }

    // --- to_wire_tools -----------------------------------------------------

    #[test]
    fn to_wire_tools_maps_name_description_and_parameters() {
        let params = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let tools = vec![tool(
            "get_weather",
            Some("Look up the weather"),
            Some(params.clone()),
        )];
        let wire = to_wire_tools(&tools);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].name, "get_weather");
        assert_eq!(wire[0].description, Some("Look up the weather"));
        assert_eq!(wire[0].input_schema, params);
    }

    #[test]
    fn to_wire_tools_defaults_input_schema_when_parameters_absent() {
        let tools = vec![tool("no_args_tool", None, None)];
        let wire = to_wire_tools(&tools);
        assert_eq!(
            wire[0].input_schema,
            json!({"type": "object", "properties": {}})
        );
        assert_eq!(wire[0].description, None);
    }

    #[test]
    fn to_wire_tools_preserves_order_across_multiple_tools() {
        let tools = vec![tool("first", None, None), tool("second", None, None)];
        let wire = to_wire_tools(&tools);
        assert_eq!(wire[0].name, "first");
        assert_eq!(wire[1].name, "second");
    }

    // --- forced_structured_output_tool ----------------------------------------

    fn request_with_response_format(response_format: Option<ResponseFormat>) -> ChatRequest {
        ChatRequest {
            model: "anthropic/claude-sonnet-5".to_string(),
            models: None,
            messages: vec![ChatMessage::user("hi")],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: None,
            user: None,
            tools: None,
            tool_choice: None,
            provider: None,
            response_format,
            reasoning: None,
            top_k: None,
            min_p: None,
            top_a: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            logit_bias: None,
            seed: None,
            transforms: None,
        }
    }

    #[test]
    fn forced_structured_output_tool_is_none_without_a_response_format() {
        let req = request_with_response_format(None);
        assert!(forced_structured_output_tool(&req).unwrap().is_none());
    }

    #[test]
    fn forced_structured_output_tool_is_none_for_text() {
        let req = request_with_response_format(Some(ResponseFormat::Text));
        assert!(forced_structured_output_tool(&req).unwrap().is_none());
    }

    #[test]
    fn forced_structured_output_tool_errs_as_unsupported_for_json_object() {
        let req = request_with_response_format(Some(ResponseFormat::JsonObject));
        let err = forced_structured_output_tool(&req).unwrap_err();
        assert!(matches!(err, ProviderError::UnsupportedFeature(_)));
        assert!(
            err.is_retryable(),
            "a fallback chain should move on to a provider with schema-less JSON mode"
        );
    }

    #[test]
    fn forced_structured_output_tool_builds_a_tool_from_the_schema() {
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let req = request_with_response_format(Some(ResponseFormat::JsonSchema {
            json_schema: rp_core::JsonSchemaFormat {
                name: "weather_report".to_string(),
                description: Some("A weather report".to_string()),
                schema: schema.clone(),
                strict: Some(true),
            },
        }));
        let tool = forced_structured_output_tool(&req).unwrap().unwrap();
        assert_eq!(tool.name, "weather_report");
        assert_eq!(tool.description, Some("A weather report"));
        assert_eq!(tool.input_schema, schema);
    }

    // --- WireRequest::from_core: structured output wiring ----------------------

    #[test]
    fn from_core_forces_tool_choice_to_the_schema_tool_and_records_its_name() {
        let req = request_with_response_format(Some(ResponseFormat::JsonSchema {
            json_schema: rp_core::JsonSchemaFormat {
                name: "weather_report".to_string(),
                description: None,
                schema: json!({"type": "object"}),
                strict: None,
            },
        }));
        let wire = WireRequest::from_core(&req, "claude-sonnet-5", false).unwrap();
        assert_eq!(
            wire.forced_output_tool_name.as_deref(),
            Some("weather_report")
        );
        assert_eq!(wire.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            wire.tool_choice,
            Some(json!({"type": "tool", "name": "weather_report"}))
        );
    }

    #[test]
    fn from_core_leaves_tools_and_forced_name_alone_without_a_response_format() {
        let req = request_with_response_format(None);
        let wire = WireRequest::from_core(&req, "claude-sonnet-5", false).unwrap();
        assert!(wire.forced_output_tool_name.is_none());
        assert!(wire.tools.is_none());
        assert!(wire.tool_choice.is_none());
    }

    // --- to_wire_tool_choice -------------------------------------------------

    #[test]
    fn to_wire_tool_choice_maps_auto() {
        assert_eq!(to_wire_tool_choice(&json!("auto")), json!({"type": "auto"}));
    }

    #[test]
    fn to_wire_tool_choice_maps_none() {
        assert_eq!(to_wire_tool_choice(&json!("none")), json!({"type": "none"}));
    }

    #[test]
    fn to_wire_tool_choice_maps_required_to_any() {
        assert_eq!(
            to_wire_tool_choice(&json!("required")),
            json!({"type": "any"})
        );
    }

    #[test]
    fn to_wire_tool_choice_maps_named_function() {
        let choice = json!({"type": "function", "function": {"name": "get_weather"}});
        assert_eq!(
            to_wire_tool_choice(&choice),
            json!({"type": "tool", "name": "get_weather"})
        );
    }

    #[test]
    fn to_wire_tool_choice_falls_back_to_auto_for_an_object_without_a_function_name() {
        let choice = json!({"type": "function"});
        assert_eq!(to_wire_tool_choice(&choice), json!({"type": "auto"}));
    }

    #[test]
    fn to_wire_tool_choice_falls_back_to_auto_for_an_unrecognized_shape() {
        assert_eq!(to_wire_tool_choice(&json!(null)), json!({"type": "auto"}));
        assert_eq!(
            to_wire_tool_choice(&json!("something-else")),
            json!({"type": "auto"})
        );
    }

    // --- map_stop_reason -----------------------------------------------------

    #[test]
    fn map_stop_reason_maps_tool_use_to_tool_calls() {
        assert_eq!(map_stop_reason("tool_use"), "tool_calls");
    }

    #[test]
    fn map_stop_reason_maps_every_documented_reason() {
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("unknown_future_reason"), "stop");
    }

    // --- usage_from_wire --------------------------------------------------------

    #[test]
    fn usage_from_wire_folds_cache_counters_into_a_cache_inclusive_prompt_total() {
        let wire = WireUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 20,
            cache_read_input_tokens: 30,
        };
        let usage = usage_from_wire(&wire);
        assert_eq!(usage.prompt_tokens, 150);
        assert_eq!(usage.total_tokens, 200);
        assert_eq!(usage.cached_tokens, Some(30));
        assert_eq!(usage.cache_creation_tokens, Some(20));
    }

    #[test]
    fn usage_from_wire_leaves_cache_fields_none_when_nothing_was_cached() {
        let wire = WireUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        };
        let usage = usage_from_wire(&wire);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.cached_tokens, None);
        assert_eq!(usage.cache_creation_tokens, None);
    }

    // --- parse_data_uri --------------------------------------------------------

    #[test]
    fn parse_data_uri_extracts_media_type_and_data() {
        assert_eq!(
            parse_data_uri("data:image/png;base64,aGVsbG8="),
            Some(("image/png", "aGVsbG8="))
        );
    }

    #[test]
    fn parse_data_uri_rejects_a_plain_url() {
        assert_eq!(parse_data_uri("https://example.com/a.png"), None);
    }

    #[test]
    fn parse_data_uri_rejects_a_non_base64_data_uri() {
        assert_eq!(parse_data_uri("data:text/plain,hello"), None);
    }

    // --- image_block -----------------------------------------------------------

    #[test]
    fn image_block_uses_base64_source_for_a_data_uri() {
        assert_eq!(
            image_block("data:image/jpeg;base64,aGVsbG8="),
            json!({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "aGVsbG8="},
            })
        );
    }

    #[test]
    fn image_block_uses_url_source_for_an_https_url() {
        assert_eq!(
            image_block("https://example.com/a.png"),
            json!({
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/a.png"},
            })
        );
    }

    // --- apply_cache_control -----------------------------------------------------

    #[test]
    fn apply_cache_control_is_a_no_op_when_unset() {
        let mut blocks = vec![json!({"type": "text", "text": "hi"})];
        apply_cache_control(&mut blocks, None);
        assert_eq!(blocks, vec![json!({"type": "text", "text": "hi"})]);
    }

    #[test]
    fn apply_cache_control_stamps_only_the_last_block() {
        let mut blocks = vec![
            json!({"type": "text", "text": "first"}),
            json!({"type": "text", "text": "second"}),
        ];
        apply_cache_control(&mut blocks, Some(&CacheControl::Ephemeral));
        assert_eq!(blocks[0].get("cache_control"), None);
        assert_eq!(blocks[1]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn apply_cache_control_on_an_empty_slice_does_nothing() {
        let mut blocks: Vec<Value> = vec![];
        apply_cache_control(&mut blocks, Some(&CacheControl::Ephemeral));
        assert!(blocks.is_empty());
    }

    // --- build_messages: cache_control -------------------------------------------

    #[test]
    fn build_messages_leaves_system_as_a_plain_string_without_cache_control() {
        let messages = vec![ChatMessage::system("be concise"), ChatMessage::user("hi")];
        let (system, _) = build_messages(&messages).unwrap();
        assert_eq!(system, Some(json!("be concise")));
    }

    #[test]
    fn build_messages_turns_system_into_a_block_array_with_cache_control_when_requested() {
        let mut system_msg = ChatMessage::system("be concise");
        system_msg.cache_control = Some(CacheControl::Ephemeral);
        let messages = vec![system_msg, ChatMessage::user("hi")];
        let (system, _) = build_messages(&messages).unwrap();
        assert_eq!(
            system,
            Some(json!([{
                "type": "text",
                "text": "be concise",
                "cache_control": {"type": "ephemeral"},
            }]))
        );
    }

    #[test]
    fn build_messages_stamps_cache_control_on_a_user_turns_last_block() {
        let mut user_msg = ChatMessage::user("hi");
        user_msg.cache_control = Some(CacheControl::Ephemeral);
        let (_, turns) = build_messages(&[user_msg]).unwrap();
        assert_eq!(
            turns[0].content,
            vec![json!({"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}})]
        );
    }

    #[test]
    fn build_messages_stamps_cache_control_on_a_tool_result_block() {
        let tool_msg = ChatMessage {
            role: Role::Tool,
            content: Some(MessageContent::text("42")),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning: None,
            cache_control: Some(CacheControl::Ephemeral),
        };
        let (_, turns) = build_messages(&[tool_msg]).unwrap();
        assert_eq!(
            turns[0].content[0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    // --- content_to_blocks -------------------------------------------------------

    #[test]
    fn content_to_blocks_wraps_plain_text_as_a_single_text_block() {
        let content = MessageContent::text("hi");
        assert_eq!(
            content_to_blocks(&content).unwrap(),
            vec![json!({"type": "text", "text": "hi"})]
        );
    }

    #[test]
    fn content_to_blocks_translates_mixed_text_and_image_parts() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "what's in this image?".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: rp_core::ImageUrl {
                    url: "data:image/png;base64,aGVsbG8=".to_string(),
                    detail: None,
                },
            },
        ]);
        assert_eq!(
            content_to_blocks(&content).unwrap(),
            vec![
                json!({"type": "text", "text": "what's in this image?"}),
                json!({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
                }),
            ]
        );
    }

    #[test]
    fn content_to_blocks_errs_on_an_audio_part() {
        let content = MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: rp_core::InputAudio {
                data: "aGVsbG8=".to_string(),
                format: "wav".to_string(),
            },
        }]);
        let err = content_to_blocks(&content).unwrap_err();
        assert!(matches!(err, ProviderError::UnsupportedContent(_)));
    }

    // --- build_messages: image content -------------------------------------------

    #[test]
    fn build_messages_translates_a_user_image_content_part() {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: rp_core::ImageUrl {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                },
            }])),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
            cache_control: None,
        }];
        let (_, turns) = build_messages(&messages).unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(
            turns[0].content,
            vec![json!({
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/a.png"},
            })]
        );
    }

    #[test]
    fn build_messages_collapses_assistant_and_system_image_parts_to_plain_text() {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: Some(MessageContent::Parts(vec![ContentPart::Text {
                    text: "be concise".to_string(),
                }])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
                cache_control: None,
            },
            ChatMessage::assistant("ok"),
        ];
        let (system, turns) = build_messages(&messages).unwrap();
        assert_eq!(system, Some(json!("be concise")));
        assert_eq!(
            turns[0].content,
            vec![json!({"type": "text", "text": "ok"})]
        );
    }

    // --- build_messages / chat: audio content is unsupported ----------------------

    #[test]
    fn build_messages_errs_on_a_user_audio_content_part() {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: Some(MessageContent::Parts(vec![ContentPart::InputAudio {
                input_audio: rp_core::InputAudio {
                    data: "aGVsbG8=".to_string(),
                    format: "wav".to_string(),
                },
            }])),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
            cache_control: None,
        }];
        let err = build_messages(&messages).unwrap_err();
        assert!(matches!(err, ProviderError::UnsupportedContent(_)));
        assert!(
            err.is_retryable(),
            "a fallback chain should move on to a candidate that might support audio"
        );
    }

    #[tokio::test]
    async fn chat_errs_on_audio_content_without_making_any_http_request() {
        // No mock server is started at all -- if this somehow tried to make
        // a real HTTP call it would fail with a connection error, not
        // UnsupportedContent, so this also proves the check happens before
        // any request is sent.
        let provider = AnthropicProvider::new("http://127.0.0.1:1", "test-key");
        let req = ChatRequest {
            model: "anthropic/claude-sonnet-5".to_string(),
            models: None,
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some(MessageContent::Parts(vec![ContentPart::InputAudio {
                    input_audio: rp_core::InputAudio {
                        data: "aGVsbG8=".to_string(),
                        format: "wav".to_string(),
                    },
                }])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning: None,
                cache_control: None,
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: None,
            user: None,
            tools: None,
            tool_choice: None,
            provider: None,
            response_format: None,
            reasoning: None,
            top_k: None,
            min_p: None,
            top_a: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            logit_bias: None,
            seed: None,
            transforms: None,
        };
        let err = provider.chat(&req, "claude-sonnet-5").await.unwrap_err();
        assert!(matches!(err, ProviderError::UnsupportedContent(_)));
    }
}
