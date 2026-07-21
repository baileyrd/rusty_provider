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
    ChatChunk, ChatMessage, ChatMessageDelta, ChatRequest, ChatResponse, ChatStream, Choice,
    ChunkChoice, ContentPart, FunctionCallDelta, MessageContent, Provider, ProviderError, Role,
    Tool, ToolCall, ToolCallDelta, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::http::{map_error_response, map_reqwest_error};
use crate::util::{gen_id, now};

const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic requires `max_tokens`; used when the client doesn't specify one.
const DEFAULT_MAX_TOKENS: u32 = 4096;

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

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    content: Vec<Value>,
}

#[derive(Serialize)]
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
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<&'a [String]>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<WireTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

/// Split messages into Anthropic's shape: a single top-level `system`
/// string (concatenated from any `Role::System` messages) plus a list of
/// user/assistant turns, each with content as typed blocks — text,
/// `tool_use` (an assistant message's `tool_calls`), or `tool_result` (a
/// `Role::Tool` message answering one).
fn build_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<WireMessage>) {
    let mut system_parts = Vec::new();
    let mut turns = Vec::new();

    for m in messages {
        match m.role {
            Role::System => {
                if let Some(c) = &m.content {
                    system_parts.push(c.as_plain_text());
                }
            }
            Role::Tool => {
                turns.push(WireMessage {
                    role: "user",
                    content: vec![json!({
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.content.as_ref().map(MessageContent::as_plain_text).unwrap_or_default(),
                    })],
                });
            }
            Role::User => {
                let content = m
                    .content
                    .as_ref()
                    .map(content_to_blocks)
                    .unwrap_or_else(|| vec![json!({"type": "text", "text": ""})]);
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
                turns.push(WireMessage {
                    role: "assistant",
                    content: blocks,
                });
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, turns)
}

/// Translates a message's content into Anthropic content blocks, turning
/// `image_url` parts into `image` blocks: a `data:<mime>;base64,<data>`
/// URI becomes a `base64` source, anything else (an `https://` URL) an
/// `url` source.
fn content_to_blocks(content: &MessageContent) -> Vec<Value> {
    match content {
        MessageContent::Text(text) => vec![json!({"type": "text", "text": text})],
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => json!({"type": "text", "text": text}),
                ContentPart::ImageUrl { image_url } => image_block(&image_url.url),
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

fn map_stop_reason(reason: &str) -> &'static str {
    match reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        _ => "stop",
    }
}

impl<'a> WireRequest<'a> {
    fn from_core(req: &'a ChatRequest, model: &'a str, stream: bool) -> Self {
        let (system, messages) = build_messages(&req.messages);
        Self {
            model,
            max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages,
            system,
            temperature: req.temperature,
            top_p: req.top_p,
            stop_sequences: req.stop.as_deref(),
            stream,
            tools: req.tools.as_deref().map(to_wire_tools),
            tool_choice: req.tool_choice.as_ref().map(to_wire_tool_choice),
        }
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
}

#[derive(Deserialize)]
struct WireUsage {
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
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
        let body = WireRequest::from_core(req, model, false);

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
        let mut tool_calls = Vec::new();
        for b in &wire.content {
            match b.kind.as_str() {
                "text" => text.push_str(&b.text),
                "tool_use" => {
                    let id = b.id.clone().unwrap_or_default();
                    let tool_name = b.name.clone().unwrap_or_default();
                    let arguments = b.input.clone().unwrap_or_else(|| json!({})).to_string();
                    tool_calls.push(ToolCall::function(id, tool_name, arguments));
                }
                _ => {}
            }
        }

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
                },
                finish_reason: wire
                    .stop_reason
                    .as_deref()
                    .map(map_stop_reason)
                    .map(str::to_string),
            }],
            usage: Some(Usage {
                prompt_tokens: wire.usage.input_tokens,
                completion_tokens: wire.usage.output_tokens,
                total_tokens: wire.usage.input_tokens + wire.usage.output_tokens,
            }),
            cost_usd: None,
        })
    }

    async fn chat_stream(
        &self,
        req: &ChatRequest,
        model: &str,
    ) -> Result<ChatStream, ProviderError> {
        let body = WireRequest::from_core(req, model, true);

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
                                Some(Ok(empty_chunk(
                                    &full_model,
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
                                    },
                                    None,
                                )))
                            }
                            _ => None,
                        }
                    }
                    "message_delta" => {
                        let stop_reason = value
                            .pointer("/delta/stop_reason")
                            .and_then(Value::as_str)
                            .map(map_stop_reason)
                            .map(str::to_string);
                        let output_tokens = value
                            .pointer("/usage/output_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as u32;
                        let mut chunk =
                            empty_chunk(&full_model, ChatMessageDelta::default(), stop_reason);
                        chunk.usage = Some(Usage {
                            prompt_tokens: input_tokens,
                            completion_tokens: output_tokens,
                            total_tokens: input_tokens + output_tokens,
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

    // --- content_to_blocks -------------------------------------------------------

    #[test]
    fn content_to_blocks_wraps_plain_text_as_a_single_text_block() {
        let content = MessageContent::text("hi");
        assert_eq!(
            content_to_blocks(&content),
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
            content_to_blocks(&content),
            vec![
                json!({"type": "text", "text": "what's in this image?"}),
                json!({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
                }),
            ]
        );
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
        }];
        let (_, turns) = build_messages(&messages);
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
            },
            ChatMessage::assistant("ok"),
        ];
        let (system, turns) = build_messages(&messages);
        assert_eq!(system, Some("be concise".to_string()));
        assert_eq!(
            turns[0].content,
            vec![json!({"type": "text", "text": "ok"})]
        );
    }
}
