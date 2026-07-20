//! Adapter for the Anthropic Messages API (`/v1/messages`). Differs from
//! the OpenAI shape in three ways this module has to bridge: system
//! prompts are a top-level field rather than a message role, `max_tokens`
//! is required rather than optional, and streaming is a sequence of typed
//! SSE events rather than uniform delta chunks.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use rp_core::{
    ChatChunk, ChatMessage, ChatMessageDelta, ChatRequest, ChatResponse, ChatStream, Choice,
    ChunkChoice, Provider, ProviderError, Role, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<&'a [String]>,
    stream: bool,
}

/// Split out system messages (concatenated, since Anthropic takes a single
/// system string) from the conversational turns.
fn split_system(messages: &[ChatMessage]) -> (Option<String>, Vec<WireMessage<'_>>) {
    let mut system_parts = Vec::new();
    let mut turns = Vec::new();
    for m in messages {
        let content = m.content.as_deref().unwrap_or("");
        match m.role {
            Role::System => system_parts.push(content),
            Role::User | Role::Tool => turns.push(WireMessage {
                role: "user",
                content,
            }),
            Role::Assistant => turns.push(WireMessage {
                role: "assistant",
                content,
            }),
        }
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, turns)
}

fn map_stop_reason(reason: &str) -> &'static str {
    match reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        _ => "stop",
    }
}

impl<'a> WireRequest<'a> {
    fn from_core(
        req: &'a ChatRequest,
        model: &'a str,
        system: Option<&'a str>,
        turns: Vec<WireMessage<'a>>,
        stream: bool,
    ) -> Self {
        Self {
            model,
            max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages: turns,
            system,
            temperature: req.temperature,
            top_p: req.top_p,
            stop_sequences: req.stop.as_deref(),
            stream,
        }
    }
}

#[derive(Deserialize)]
struct WireContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
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
        let (system, turns) = split_system(&req.messages);
        let body = WireRequest::from_core(req, model, system.as_deref(), turns, false);

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
        let text: String = wire
            .content
            .iter()
            .filter(|b| b.kind == "text")
            .map(|b| b.text.as_str())
            .collect();

        Ok(ChatResponse {
            id: gen_id("chatcmpl"),
            object: "chat.completion",
            created: now(),
            model: format!("anthropic/{model}"),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: Some(text),
                    name: None,
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
        })
    }

    async fn chat_stream(
        &self,
        req: &ChatRequest,
        model: &str,
    ) -> Result<ChatStream, ProviderError> {
        let (system, turns) = split_system(&req.messages);
        let body = WireRequest::from_core(req, model, system.as_deref(), turns, true);

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
                                content: None,
                            },
                            None,
                        )))
                    }
                    "content_block_delta" => {
                        let text = value
                            .pointer("/delta/text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        Some(Ok(empty_chunk(
                            &full_model,
                            ChatMessageDelta {
                                role: None,
                                content: Some(text),
                            },
                            None,
                        )))
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
                    "message_stop" | "content_block_start" | "content_block_stop" | "ping" => None,
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
    }
}
