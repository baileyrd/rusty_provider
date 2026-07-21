//! Adapter for any backend that speaks the OpenAI `/chat/completions`
//! wire format: OpenAI itself, Groq, Together AI, Fireworks, and most
//! other "OpenAI-compatible" inference APIs. Only the base URL, API key,
//! and provider name differ between them.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use rp_core::{
    ChatChunk, ChatMessage, ChatMessageDelta, ChatRequest, ChatResponse, ChatStream, Choice,
    ChunkChoice, Provider, ProviderError, Role, Tool, ToolCall, ToolCallDelta, Usage,
};
use serde::{Deserialize, Serialize};

use crate::http::{map_error_response, map_reqwest_error};
use crate::util::{gen_id, now};

pub struct OpenAiCompatibleProvider {
    name: String,
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl OpenAiCompatibleProvider {
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a [String]>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Tool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
}

impl<'a> WireRequest<'a> {
    fn from_core(req: &'a ChatRequest, model: &'a str, stream: bool) -> Self {
        Self {
            model,
            messages: &req.messages,
            temperature: req.temperature,
            top_p: req.top_p,
            max_tokens: req.max_tokens,
            stop: req.stop.as_deref(),
            stream,
            tools: req.tools.as_deref(),
            tool_choice: req.tool_choice.as_ref(),
        }
    }
}

#[derive(Deserialize)]
struct WireUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<WireUsage> for Usage {
    fn from(u: WireUsage) -> Self {
        Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

fn parse_role(role: &str) -> Role {
    match role {
        "system" => Role::System,
        "user" => Role::User,
        "tool" => Role::Tool,
        _ => Role::Assistant,
    }
}

#[derive(Deserialize)]
struct WireMessage {
    role: String,
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize)]
struct WireChoice {
    index: u32,
    message: WireMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireResponse {
    choices: Vec<WireChoice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize, Default)]
struct WireDelta {
    role: Option<String>,
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct WireChunkChoice {
    index: u32,
    #[serde(default)]
    delta: WireDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireChunk {
    choices: Vec<WireChunkChoice>,
    usage: Option<WireUsage>,
}

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(&self, req: &ChatRequest, model: &str) -> Result<ChatResponse, ProviderError> {
        let body = WireRequest::from_core(req, model, false);
        let resp = self
            .client
            .post(self.endpoint())
            .bearer_auth(&self.api_key)
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
        let full_model = format!("{}/{}", self.name, model);

        Ok(ChatResponse {
            id: gen_id("chatcmpl"),
            object: "chat.completion",
            created: now(),
            model: full_model,
            choices: wire
                .choices
                .into_iter()
                .map(|c| Choice {
                    index: c.index,
                    message: ChatMessage {
                        role: parse_role(&c.message.role),
                        content: c.message.content,
                        name: None,
                        tool_calls: c.message.tool_calls,
                        tool_call_id: None,
                    },
                    finish_reason: c.finish_reason,
                })
                .collect(),
            usage: wire.usage.map(Into::into),
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
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            return Err(map_error_response(resp).await);
        }

        let full_model = format!("{}/{}", self.name, model);
        let stream = resp.bytes_stream().eventsource().filter_map(move |ev| {
            let full_model = full_model.clone();
            async move {
                let ev = match ev {
                    Ok(ev) => ev,
                    Err(e) => return Some(Err(ProviderError::Decode(e.to_string()))),
                };
                if ev.data == "[DONE]" {
                    return None;
                }
                let wire: WireChunk = match serde_json::from_str(&ev.data) {
                    Ok(w) => w,
                    Err(e) => return Some(Err(ProviderError::Decode(e.to_string()))),
                };
                Some(Ok(ChatChunk {
                    id: gen_id("chatcmpl"),
                    object: "chat.completion.chunk",
                    created: now(),
                    model: full_model,
                    choices: wire
                        .choices
                        .into_iter()
                        .map(|c| ChunkChoice {
                            index: c.index,
                            delta: ChatMessageDelta {
                                role: c.delta.role.as_deref().map(parse_role),
                                content: c.delta.content,
                                tool_calls: c.delta.tool_calls,
                            },
                            finish_reason: c.finish_reason,
                        })
                        .collect(),
                    usage: wire.usage.map(Into::into),
                    cost_usd: None,
                }))
            }
        });

        Ok(Box::pin(stream))
    }
}
