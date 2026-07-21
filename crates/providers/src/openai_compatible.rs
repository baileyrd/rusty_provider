//! Adapter for any backend that speaks the OpenAI `/chat/completions`
//! wire format: OpenAI itself, Groq, Together AI, Fireworks, and most
//! other "OpenAI-compatible" inference APIs. Only the base URL, API key,
//! and provider name differ between them.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use rp_core::{
    ChatChunk, ChatMessage, ChatMessageDelta, ChatRequest, ChatResponse, ChatStream, Choice,
    ChunkChoice, MessageContent, Provider, ProviderError, ResponseFormat, Role, Tool, ToolCall,
    ToolCallDelta, Usage,
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
    /// Matches the OpenAI wire format exactly, so this is a direct
    /// passthrough -- no translation needed, unlike Anthropic and Gemini.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'a ResponseFormat>,
    /// The widely-adopted convention across DeepSeek/Groq/etc.
    /// OpenAI-compatible reasoning models: a top-level effort hint, direct
    /// passthrough from `reasoning.effort`.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    /// Not part of OpenAI's own API, but common on OpenAI-compatible
    /// inference servers (Groq, Together, Fireworks, vLLM, etc.) -- passed
    /// through unconditionally rather than guessing which backend
    /// supports it, matching this adapter's "no translation needed"
    /// approach elsewhere.
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_a: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<&'a std::collections::HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    /// Matches the OpenAI wire format exactly -- direct passthrough, same
    /// as `response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
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
            response_format: req.response_format.as_ref(),
            reasoning_effort: req.reasoning.as_ref().and_then(|r| r.effort.as_deref()),
            top_k: req.top_k,
            min_p: req.min_p,
            top_a: req.top_a,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            repetition_penalty: req.repetition_penalty,
            logit_bias: req.logit_bias.as_ref(),
            seed: req.seed,
            logprobs: req.logprobs,
            top_logprobs: req.top_logprobs,
        }
    }
}

/// The `prompt_tokens_details.cached_tokens` convention OpenAI (and most
/// OpenAI-compatible backends) report automatic prompt-cache hits under.
#[derive(Deserialize, Default)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Deserialize)]
struct WireUsage {
    /// Already inclusive of any cached tokens -- `prompt_tokens_details`
    /// below is a breakdown of this total, not additive on top of it.
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

impl From<WireUsage> for Usage {
    fn from(u: WireUsage) -> Self {
        let cached_tokens = u
            .prompt_tokens_details
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cached_tokens: (cached_tokens > 0).then_some(cached_tokens),
            // OpenAI-compatible caching is fully automatic -- no separate
            // cache-write cost/accounting to surface here.
            cache_creation_tokens: None,
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
    /// The `reasoning_content` convention adopted across DeepSeek/Groq/etc.
    /// OpenAI-compatible reasoning models.
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct WireChoice {
    index: u32,
    message: WireMessage,
    finish_reason: Option<String>,
    /// Already matches `rp_core::ChoiceLogprobs`'s shape exactly -- direct
    /// passthrough, same as `response_format` on the request side.
    #[serde(default)]
    logprobs: Option<rp_core::ChoiceLogprobs>,
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
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct WireChunkChoice {
    index: u32,
    #[serde(default)]
    delta: WireDelta,
    finish_reason: Option<String>,
    #[serde(default)]
    logprobs: Option<rp_core::ChoiceLogprobs>,
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
        let exclude_reasoning = req
            .reasoning
            .as_ref()
            .and_then(|r| r.exclude)
            .unwrap_or(false);

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
                        content: c.message.content.map(MessageContent::Text),
                        name: None,
                        tool_calls: c.message.tool_calls,
                        tool_call_id: None,
                        reasoning: if exclude_reasoning {
                            None
                        } else {
                            c.message.reasoning_content
                        },
                        cache_control: None,
                    },
                    finish_reason: c.finish_reason,
                    logprobs: c.logprobs,
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
        let exclude_reasoning = req
            .reasoning
            .as_ref()
            .and_then(|r| r.exclude)
            .unwrap_or(false);
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
                                reasoning: if exclude_reasoning {
                                    None
                                } else {
                                    c.delta.reasoning_content
                                },
                            },
                            finish_reason: c.finish_reason,
                            logprobs: c.logprobs,
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
