//! Adapter for the Google Gemini `generateContent` / `streamGenerateContent`
//! API. Differs from OpenAI/Anthropic in role naming ("model" instead of
//! "assistant"), a nested `parts` content structure, and the API key being
//! passed as a query parameter rather than a header.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use rp_core::{
    ChatChunk, ChatMessage, ChatMessageDelta, ChatRequest, ChatResponse, ChatStream, Choice,
    ChunkChoice, Provider, ProviderError, Role, Usage,
};
use serde::{Deserialize, Serialize};

use crate::http::map_reqwest_error;
use crate::util::{gen_id, now};

pub struct GeminiProvider {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl GeminiProvider {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    fn endpoint(&self, model: &str, method: &str) -> String {
        format!("{}/v1beta/models/{model}:{method}", self.base_url)
    }
}

#[derive(Serialize)]
struct Part<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct Content<'a> {
    role: &'a str,
    parts: Vec<Part<'a>>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<OwnedPart>,
}

#[derive(Serialize)]
struct OwnedPart {
    text: String,
}

#[derive(Serialize, Default)]
struct GenerationConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "stopSequences")]
    stop_sequences: Option<&'a [String]>,
}

#[derive(Serialize)]
struct WireRequest<'a> {
    contents: Vec<Content<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "systemInstruction")]
    system_instruction: Option<SystemInstruction>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig<'a>,
}

fn gemini_role(role: &Role) -> &'static str {
    match role {
        Role::Assistant => "model",
        _ => "user",
    }
}

fn build_request<'a>(req: &'a ChatRequest) -> WireRequest<'a> {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    for m in &req.messages {
        let text = m.content.as_deref().unwrap_or("");
        if matches!(m.role, Role::System) {
            system_parts.push(text);
        } else {
            contents.push(Content {
                role: gemini_role(&m.role),
                parts: vec![Part { text }],
            });
        }
    }
    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(SystemInstruction {
            parts: vec![OwnedPart {
                text: system_parts.join("\n\n"),
            }],
        })
    };

    WireRequest {
        contents,
        system_instruction,
        generation_config: GenerationConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            max_output_tokens: req.max_tokens,
            stop_sequences: req.stop.as_deref(),
        },
    }
}

fn map_finish_reason(reason: &str) -> &'static str {
    match reason {
        "MAX_TOKENS" => "length",
        _ => "stop",
    }
}

#[derive(Deserialize, Default)]
struct WirePart {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize, Default)]
struct WireContent {
    #[serde(default)]
    parts: Vec<WirePart>,
}

#[derive(Deserialize)]
struct WireCandidate {
    #[serde(default)]
    content: WireContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct WireUsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
    #[serde(rename = "totalTokenCount", default)]
    total_token_count: u32,
}

#[derive(Deserialize)]
struct WireResponse {
    #[serde(default)]
    candidates: Vec<WireCandidate>,
    #[serde(rename = "usageMetadata", default)]
    usage_metadata: WireUsageMetadata,
}

fn candidate_text(c: &WireCandidate) -> String {
    c.content.parts.iter().map(|p| p.text.as_str()).collect()
}

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    async fn chat(&self, req: &ChatRequest, model: &str) -> Result<ChatResponse, ProviderError> {
        let body = build_request(req);
        let resp = self
            .client
            .post(self.endpoint(model, "generateContent"))
            .query(&[("key", self.api_key.as_str())])
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            return Err(crate::http::map_error_response(resp).await);
        }

        let wire: WireResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Decode(e.to_string()))?;
        let candidate = wire.candidates.into_iter().next();
        let (text, finish_reason) = match &candidate {
            Some(c) => (
                candidate_text(c),
                c.finish_reason
                    .as_deref()
                    .map(map_finish_reason)
                    .map(str::to_string),
            ),
            None => (String::new(), None),
        };

        Ok(ChatResponse {
            id: gen_id("chatcmpl"),
            object: "chat.completion",
            created: now(),
            model: format!("gemini/{model}"),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: Some(text),
                    name: None,
                },
                finish_reason,
            }],
            usage: Some(Usage {
                prompt_tokens: wire.usage_metadata.prompt_token_count,
                completion_tokens: wire.usage_metadata.candidates_token_count,
                total_tokens: wire.usage_metadata.total_token_count,
            }),
        })
    }

    async fn chat_stream(
        &self,
        req: &ChatRequest,
        model: &str,
    ) -> Result<ChatStream, ProviderError> {
        let body = build_request(req);
        let resp = self
            .client
            .post(self.endpoint(model, "streamGenerateContent"))
            .query(&[("key", self.api_key.as_str()), ("alt", "sse")])
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            return Err(crate::http::map_error_response(resp).await);
        }

        let full_model = format!("gemini/{model}");
        let stream = resp.bytes_stream().eventsource().filter_map(move |ev| {
            let full_model = full_model.clone();
            async move {
                let ev = match ev {
                    Ok(ev) => ev,
                    Err(e) => return Some(Err(ProviderError::Decode(e.to_string()))),
                };
                let wire: WireResponse = match serde_json::from_str(&ev.data) {
                    Ok(w) => w,
                    Err(e) => return Some(Err(ProviderError::Decode(e.to_string()))),
                };
                let candidate = wire.candidates.into_iter().next();
                let (text, finish_reason) = match &candidate {
                    Some(c) => (
                        candidate_text(c),
                        c.finish_reason
                            .as_deref()
                            .map(map_finish_reason)
                            .map(str::to_string),
                    ),
                    None => (String::new(), None),
                };
                let usage = if wire.usage_metadata.total_token_count > 0 {
                    Some(Usage {
                        prompt_tokens: wire.usage_metadata.prompt_token_count,
                        completion_tokens: wire.usage_metadata.candidates_token_count,
                        total_tokens: wire.usage_metadata.total_token_count,
                    })
                } else {
                    None
                };

                Some(Ok(ChatChunk {
                    id: gen_id("chatcmpl"),
                    object: "chat.completion.chunk",
                    created: now(),
                    model: full_model,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChatMessageDelta {
                            role: Some(Role::Assistant),
                            content: Some(text),
                        },
                        finish_reason,
                    }],
                    usage,
                }))
            }
        });

        Ok(Box::pin(stream))
    }
}
