//! Adapter for the Google Gemini `generateContent` / `streamGenerateContent`
//! API. Differs from OpenAI/Anthropic in role naming ("model" instead of
//! "assistant"), a nested `parts` content structure (including
//! `functionCall`/`functionResponse` parts for tool use), and the API key
//! being passed as a query parameter rather than a header.

use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use rp_core::{
    ChatChunk, ChatMessage, ChatMessageDelta, ChatRequest, ChatResponse, ChatStream, Choice,
    ChunkChoice, FunctionCallDelta, Provider, ProviderError, Role, Tool, ToolCall, ToolCallDelta,
    Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
struct Content {
    role: &'static str,
    parts: Vec<Value>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Value>,
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
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "systemInstruction")]
    system_instruction: Option<SystemInstruction>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "toolConfig")]
    tool_config: Option<Value>,
}

fn gemini_role(role: &Role) -> &'static str {
    match role {
        Role::Assistant => "model",
        _ => "user",
    }
}

/// Build the `contents` list, tracking each assistant tool call's id ->
/// function name (Gemini's `functionResponse` needs the name, but our
/// `Role::Tool` messages, like OpenAI's, only carry the call id).
fn build_contents(messages: &[ChatMessage]) -> (Option<String>, Vec<Content>) {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut call_names: HashMap<String, String> = HashMap::new();

    for m in messages {
        match m.role {
            Role::System => {
                if let Some(c) = &m.content {
                    system_parts.push(c.clone());
                }
            }
            Role::Tool => {
                let name = m
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| call_names.get(id))
                    .cloned()
                    .unwrap_or_default();
                contents.push(Content {
                    role: "user",
                    parts: vec![json!({
                        "functionResponse": {
                            "name": name,
                            "response": {"content": m.content.clone().unwrap_or_default()},
                        }
                    })],
                });
            }
            Role::User => {
                contents.push(Content {
                    role: gemini_role(&m.role),
                    parts: vec![json!({"text": m.content.clone().unwrap_or_default()})],
                });
            }
            Role::Assistant => {
                let mut parts = Vec::new();
                if let Some(text) = &m.content {
                    if !text.is_empty() {
                        parts.push(json!({"text": text}));
                    }
                }
                for tc in m.tool_calls.iter().flatten() {
                    call_names.insert(tc.id.clone(), tc.function.name.clone());
                    let args: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                    parts.push(json!({"functionCall": {"name": tc.function.name, "args": args}}));
                }
                if parts.is_empty() {
                    parts.push(json!({"text": ""}));
                }
                contents.push(Content {
                    role: gemini_role(&m.role),
                    parts,
                });
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, contents)
}

fn to_wire_tools(tools: &[Tool]) -> Value {
    json!([{
        "functionDeclarations": tools.iter().map(|t| json!({
            "name": t.function.name,
            "description": t.function.description,
            "parameters": t.function.parameters.clone().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        })).collect::<Vec<_>>(),
    }])
}

/// Map OpenAI's `tool_choice` vocabulary onto Gemini's
/// `toolConfig.functionCallingConfig` (`AUTO`/`ANY`/`NONE`, optionally
/// restricted to a named function).
fn to_wire_tool_choice(choice: &Value) -> Value {
    match choice {
        Value::String(s) if s == "none" => json!({"functionCallingConfig": {"mode": "NONE"}}),
        Value::String(s) if s == "required" => json!({"functionCallingConfig": {"mode": "ANY"}}),
        Value::Object(_) => match choice.pointer("/function/name").and_then(Value::as_str) {
            Some(name) => {
                json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}})
            }
            None => json!({"functionCallingConfig": {"mode": "AUTO"}}),
        },
        _ => json!({"functionCallingConfig": {"mode": "AUTO"}}),
    }
}

fn build_request(req: &ChatRequest) -> WireRequest<'_> {
    let (system, contents) = build_contents(&req.messages);

    WireRequest {
        contents,
        system_instruction: system.map(|s| SystemInstruction {
            parts: vec![json!({"text": s})],
        }),
        generation_config: GenerationConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            max_output_tokens: req.max_tokens,
            stop_sequences: req.stop.as_deref(),
        },
        tools: req.tools.as_deref().map(to_wire_tools),
        tool_config: req.tool_choice.as_ref().map(to_wire_tool_choice),
    }
}

fn map_finish_reason(reason: &str) -> &'static str {
    match reason {
        "MAX_TOKENS" => "length",
        _ => "stop",
    }
}

#[derive(Deserialize)]
struct WireFunctionCall {
    name: String,
    #[serde(default)]
    args: Value,
}

#[derive(Deserialize, Default)]
struct WirePart {
    #[serde(default)]
    text: String,
    #[serde(default, rename = "functionCall")]
    function_call: Option<WireFunctionCall>,
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

fn candidate_tool_calls(c: &WireCandidate) -> Vec<ToolCall> {
    c.content
        .parts
        .iter()
        .filter_map(|p| p.function_call.as_ref())
        .map(|fc| ToolCall::function(gen_id("call"), fc.name.clone(), fc.args.to_string()))
        .collect()
}

/// A tool call in this candidate overrides the raw `finishReason`, since
/// Gemini reports `STOP` for both a normal end-of-turn and a function call.
fn resolve_finish_reason(candidate: &WireCandidate, tool_calls: &[ToolCall]) -> Option<String> {
    if !tool_calls.is_empty() {
        return Some("tool_calls".to_string());
    }
    candidate
        .finish_reason
        .as_deref()
        .map(map_finish_reason)
        .map(str::to_string)
}

fn tool_call_deltas(tool_calls: &[ToolCall]) -> Option<Vec<ToolCallDelta>> {
    if tool_calls.is_empty() {
        return None;
    }
    Some(
        tool_calls
            .iter()
            .enumerate()
            .map(|(i, tc)| ToolCallDelta {
                index: i as u32,
                id: Some(tc.id.clone()),
                kind: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(tc.function.name.clone()),
                    arguments: Some(tc.function.arguments.clone()),
                }),
            })
            .collect(),
    )
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
        let (text, tool_calls, finish_reason) = match &candidate {
            Some(c) => {
                let tool_calls = candidate_tool_calls(c);
                let finish_reason = resolve_finish_reason(c, &tool_calls);
                (candidate_text(c), tool_calls, finish_reason)
            }
            None => (String::new(), Vec::new(), None),
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
                    content: if text.is_empty() { None } else { Some(text) },
                    name: None,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                },
                finish_reason,
            }],
            usage: Some(Usage {
                prompt_tokens: wire.usage_metadata.prompt_token_count,
                completion_tokens: wire.usage_metadata.candidates_token_count,
                total_tokens: wire.usage_metadata.total_token_count,
            }),
            cost_usd: None,
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
                let (text, tool_calls, finish_reason) = match &candidate {
                    Some(c) => {
                        let tool_calls = candidate_tool_calls(c);
                        let finish_reason = resolve_finish_reason(c, &tool_calls);
                        (candidate_text(c), tool_calls, finish_reason)
                    }
                    None => (String::new(), Vec::new(), None),
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
                            content: if text.is_empty() { None } else { Some(text) },
                            tool_calls: tool_call_deltas(&tool_calls),
                        },
                        finish_reason,
                    }],
                    usage,
                    cost_usd: None,
                }))
            }
        });

        Ok(Box::pin(stream))
    }
}
