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
    ChunkChoice, ContentPart, FunctionCallDelta, InputAudio, MessageContent, Provider,
    ProviderError, ReasoningConfig, ResponseFormat, Role, Tool, ToolCall, ToolCallDelta, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::http::map_reqwest_error;
use crate::util::{effort_thinking_budget, gen_id, now};

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
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseMimeType")]
    response_mime_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseSchema")]
    response_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingConfig")]
    thinking_config: Option<Value>,
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
                    system_parts.push(c.as_plain_text());
                }
            }
            Role::Tool => {
                let name = m
                    .tool_call_id
                    .as_ref()
                    .and_then(|id| call_names.get(id))
                    .cloned()
                    .unwrap_or_default();
                let text = m
                    .content
                    .as_ref()
                    .map(MessageContent::as_plain_text)
                    .unwrap_or_default();
                contents.push(Content {
                    role: "user",
                    parts: vec![json!({
                        "functionResponse": {
                            "name": name,
                            "response": {"content": text},
                        }
                    })],
                });
            }
            Role::User => {
                let parts = m
                    .content
                    .as_ref()
                    .map(content_to_parts)
                    .unwrap_or_else(|| vec![json!({"text": ""})]);
                contents.push(Content {
                    role: gemini_role(&m.role),
                    parts,
                });
            }
            Role::Assistant => {
                let mut parts = Vec::new();
                if let Some(content) = &m.content {
                    let text = content.as_plain_text();
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

/// Translates a message's content into Gemini `parts`, turning
/// `image_url` parts into `inlineData` (for a `data:<mime>;base64,<data>`
/// URI) or `fileData` (for an `https://` URL, whose MIME type is guessed
/// from the URL's extension since Gemini requires one), and `input_audio`
/// parts into `inlineData` directly -- Gemini accepts audio the same way
/// it accepts inline images, and OpenAI's `input_audio.data` is already
/// raw base64 with no URI wrapper to unwrap.
fn content_to_parts(content: &MessageContent) -> Vec<Value> {
    match content {
        MessageContent::Text(text) => vec![json!({"text": text})],
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => json!({"text": text}),
                ContentPart::ImageUrl { image_url } => image_part(&image_url.url),
                ContentPart::InputAudio { input_audio } => audio_part(input_audio),
            })
            .collect(),
    }
}

fn audio_part(audio: &InputAudio) -> Value {
    json!({
        "inlineData": {"mimeType": format!("audio/{}", audio.format), "data": audio.data},
    })
}

fn image_part(url: &str) -> Value {
    match parse_data_uri(url) {
        Some((mime_type, data)) => json!({
            "inlineData": {"mimeType": mime_type, "data": data},
        }),
        None => json!({
            "fileData": {"mimeType": guess_mime_type(url), "fileUri": url},
        }),
    }
}

/// Parses a `data:<mime>;base64,<data>` URI into its `(mime, data)` parts.
/// Returns `None` for anything else (e.g. a plain `https://` URL).
fn parse_data_uri(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let mime_type = meta.strip_suffix(";base64")?;
    Some((mime_type, data))
}

/// Best-effort MIME type guess from a URL's extension, since Gemini's
/// `fileData` requires one and a plain image URL doesn't carry it.
fn guess_mime_type(url: &str) -> &'static str {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "image/jpeg",
    }
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

/// Gemini has native support for constrained JSON output, unlike
/// Anthropic -- both `ResponseFormat::JsonObject` (mime type only) and
/// `ResponseFormat::JsonSchema` (mime type plus an OpenAPI-flavored schema,
/// close enough to plain JSON Schema for typical use) map directly onto
/// `generationConfig` fields, so this never needs to reject a request the
/// way the Anthropic adapter sometimes does.
fn response_format_config(fmt: &ResponseFormat) -> (Option<&'static str>, Option<Value>) {
    match fmt {
        ResponseFormat::Text => (None, None),
        ResponseFormat::JsonObject => (Some("application/json"), None),
        ResponseFormat::JsonSchema { json_schema } => {
            (Some("application/json"), Some(json_schema.schema.clone()))
        }
    }
}

fn thinking_config(reasoning: &ReasoningConfig, max_tokens: Option<u32>) -> Value {
    let budget = reasoning
        .max_tokens
        .unwrap_or_else(|| effort_thinking_budget(reasoning.effort.as_deref(), max_tokens));
    json!({
        "thinkingBudget": budget,
        "includeThoughts": !reasoning.exclude.unwrap_or(false),
    })
}

fn build_request(req: &ChatRequest) -> WireRequest<'_> {
    let (system, contents) = build_contents(&req.messages);
    let (response_mime_type, response_schema) = req
        .response_format
        .as_ref()
        .map(response_format_config)
        .unwrap_or((None, None));

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
            response_mime_type,
            response_schema,
            thinking_config: req
                .reasoning
                .as_ref()
                .map(|r| thinking_config(r, req.max_tokens)),
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
    /// Marks this part as a thought summary rather than the actual answer
    /// -- only present (and only ever `true`) when `thinkingConfig.
    /// includeThoughts` was set in the request.
    #[serde(default)]
    thought: bool,
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
    c.content
        .parts
        .iter()
        .filter(|p| !p.thought)
        .map(|p| p.text.as_str())
        .collect()
}

/// The model's thought-summary text (parts marked `thought: true`),
/// separate from the actual answer -- `None` if the candidate has none
/// (either the request didn't ask for reasoning, or this model doesn't
/// support it).
fn candidate_reasoning(c: &WireCandidate) -> Option<String> {
    let text: String = c
        .content
        .parts
        .iter()
        .filter(|p| p.thought)
        .map(|p| p.text.as_str())
        .collect();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
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
        let (text, reasoning, tool_calls, finish_reason) = match &candidate {
            Some(c) => {
                let tool_calls = candidate_tool_calls(c);
                let finish_reason = resolve_finish_reason(c, &tool_calls);
                (
                    candidate_text(c),
                    candidate_reasoning(c),
                    tool_calls,
                    finish_reason,
                )
            }
            None => (String::new(), None, Vec::new(), None),
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
                let (text, reasoning, tool_calls, finish_reason) = match &candidate {
                    Some(c) => {
                        let tool_calls = candidate_tool_calls(c);
                        let finish_reason = resolve_finish_reason(c, &tool_calls);
                        (
                            candidate_text(c),
                            candidate_reasoning(c),
                            tool_calls,
                            finish_reason,
                        )
                    }
                    None => (String::new(), None, Vec::new(), None),
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
                            reasoning,
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
        assert_eq!(
            wire,
            json!([{
                "functionDeclarations": [{
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": params,
                }]
            }])
        );
    }

    #[test]
    fn to_wire_tools_defaults_parameters_when_absent() {
        let tools = vec![tool("no_args_tool", None, None)];
        let wire = to_wire_tools(&tools);
        assert_eq!(
            wire[0]["functionDeclarations"][0]["parameters"],
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn to_wire_tools_wraps_every_tool_in_a_single_function_declarations_entry() {
        let tools = vec![tool("first", None, None), tool("second", None, None)];
        let wire = to_wire_tools(&tools);
        let declarations = wire[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(declarations.len(), 2);
        assert_eq!(declarations[0]["name"], "first");
        assert_eq!(declarations[1]["name"], "second");
    }

    // --- to_wire_tool_choice -------------------------------------------------

    #[test]
    fn to_wire_tool_choice_maps_auto() {
        assert_eq!(
            to_wire_tool_choice(&json!("auto")),
            json!({"functionCallingConfig": {"mode": "AUTO"}})
        );
    }

    #[test]
    fn to_wire_tool_choice_maps_none() {
        assert_eq!(
            to_wire_tool_choice(&json!("none")),
            json!({"functionCallingConfig": {"mode": "NONE"}})
        );
    }

    #[test]
    fn to_wire_tool_choice_maps_required_to_any() {
        assert_eq!(
            to_wire_tool_choice(&json!("required")),
            json!({"functionCallingConfig": {"mode": "ANY"}})
        );
    }

    #[test]
    fn to_wire_tool_choice_maps_named_function_to_allowed_function_names() {
        let choice = json!({"type": "function", "function": {"name": "get_weather"}});
        assert_eq!(
            to_wire_tool_choice(&choice),
            json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}})
        );
    }

    #[test]
    fn to_wire_tool_choice_falls_back_to_auto_for_an_object_without_a_function_name() {
        let choice = json!({"type": "function"});
        assert_eq!(
            to_wire_tool_choice(&choice),
            json!({"functionCallingConfig": {"mode": "AUTO"}})
        );
    }

    #[test]
    fn to_wire_tool_choice_falls_back_to_auto_for_an_unrecognized_shape() {
        assert_eq!(
            to_wire_tool_choice(&json!(null)),
            json!({"functionCallingConfig": {"mode": "AUTO"}})
        );
    }

    // --- response_format_config ---------------------------------------------

    #[test]
    fn response_format_config_text_sets_no_mime_type_or_schema() {
        assert_eq!(response_format_config(&ResponseFormat::Text), (None, None));
    }

    #[test]
    fn response_format_config_json_object_sets_mime_type_with_no_schema() {
        assert_eq!(
            response_format_config(&ResponseFormat::JsonObject),
            (Some("application/json"), None)
        );
    }

    #[test]
    fn response_format_config_json_schema_sets_mime_type_and_schema() {
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let fmt = ResponseFormat::JsonSchema {
            json_schema: rp_core::JsonSchemaFormat {
                name: "weather_report".to_string(),
                description: None,
                schema: schema.clone(),
                strict: None,
            },
        };
        assert_eq!(
            response_format_config(&fmt),
            (Some("application/json"), Some(schema))
        );
    }

    // --- effort_thinking_budget / thinking_config -----------------------------

    #[test]
    fn effort_thinking_budget_takes_a_fraction_of_max_tokens_when_set() {
        assert_eq!(effort_thinking_budget(Some("high"), Some(1000)), 800);
        assert_eq!(effort_thinking_budget(Some("medium"), Some(1000)), 500);
        assert_eq!(effort_thinking_budget(None, Some(1000)), 500);
        assert_eq!(effort_thinking_budget(Some("low"), Some(1000)), 200);
    }

    #[test]
    fn effort_thinking_budget_falls_back_to_a_flat_default_without_max_tokens() {
        assert_eq!(effort_thinking_budget(Some("high"), None), 24_576);
        assert_eq!(effort_thinking_budget(None, None), 8_192);
        assert_eq!(effort_thinking_budget(Some("low"), None), 1_024);
    }

    #[test]
    fn thinking_config_uses_max_tokens_directly_when_set() {
        let reasoning = rp_core::ReasoningConfig {
            effort: Some("high".to_string()),
            max_tokens: Some(2048),
            exclude: None,
        };
        assert_eq!(
            thinking_config(&reasoning, Some(10_000)),
            json!({"thinkingBudget": 2048, "includeThoughts": true})
        );
    }

    #[test]
    fn thinking_config_sets_include_thoughts_false_when_excluded() {
        let reasoning = rp_core::ReasoningConfig {
            effort: None,
            max_tokens: Some(1024),
            exclude: Some(true),
        };
        assert_eq!(
            thinking_config(&reasoning, None),
            json!({"thinkingBudget": 1024, "includeThoughts": false})
        );
    }

    // --- candidate_reasoning -------------------------------------------------

    #[test]
    fn candidate_reasoning_collects_only_thought_parts() {
        let candidate = WireCandidate {
            content: WireContent {
                parts: vec![
                    WirePart {
                        text: "Let me think...".to_string(),
                        function_call: None,
                        thought: true,
                    },
                    WirePart {
                        text: "The answer is 42.".to_string(),
                        function_call: None,
                        thought: false,
                    },
                ],
            },
            finish_reason: Some("STOP".to_string()),
        };
        assert_eq!(
            candidate_reasoning(&candidate),
            Some("Let me think...".to_string())
        );
        assert_eq!(candidate_text(&candidate), "The answer is 42.");
    }

    #[test]
    fn candidate_reasoning_is_none_without_any_thought_parts() {
        let candidate = candidate_with_text_only("hello", "STOP");
        assert_eq!(candidate_reasoning(&candidate), None);
    }

    // --- candidate_tool_calls / resolve_finish_reason / tool_call_deltas ------

    fn candidate_with_function_call(name: &str, args: Value) -> WireCandidate {
        WireCandidate {
            content: WireContent {
                parts: vec![WirePart {
                    text: String::new(),
                    function_call: Some(WireFunctionCall {
                        name: name.to_string(),
                        args,
                    }),
                    thought: false,
                }],
            },
            finish_reason: Some("STOP".to_string()),
        }
    }

    fn candidate_with_text_only(text: &str, finish_reason: &str) -> WireCandidate {
        WireCandidate {
            content: WireContent {
                parts: vec![WirePart {
                    text: text.to_string(),
                    function_call: None,
                    thought: false,
                }],
            },
            finish_reason: Some(finish_reason.to_string()),
        }
    }

    #[test]
    fn candidate_tool_calls_extracts_function_call_parts_and_ignores_text_parts() {
        let candidate = candidate_with_function_call("get_weather", json!({"city": "Boston"}));
        let calls = candidate_tool_calls(&candidate);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, json!({"city": "Boston"}));
    }

    #[test]
    fn candidate_tool_calls_is_empty_for_a_text_only_candidate() {
        let candidate = candidate_with_text_only("hello", "STOP");
        assert!(candidate_tool_calls(&candidate).is_empty());
    }

    #[test]
    fn resolve_finish_reason_overrides_stop_to_tool_calls_when_a_tool_call_is_present() {
        let candidate = candidate_with_function_call("get_weather", json!({}));
        let tool_calls = candidate_tool_calls(&candidate);
        assert_eq!(
            resolve_finish_reason(&candidate, &tool_calls),
            Some("tool_calls".to_string())
        );
    }

    #[test]
    fn resolve_finish_reason_falls_back_to_the_raw_reason_without_tool_calls() {
        let candidate = candidate_with_text_only("hello", "MAX_TOKENS");
        assert_eq!(
            resolve_finish_reason(&candidate, &[]),
            Some("length".to_string())
        );
    }

    #[test]
    fn tool_call_deltas_is_none_for_an_empty_slice() {
        assert!(tool_call_deltas(&[]).is_none());
    }

    #[test]
    fn tool_call_deltas_carries_the_full_call_in_one_delta_per_index() {
        let calls = vec![
            ToolCall::function("call_1", "get_weather", "{\"city\":\"Boston\"}"),
            ToolCall::function("call_2", "get_time", "{}"),
        ];
        let deltas = tool_call_deltas(&calls).unwrap();
        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].index, 0);
        assert_eq!(deltas[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            deltas[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            deltas[0].function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"city\":\"Boston\"}")
        );
        assert_eq!(deltas[1].index, 1);
        assert_eq!(deltas[1].id.as_deref(), Some("call_2"));
    }

    // --- parse_data_uri --------------------------------------------------------

    #[test]
    fn parse_data_uri_extracts_mime_type_and_data() {
        assert_eq!(
            parse_data_uri("data:image/png;base64,aGVsbG8="),
            Some(("image/png", "aGVsbG8="))
        );
    }

    #[test]
    fn parse_data_uri_rejects_a_plain_url() {
        assert_eq!(parse_data_uri("https://example.com/a.png"), None);
    }

    // --- guess_mime_type ---------------------------------------------------------

    #[test]
    fn guess_mime_type_recognizes_common_extensions() {
        assert_eq!(guess_mime_type("https://example.com/a.png"), "image/png");
        assert_eq!(guess_mime_type("https://example.com/a.gif"), "image/gif");
        assert_eq!(guess_mime_type("https://example.com/a.webp"), "image/webp");
        assert_eq!(guess_mime_type("https://example.com/a.jpg"), "image/jpeg");
    }

    #[test]
    fn guess_mime_type_ignores_query_and_fragment_when_choosing_the_extension() {
        assert_eq!(
            guess_mime_type("https://example.com/a.png?w=100#frag"),
            "image/png"
        );
    }

    #[test]
    fn guess_mime_type_defaults_to_jpeg_for_an_unknown_extension() {
        assert_eq!(guess_mime_type("https://example.com/a.bmp"), "image/jpeg");
        assert_eq!(guess_mime_type("https://example.com/a"), "image/jpeg");
    }

    // --- image_part --------------------------------------------------------------

    #[test]
    fn image_part_uses_inline_data_for_a_data_uri() {
        assert_eq!(
            image_part("data:image/jpeg;base64,aGVsbG8="),
            json!({"inlineData": {"mimeType": "image/jpeg", "data": "aGVsbG8="}})
        );
    }

    #[test]
    fn image_part_uses_file_data_for_an_https_url() {
        assert_eq!(
            image_part("https://example.com/a.png"),
            json!({"fileData": {"mimeType": "image/png", "fileUri": "https://example.com/a.png"}})
        );
    }

    // --- audio_part ----------------------------------------------------------------

    #[test]
    fn audio_part_uses_inline_data_with_a_derived_mime_type() {
        assert_eq!(
            audio_part(&InputAudio {
                data: "aGVsbG8=".to_string(),
                format: "wav".to_string(),
            }),
            json!({"inlineData": {"mimeType": "audio/wav", "data": "aGVsbG8="}})
        );
        assert_eq!(
            audio_part(&InputAudio {
                data: "aGVsbG8=".to_string(),
                format: "mp3".to_string(),
            }),
            json!({"inlineData": {"mimeType": "audio/mp3", "data": "aGVsbG8="}})
        );
    }

    // --- content_to_parts ---------------------------------------------------------

    #[test]
    fn content_to_parts_wraps_plain_text_as_a_single_text_part() {
        let content = MessageContent::text("hi");
        assert_eq!(content_to_parts(&content), vec![json!({"text": "hi"})]);
    }

    #[test]
    fn content_to_parts_translates_mixed_text_and_image_parts() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "what's this?".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: rp_core::ImageUrl {
                    url: "data:image/png;base64,aGVsbG8=".to_string(),
                    detail: None,
                },
            },
        ]);
        assert_eq!(
            content_to_parts(&content),
            vec![
                json!({"text": "what's this?"}),
                json!({"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}),
            ]
        );
    }

    #[test]
    fn content_to_parts_translates_an_audio_part() {
        let content = MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: "aGVsbG8=".to_string(),
                format: "wav".to_string(),
            },
        }]);
        assert_eq!(
            content_to_parts(&content),
            vec![json!({"inlineData": {"mimeType": "audio/wav", "data": "aGVsbG8="}})]
        );
    }

    // --- build_contents: image content ----------------------------------------------

    #[test]
    fn build_contents_translates_a_user_image_content_part() {
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
        }];
        let (_, contents) = build_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(
            contents[0].parts,
            vec![
                json!({"fileData": {"mimeType": "image/png", "fileUri": "https://example.com/a.png"}})
            ]
        );
    }

    #[test]
    fn build_contents_collapses_assistant_and_system_image_parts_to_plain_text() {
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
            },
            ChatMessage::assistant("ok"),
        ];
        let (system, contents) = build_contents(&messages);
        assert_eq!(system, Some("be concise".to_string()));
        assert_eq!(contents[0].parts, vec![json!({"text": "ok"})]);
    }

    #[test]
    fn build_contents_translates_a_user_audio_content_part() {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: Some(MessageContent::Parts(vec![ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "aGVsbG8=".to_string(),
                    format: "wav".to_string(),
                },
            }])),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }];
        let (_, contents) = build_contents(&messages);
        assert_eq!(
            contents[0].parts,
            vec![json!({"inlineData": {"mimeType": "audio/wav", "data": "aGVsbG8="}})]
        );
    }
}
