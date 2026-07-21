use serde::{Deserialize, Serialize};

/// The unified request/response schema follows the OpenAI chat completions
/// shape, since that's also what the HTTP server exposes to clients. Every
/// provider adapter translates to/from this type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Set on assistant messages that invoke one or more tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Set on `Role::Tool` messages: the id (from the assistant's
    /// `tool_calls`) of the call this message's `content` answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(MessageContent::text(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(MessageContent::text(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(MessageContent::text(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// A chat message's `content`, matching the OpenAI wire shape where it may
/// be either a plain string or an array of typed parts (for multimodal
/// messages). `#[serde(untagged)]` makes both forms transparently
/// interchangeable on the wire: a plain JSON string deserializes as
/// `Text`, a JSON array as `Parts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(s) => s.is_empty(),
            Self::Parts(parts) => parts.is_empty(),
        }
    }

    /// Collapses this content down to its plain text, discarding any image
    /// or audio parts. Used by adapters/roles that only understand text
    /// (e.g. tool results, or providers translating non-user roles).
    pub fn as_plain_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } | ContentPart::InputAudio { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// One part of a multimodal message's content array, in the OpenAI
/// `content: [{"type": "text", ...}, {"type": "image_url", ...},
/// {"type": "input_audio", ...}]` shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// Either a `data:<mime>;base64,<data>` URI or an `https://` URL,
    /// per the OpenAI convention.
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputAudio {
    /// Raw base64-encoded audio (not a `data:` URI — just the payload),
    /// per the OpenAI convention.
    pub data: String,
    /// e.g. `"wav"`, `"mp3"`.
    pub format: String,
}

/// A tool the model may call, in the OpenAI function-calling shape.
/// `parameters` is a JSON Schema object describing the function's
/// arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A single invocation of a tool, as requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments, per the OpenAI convention (a string, not a
    /// nested object) so it can be streamed as accumulating text.
    pub arguments: String,
}

impl ToolCall {
    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Either "provider/model" (e.g. "anthropic/claude-sonnet-5") to target
    /// one provider directly, or a router alias defined in the config's
    /// fallback chains (e.g. "smart").
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// `"auto"` | `"none"` | `"required"` | `{"type":"function","function":{"name":...}}`,
    /// per the OpenAI convention. Left as raw JSON since providers each
    /// have a slightly different vocabulary here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Constrains/orders which providers in a resolved fallback chain the
    /// router is allowed to try, à la OpenRouter's `provider` request field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    /// Constrains the shape of the model's output, per the OpenAI
    /// `response_format` convention. Not every provider can represent every
    /// variant natively -- see each adapter for its translation (or
    /// rejection via `ProviderError::UnsupportedFeature`, which a fallback
    /// chain treats the same as any other retryable error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

/// Constrains a model's output shape, matching the OpenAI
/// `response_format` wire convention exactly (`{"type": "text"}` /
/// `{"type": "json_object"}` / `{"type": "json_schema", "json_schema": {...}}`)
/// so the OpenAI-compatible adapter can pass it through unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// The default -- unconstrained free-form text (or a tool call).
    Text,
    /// Loose JSON mode: the model must emit syntactically valid JSON, but
    /// with no particular shape enforced.
    JsonObject,
    /// Strict schema-constrained JSON: the model's output must validate
    /// against `json_schema.schema`.
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaFormat {
    /// Identifies the schema -- required by the OpenAI wire format, and
    /// reused by the Anthropic adapter as the name of the synthetic tool it
    /// forces the model to call (see that adapter for why).
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// A JSON Schema object describing the required output shape.
    pub schema: serde_json::Value,
    /// OpenAI-specific strict-mode flag; providers that don't have a
    /// concept of "strict" schema adherence ignore this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Per-request routing constraints, applied to a resolved "provider/model"
/// chain (whether that came from a direct `"provider/model"` request or a
/// config route alias) before the router tries any of it.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderPreferences {
    /// If set, only these provider names (matching config keys, e.g.
    /// `"anthropic"`) are eligible; everything else is dropped from the
    /// chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// Provider names to drop from the chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// `"price"` stable-sorts the remaining chain ascending by the
    /// prompt-token price configured for each "provider/model" entry in
    /// `[[pricing]]`; entries with no configured price sort last. Any
    /// other value (or unset) leaves the chain in its configured order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<String>,
    /// If `true`, drop every candidate whose provider isn't marked
    /// `zdr = true` in config — i.e. only route to providers the operator
    /// has declared a Zero Data Retention agreement with. This is a
    /// self-declared config flag the router trusts, not something it
    /// verifies against the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
}

impl ChatRequest {
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    #[serde(skip_deserializing, default = "chat_completion_object")]
    pub object: &'static str,
    pub created: i64,
    /// The fully-qualified "provider/model" that actually served the
    /// request (useful when a fallback chain skipped the first choice).
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Estimated USD cost of this response, computed by the router from
    /// its `[[pricing]]` config. Not part of the OpenAI schema — an
    /// OpenAI SDK/client will just ignore it. `None` if the model that
    /// served this request has no configured pricing. Always unset here;
    /// provider adapters never populate it themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatMessageDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// An incremental piece of a tool call, streamed across possibly many
/// chunks: `id`/`function.name` arrive once, `function.arguments` arrives
/// as accumulating string fragments to be concatenated by the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FunctionCallDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    pub id: String,
    #[serde(skip_deserializing, default = "chat_completion_chunk_object")]
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Same as `ChatResponse::cost_usd`, set on whichever chunk carries the
    /// final `usage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

/// Metadata describing a model exposed by the router, for `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(skip_deserializing, default = "model_object")]
    pub object: &'static str,
    pub owned_by: String,
}

fn chat_completion_object() -> &'static str {
    "chat.completion"
}

fn chat_completion_chunk_object() -> &'static str {
    "chat.completion.chunk"
}

fn model_object() -> &'static str {
    "model"
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MessageContent (de)serialization ---------------------------------

    #[test]
    fn message_content_deserializes_a_plain_string_as_text() {
        let content: MessageContent = serde_json::from_str(r#""hello""#).unwrap();
        assert_eq!(content, MessageContent::Text("hello".to_string()));
    }

    #[test]
    fn message_content_deserializes_an_array_as_parts() {
        let content: MessageContent = serde_json::from_str(
            r#"[{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]"#,
        )
        .unwrap();
        assert_eq!(
            content,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "hi".to_string()
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/a.png".to_string(),
                        detail: None,
                    }
                },
            ])
        );
    }

    #[test]
    fn message_content_text_serializes_as_a_plain_string() {
        let json = serde_json::to_value(MessageContent::text("hi")).unwrap();
        assert_eq!(json, serde_json::json!("hi"));
    }

    #[test]
    fn message_content_parts_serializes_as_an_array() {
        let content = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/a.png".to_string(),
                detail: Some("high".to_string()),
            },
        }]);
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(
            json,
            serde_json::json!([{
                "type": "image_url",
                "image_url": {"url": "https://example.com/a.png", "detail": "high"}
            }])
        );
    }

    #[test]
    fn message_content_round_trips_through_a_chat_message() {
        let msg = ChatMessage {
            role: Role::User,
            content: Some(MessageContent::Parts(vec![ContentPart::Text {
                text: "describe this".to_string(),
            }])),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let round_tripped: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped.content, msg.content);
    }

    // --- MessageContent helpers ---------------------------------------------

    #[test]
    fn text_is_empty_reflects_the_string() {
        assert!(MessageContent::text("").is_empty());
        assert!(!MessageContent::text("hi").is_empty());
    }

    #[test]
    fn parts_is_empty_reflects_the_vec() {
        assert!(MessageContent::Parts(vec![]).is_empty());
        assert!(!MessageContent::Parts(vec![ContentPart::Text {
            text: "hi".to_string()
        }])
        .is_empty());
    }

    #[test]
    fn as_plain_text_returns_the_string_for_text_content() {
        assert_eq!(MessageContent::text("hello").as_plain_text(), "hello");
    }

    #[test]
    fn as_plain_text_concatenates_text_parts_and_drops_images() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "look at this: ".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                },
            },
            ContentPart::Text {
                text: "neat, right?".to_string(),
            },
        ]);
        assert_eq!(content.as_plain_text(), "look at this: neat, right?");
    }

    #[test]
    fn as_plain_text_is_empty_when_parts_has_no_text() {
        let content = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/a.png".to_string(),
                detail: None,
            },
        }]);
        assert_eq!(content.as_plain_text(), "");
    }

    // --- ContentPart::InputAudio ---------------------------------------------

    #[test]
    fn content_part_deserializes_input_audio() {
        let content: MessageContent = serde_json::from_str(
            r#"[{"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}}]"#,
        )
        .unwrap();
        assert_eq!(
            content,
            MessageContent::Parts(vec![ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "aGVsbG8=".to_string(),
                    format: "wav".to_string(),
                }
            }])
        );
    }

    #[test]
    fn content_part_serializes_input_audio() {
        let content = MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: "aGVsbG8=".to_string(),
                format: "mp3".to_string(),
            },
        }]);
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(
            json,
            serde_json::json!([{
                "type": "input_audio",
                "input_audio": {"data": "aGVsbG8=", "format": "mp3"}
            }])
        );
    }

    #[test]
    fn as_plain_text_drops_audio_parts_the_same_as_images() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "listen: ".to_string(),
            },
            ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "aGVsbG8=".to_string(),
                    format: "wav".to_string(),
                },
            },
        ]);
        assert_eq!(content.as_plain_text(), "listen: ");
    }

    // --- constructors wrap in MessageContent::Text ---------------------------

    #[test]
    fn user_constructor_wraps_content_as_text() {
        let msg = ChatMessage::user("hi");
        assert_eq!(msg.content, Some(MessageContent::text("hi")));
    }
}
