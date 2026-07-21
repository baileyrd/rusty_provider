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
    pub content: Option<String>,
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
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
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
