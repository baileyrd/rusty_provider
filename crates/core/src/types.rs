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
    /// The model's reasoning/thinking trace behind this message, if the
    /// request opted in via `ChatRequest.reasoning` and the provider
    /// returned one. Plain text -- providers that expose richer structure
    /// (e.g. Anthropic's signed thinking blocks, needed to replay a prior
    /// turn in a later request) don't get full fidelity here, only the
    /// human-readable trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Marks this message as the end of a cacheable prefix, à la
    /// Anthropic's `cache_control` breakpoints. Only meaningful to
    /// providers with an explicit cache-breakpoint API (Anthropic); others
    /// (OpenAI-compatible, Gemini) cache automatically with no request-side
    /// marker, so this is silently a no-op there rather than an error --
    /// it's a hint/optimization, not a correctness requirement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(MessageContent::text(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
            cache_control: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(MessageContent::text(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
            cache_control: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(MessageContent::text(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
            cache_control: None,
        }
    }
}

/// A cache breakpoint hint on a message, matching Anthropic's
/// `cache_control` block shape directly (`{"type": "ephemeral"}`) so
/// Anthropic's adapter can pass it through with no translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    Ephemeral,
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
                    ContentPart::ImageUrl { .. }
                    | ContentPart::InputAudio { .. }
                    | ContentPart::File { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// One part of a multimodal message's content array, in the OpenAI
/// `content: [{"type": "text", ...}, {"type": "image_url", ...},
/// {"type": "input_audio", ...}, {"type": "file", ...}]` shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
    File { file: FileData },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// Either a `data:<mime>;base64,<data>` URI or an `https://` URL,
    /// per the OpenAI convention.
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A file attachment (currently only exercised for PDFs), per the OpenAI
/// convention -- same shape as `ImageUrl`: `file_data` is either a
/// `data:<mime>;base64,<data>` URI or an `https://` URL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileData {
    pub file_data: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
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
    /// An ad-hoc fallback chain for just this request, each entry a
    /// "provider/model" string tried in order after `model` if it fails --
    /// à la OpenRouter's `models` field. When non-empty, this entirely
    /// bypasses `[[routes]]` alias lookup for `model` (so `model` itself
    /// must be a direct "provider/model" here, not an alias); every other
    /// request keeps resolving `model` through configured route aliases as
    /// before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
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
    /// Requests a visible reasoning/thinking trace from models that
    /// support one. `None` means "use the model's default" -- most
    /// reasoning models still reason internally either way; this only
    /// controls how much budget/effort goes into it and whether the trace
    /// is surfaced back in `ChatMessage.reasoning`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// Restricts sampling to the `top_k` highest-probability tokens.
    /// Native to Anthropic and Gemini; not part of OpenAI's own API but
    /// common on OpenAI-compatible inference servers (Groq, Together,
    /// Fireworks, vLLM, etc.), so the OpenAI-compatible adapter passes it
    /// through unconditionally rather than guessing which backend
    /// supports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Nucleus-style cutoff by minimum token probability (relative to the
    /// most likely token), rather than `top_p`'s cumulative-probability
    /// mass. Not supported by Anthropic or Gemini's native APIs -- silently
    /// ignored there, same rationale as `cache_control` on a provider with
    /// no cache-breakpoint API: a sampling hint, not a correctness
    /// requirement, so the request still produces a valid response either
    /// way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    /// Dynamic nucleus cutoff scaled by the top token's own probability
    /// (used by some open-source inference servers). Not supported by
    /// Anthropic or Gemini's native APIs -- silently ignored there, same
    /// rationale as `min_p`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f32>,
    /// Penalizes tokens by how often they've already appeared, per the
    /// OpenAI convention. Native to OpenAI and Gemini; not supported by
    /// Anthropic's API -- silently ignored there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Penalizes tokens that have appeared at all, per the OpenAI
    /// convention. Native to OpenAI and Gemini; not supported by
    /// Anthropic's API -- silently ignored there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// A multiplicative variant of frequency/presence penalties common on
    /// open-source inference servers. Not supported by Anthropic or
    /// Gemini's native APIs -- silently ignored there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Per-token-ID logit bias, per the OpenAI convention (keys are
    /// provider-specific token ID strings, so this only round-trips
    /// meaningfully with the same tokenizer that produced the IDs).
    /// Native to OpenAI; not supported by Anthropic or Gemini's native
    /// APIs -- silently ignored there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Best-effort determinism across repeated requests with identical
    /// parameters, per the OpenAI convention (no provider guarantees
    /// bit-exact reproducibility). Native to OpenAI and Gemini; not
    /// supported by Anthropic's API -- silently ignored there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

/// Requests and tunes a model's reasoning/thinking trace. `effort` and
/// `max_tokens` are alternative ways of specifying the same budget --
/// `effort` (OpenAI's convention: `"low"`/`"medium"`/`"high"`) for
/// providers with no direct token-budget knob, `max_tokens` (Anthropic/
/// Gemini's convention) for providers that take one directly. If both are
/// set, a provider that supports `max_tokens` uses it directly; one that
/// only understands `effort` ignores `max_tokens`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// If `true`, the model still reasons internally (where supported) but
    /// the trace is withheld from `ChatMessage.reasoning` -- saves
    /// response size/bandwidth without giving up whatever quality benefit
    /// reasoning provides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
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
    /// Drop every candidate priced above this, in USD per million prompt
    /// tokens -- the same `prompt_per_million` figure `sort: "price"`
    /// consults from `[[pricing]]`. A candidate with no configured price
    /// is dropped too, on the same reasoning as `zdr`: an unverifiable
    /// claim isn't a satisfied one, so with a hard ceiling in effect an
    /// unpriced entry can't be trusted to be under it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price: Option<f64>,
    /// If `true`, drop every candidate whose provider adapter doesn't
    /// actually give an effect to every field this specific request sets
    /// (`tools`, `response_format`, `top_k`, a message's `cache_control`,
    /// etc. -- see `GET /v1/models`' `supported_params` for the exact
    /// per-provider list). Without this, an unsupported field is silently
    /// dropped or, for a structural field like `response_format`,
    /// rejected only after a wasted round trip; this filters those
    /// candidates out before dispatch instead. `stop`/`temperature`/
    /// `top_p`/`max_tokens` never disqualify a candidate -- every
    /// provider kind supports all four natively.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
}

impl ChatRequest {
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Total input tokens, including any served from or written to a
    /// cache -- `cached_tokens` and `cache_creation_tokens` below are a
    /// breakdown of this total, not additive on top of it.
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// The portion of `prompt_tokens` served from a cache instead of
    /// freshly processed, if the provider reports this breakdown. `None`
    /// when the provider exposes no cache accounting (or nothing was
    /// cached).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    /// The portion of `prompt_tokens` newly written into a cache by this
    /// request, billed at a premium over a normal prompt token on
    /// providers that separately account for it (Anthropic). `None` on
    /// providers with no separate cache-write cost (OpenAI-compatible,
    /// Gemini bill a cache write the same as a normal prompt token).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u32>,
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
    /// Incremental piece of the model's reasoning/thinking trace, streamed
    /// the same way `content` is -- see `ChatMessage.reasoning`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
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
/// `context_length`/`pricing`/`supported_params` are only known for a
/// concrete "provider/model" with a `[[pricing]]` entry -- absent for a
/// `[[routes]]` alias (which can span models with different context
/// windows and pricing) and for a `"{provider}/*"` wildcard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(skip_deserializing, default = "model_object")]
    pub object: &'static str,
    pub owned_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<ModelPricing>,
    /// Which `ChatRequest` fields this model's provider adapter actually
    /// gives an effect to -- see each adapter's own docs/README for the
    /// per-provider translation or silent-no-op behavior of each.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supported_params: Option<Vec<String>>,
}

/// USD per million tokens, mirroring a `[[pricing]]` entry -- with
/// `cache_read`/`cache_write` already defaulted to `prompt` when the
/// operator left them unset in config, same as `cost_usd` computation
/// uses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelPricing {
    pub prompt: f64,
    pub completion: f64,
    pub cache_read: f64,
    pub cache_write: f64,
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
            reasoning: None,
            cache_control: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let round_tripped: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped.content, msg.content);
    }

    // --- CacheControl ---------------------------------------------------------

    #[test]
    fn cache_control_ephemeral_serializes_matching_anthropics_wire_shape() {
        let json = serde_json::to_value(CacheControl::Ephemeral).unwrap();
        assert_eq!(json, serde_json::json!({"type": "ephemeral"}));
    }

    #[test]
    fn chat_message_cache_control_round_trips_through_json() {
        let mut msg = ChatMessage::user("hi");
        msg.cache_control = Some(CacheControl::Ephemeral);
        let json = serde_json::to_string(&msg).unwrap();
        let round_tripped: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped.cache_control, Some(CacheControl::Ephemeral));
    }

    #[test]
    fn chat_message_cache_control_is_absent_from_json_when_unset() {
        let msg = ChatMessage::user("hi");
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json.get("cache_control").is_none());
    }

    // --- Usage: cache-token fields ---------------------------------------------

    #[test]
    fn usage_cache_fields_are_absent_from_json_when_unset() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cached_tokens: None,
            cache_creation_tokens: None,
        };
        let json = serde_json::to_value(&usage).unwrap();
        assert!(json.get("cached_tokens").is_none());
        assert!(json.get("cache_creation_tokens").is_none());
    }

    #[test]
    fn usage_cache_fields_serialize_when_set() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cached_tokens: Some(8),
            cache_creation_tokens: Some(2),
        };
        let json = serde_json::to_value(&usage).unwrap();
        assert_eq!(json["cached_tokens"], 8);
        assert_eq!(json["cache_creation_tokens"], 2);
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
