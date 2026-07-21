use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::ProviderError;
use crate::types::{ChatChunk, ChatRequest, ChatResponse, EmbeddingsRequest, EmbeddingsResponse};

pub type ChatStream = BoxStream<'static, Result<ChatChunk, ProviderError>>;

/// Implemented by each backend (OpenAI-compatible, Anthropic, Gemini, ...).
/// `model` is the bare upstream model name, already stripped of the
/// "provider/" prefix by the router.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Short identifier used in config and in "provider/model" routing
    /// strings, e.g. "openai", "anthropic", "groq".
    fn name(&self) -> &str;

    /// `api_key_override`, when set, is used for this call's own
    /// authentication instead of the key this provider was constructed
    /// with -- how BYOK (`provider.byok` on the request, see
    /// `rp_core::ProviderPreferences`) reaches the actual HTTP call.
    /// `None` is the ordinary case: use the operator-configured key.
    async fn chat(
        &self,
        req: &ChatRequest,
        model: &str,
        api_key_override: Option<&str>,
    ) -> Result<ChatResponse, ProviderError>;

    async fn chat_stream(
        &self,
        req: &ChatRequest,
        model: &str,
        api_key_override: Option<&str>,
    ) -> Result<ChatStream, ProviderError>;

    /// Embeds `req.input` into vector(s). A provider with no embeddings
    /// API at all (Anthropic) returns
    /// `Err(ProviderError::UnsupportedFeature(_))` -- retryable, so a
    /// fallback chain moves on to a candidate that does support it,
    /// rather than failing the whole request outright.
    async fn embeddings(
        &self,
        req: &EmbeddingsRequest,
        model: &str,
        api_key_override: Option<&str>,
    ) -> Result<EmbeddingsResponse, ProviderError>;
}
