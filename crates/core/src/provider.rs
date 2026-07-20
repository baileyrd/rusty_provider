use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::ProviderError;
use crate::types::{ChatChunk, ChatRequest, ChatResponse};

pub type ChatStream = BoxStream<'static, Result<ChatChunk, ProviderError>>;

/// Implemented by each backend (OpenAI-compatible, Anthropic, Gemini, ...).
/// `model` is the bare upstream model name, already stripped of the
/// "provider/" prefix by the router.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Short identifier used in config and in "provider/model" routing
    /// strings, e.g. "openai", "anthropic", "groq".
    fn name(&self) -> &str;

    async fn chat(&self, req: &ChatRequest, model: &str) -> Result<ChatResponse, ProviderError>;

    async fn chat_stream(
        &self,
        req: &ChatRequest,
        model: &str,
    ) -> Result<ChatStream, ProviderError>;
}
