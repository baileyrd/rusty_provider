use thiserror::Error;

/// Error returned by a provider adapter. `is_retryable` tells the router
/// whether it's worth falling back to the next provider in a chain.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("authentication failed: {0}")]
    Auth(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("rate limited{}", .retry_after_secs.map(|s| format!(", retry after {s}s")).unwrap_or_default())]
    RateLimited { retry_after_secs: Option<u64> },

    #[error("upstream error (status {status}): {message}")]
    Upstream { status: u16, message: String },

    #[error("request timed out")]
    Timeout,

    #[error("network error: {0}")]
    Network(String),

    #[error("failed to decode provider response: {0}")]
    Decode(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("provider error: {0}")]
    Other(String),
}

impl ProviderError {
    /// Whether the router should try the next provider in a fallback chain
    /// after this error, as opposed to surfacing it straight to the client.
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::RateLimited { .. } => true,
            ProviderError::Timeout => true,
            ProviderError::Network(_) => true,
            ProviderError::Upstream { status, .. } => *status >= 500 || *status == 429,
            ProviderError::Auth(_) => false,
            ProviderError::InvalidRequest(_) => false,
            ProviderError::ModelNotFound(_) => false,
            ProviderError::Decode(_) => false,
            ProviderError::Other(_) => false,
        }
    }

    /// Best-effort HTTP status to use if this error must be surfaced to a
    /// client directly (e.g. every provider in a chain was exhausted).
    pub fn status_code(&self) -> u16 {
        match self {
            ProviderError::Auth(_) => 401,
            ProviderError::InvalidRequest(_) => 400,
            ProviderError::RateLimited { .. } => 429,
            ProviderError::Upstream { status, .. } => *status,
            ProviderError::Timeout => 504,
            ProviderError::Network(_) => 502,
            ProviderError::Decode(_) => 502,
            ProviderError::ModelNotFound(_) => 404,
            ProviderError::Other(_) => 500,
        }
    }
}
