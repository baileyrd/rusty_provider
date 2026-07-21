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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_maps_to_401() {
        assert_eq!(
            ProviderError::Auth("bad key".to_string()).status_code(),
            401
        );
    }

    #[test]
    fn invalid_request_maps_to_400() {
        assert_eq!(
            ProviderError::InvalidRequest("bad body".to_string()).status_code(),
            400
        );
    }

    #[test]
    fn rate_limited_maps_to_429_regardless_of_retry_after() {
        assert_eq!(
            ProviderError::RateLimited {
                retry_after_secs: Some(30)
            }
            .status_code(),
            429
        );
        assert_eq!(
            ProviderError::RateLimited {
                retry_after_secs: None
            }
            .status_code(),
            429
        );
    }

    #[test]
    fn upstream_passes_the_provider_status_through_unchanged() {
        for status in [400, 401, 403, 404, 429, 500, 502, 503, 529] {
            assert_eq!(
                ProviderError::Upstream {
                    status,
                    message: "boom".to_string()
                }
                .status_code(),
                status,
                "upstream status {status} should pass through as-is"
            );
        }
    }

    #[test]
    fn timeout_maps_to_504() {
        assert_eq!(ProviderError::Timeout.status_code(), 504);
    }

    #[test]
    fn network_maps_to_502() {
        assert_eq!(
            ProviderError::Network("connection reset".to_string()).status_code(),
            502
        );
    }

    #[test]
    fn decode_maps_to_502() {
        assert_eq!(
            ProviderError::Decode("invalid json".to_string()).status_code(),
            502
        );
    }

    #[test]
    fn model_not_found_maps_to_404() {
        assert_eq!(
            ProviderError::ModelNotFound("gpt-9".to_string()).status_code(),
            404
        );
    }

    #[test]
    fn other_maps_to_500() {
        assert_eq!(
            ProviderError::Other("unclassified".to_string()).status_code(),
            500
        );
    }

    #[test]
    fn every_status_code_falls_in_the_valid_http_status_range() {
        let variants = [
            ProviderError::Auth("x".to_string()),
            ProviderError::InvalidRequest("x".to_string()),
            ProviderError::RateLimited {
                retry_after_secs: None,
            },
            ProviderError::Upstream {
                status: 503,
                message: "x".to_string(),
            },
            ProviderError::Timeout,
            ProviderError::Network("x".to_string()),
            ProviderError::Decode("x".to_string()),
            ProviderError::ModelNotFound("x".to_string()),
            ProviderError::Other("x".to_string()),
        ];
        for variant in variants {
            let status = variant.status_code();
            assert!(
                (100..=599).contains(&status),
                "status {status} out of valid HTTP range for {variant:?}"
            );
        }
    }

    #[test]
    fn retryable_errors_never_map_to_a_4xx_client_error_status() {
        // The router falls back to the next provider for retryable errors
        // rather than surfacing status_code() straight to the client, but
        // if it ever did, a 4xx here would wrongly blame the caller for
        // something transient/upstream.
        let retryable = [
            ProviderError::RateLimited {
                retry_after_secs: Some(5),
            },
            ProviderError::Timeout,
            ProviderError::Network("x".to_string()),
            ProviderError::Upstream {
                status: 503,
                message: "x".to_string(),
            },
        ];
        for err in retryable {
            assert!(err.is_retryable());
            let status = err.status_code();
            assert!(
                !(400..500).contains(&status) || status == 429,
                "retryable error {err:?} unexpectedly mapped to client-error status {status}"
            );
        }
    }
}
