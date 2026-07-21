use rp_core::ProviderError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RouterError {
    #[error("invalid model \"{0}\": expected \"provider/model\" or a configured route alias")]
    InvalidModel(String),

    #[error("provider \"{0}\" is unknown or not configured (missing API key?)")]
    ProviderNotConfigured(String),

    #[error(
        "no provider for \"{0}\" survives the request's provider.only/ignore/zdr/data_collection filter"
    )]
    NoEligibleProvider(String),

    #[error("request blocked by guardrail \"{0}\"")]
    GuardrailBlocked(String),

    #[error(transparent)]
    Provider(#[from] ProviderError),
}

impl RouterError {
    pub fn status_code(&self) -> u16 {
        match self {
            RouterError::InvalidModel(_) => 400,
            RouterError::ProviderNotConfigured(_) => 424,
            RouterError::NoEligibleProvider(_) => 400,
            RouterError::GuardrailBlocked(_) => 400,
            RouterError::Provider(e) => e.status_code(),
        }
    }

    /// Seconds the client should wait before retrying, if known — set when
    /// this wraps a `ProviderError::RateLimited` (from the upstream
    /// provider itself, or from this router's own outbound self-throttle).
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            RouterError::Provider(ProviderError::RateLimited { retry_after_secs }) => {
                *retry_after_secs
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_model_maps_to_400() {
        assert_eq!(
            RouterError::InvalidModel("bogus".to_string()).status_code(),
            400
        );
    }

    #[test]
    fn provider_not_configured_maps_to_424() {
        assert_eq!(
            RouterError::ProviderNotConfigured("openai".to_string()).status_code(),
            424
        );
    }

    #[test]
    fn no_eligible_provider_maps_to_400() {
        assert_eq!(
            RouterError::NoEligibleProvider("anthropic/claude".to_string()).status_code(),
            400
        );
    }

    #[test]
    fn guardrail_blocked_maps_to_400() {
        assert_eq!(
            RouterError::GuardrailBlocked("no-ssn".to_string()).status_code(),
            400
        );
    }

    #[test]
    fn provider_variant_delegates_to_the_wrapped_providererror_status_code() {
        let cases: [(ProviderError, u16); 6] = [
            (ProviderError::Auth("x".to_string()), 401),
            (ProviderError::InvalidRequest("x".to_string()), 400),
            (
                ProviderError::RateLimited {
                    retry_after_secs: Some(10),
                },
                429,
            ),
            (
                ProviderError::Upstream {
                    status: 503,
                    message: "x".to_string(),
                },
                503,
            ),
            (ProviderError::Timeout, 504),
            (ProviderError::ModelNotFound("x".to_string()), 404),
        ];
        for (provider_err, expected) in cases {
            let router_err = RouterError::from(provider_err);
            assert_eq!(router_err.status_code(), expected);
        }
    }
}
