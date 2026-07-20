use rp_core::ProviderError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RouterError {
    #[error("invalid model \"{0}\": expected \"provider/model\" or a configured route alias")]
    InvalidModel(String),

    #[error("provider \"{0}\" is unknown or not configured (missing API key?)")]
    ProviderNotConfigured(String),

    #[error(transparent)]
    Provider(#[from] ProviderError),
}

impl RouterError {
    pub fn status_code(&self) -> u16 {
        match self {
            RouterError::InvalidModel(_) => 400,
            RouterError::ProviderNotConfigured(_) => 424,
            RouterError::Provider(e) => e.status_code(),
        }
    }
}
