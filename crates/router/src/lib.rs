mod config;
mod error;

use std::collections::HashMap;
use std::sync::Arc;

pub use config::{Config, ProviderConfig, ProviderKind, RouteAlias, ServerConfig};
pub use error::RouterError;

use rp_core::{ChatRequest, ChatResponse, ChatStream, Provider};
use rp_providers::{AnthropicProvider, GeminiProvider, OpenAiCompatibleProvider};

/// Holds every provider adapter that could be built from config (i.e. its
/// API key env var was set), plus the named fallback-chain aliases. Model
/// strings are resolved to a chain of (provider, model) pairs and tried in
/// order, falling back on retryable errors (rate limits, timeouts, 5xxs).
pub struct Router {
    providers: HashMap<String, Arc<dyn Provider>>,
    routes: HashMap<String, Vec<String>>,
}

impl Router {
    pub fn from_config(config: &Config) -> Self {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();

        for (name, cfg) in &config.providers {
            let key = match std::env::var(&cfg.api_key_env) {
                Ok(k) if !k.is_empty() => k,
                _ => {
                    tracing::warn!(provider = %name, env_var = %cfg.api_key_env, "skipping provider: API key env var not set");
                    continue;
                }
            };

            let provider: Arc<dyn Provider> = match cfg.kind {
                ProviderKind::Openai => Arc::new(OpenAiCompatibleProvider::new(
                    name.clone(),
                    cfg.base_url.clone(),
                    key,
                )),
                ProviderKind::Anthropic => {
                    Arc::new(AnthropicProvider::new(cfg.base_url.clone(), key))
                }
                ProviderKind::Gemini => Arc::new(GeminiProvider::new(cfg.base_url.clone(), key)),
            };
            providers.insert(name.clone(), provider);
        }

        let routes = config
            .routes
            .iter()
            .map(|r| (r.alias.clone(), r.chain.clone()))
            .collect();

        Self { providers, routes }
    }

    pub fn configured_providers(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    pub fn route_aliases(&self) -> impl Iterator<Item = &str> {
        self.routes.keys().map(String::as_str)
    }

    /// Resolve a client-supplied `model` string into an ordered chain of
    /// (provider, model) pairs: either a configured alias's fallback chain,
    /// or a single "provider/model" entry.
    fn resolve_chain(&self, model: &str) -> Result<Vec<(String, String)>, RouterError> {
        let entries: Vec<String> = match self.routes.get(model) {
            Some(chain) => chain.clone(),
            None => vec![model.to_string()],
        };

        entries
            .into_iter()
            .map(|entry| {
                entry
                    .split_once('/')
                    .map(|(p, m)| (p.to_string(), m.to_string()))
                    .ok_or_else(|| RouterError::InvalidModel(entry.clone()))
            })
            .collect()
    }

    fn get_provider(&self, name: &str) -> Result<&Arc<dyn Provider>, RouterError> {
        self.providers
            .get(name)
            .ok_or_else(|| RouterError::ProviderNotConfigured(name.to_string()))
    }

    pub async fn dispatch(&self, req: &ChatRequest) -> Result<ChatResponse, RouterError> {
        let chain = self.resolve_chain(&req.model)?;
        let mut last_err: Option<RouterError> = None;

        for (provider_name, model_name) in &chain {
            let provider = match self.get_provider(provider_name) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(provider = %provider_name, "skipping candidate: {e}");
                    last_err = Some(e);
                    continue;
                }
            };

            match provider.chat(req, model_name).await {
                Ok(resp) => return Ok(resp),
                Err(e) if e.is_retryable() => {
                    tracing::warn!(provider = %provider_name, model = %model_name, "provider failed, falling back: {e}");
                    last_err = Some(RouterError::Provider(e));
                }
                Err(e) => return Err(RouterError::Provider(e)),
            }
        }

        Err(last_err.unwrap_or_else(|| RouterError::InvalidModel(req.model.clone())))
    }

    pub async fn dispatch_stream(&self, req: &ChatRequest) -> Result<ChatStream, RouterError> {
        let chain = self.resolve_chain(&req.model)?;
        let mut last_err: Option<RouterError> = None;

        for (provider_name, model_name) in &chain {
            let provider = match self.get_provider(provider_name) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(provider = %provider_name, "skipping candidate: {e}");
                    last_err = Some(e);
                    continue;
                }
            };

            match provider.chat_stream(req, model_name).await {
                Ok(stream) => return Ok(stream),
                Err(e) if e.is_retryable() => {
                    tracing::warn!(provider = %provider_name, model = %model_name, "provider failed, falling back: {e}");
                    last_err = Some(RouterError::Provider(e));
                }
                Err(e) => return Err(RouterError::Provider(e)),
            }
        }

        Err(last_err.unwrap_or_else(|| RouterError::InvalidModel(req.model.clone())))
    }
}
