mod config;
mod error;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

pub use config::{Config, PricingEntry, ProviderConfig, ProviderKind, RouteAlias, ServerConfig};
pub use error::RouterError;

use futures::stream::StreamExt;
use rp_core::{ChatRequest, ChatResponse, ChatStream, Provider, ProviderPreferences};
use rp_providers::{AnthropicProvider, GeminiProvider, OpenAiCompatibleProvider};

/// Weight given to each new latency/throughput sample in its running
/// average — higher reacts faster to recent conditions, lower smooths out
/// noise.
const EWMA_ALPHA: f64 = 0.3;

/// Holds every provider adapter that could be built from config (i.e. its
/// API key env var was set), the named fallback-chain aliases, static
/// per-model pricing (used only for `provider.sort: "price"` requests),
/// and running averages of this router's own observed per-model response
/// latency and generation throughput (used for `provider.sort: "latency"`
/// / `"throughput"` requests). Model strings are resolved to a chain of
/// (provider, model) pairs and tried in order, falling back on retryable
/// errors (rate limits, timeouts, 5xxs).
pub struct Router {
    providers: HashMap<String, Arc<dyn Provider>>,
    routes: HashMap<String, Vec<String>>,
    /// "provider/model" -> prompt price per million tokens.
    pricing: HashMap<String, f64>,
    /// Provider names with `zdr = true` in config.
    zdr_providers: HashSet<String>,
    /// "provider/model" -> EWMA response latency in milliseconds, measured
    /// from this process's own successful dispatches. In-memory only —
    /// resets on restart and isn't a live external feed.
    latency: RwLock<HashMap<String, f64>>,
    /// "provider/model" -> EWMA completion tokens/sec, measured the same
    /// way. `Arc`-wrapped (unlike `latency`) so it can be shared into a
    /// streaming response's instrumentation, which outlives `dispatch_stream`
    /// itself — the router hands the stream off to the HTTP layer rather
    /// than consuming it.
    throughput: Arc<RwLock<HashMap<String, f64>>>,
}

/// Record a new EWMA sample under `key`, seeding the average on first
/// observation.
fn ewma_record(map: &RwLock<HashMap<String, f64>>, key: String, sample: f64) {
    let mut map = map.write().unwrap();
    map.entry(key)
        .and_modify(|avg| *avg = EWMA_ALPHA * sample + (1.0 - EWMA_ALPHA) * *avg)
        .or_insert(sample);
}

/// Look up an EWMA sample for "provider/model", or `missing` if this
/// router has no observation for it yet.
fn ewma_lookup(
    map: &RwLock<HashMap<String, f64>>,
    provider: &str,
    model: &str,
    missing: f64,
) -> f64 {
    map.read()
        .unwrap()
        .get(&format!("{provider}/{model}"))
        .copied()
        .unwrap_or(missing)
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

        let pricing = config
            .pricing
            .iter()
            .map(|p| (p.model.clone(), p.prompt_per_million))
            .collect();

        let zdr_providers = config
            .providers
            .iter()
            .filter(|(_, cfg)| cfg.zdr)
            .map(|(name, _)| name.clone())
            .collect();

        Self {
            providers,
            routes,
            pricing,
            zdr_providers,
            latency: RwLock::new(HashMap::new()),
            throughput: Arc::new(RwLock::new(HashMap::new())),
        }
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

    /// Apply a request's `provider.only`/`provider.ignore`/`provider.zdr`/
    /// `provider.sort` constraints to a resolved chain, in that order:
    /// filter, then sort.
    fn apply_preferences(
        &self,
        model: &str,
        mut chain: Vec<(String, String)>,
        prefs: Option<&ProviderPreferences>,
    ) -> Result<Vec<(String, String)>, RouterError> {
        let Some(prefs) = prefs else { return Ok(chain) };

        if let Some(only) = &prefs.only {
            chain.retain(|(provider, _)| only.iter().any(|p| p == provider));
        }
        if let Some(ignore) = &prefs.ignore {
            chain.retain(|(provider, _)| !ignore.iter().any(|p| p == provider));
        }
        if prefs.zdr == Some(true) {
            chain.retain(|(provider, _)| self.zdr_providers.contains(provider));
        }
        if chain.is_empty() {
            return Err(RouterError::NoEligibleProvider(model.to_string()));
        }

        match prefs.sort.as_deref() {
            Some("price") => chain.sort_by(|a, b| {
                let price_of = |entry: &(String, String)| {
                    self.pricing
                        .get(&format!("{}/{}", entry.0, entry.1))
                        .copied()
                        .unwrap_or(f64::MAX)
                };
                price_of(a).total_cmp(&price_of(b))
            }),
            // Ascending: lower is better, and unobserved entries (f64::MAX) sort last.
            Some("latency") => chain.sort_by(|a, b| {
                ewma_lookup(&self.latency, &a.0, &a.1, f64::MAX).total_cmp(&ewma_lookup(
                    &self.latency,
                    &b.0,
                    &b.1,
                    f64::MAX,
                ))
            }),
            // Descending: higher tokens/sec is better, and unobserved entries (0.0) sort last.
            Some("throughput") => chain.sort_by(|a, b| {
                ewma_lookup(&self.throughput, &b.0, &b.1, 0.0).total_cmp(&ewma_lookup(
                    &self.throughput,
                    &a.0,
                    &a.1,
                    0.0,
                ))
            }),
            _ => {}
        }

        Ok(chain)
    }

    fn get_provider(&self, name: &str) -> Result<&Arc<dyn Provider>, RouterError> {
        self.providers
            .get(name)
            .ok_or_else(|| RouterError::ProviderNotConfigured(name.to_string()))
    }

    /// Wrap a streaming response so that whichever chunk carries the final
    /// `usage` (completion token count) also records a throughput sample,
    /// without otherwise touching the stream — the router hands the stream
    /// off to the HTTP layer to consume, it doesn't read it itself.
    fn instrument_throughput(
        &self,
        provider_name: String,
        model_name: String,
        started_at: Instant,
        stream: ChatStream,
    ) -> ChatStream {
        let throughput = self.throughput.clone();
        let instrumented = stream.inspect(move |item| {
            let Ok(chunk) = item else { return };
            let Some(usage) = &chunk.usage else { return };
            if usage.completion_tokens == 0 {
                return;
            }
            let elapsed_secs = started_at.elapsed().as_secs_f64();
            if elapsed_secs <= 0.0 {
                return;
            }
            let tps = usage.completion_tokens as f64 / elapsed_secs;
            ewma_record(&throughput, format!("{provider_name}/{model_name}"), tps);
            tracing::debug!(provider = %provider_name, model = %model_name, tokens_per_sec = tps, "recorded throughput");
        });
        Box::pin(instrumented)
    }

    pub async fn dispatch(&self, req: &ChatRequest) -> Result<ChatResponse, RouterError> {
        let chain = self.resolve_chain(&req.model)?;
        let chain = self.apply_preferences(&req.model, chain, req.provider.as_ref())?;
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

            let started_at = Instant::now();
            match provider.chat(req, model_name).await {
                Ok(resp) => {
                    let elapsed_secs = started_at.elapsed().as_secs_f64();
                    ewma_record(
                        &self.latency,
                        format!("{provider_name}/{model_name}"),
                        elapsed_secs * 1000.0,
                    );
                    tracing::debug!(provider = %provider_name, model = %model_name, elapsed_ms = elapsed_secs * 1000.0, "recorded latency");

                    if let Some(usage) = &resp.usage {
                        if usage.completion_tokens > 0 && elapsed_secs > 0.0 {
                            let tps = usage.completion_tokens as f64 / elapsed_secs;
                            ewma_record(
                                &self.throughput,
                                format!("{provider_name}/{model_name}"),
                                tps,
                            );
                            tracing::debug!(provider = %provider_name, model = %model_name, tokens_per_sec = tps, "recorded throughput");
                        }
                    }

                    return Ok(resp);
                }
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
        let chain = self.apply_preferences(&req.model, chain, req.provider.as_ref())?;
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

            let started_at = Instant::now();
            match provider.chat_stream(req, model_name).await {
                Ok(stream) => {
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    ewma_record(
                        &self.latency,
                        format!("{provider_name}/{model_name}"),
                        elapsed_ms,
                    );
                    tracing::debug!(provider = %provider_name, model = %model_name, elapsed_ms, "recorded latency (time to first byte)");

                    return Ok(self.instrument_throughput(
                        provider_name.clone(),
                        model_name.clone(),
                        started_at,
                        stream,
                    ));
                }
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
