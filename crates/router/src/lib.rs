mod config;
mod error;
mod metrics;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

pub use config::{Config, PricingEntry, ProviderConfig, ProviderKind, RouteAlias, ServerConfig};
pub use error::RouterError;
pub use metrics::Metrics;

use futures::stream::StreamExt;
use rp_core::{
    ChatRequest, ChatResponse, ChatStream, Provider, ProviderError, ProviderPreferences,
    RateLimiter, Usage,
};
use rp_providers::{AnthropicProvider, GeminiProvider, OpenAiCompatibleProvider};

/// Weight given to each new latency/throughput sample in its running
/// average — higher reacts faster to recent conditions, lower smooths out
/// noise.
const EWMA_ALPHA: f64 = 0.3;

/// Cumulative request/token/cost counters for one "provider/model", as
/// returned by `GET /v1/usage`.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct UsageStats {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Sum of every response's estimated cost. Only accumulates for
    /// responses whose model had a `[[pricing]]` entry — requests to an
    /// unpriced model still count toward `requests`/`*_tokens` but leave
    /// this at 0.0, which is "unknown," not "free."
    pub cost_usd: f64,
}

/// Holds every provider adapter that could be built from config (i.e. its
/// API key env var was set), the named fallback-chain aliases, static
/// per-model pricing (used for `provider.sort: "price"` and for computing
/// each response's `cost_usd`), and running metrics — latency/throughput
/// averages, cumulative usage/cost, and the Prometheus registry backing
/// them — from this router's own observed traffic. Model strings are
/// resolved to a chain of (provider, model) pairs and tried in order,
/// falling back on retryable errors (rate limits, timeouts, 5xxs).
pub struct Router {
    providers: HashMap<String, Arc<dyn Provider>>,
    routes: HashMap<String, Vec<String>>,
    /// "provider/model" -> (prompt price, completion price) per million tokens.
    pricing: Arc<HashMap<String, (f64, f64)>>,
    /// Provider names with `zdr = true` in config.
    zdr_providers: HashSet<String>,
    /// "provider/model" -> EWMA response latency in milliseconds, measured
    /// from this process's own successful dispatches. In-memory only —
    /// resets on restart and isn't a live external feed.
    latency: RwLock<HashMap<String, f64>>,
    /// "provider/model" -> EWMA completion tokens/sec, measured the same
    /// way.
    throughput: Arc<RwLock<HashMap<String, f64>>>,
    /// "provider/model" -> cumulative usage/cost. `Arc`-wrapped (like
    /// `throughput`, unlike `latency`) so it can be shared into a
    /// streaming response's instrumentation, which outlives
    /// `dispatch_stream` itself — the router hands the stream off to the
    /// HTTP layer rather than consuming it.
    usage: Arc<RwLock<HashMap<String, UsageStats>>>,
    /// Prometheus counters/histograms/gauges for `GET /metrics`, updated at
    /// the same points as `latency`/`throughput`/`usage` above.
    metrics: Metrics,
    /// Provider names with a self-imposed `requests_per_minute` in config.
    provider_rpm: HashMap<String, u32>,
    /// Backs `provider_rpm`'s outbound self-throttling — one bucket per
    /// provider name, checked before every dispatch attempt.
    outbound_limiter: RateLimiter,
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

/// Compute a response's estimated USD cost (if `pricing` has an entry for
/// "provider/model") and fold it, along with the raw token counts, into
/// that entry's cumulative `UsageStats`. Returns the computed cost so the
/// caller can attach it to the response/chunk sent back to the client.
fn record_usage(
    usage_map: &RwLock<HashMap<String, UsageStats>>,
    pricing: &HashMap<String, (f64, f64)>,
    provider: &str,
    model: &str,
    usage: &Usage,
) -> Option<f64> {
    let key = format!("{provider}/{model}");
    let cost = pricing.get(&key).map(|(prompt_ppm, completion_ppm)| {
        (usage.prompt_tokens as f64 * prompt_ppm + usage.completion_tokens as f64 * completion_ppm)
            / 1_000_000.0
    });

    let mut map = usage_map.write().unwrap();
    let stats = map.entry(key).or_default();
    stats.requests += 1;
    stats.prompt_tokens += usage.prompt_tokens as u64;
    stats.completion_tokens += usage.completion_tokens as u64;
    if let Some(cost) = cost {
        stats.cost_usd += cost;
    }

    cost
}

impl Router {
    pub fn from_config(config: &Config) -> Self {
        let metrics = Metrics::new();
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();

        for (name, cfg) in &config.providers {
            let key = match std::env::var(&cfg.api_key_env) {
                Ok(k) if !k.is_empty() => k,
                _ => {
                    tracing::warn!(provider = %name, env_var = %cfg.api_key_env, "skipping provider: API key env var not set");
                    metrics.set_provider_configured(name, false);
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
            metrics.set_provider_configured(name, true);
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
            .map(|p| {
                (
                    p.model.clone(),
                    (p.prompt_per_million, p.completion_per_million),
                )
            })
            .collect();

        let zdr_providers = config
            .providers
            .iter()
            .filter(|(_, cfg)| cfg.zdr)
            .map(|(name, _)| name.clone())
            .collect();

        let provider_rpm = config
            .providers
            .iter()
            .filter_map(|(name, cfg)| cfg.requests_per_minute.map(|rpm| (name.clone(), rpm)))
            .collect();

        Self {
            providers,
            routes,
            pricing: Arc::new(pricing),
            zdr_providers,
            latency: RwLock::new(HashMap::new()),
            throughput: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(HashMap::new())),
            metrics,
            provider_rpm,
            outbound_limiter: RateLimiter::new(),
        }
    }

    pub fn configured_providers(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    pub fn route_aliases(&self) -> impl Iterator<Item = &str> {
        self.routes.keys().map(String::as_str)
    }

    /// Snapshot of cumulative usage/cost per "provider/model", for
    /// `GET /v1/usage`.
    pub fn usage_snapshot(&self) -> HashMap<String, UsageStats> {
        self.usage.read().unwrap().clone()
    }

    /// Every registered metric rendered in the Prometheus text exposition
    /// format, for `GET /metrics`.
    pub fn render_prometheus_metrics(&self) -> String {
        self.metrics.render()
    }

    /// Record an inbound request rejected by the HTTP layer's own
    /// per-client/per-IP rate limiter, so it shows up in `GET /metrics`
    /// alongside every other counter this router tracks.
    pub fn record_inbound_rate_limit_rejection(&self, identity: &str) {
        self.metrics.record_inbound_rate_limit_rejection(identity);
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
                        .map(|(prompt_ppm, _)| *prompt_ppm)
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

    /// Self-imposed outbound throttle: if `provider_name` has a configured
    /// `requests_per_minute`, consume one token from its bucket. A no-op
    /// (always `Ok`) for providers with no configured limit. This never
    /// talks to the provider itself — it's purely so this router doesn't
    /// exceed limits it already knows about and get 429'd/banned.
    fn check_outbound_rate_limit(&self, provider_name: &str) -> Result<(), ProviderError> {
        let Some(&rpm) = self.provider_rpm.get(provider_name) else {
            return Ok(());
        };
        self.outbound_limiter
            .check(provider_name, rpm)
            .map_err(|retry_after_secs| ProviderError::RateLimited {
                retry_after_secs: Some(retry_after_secs.ceil() as u64),
            })
    }

    /// Wrap a streaming response so that whichever chunk carries the final
    /// `usage` (completion token count) also records a throughput sample
    /// and cumulative usage/cost/metrics, stamping the chunk's own
    /// `cost_usd` in the process — the router hands the stream off to the
    /// HTTP layer to consume, so this is the only point where it gets to
    /// touch it.
    fn instrument_stream(
        &self,
        provider_name: String,
        model_name: String,
        started_at: Instant,
        stream: ChatStream,
    ) -> ChatStream {
        let throughput = self.throughput.clone();
        let usage_map = self.usage.clone();
        let pricing = self.pricing.clone();
        let metrics = self.metrics.clone();

        let instrumented = stream.map(move |mut item| {
            if let Ok(chunk) = &mut item {
                if let Some(usage) = chunk.usage.clone() {
                    if usage.completion_tokens > 0 {
                        let elapsed_secs = started_at.elapsed().as_secs_f64();
                        if elapsed_secs > 0.0 {
                            let tps = usage.completion_tokens as f64 / elapsed_secs;
                            ewma_record(&throughput, format!("{provider_name}/{model_name}"), tps);
                            metrics.observe_throughput_tps(&provider_name, &model_name, tps);
                            tracing::debug!(provider = %provider_name, model = %model_name, tokens_per_sec = tps, "recorded throughput");
                        }
                        let cost = record_usage(&usage_map, &pricing, &provider_name, &model_name, &usage);
                        metrics.record_tokens_and_cost(&provider_name, &model_name, usage.prompt_tokens, usage.completion_tokens, cost);
                        chunk.cost_usd = cost;
                    }
                }
            }
            item
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
                    self.metrics
                        .record_attempt(provider_name, model_name, "not_configured");
                    last_err = Some(e);
                    continue;
                }
            };

            if let Err(e) = self.check_outbound_rate_limit(provider_name) {
                tracing::warn!(provider = %provider_name, model = %model_name, "outbound rate limit hit, falling back: {e}");
                self.metrics
                    .record_attempt(provider_name, model_name, "rate_limited");
                last_err = Some(RouterError::Provider(e));
                continue;
            }

            let started_at = Instant::now();
            match provider.chat(req, model_name).await {
                Ok(mut resp) => {
                    let elapsed_secs = started_at.elapsed().as_secs_f64();
                    ewma_record(
                        &self.latency,
                        format!("{provider_name}/{model_name}"),
                        elapsed_secs * 1000.0,
                    );
                    self.metrics
                        .observe_latency_seconds(provider_name, model_name, elapsed_secs);
                    self.metrics
                        .record_attempt(provider_name, model_name, "success");
                    tracing::debug!(provider = %provider_name, model = %model_name, elapsed_ms = elapsed_secs * 1000.0, "recorded latency");

                    if let Some(usage) = resp.usage.clone() {
                        if usage.completion_tokens > 0 && elapsed_secs > 0.0 {
                            let tps = usage.completion_tokens as f64 / elapsed_secs;
                            ewma_record(
                                &self.throughput,
                                format!("{provider_name}/{model_name}"),
                                tps,
                            );
                            self.metrics
                                .observe_throughput_tps(provider_name, model_name, tps);
                            tracing::debug!(provider = %provider_name, model = %model_name, tokens_per_sec = tps, "recorded throughput");
                        }
                        let cost = record_usage(
                            &self.usage,
                            &self.pricing,
                            provider_name,
                            model_name,
                            &usage,
                        );
                        self.metrics.record_tokens_and_cost(
                            provider_name,
                            model_name,
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            cost,
                        );
                        resp.cost_usd = cost;
                    }

                    return Ok(resp);
                }
                Err(e) if e.is_retryable() => {
                    tracing::warn!(provider = %provider_name, model = %model_name, "provider failed, falling back: {e}");
                    self.metrics
                        .record_attempt(provider_name, model_name, "retryable_error");
                    last_err = Some(RouterError::Provider(e));
                }
                Err(e) => {
                    self.metrics
                        .record_attempt(provider_name, model_name, "error");
                    return Err(RouterError::Provider(e));
                }
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
                    self.metrics
                        .record_attempt(provider_name, model_name, "not_configured");
                    last_err = Some(e);
                    continue;
                }
            };

            if let Err(e) = self.check_outbound_rate_limit(provider_name) {
                tracing::warn!(provider = %provider_name, model = %model_name, "outbound rate limit hit, falling back: {e}");
                self.metrics
                    .record_attempt(provider_name, model_name, "rate_limited");
                last_err = Some(RouterError::Provider(e));
                continue;
            }

            let started_at = Instant::now();
            match provider.chat_stream(req, model_name).await {
                Ok(stream) => {
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    ewma_record(
                        &self.latency,
                        format!("{provider_name}/{model_name}"),
                        elapsed_ms,
                    );
                    self.metrics.observe_latency_seconds(
                        provider_name,
                        model_name,
                        elapsed_ms / 1000.0,
                    );
                    self.metrics
                        .record_attempt(provider_name, model_name, "success");
                    tracing::debug!(provider = %provider_name, model = %model_name, elapsed_ms, "recorded latency (time to first byte)");

                    return Ok(self.instrument_stream(
                        provider_name.clone(),
                        model_name.clone(),
                        started_at,
                        stream,
                    ));
                }
                Err(e) if e.is_retryable() => {
                    tracing::warn!(provider = %provider_name, model = %model_name, "provider failed, falling back: {e}");
                    self.metrics
                        .record_attempt(provider_name, model_name, "retryable_error");
                    last_err = Some(RouterError::Provider(e));
                }
                Err(e) => {
                    self.metrics
                        .record_attempt(provider_name, model_name, "error");
                    return Err(RouterError::Provider(e));
                }
            }
        }

        Err(last_err.unwrap_or_else(|| RouterError::InvalidModel(req.model.clone())))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures::stream;
    use rp_core::{ChatChunk, ChatMessage, ChatMessageDelta, Choice, ChunkChoice, Role};

    use super::*;

    /// Directly construct a `Router` with arbitrary private-field state,
    /// bypassing `from_config` (which only ever builds real provider
    /// adapters from env vars) so tests can inject `MockProvider`s and
    /// pre-seed pricing/zdr/rate-limit state without any network I/O.
    fn test_router(
        providers: Vec<(&str, Arc<dyn Provider>)>,
        routes: Vec<(&str, Vec<&str>)>,
        pricing: Vec<(&str, f64, f64)>,
        zdr_providers: Vec<&str>,
        provider_rpm: Vec<(&str, u32)>,
    ) -> Router {
        Router {
            providers: providers
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            routes: routes
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.into_iter().map(String::from).collect()))
                .collect(),
            pricing: Arc::new(
                pricing
                    .into_iter()
                    .map(|(k, p, c)| (k.to_string(), (p, c)))
                    .collect(),
            ),
            zdr_providers: zdr_providers.into_iter().map(String::from).collect(),
            latency: RwLock::new(HashMap::new()),
            throughput: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(HashMap::new())),
            metrics: Metrics::new(),
            provider_rpm: provider_rpm
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            outbound_limiter: RateLimiter::new(),
        }
    }

    fn chain(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries
            .iter()
            .map(|(p, m)| (p.to_string(), m.to_string()))
            .collect()
    }

    fn test_request(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage::user("hi")],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: None,
            user: None,
            tools: None,
            tool_choice: None,
            provider: None,
        }
    }

    enum MockBehavior {
        Succeed,
        FailRetryable,
        FailFatal,
    }

    /// A `Provider` with scripted, network-free behavior and a call
    /// counter, so dispatch/fallback logic can be tested in isolation from
    /// any real adapter or HTTP call.
    struct MockProvider {
        name: String,
        behavior: MockBehavior,
        calls: Arc<AtomicUsize>,
    }

    impl MockProvider {
        fn canned_error(&self) -> ProviderError {
            match self.behavior {
                MockBehavior::FailRetryable => ProviderError::Upstream {
                    status: 503,
                    message: "mock retryable failure".to_string(),
                },
                MockBehavior::FailFatal => {
                    ProviderError::InvalidRequest("mock fatal failure".to_string())
                }
                MockBehavior::Succeed => {
                    unreachable!("canned_error only called for failure behaviors")
                }
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }

        async fn chat(
            &self,
            _req: &ChatRequest,
            model: &str,
        ) -> Result<ChatResponse, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.behavior {
                MockBehavior::Succeed => Ok(ChatResponse {
                    id: "test-id".to_string(),
                    object: "chat.completion",
                    created: 0,
                    model: format!("{}/{model}", self.name),
                    choices: vec![Choice {
                        index: 0,
                        message: ChatMessage::assistant("ok"),
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Some(Usage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    }),
                    cost_usd: None,
                }),
                _ => Err(self.canned_error()),
            }
        }

        async fn chat_stream(
            &self,
            _req: &ChatRequest,
            model: &str,
        ) -> Result<ChatStream, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.behavior {
                MockBehavior::Succeed => {
                    let chunk = ChatChunk {
                        id: "test-id".to_string(),
                        object: "chat.completion.chunk",
                        created: 0,
                        model: format!("{}/{model}", self.name),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChatMessageDelta {
                                role: Some(Role::Assistant),
                                content: Some("ok".to_string()),
                                tool_calls: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                        usage: Some(Usage {
                            prompt_tokens: 1,
                            completion_tokens: 1,
                            total_tokens: 2,
                        }),
                        cost_usd: None,
                    };
                    Ok(Box::pin(stream::once(async { Ok(chunk) })))
                }
                _ => Err(self.canned_error()),
            }
        }
    }

    /// A `Provider` whose `chat_stream` replays an exact, caller-supplied
    /// sequence of chunks, so `instrument_stream`'s per-chunk usage/cost
    /// bookkeeping can be exercised with precise control over which chunks
    /// carry usage and how many completion tokens each one reports.
    struct ScriptedStreamProvider {
        name: String,
        chunks: Vec<ChatChunk>,
    }

    #[async_trait]
    impl Provider for ScriptedStreamProvider {
        fn name(&self) -> &str {
            &self.name
        }

        async fn chat(
            &self,
            _req: &ChatRequest,
            _model: &str,
        ) -> Result<ChatResponse, ProviderError> {
            unreachable!("ScriptedStreamProvider only implements chat_stream")
        }

        async fn chat_stream(
            &self,
            _req: &ChatRequest,
            _model: &str,
        ) -> Result<ChatStream, ProviderError> {
            let chunks = self.chunks.clone();
            Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
        }
    }

    fn scripted_chunk(prompt_tokens: u32, completion_tokens: u32) -> ChatChunk {
        ChatChunk {
            id: "test-id".to_string(),
            object: "chat.completion.chunk",
            created: 0,
            model: "anthropic/m1".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: Some(Role::Assistant),
                    content: Some("ok".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
            cost_usd: None,
        }
    }

    fn scripted_chunk_without_usage() -> ChatChunk {
        ChatChunk {
            usage: None,
            ..scripted_chunk(0, 0)
        }
    }

    // --- resolve_chain -----------------------------------------------------

    #[test]
    fn resolve_chain_direct_model_string() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let result = router.resolve_chain("anthropic/claude-sonnet-5").unwrap();
        assert_eq!(result, chain(&[("anthropic", "claude-sonnet-5")]));
    }

    #[test]
    fn resolve_chain_alias_returns_configured_order() {
        let router = test_router(
            vec![],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            vec![],
        );
        let result = router.resolve_chain("smart").unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
    }

    #[test]
    fn resolve_chain_rejects_model_without_a_slash() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let err = router.resolve_chain("not-a-valid-model").unwrap_err();
        assert!(matches!(err, RouterError::InvalidModel(_)));
    }

    // --- apply_preferences ---------------------------------------------------

    #[test]
    fn apply_preferences_no_prefs_is_a_no_op() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let input = chain(&[("anthropic", "m1"), ("openai", "m2")]);
        let result = router
            .apply_preferences("smart", input.clone(), None)
            .unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn apply_preferences_only_filters_chain() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            only: Some(vec!["openai".to_string()]),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2")]));
    }

    #[test]
    fn apply_preferences_ignore_filters_chain() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            ignore: Some(vec!["anthropic".to_string()]),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2")]));
    }

    #[test]
    fn apply_preferences_empty_after_filter_is_an_error() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            only: Some(vec!["gemini".to_string()]),
            ..Default::default()
        };
        let err = router
            .apply_preferences("smart", chain(&[("anthropic", "m1")]), Some(&prefs))
            .unwrap_err();
        assert!(matches!(err, RouterError::NoEligibleProvider(_)));
    }

    #[test]
    fn apply_preferences_zdr_filters_to_flagged_providers_only() {
        let router = test_router(vec![], vec![], vec![], vec!["anthropic"], vec![]);
        let prefs = ProviderPreferences {
            zdr: Some(true),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1")]));
    }

    #[test]
    fn apply_preferences_zdr_false_is_a_no_op() {
        let router = test_router(vec![], vec![], vec![], vec!["anthropic"], vec![]);
        let prefs = ProviderPreferences {
            zdr: Some(false),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(
            result, input,
            "zdr: false must not filter out non-ZDR providers"
        );
    }

    #[test]
    fn apply_preferences_zdr_unset_is_a_no_op() {
        let router = test_router(vec![], vec![], vec![], vec!["anthropic"], vec![]);
        // `zdr` left unset within an otherwise-present preferences object,
        // as opposed to `prefs` being `None` entirely.
        let prefs = ProviderPreferences {
            only: Some(vec!["anthropic".to_string(), "openai".to_string()]),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn apply_preferences_zdr_keeps_every_flagged_provider() {
        let router = test_router(vec![], vec![], vec![], vec!["anthropic", "gemini"], vec![]);
        let prefs = ProviderPreferences {
            zdr: Some(true),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("gemini", "m3")]));
    }

    #[test]
    fn apply_preferences_zdr_with_no_flagged_providers_empties_the_chain_and_errors() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            zdr: Some(true),
            ..Default::default()
        };
        let err = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                Some(&prefs),
            )
            .unwrap_err();
        assert!(matches!(err, RouterError::NoEligibleProvider(_)));
    }

    #[test]
    fn apply_preferences_zdr_combines_with_only_filter() {
        // "openai" passes `only` but isn't ZDR-flagged, so it must still be
        // dropped -- the two filters are independent, not either/or.
        let router = test_router(vec![], vec![], vec![], vec!["anthropic"], vec![]);
        let prefs = ProviderPreferences {
            only: Some(vec!["anthropic".to_string(), "openai".to_string()]),
            zdr: Some(true),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1")]));
    }

    #[test]
    fn apply_preferences_zdr_combines_with_ignore_filter() {
        // Both "anthropic" and "gemini" are ZDR-flagged, but "ignore" drops
        // "anthropic" first, leaving only "gemini".
        let router = test_router(vec![], vec![], vec![], vec!["anthropic", "gemini"], vec![]);
        let prefs = ProviderPreferences {
            ignore: Some(vec!["anthropic".to_string()]),
            zdr: Some(true),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("gemini", "m3")]));
    }

    #[test]
    fn apply_preferences_zdr_filters_before_price_sort() {
        // The cheapest candidate ("gemini") isn't ZDR-flagged and must be
        // dropped before price sorting ever sees it, not merely sorted last.
        let router = test_router(
            vec![],
            vec![],
            vec![
                ("anthropic/m1", 3.0, 15.0),
                ("openai/m2", 1.0, 5.0),
                ("gemini/m3", 0.1, 0.4),
            ],
            vec!["anthropic", "openai"],
            vec![],
        );
        let prefs = ProviderPreferences {
            zdr: Some(true),
            sort: Some("price".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2"), ("anthropic", "m1")]));
    }

    #[test]
    fn apply_preferences_sorts_ascending_by_price() {
        let router = test_router(
            vec![],
            vec![],
            vec![("anthropic/m1", 3.0, 15.0), ("openai/m2", 1.0, 5.0)],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            sort: Some("price".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2"), ("anthropic", "m1")]));
    }

    #[test]
    fn apply_preferences_price_sort_orders_three_or_more_entries_correctly() {
        let router = test_router(
            vec![],
            vec![],
            vec![
                ("anthropic/m1", 3.0, 15.0),
                ("openai/m2", 1.0, 5.0),
                ("gemini/m3", 2.0, 4.0),
            ],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            sort: Some("price".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(
            result,
            chain(&[("openai", "m2"), ("gemini", "m3"), ("anthropic", "m1")])
        );
    }

    #[test]
    fn apply_preferences_price_sort_puts_unpriced_entries_last() {
        // Only "openai/m2" has a configured price; "anthropic/m1" and
        // "gemini/m3" have none and should sort after it, in their
        // original relative order (stable sort, both tied at f64::MAX).
        let router = test_router(
            vec![],
            vec![],
            vec![("openai/m2", 1.0, 5.0)],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            sort: Some("price".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(
            result,
            chain(&[("openai", "m2"), ("anthropic", "m1"), ("gemini", "m3")])
        );
    }

    #[test]
    fn apply_preferences_price_sort_with_all_unpriced_preserves_original_order() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            sort: Some("price".to_string()),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn apply_preferences_price_sort_ranks_by_prompt_price_only_ignoring_completion_price() {
        // "anthropic/m1" has a far higher completion price but a lower
        // prompt price -- sort:"price" only ever consults prompt_per_million.
        let router = test_router(
            vec![],
            vec![],
            vec![("anthropic/m1", 1.0, 100.0), ("openai/m2", 2.0, 1.0)],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            sort: Some("price".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("openai", "m2"), ("anthropic", "m1")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
    }

    #[test]
    fn apply_preferences_price_sort_is_stable_for_equal_prices() {
        let router = test_router(
            vec![],
            vec![],
            vec![("anthropic/m1", 1.0, 1.0), ("openai/m2", 1.0, 1.0)],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            sort: Some("price".to_string()),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(result, input, "tied prices should preserve original order");
    }

    #[test]
    fn apply_preferences_price_sort_applies_after_only_filter() {
        // A denied-by-filter provider must not influence the sorted output,
        // even if it would otherwise be the cheapest.
        let router = test_router(
            vec![],
            vec![],
            vec![
                ("anthropic/m1", 3.0, 15.0),
                ("openai/m2", 1.0, 5.0),
                ("gemini/m3", 0.1, 0.4),
            ],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            only: Some(vec!["anthropic".to_string(), "openai".to_string()]),
            sort: Some("price".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2"), ("anthropic", "m1")]));
    }

    #[test]
    fn apply_preferences_sorts_ascending_by_latency() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router
            .latency
            .write()
            .unwrap()
            .insert("anthropic/m1".to_string(), 2000.0);
        router
            .latency
            .write()
            .unwrap()
            .insert("openai/m2".to_string(), 500.0);
        let prefs = ProviderPreferences {
            sort: Some("latency".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2"), ("anthropic", "m1")]));
    }

    #[test]
    fn apply_preferences_sorts_descending_by_throughput() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router
            .throughput
            .write()
            .unwrap()
            .insert("anthropic/m1".to_string(), 20.0);
        router
            .throughput
            .write()
            .unwrap()
            .insert("openai/m2".to_string(), 80.0);
        let prefs = ProviderPreferences {
            sort: Some("throughput".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2"), ("anthropic", "m1")]));
    }

    #[test]
    fn apply_preferences_unobserved_latency_sorts_last() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router
            .latency
            .write()
            .unwrap()
            .insert("anthropic/m1".to_string(), 500.0);
        // "openai/m2" has no observed latency -- despite being first in the
        // chain, it should sort after the entry with real data.
        let prefs = ProviderPreferences {
            sort: Some("latency".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("openai", "m2"), ("anthropic", "m1")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
    }

    #[test]
    fn apply_preferences_latency_sort_orders_three_or_more_entries_correctly() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut latency = router.latency.write().unwrap();
            latency.insert("anthropic/m1".to_string(), 2000.0);
            latency.insert("openai/m2".to_string(), 500.0);
            latency.insert("gemini/m3".to_string(), 1000.0);
        }
        let prefs = ProviderPreferences {
            sort: Some("latency".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(
            result,
            chain(&[("openai", "m2"), ("gemini", "m3"), ("anthropic", "m1")])
        );
    }

    #[test]
    fn apply_preferences_latency_sort_with_all_unobserved_preserves_original_order() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            sort: Some("latency".to_string()),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn apply_preferences_latency_sort_is_stable_for_equal_latencies() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut latency = router.latency.write().unwrap();
            latency.insert("anthropic/m1".to_string(), 500.0);
            latency.insert("openai/m2".to_string(), 500.0);
        }
        let prefs = ProviderPreferences {
            sort: Some("latency".to_string()),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(
            result, input,
            "tied latencies should preserve original order"
        );
    }

    #[test]
    fn apply_preferences_latency_sort_applies_after_only_filter() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut latency = router.latency.write().unwrap();
            latency.insert("anthropic/m1".to_string(), 2000.0);
            latency.insert("openai/m2".to_string(), 500.0);
            // Fastest of all three, but filtered out by `only` below.
            latency.insert("gemini/m3".to_string(), 10.0);
        }
        let prefs = ProviderPreferences {
            only: Some(vec!["anthropic".to_string(), "openai".to_string()]),
            sort: Some("latency".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2"), ("anthropic", "m1")]));
    }

    #[test]
    fn apply_preferences_throughput_sort_orders_three_or_more_entries_correctly() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut throughput = router.throughput.write().unwrap();
            throughput.insert("anthropic/m1".to_string(), 20.0);
            throughput.insert("openai/m2".to_string(), 80.0);
            throughput.insert("gemini/m3".to_string(), 50.0);
        }
        let prefs = ProviderPreferences {
            sort: Some("throughput".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(
            result,
            chain(&[("openai", "m2"), ("gemini", "m3"), ("anthropic", "m1")])
        );
    }

    #[test]
    fn apply_preferences_unobserved_throughput_sorts_last() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router
            .throughput
            .write()
            .unwrap()
            .insert("anthropic/m1".to_string(), 50.0);
        // "openai/m2" has no observed throughput -- despite being first in
        // the chain, it should sort after the entry with real data (missing
        // treated as 0.0 tokens/sec, worse than any real observation).
        let prefs = ProviderPreferences {
            sort: Some("throughput".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("openai", "m2"), ("anthropic", "m1")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
    }

    #[test]
    fn apply_preferences_throughput_sort_with_all_unobserved_preserves_original_order() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            sort: Some("throughput".to_string()),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn apply_preferences_throughput_sort_is_stable_for_equal_throughput() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut throughput = router.throughput.write().unwrap();
            throughput.insert("anthropic/m1".to_string(), 50.0);
            throughput.insert("openai/m2".to_string(), 50.0);
        }
        let prefs = ProviderPreferences {
            sort: Some("throughput".to_string()),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(
            result, input,
            "tied throughput should preserve original order"
        );
    }

    #[test]
    fn apply_preferences_throughput_sort_applies_after_only_filter() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut throughput = router.throughput.write().unwrap();
            throughput.insert("anthropic/m1".to_string(), 20.0);
            throughput.insert("openai/m2".to_string(), 80.0);
            // Fastest of all three, but filtered out by `only` below.
            throughput.insert("gemini/m3".to_string(), 500.0);
        }
        let prefs = ProviderPreferences {
            only: Some(vec!["anthropic".to_string(), "openai".to_string()]),
            sort: Some("throughput".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2"), ("anthropic", "m1")]));
    }

    // --- ewma_record / ewma_lookup --------------------------------------------

    #[test]
    fn ewma_lookup_returns_the_missing_default_for_an_unrecorded_key() {
        let map = RwLock::new(HashMap::new());
        assert_eq!(ewma_lookup(&map, "anthropic", "m1", -1.0), -1.0);
    }

    #[test]
    fn ewma_record_seeds_the_average_on_first_observation() {
        let map = RwLock::new(HashMap::new());
        ewma_record(&map, "anthropic/m1".to_string(), 1000.0);
        assert_eq!(ewma_lookup(&map, "anthropic", "m1", -1.0), 1000.0);
    }

    #[test]
    fn ewma_record_blends_subsequent_samples_by_the_configured_alpha() {
        let map = RwLock::new(HashMap::new());
        ewma_record(&map, "anthropic/m1".to_string(), 1000.0);
        ewma_record(&map, "anthropic/m1".to_string(), 0.0);
        // EWMA_ALPHA = 0.3: 0.3 * 0.0 + 0.7 * 1000.0 = 700.0.
        let observed = ewma_lookup(&map, "anthropic", "m1", -1.0);
        assert!(
            (observed - 700.0).abs() < 1e-9,
            "expected ~700.0, got {observed}"
        );
    }

    #[test]
    fn ewma_record_keys_are_independent_per_provider_model() {
        let map = RwLock::new(HashMap::new());
        ewma_record(&map, "anthropic/m1".to_string(), 100.0);
        ewma_record(&map, "openai/m2".to_string(), 900.0);
        assert_eq!(ewma_lookup(&map, "anthropic", "m1", -1.0), 100.0);
        assert_eq!(ewma_lookup(&map, "openai", "m2", -1.0), 900.0);
    }

    // --- record_usage ------------------------------------------------------------

    fn usage(prompt_tokens: u32, completion_tokens: u32) -> Usage {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    #[test]
    fn record_usage_computes_cost_from_per_million_token_pricing() {
        let usage_map = RwLock::new(HashMap::new());
        let mut pricing = HashMap::new();
        // $2/1M prompt tokens, $10/1M completion tokens.
        pricing.insert("anthropic/m1".to_string(), (2.0, 10.0));

        let cost = record_usage(
            &usage_map,
            &pricing,
            "anthropic",
            "m1",
            &usage(500_000, 100_000),
        );

        // (500_000 * 2.0 + 100_000 * 10.0) / 1_000_000 = (1_000_000 + 1_000_000) / 1_000_000.
        assert!((cost.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn record_usage_returns_none_and_leaves_cost_at_zero_when_unpriced() {
        let usage_map = RwLock::new(HashMap::new());
        let pricing = HashMap::new();

        let cost = record_usage(&usage_map, &pricing, "anthropic", "m1", &usage(100, 50));

        assert!(cost.is_none());
        let stats = usage_map.read().unwrap();
        let entry = &stats["anthropic/m1"];
        assert_eq!(entry.requests, 1);
        assert_eq!(entry.prompt_tokens, 100);
        assert_eq!(entry.completion_tokens, 50);
        assert_eq!(entry.cost_usd, 0.0, "unpriced usage is unknown, not free");
    }

    #[test]
    fn record_usage_zero_tokens_with_pricing_still_returns_some_zero_cost() {
        let usage_map = RwLock::new(HashMap::new());
        let mut pricing = HashMap::new();
        pricing.insert("anthropic/m1".to_string(), (2.0, 10.0));

        let cost = record_usage(&usage_map, &pricing, "anthropic", "m1", &usage(0, 0));

        assert_eq!(
            cost,
            Some(0.0),
            "a priced model with 0 tokens costs $0, not unknown"
        );
    }

    #[test]
    fn record_usage_accumulates_across_multiple_calls_for_the_same_key() {
        let usage_map = RwLock::new(HashMap::new());
        let mut pricing = HashMap::new();
        pricing.insert("anthropic/m1".to_string(), (1.0, 1.0));

        record_usage(&usage_map, &pricing, "anthropic", "m1", &usage(100, 50));
        record_usage(&usage_map, &pricing, "anthropic", "m1", &usage(200, 100));

        let stats = usage_map.read().unwrap();
        let entry = &stats["anthropic/m1"];
        assert_eq!(entry.requests, 2);
        assert_eq!(entry.prompt_tokens, 300);
        assert_eq!(entry.completion_tokens, 150);
        // (100+50)/1e6 + (200+100)/1e6 = 150/1e6 + 300/1e6.
        assert!((entry.cost_usd - (150.0 + 300.0) / 1_000_000.0).abs() < 1e-12);
    }

    #[test]
    fn record_usage_keys_are_independent_per_provider_model() {
        let usage_map = RwLock::new(HashMap::new());
        let mut pricing = HashMap::new();
        pricing.insert("anthropic/m1".to_string(), (1.0, 1.0));
        pricing.insert("openai/m2".to_string(), (5.0, 5.0));

        record_usage(&usage_map, &pricing, "anthropic", "m1", &usage(100, 0));
        record_usage(&usage_map, &pricing, "openai", "m2", &usage(200, 0));

        let stats = usage_map.read().unwrap();
        assert_eq!(stats["anthropic/m1"].requests, 1);
        assert_eq!(stats["anthropic/m1"].prompt_tokens, 100);
        assert_eq!(stats["openai/m2"].requests, 1);
        assert_eq!(stats["openai/m2"].prompt_tokens, 200);
    }

    // --- check_outbound_rate_limit ----------------------------------------------

    #[test]
    fn check_outbound_rate_limit_is_a_noop_for_a_provider_with_no_configured_limit() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..10 {
            assert!(router.check_outbound_rate_limit("anthropic").is_ok());
        }
    }

    #[test]
    fn check_outbound_rate_limit_allows_up_to_capacity_then_rejects() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![("anthropic", 2)]);
        assert!(router.check_outbound_rate_limit("anthropic").is_ok());
        assert!(router.check_outbound_rate_limit("anthropic").is_ok());

        let err = router.check_outbound_rate_limit("anthropic").unwrap_err();
        assert!(matches!(err, ProviderError::RateLimited { .. }));
        assert!(err.is_retryable());
    }

    #[test]
    fn check_outbound_rate_limit_rejection_reports_a_positive_retry_after() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![("anthropic", 1)]);
        router.check_outbound_rate_limit("anthropic").unwrap();
        let err = router.check_outbound_rate_limit("anthropic").unwrap_err();
        match err {
            ProviderError::RateLimited { retry_after_secs } => {
                assert!(retry_after_secs.unwrap_or(0) > 0);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn check_outbound_rate_limit_zero_rpm_always_rejects() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![("anthropic", 0)]);
        let err = router.check_outbound_rate_limit("anthropic").unwrap_err();
        assert!(matches!(
            err,
            ProviderError::RateLimited {
                retry_after_secs: Some(_)
            }
        ));
    }

    #[test]
    fn check_outbound_rate_limit_buckets_are_independent_per_provider() {
        let router = test_router(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![("anthropic", 1), ("openai", 1)],
        );
        router.check_outbound_rate_limit("anthropic").unwrap();
        assert!(router.check_outbound_rate_limit("anthropic").is_err());
        // "openai" has its own bucket and is untouched by "anthropic"'s.
        assert!(router.check_outbound_rate_limit("openai").is_ok());
    }

    // --- dispatch ------------------------------------------------------------

    #[tokio::test]
    async fn dispatch_returns_success_from_first_provider() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = test_router(vec![("anthropic", mock)], vec![], vec![], vec![], vec![]);

        let resp = router
            .dispatch(&test_request("anthropic/claude-sonnet-5"))
            .await
            .expect("dispatch should succeed");

        assert_eq!(resp.model, "anthropic/claude-sonnet-5");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_stamps_cost_and_records_usage_when_the_model_is_priced() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls,
        });
        // MockProvider's Succeed response carries Usage { prompt_tokens: 1,
        // completion_tokens: 1 }.
        let router = test_router(
            vec![("anthropic", mock)],
            vec![],
            vec![("anthropic/m1", 1.0, 1.0)],
            vec![],
            vec![],
        );

        let resp = router
            .dispatch(&test_request("anthropic/m1"))
            .await
            .expect("dispatch should succeed");

        let expected_cost = 2.0 / 1_000_000.0;
        assert!((resp.cost_usd.unwrap() - expected_cost).abs() < 1e-12);

        let snapshot = router.usage_snapshot();
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(stats.requests, 1);
        assert_eq!(stats.prompt_tokens, 1);
        assert_eq!(stats.completion_tokens, 1);
        assert!((stats.cost_usd - expected_cost).abs() < 1e-12);
    }

    #[tokio::test]
    async fn dispatch_leaves_cost_usd_none_but_still_records_usage_when_unpriced() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls,
        });
        let router = test_router(vec![("anthropic", mock)], vec![], vec![], vec![], vec![]);

        let resp = router
            .dispatch(&test_request("anthropic/m1"))
            .await
            .expect("dispatch should succeed");

        assert!(resp.cost_usd.is_none());
        let snapshot = router.usage_snapshot();
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(stats.requests, 1);
        assert_eq!(stats.cost_usd, 0.0);
    }

    #[tokio::test]
    async fn usage_snapshot_accumulates_across_multiple_dispatches() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls,
        });
        let router = test_router(vec![("anthropic", mock)], vec![], vec![], vec![], vec![]);
        let req = test_request("anthropic/m1");

        router.dispatch(&req).await.unwrap();
        router.dispatch(&req).await.unwrap();
        router.dispatch(&req).await.unwrap();

        let snapshot = router.usage_snapshot();
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(stats.requests, 3);
        assert_eq!(stats.prompt_tokens, 3);
        assert_eq!(stats.completion_tokens, 3);
    }

    #[tokio::test]
    async fn dispatch_stream_stamps_cost_and_records_usage_on_the_final_chunk() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls,
        });
        // MockProvider's chat_stream yields a single chunk carrying the same
        // Usage { prompt_tokens: 1, completion_tokens: 1 } as chat().
        let router = test_router(
            vec![("anthropic", mock)],
            vec![],
            vec![("anthropic/m1", 3.0, 9.0)],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);

        let mut stream = router
            .dispatch_stream(&req)
            .await
            .expect("dispatch_stream should succeed");
        let chunk = stream
            .next()
            .await
            .expect("stream should yield one chunk")
            .expect("chunk should be Ok");

        let expected_cost = (3.0 + 9.0) / 1_000_000.0;
        assert!((chunk.cost_usd.unwrap() - expected_cost).abs() < 1e-12);

        let snapshot = router.usage_snapshot();
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(stats.requests, 1);
        assert!((stats.cost_usd - expected_cost).abs() < 1e-12);
    }

    // --- dispatch_stream usage instrumentation ------------------------------

    #[tokio::test]
    async fn dispatch_stream_skips_usage_and_cost_for_a_chunk_with_zero_completion_tokens() {
        let provider = Arc::new(ScriptedStreamProvider {
            name: "anthropic".to_string(),
            // prompt_tokens carried, but completion_tokens is 0 -- the
            // instrumentation gate is on completion_tokens, not on usage
            // simply being present.
            chunks: vec![scripted_chunk(10, 0)],
        });
        let router = test_router(
            vec![("anthropic", provider)],
            vec![],
            vec![("anthropic/m1", 5.0, 5.0)],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);

        let mut stream = router
            .dispatch_stream(&req)
            .await
            .expect("dispatch_stream should succeed");
        let chunk = stream.next().await.unwrap().unwrap();

        assert!(
            chunk.cost_usd.is_none(),
            "a chunk with 0 completion tokens must not be cost-stamped"
        );
        assert!(
            !router.usage_snapshot().contains_key("anthropic/m1"),
            "0-completion-token chunks must not create a usage_snapshot entry"
        );
    }

    #[tokio::test]
    async fn dispatch_stream_accumulates_usage_across_multiple_usage_bearing_chunks() {
        let provider = Arc::new(ScriptedStreamProvider {
            name: "anthropic".to_string(),
            chunks: vec![scripted_chunk(10, 5), scripted_chunk(20, 3)],
        });
        let router = test_router(
            vec![("anthropic", provider)],
            vec![],
            vec![("anthropic/m1", 1.0, 1.0)],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);

        let stream = router
            .dispatch_stream(&req)
            .await
            .expect("dispatch_stream should succeed");
        let chunks: Vec<_> = stream.map(|c| c.unwrap()).collect().await;

        assert_eq!(chunks.len(), 2);
        // Each chunk is cost-stamped from its own usage, not a running total.
        assert!((chunks[0].cost_usd.unwrap() - 15.0 / 1_000_000.0).abs() < 1e-12);
        assert!((chunks[1].cost_usd.unwrap() - 23.0 / 1_000_000.0).abs() < 1e-12);

        let snapshot = router.usage_snapshot();
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(
            stats.requests, 2,
            "each usage-bearing chunk is one record_usage call"
        );
        assert_eq!(stats.prompt_tokens, 30);
        assert_eq!(stats.completion_tokens, 8);
        assert!((stats.cost_usd - 38.0 / 1_000_000.0).abs() < 1e-12);
    }

    #[tokio::test]
    async fn dispatch_stream_leaves_a_chunk_without_usage_untouched() {
        let provider = Arc::new(ScriptedStreamProvider {
            name: "anthropic".to_string(),
            chunks: vec![scripted_chunk_without_usage()],
        });
        let router = test_router(
            vec![("anthropic", provider)],
            vec![],
            vec![("anthropic/m1", 1.0, 1.0)],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);

        let mut stream = router
            .dispatch_stream(&req)
            .await
            .expect("dispatch_stream should succeed");
        let chunk = stream.next().await.unwrap().unwrap();

        assert!(chunk.cost_usd.is_none());
        assert!(chunk.usage.is_none());
        assert!(!router.usage_snapshot().contains_key("anthropic/m1"));
    }

    #[tokio::test]
    async fn dispatch_stream_records_throughput_ewma_when_completion_tokens_positive() {
        let provider = Arc::new(ScriptedStreamProvider {
            name: "anthropic".to_string(),
            chunks: vec![scripted_chunk(10, 5)],
        });
        let router = test_router(
            vec![("anthropic", provider)],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);

        let mut stream = router
            .dispatch_stream(&req)
            .await
            .expect("dispatch_stream should succeed");
        stream.next().await.unwrap().unwrap();

        let throughput = router.throughput.read().unwrap();
        assert!(
            throughput.contains_key("anthropic/m1"),
            "a completion-bearing chunk should record a throughput sample"
        );
        assert!(throughput["anthropic/m1"] > 0.0);
    }

    #[tokio::test]
    async fn dispatch_stream_updates_prometheus_metrics_from_streamed_usage() {
        let provider = Arc::new(ScriptedStreamProvider {
            name: "anthropic".to_string(),
            chunks: vec![scripted_chunk(10, 5)],
        });
        let router = test_router(
            vec![("anthropic", provider)],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);

        let mut stream = router
            .dispatch_stream(&req)
            .await
            .expect("dispatch_stream should succeed");
        stream.next().await.unwrap().unwrap();

        let metrics = router.render_prometheus_metrics();
        assert!(metrics.contains("rusty_provider_completion_tokens_total"));
        assert!(metrics.contains(r#"provider="anthropic""#));
        assert!(metrics.contains(r#"model="m1""#));
    }

    #[tokio::test]
    async fn dispatch_falls_back_to_next_candidate_on_retryable_error() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let failing = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::FailRetryable,
            calls: calls_a.clone(),
        });
        let succeeding = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls_b.clone(),
        });
        let router = test_router(
            vec![("anthropic", failing), ("openai", succeeding)],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            vec![],
        );

        let resp = router
            .dispatch(&test_request("smart"))
            .await
            .expect("should fall through to openai");

        assert_eq!(resp.model, "openai/m2");
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_aborts_immediately_on_fatal_error() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let failing = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::FailFatal,
            calls: calls_a.clone(),
        });
        let never_called = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls_b.clone(),
        });
        let router = test_router(
            vec![("anthropic", failing), ("openai", never_called)],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            vec![],
        );

        let err = router.dispatch(&test_request("smart")).await.unwrap_err();

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::InvalidRequest(_))
        ));
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(
            calls_b.load(Ordering::SeqCst),
            0,
            "a fatal error must not fall through to the next candidate"
        );
    }

    #[tokio::test]
    async fn dispatch_returns_last_error_when_every_candidate_fails() {
        let a = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::FailRetryable,
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let b = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::FailRetryable,
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let router = test_router(
            vec![("anthropic", a), ("openai", b)],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            vec![],
        );

        let err = router.dispatch(&test_request("smart")).await.unwrap_err();

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::Upstream { .. })
        ));
    }

    #[tokio::test]
    async fn dispatch_skips_a_chain_entry_with_no_registered_provider() {
        let calls = Arc::new(AtomicUsize::new(0));
        let configured = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        // "anthropic" is referenced by the alias but never registered.
        let router = test_router(
            vec![("openai", configured)],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            vec![],
        );

        let resp = router
            .dispatch(&test_request("smart"))
            .await
            .expect("should fall through to the configured provider");

        assert_eq!(resp.model, "openai/m2");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_respects_outbound_rate_limit_and_reports_it_as_retryable() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = test_router(
            vec![("anthropic", mock)],
            vec![],
            vec![],
            vec![],
            vec![("anthropic", 1)],
        );
        let req = test_request("anthropic/m1");

        router
            .dispatch(&req)
            .await
            .expect("first request is within the 1/min budget");
        let err = router.dispatch(&req).await.unwrap_err();

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::RateLimited { .. })
        ));
        assert!(err.retry_after_secs().is_some());
        // The mock was only actually invoked once -- the second dispatch
        // was stopped by the outbound limiter before ever calling it.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_falls_back_to_next_provider_when_outbound_limit_is_exhausted() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let limited = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls_a.clone(),
        });
        let unlimited = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls_b.clone(),
        });
        let router = test_router(
            vec![("anthropic", limited), ("openai", unlimited)],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            // A 0/min budget for "anthropic" means it is rejected on every
            // attempt, forcing every dispatch to fall through to "openai".
            vec![("anthropic", 0)],
        );

        let resp = router
            .dispatch(&test_request("smart"))
            .await
            .expect("should fall through to the unlimited provider");

        assert_eq!(resp.model, "openai/m2");
        assert_eq!(
            calls_a.load(Ordering::SeqCst),
            0,
            "the outbound-limited provider must never actually be called"
        );
        assert_eq!(calls_b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_records_the_rate_limited_outcome_in_metrics() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = test_router(
            vec![("anthropic", mock)],
            vec![],
            vec![],
            vec![],
            vec![("anthropic", 0)],
        );

        let _ = router.dispatch(&test_request("anthropic/m1")).await;

        let metrics = router.render_prometheus_metrics();
        assert!(metrics.contains("rusty_provider_dispatch_attempts_total"));
        assert!(metrics.contains(r#"outcome="rate_limited""#));
        assert!(metrics.contains(r#"provider="anthropic""#));
    }

    #[tokio::test]
    async fn dispatch_stream_respects_outbound_rate_limit_and_reports_it_as_retryable() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = test_router(
            vec![("anthropic", mock)],
            vec![],
            vec![],
            vec![],
            vec![("anthropic", 1)],
        );
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);

        let _stream = router
            .dispatch_stream(&req)
            .await
            .expect("first request is within the 1/min budget");
        // ChatStream (the Ok side) isn't Debug, so unwrap_err() doesn't
        // typecheck here -- match instead.
        let err = match router.dispatch_stream(&req).await {
            Ok(_) => panic!("expected the second call to be outbound rate limited"),
            Err(e) => e,
        };

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::RateLimited { .. })
        ));
        assert!(err.retry_after_secs().is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    // --- RouterError -----------------------------------------------------------

    #[test]
    fn retry_after_secs_extracts_from_rate_limited_provider_error() {
        let err = RouterError::Provider(ProviderError::RateLimited {
            retry_after_secs: Some(42),
        });
        assert_eq!(err.retry_after_secs(), Some(42));
    }

    #[test]
    fn retry_after_secs_is_none_for_other_errors() {
        assert_eq!(
            RouterError::InvalidModel("x".to_string()).retry_after_secs(),
            None
        );
    }
}
