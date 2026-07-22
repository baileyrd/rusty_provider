mod auto_routing;
mod cache;
mod client_budget;
mod config;
mod error;
mod guardrails;
mod metrics;
mod moderation;
mod persistence;
mod presets;
mod web_search;
mod webhook;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use cache::ResponseCache;
use client_budget::{ClientBudgetSetting, SpendState};
pub use config::{
    AutoRoutingConfig, BudgetPeriod, CacheConfig, ClientConfig, ClientRole, Config,
    GuardrailAction, GuardrailConfig, ModerationConfig, PersistenceBackend, PersistenceConfig,
    PostgresTlsMode, PresetConfig, PricingEntry, ProviderConfig, ProviderKind, RouteAlias,
    ServerConfig, WebSearchConfig, WebhookConfig,
};
pub use error::RouterError;
use guardrails::Guardrail;
pub use metrics::Metrics;
use moderation::{ModerationClient, ModerationError};
use persistence::{Persistence, PersistenceTarget};
use web_search::WebSearchClient;
use webhook::WebhookNotifier;

use futures::stream::StreamExt;
use rp_core::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, EmbeddingsRequest, EmbeddingsResponse,
    ModelInfo, ModelPricing, Provider, ProviderError, ProviderPreferences, RateLimiter, Usage,
};
use rp_providers::{AnthropicProvider, GeminiProvider, OpenAiCompatibleProvider};

/// Weight given to each new latency/throughput sample in its running
/// average — higher reacts faster to recent conditions, lower smooths out
/// noise.
const EWMA_ALPHA: f64 = 0.3;

/// This router's own observed performance for one "provider/model", as
/// returned by `GET /v1/providers/stats` -- the same EWMA figures
/// `sort: "latency"`/`"throughput"`/`"uptime"` consult internally, finally
/// surfaced to clients instead of staying purely an internal ranking
/// signal. `None` for any figure this process hasn't observed yet, same
/// "unobserved, not zero" convention the sorts themselves use.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ProviderStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub throughput_tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uptime: Option<f64>,
}

/// Max number of individual request records `GenerationCache` retains for
/// `GET /v1/generation?id=` lookups before evicting the oldest -- a
/// recent-history cache, not a durable audit log, so an unbounded size
/// isn't the right tradeoff here.
const GENERATION_CACHE_CAPACITY: usize = 1000;

/// One completed request's token/cost breakdown, as returned by
/// `GET /v1/generation?id=` -- the OpenAI-shaped `id` from that request's
/// `ChatResponse`/final `ChatChunk` is the lookup key.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerationRecord {
    pub id: String,
    /// The fully-qualified "provider/model" that served this request.
    pub model: String,
    pub created: i64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// Same unpriced-means-absent convention as `ChatResponse::cost_usd`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

/// Fixed-capacity, insertion-order-evicting cache of recent
/// `GenerationRecord`s, keyed by request id. Not a general-purpose LRU --
/// only insertion order is tracked, so a lookup doesn't refresh an
/// entry's position; that's the right tradeoff for "how long ago was this
/// requested," which is monotonic with insertion order anyway.
#[derive(Debug, Default)]
struct GenerationCache {
    order: std::collections::VecDeque<String>,
    by_id: HashMap<String, GenerationRecord>,
}

impl GenerationCache {
    fn insert(&mut self, record: GenerationRecord) {
        if !self.by_id.contains_key(&record.id) {
            self.order.push_back(record.id.clone());
            if self.order.len() > GENERATION_CACHE_CAPACITY {
                if let Some(oldest) = self.order.pop_front() {
                    self.by_id.remove(&oldest);
                }
            }
        }
        self.by_id.insert(record.id.clone(), record);
    }

    fn get(&self, id: &str) -> Option<GenerationRecord> {
        self.by_id.get(id).cloned()
    }
}

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

/// A client's current-period spend and configured cap, returned by
/// [`Router::check_client_budget`] when it's been exceeded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClientBudgetExceeded {
    pub spent_usd: f64,
    pub budget_usd: f64,
}

/// A client's live spend against its configured budget, returned by
/// [`Router::client_spend_status`] for the admin API.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ClientSpendStatus {
    pub spent_usd: f64,
    pub budget_usd: f64,
    pub period: BudgetPeriod,
}

/// Per-million-token USD rates for one "provider/model", derived from its
/// `[[pricing]]` entry with `cache_read_per_million`/`cache_write_per_million`
/// defaulted to `prompt_per_million` when the operator left them unset.
#[derive(Debug, Clone, Copy)]
struct PriceRates {
    prompt_ppm: f64,
    completion_ppm: f64,
    cache_read_ppm: f64,
    cache_write_ppm: f64,
    context_length: Option<u32>,
}

impl From<&PricingEntry> for PriceRates {
    fn from(p: &PricingEntry) -> Self {
        Self {
            prompt_ppm: p.prompt_per_million,
            completion_ppm: p.completion_per_million,
            cache_read_ppm: p.cache_read_per_million.unwrap_or(p.prompt_per_million),
            cache_write_ppm: p.cache_write_per_million.unwrap_or(p.prompt_per_million),
            context_length: p.context_length,
        }
    }
}

/// Which `ChatRequest` fields this provider kind's adapter actually gives
/// an effect to -- natively, or (OpenAI-compatible only) via unconditional
/// passthrough to whatever inference server is behind it. Mirrors the
/// per-provider support tables in the README; a field left off a kind's
/// list is either rejected or silently a no-op there, not actually
/// influencing the response.
fn supported_params(kind: ProviderKind) -> Vec<&'static str> {
    let mut params = vec![
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
        "reasoning",
        "top_k",
    ];
    match kind {
        ProviderKind::Openai => params.extend([
            "min_p",
            "top_a",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "logit_bias",
            "seed",
            "logprobs",
        ]),
        ProviderKind::Anthropic => params.push("cache_control"),
        ProviderKind::Gemini => params.extend(["frequency_penalty", "presence_penalty", "seed"]),
    }
    params
}

/// Which of `supported_params`' names this specific request actually sets
/// -- i.e. would be silently dropped, or (for a structural field like
/// `response_format`) rejected, on a provider that doesn't support it.
/// `temperature`/`top_p`/`max_tokens`/`stop` are never included: every
/// provider kind supports all four, so they can never disqualify a
/// candidate.
fn active_params(req: &ChatRequest) -> Vec<&'static str> {
    let mut params = Vec::new();
    if req.tools.is_some() {
        params.push("tools");
    }
    if req.tool_choice.is_some() {
        params.push("tool_choice");
    }
    if req.response_format.is_some() {
        params.push("response_format");
    }
    if req.reasoning.is_some() {
        params.push("reasoning");
    }
    if req.top_k.is_some() {
        params.push("top_k");
    }
    if req.min_p.is_some() {
        params.push("min_p");
    }
    if req.top_a.is_some() {
        params.push("top_a");
    }
    if req.frequency_penalty.is_some() {
        params.push("frequency_penalty");
    }
    if req.presence_penalty.is_some() {
        params.push("presence_penalty");
    }
    if req.repetition_penalty.is_some() {
        params.push("repetition_penalty");
    }
    if req.logit_bias.is_some() {
        params.push("logit_bias");
    }
    if req.seed.is_some() {
        params.push("seed");
    }
    if req.logprobs.is_some() {
        params.push("logprobs");
    }
    if req.messages.iter().any(|m| m.cache_control.is_some()) {
        params.push("cache_control");
    }
    params
}

/// Characters per estimated token -- a crude, tokenizer-free heuristic
/// (this router has no real tokenizer for any of the three providers'
/// models), close enough for "will this roughly fit" decisions but not a
/// substitute for the provider's own accounting.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

pub(crate) fn estimate_tokens(message: &ChatMessage) -> usize {
    message
        .content
        .as_ref()
        .map(|c| c.as_plain_text().len() / CHARS_PER_TOKEN_ESTIMATE)
        .unwrap_or(0)
}

/// Applies `transforms: ["middle-out"]`: drops messages from the middle of
/// the conversation (oldest-first among the middle) until the estimated
/// total fits `budget_tokens`, or nothing more can be dropped. Always
/// keeps the first message (typically `system`) and the last one (the
/// current turn) intact -- both ends carry the most load-bearing context,
/// which is the whole point of "middle-out" over just truncating from one
/// end. A no-op when already within budget or when there are 2 or fewer
/// messages (nothing "middle" to drop).
fn apply_middle_out(messages: &[ChatMessage], budget_tokens: usize) -> Vec<ChatMessage> {
    let mut kept = messages.to_vec();
    let mut total: usize = kept.iter().map(estimate_tokens).sum();
    while total > budget_tokens && kept.len() > 2 {
        total -= estimate_tokens(&kept[1]);
        kept.remove(1);
    }
    kept
}

/// If `req` opts into `"middle-out"` and the resolved candidate has a
/// `context_length` on record (from `[[pricing]]`), estimates whether
/// `req.messages` fits and, if not, returns a truncated copy to send
/// instead. Returns `None` when no truncation is needed or possible
/// (transform not requested, or no known `context_length` for this
/// candidate to fit against), in which case the caller sends `req`
/// unmodified, same as today.
fn maybe_apply_middle_out(
    req: &ChatRequest,
    provider_name: &str,
    model_name: &str,
    pricing: &HashMap<String, PriceRates>,
) -> Option<ChatRequest> {
    let wants_middle_out = req
        .transforms
        .as_ref()
        .is_some_and(|t| t.iter().any(|s| s == "middle-out"));
    if !wants_middle_out {
        return None;
    }
    let context_length = pricing
        .get(&format!("{provider_name}/{model_name}"))
        .and_then(|rates| rates.context_length)? as usize;
    // Leave room for the response -- default reservation mirrors this
    // router's own default max_tokens fallback used elsewhere (Anthropic's
    // required-field default), a reasonable stand-in when the client
    // didn't ask for a specific completion length either.
    let reserved_for_completion = req.max_tokens.unwrap_or(4096) as usize;
    let budget_tokens = context_length.saturating_sub(reserved_for_completion);

    let total: usize = req.messages.iter().map(estimate_tokens).sum();
    if total <= budget_tokens {
        return None;
    }

    let mut truncated = req.clone();
    truncated.messages = apply_middle_out(&req.messages, budget_tokens);
    Some(truncated)
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
    /// Provider name -> its configured `kind`, for every provider in
    /// `providers` above (i.e. only ones that actually got built). Used to
    /// report `GET /v1/models`' `supported_params`, which depends only on
    /// which wire format a provider speaks, not on its specific model.
    provider_kinds: HashMap<String, ProviderKind>,
    routes: HashMap<String, Vec<String>>,
    /// "provider/model" -> per-million-token USD rates.
    pricing: Arc<HashMap<String, PriceRates>>,
    /// Provider names with `zdr = true` in config.
    zdr_providers: HashSet<String>,
    /// Provider names with `no_training = true` in config.
    no_training_providers: HashSet<String>,
    /// "provider/model" -> EWMA response latency in milliseconds, measured
    /// from this process's own successful dispatches. In-memory only —
    /// resets on restart and isn't a live external feed.
    latency: RwLock<HashMap<String, f64>>,
    /// "provider/model" -> EWMA completion tokens/sec, measured the same
    /// way.
    throughput: Arc<RwLock<HashMap<String, f64>>>,
    /// "provider/model" -> EWMA success rate (1.0 per successful attempt,
    /// 0.0 per failed one -- retryable or fatal), sampled only on an
    /// actual dispatch attempt against that provider, not on a candidate
    /// skipped locally (unconfigured, or this process's own outbound rate
    /// limit). Same in-memory-only, per-process caveats as
    /// `latency`/`throughput`.
    uptime: RwLock<HashMap<String, f64>>,
    /// "provider/model" -> cumulative usage/cost. `Arc`-wrapped (like
    /// `throughput`, unlike `latency`) so it can be shared into a
    /// streaming response's instrumentation, which outlives
    /// `dispatch_stream` itself — the router hands the stream off to the
    /// HTTP layer rather than consuming it. Always kept up to date even
    /// when `persistence` is configured (it's the fallback `usage_snapshot`
    /// reads from if a DB read fails), but `persistence`, not this map, is
    /// the source of truth for `GET /v1/usage` once it's set up.
    usage: Arc<RwLock<HashMap<String, UsageStats>>>,
    /// Durable, cross-process backing store for `usage`, if `[persistence]`
    /// is configured. `None` means the original in-memory-only behavior:
    /// `usage` above resets on restart and is never shared across
    /// processes.
    persistence: Option<Arc<Persistence>>,
    /// Most recent individual request records, for `GET /v1/generation?id=`
    /// lookups -- bounded to `GENERATION_CACHE_CAPACITY`, oldest evicted
    /// first once full. `Arc`-wrapped for the same reason as `usage`: a
    /// streaming response's instrumentation outlives `dispatch_stream`
    /// itself. Always in-memory only, with no `persistence` backing (unlike
    /// `usage`) -- this is a short-lived "what did my last request cost"
    /// lookup, not a durable audit log.
    generations: Arc<RwLock<GenerationCache>>,
    /// Prometheus counters/histograms/gauges for `GET /metrics`, updated at
    /// the same points as `latency`/`throughput`/`usage` above. Always
    /// per-process, even when `persistence` is configured — Prometheus
    /// aggregates across processes at scrape time, not here.
    metrics: Metrics,
    /// Provider names with a self-imposed `requests_per_minute` in config.
    provider_rpm: HashMap<String, u32>,
    /// Backs `provider_rpm`'s outbound self-throttling — one bucket per
    /// provider name, checked before every dispatch attempt.
    outbound_limiter: RateLimiter,
    /// `[[clients]]` entries with a configured `budget_usd`, plus any
    /// added/changed at runtime via the admin API
    /// (`set_client_budget`/`remove_client`). Absent here means
    /// unrestricted. Lock-protected since the admin API can mutate it
    /// after startup, unlike the rest of this struct's config-derived
    /// fields.
    client_budgets: RwLock<HashMap<String, ClientBudgetSetting>>,
    /// In-memory spend per budgeted client, used when `persistence` is
    /// `None`. When persistence is configured, `persistence`'s
    /// `client_spend` table is the source of truth instead and this map
    /// goes unused, mirroring how `usage`/`persistence` split their roles.
    client_spend: Mutex<HashMap<String, SpendState>>,
    /// `[webhook]` delivery, if configured. `None` means budget events are
    /// never pushed anywhere -- same as today, a `402` and a Prometheus
    /// counter are all a client/operator sees.
    webhook: Option<Arc<WebhookNotifier>>,
    /// Compiled `[[guardrails]]` entries, in config order. Empty means no
    /// guardrails configured -- every request passes through unchanged.
    guardrails: Vec<Guardrail>,
    /// `[[presets]]` entries by `name`. Duplicate names keep only the
    /// last entry, same "last one wins" convention as `[[routes]]`
    /// aliases.
    presets: HashMap<String, PresetConfig>,
    /// `[auto_routing]`, if configured -- backs `model: "auto"`. `None`
    /// means `"auto"` isn't special-cased at all.
    auto_routing: Option<AutoRoutingConfig>,
    /// `[moderation]` client, if configured and its `api_key_env`
    /// resolved. `None` means every request skips the moderation check
    /// entirely, same as before this field existed.
    moderation: Option<Arc<ModerationClient>>,
    /// `[web_search]` client, if configured and its `api_key_env`
    /// resolved. `None` means `"web_search": true` on a request is a
    /// no-op -- the same as before this field existed.
    web_search: Option<Arc<WebSearchClient>>,
    /// `[cache]`, if configured -- an exact-match cache of non-streaming
    /// `dispatch` responses (see `cache::ResponseCache`). `None` means
    /// every request always reaches a provider, same as before this
    /// field existed.
    cache: Option<Arc<RwLock<ResponseCache>>>,
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
/// that entry's cumulative `UsageStats` -- both the in-memory map and, if
/// configured, the durable `persistence` store. Returns the computed cost
/// so the caller can attach it to the response/chunk sent back to the
/// client.
fn record_usage(
    usage_map: &RwLock<HashMap<String, UsageStats>>,
    persistence: Option<&Persistence>,
    pricing: &HashMap<String, PriceRates>,
    provider: &str,
    model: &str,
    usage: &Usage,
) -> Option<f64> {
    let key = format!("{provider}/{model}");
    let cost = pricing.get(&key).map(|rates| {
        let cached = usage.cached_tokens.unwrap_or(0) as f64;
        let cache_write = usage.cache_creation_tokens.unwrap_or(0) as f64;
        let fresh_prompt = usage.prompt_tokens as f64 - cached - cache_write;
        (fresh_prompt * rates.prompt_ppm
            + cached * rates.cache_read_ppm
            + cache_write * rates.cache_write_ppm
            + usage.completion_tokens as f64 * rates.completion_ppm)
            / 1_000_000.0
    });

    {
        let mut map = usage_map.write().unwrap();
        let stats = map.entry(key.clone()).or_default();
        stats.requests += 1;
        stats.prompt_tokens += usage.prompt_tokens as u64;
        stats.completion_tokens += usage.completion_tokens as u64;
        if let Some(cost) = cost {
            stats.cost_usd += cost;
        }
    }

    if let Some(persistence) = persistence {
        persistence.record(&key, usage, cost);
    }

    cost
}

/// Resolve `config.persistence` into a connectable `PersistenceTarget`,
/// or `None` if the section is absent or missing what its backend needs
/// (an unset `sqlite_path`/`postgres_url_env`, or a `postgres_url_env`
/// that names an env var that isn't actually set) -- every such case is a
/// soft, warned-about failure, not a hard error, matching how a
/// misconfigured provider or client is skipped rather than refused at
/// startup.
fn resolve_persistence_target(config: &PersistenceConfig) -> Option<PersistenceTarget> {
    match config.backend {
        PersistenceBackend::Sqlite => match &config.sqlite_path {
            Some(path) => Some(PersistenceTarget::Sqlite(path.into())),
            None => {
                tracing::warn!(
                    "persistence backend is \"sqlite\" but sqlite_path is not set; falling back to in-memory usage tracking"
                );
                None
            }
        },
        PersistenceBackend::Postgres => match &config.postgres_url_env {
            Some(var) => match std::env::var(var) {
                Ok(url) if !url.is_empty() => Some(PersistenceTarget::Postgres {
                    url,
                    tls: config.postgres_tls,
                }),
                _ => {
                    tracing::warn!(env_var = %var, "persistence backend is \"postgres\" but the connection string env var isn't set; falling back to in-memory usage tracking");
                    None
                }
            },
            None => {
                tracing::warn!(
                    "persistence backend is \"postgres\" but postgres_url_env is not set; falling back to in-memory usage tracking"
                );
                None
            }
        },
    }
}

impl Router {
    pub async fn from_config(config: &Config) -> Self {
        let metrics = Metrics::new();
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        let mut provider_kinds: HashMap<String, ProviderKind> = HashMap::new();

        for (name, cfg) in &config.providers {
            let key = match std::env::var(&cfg.api_key_env) {
                Ok(k) if !k.is_empty() => k,
                _ => {
                    tracing::warn!(provider = %name, env_var = %cfg.api_key_env, "skipping provider: API key env var not set");
                    metrics.set_provider_configured(name, false);
                    continue;
                }
            };

            let timeout = std::time::Duration::from_secs(cfg.timeout_secs);
            let provider: Arc<dyn Provider> = match cfg.kind {
                ProviderKind::Openai => Arc::new(
                    OpenAiCompatibleProvider::new(name.clone(), cfg.base_url.clone(), key)
                        .with_timeout(timeout),
                ),
                ProviderKind::Anthropic => Arc::new(
                    AnthropicProvider::new(cfg.base_url.clone(), key).with_timeout(timeout),
                ),
                ProviderKind::Gemini => {
                    Arc::new(GeminiProvider::new(cfg.base_url.clone(), key).with_timeout(timeout))
                }
            };
            metrics.set_provider_configured(name, true);
            providers.insert(name.clone(), provider);
            provider_kinds.insert(name.clone(), cfg.kind);
        }

        let routes = config
            .routes
            .iter()
            .map(|r| (r.alias.clone(), r.chain.clone()))
            .collect();

        let pricing = config
            .pricing
            .iter()
            .map(|p| (p.model.clone(), PriceRates::from(p)))
            .collect();

        let zdr_providers = config
            .providers
            .iter()
            .filter(|(_, cfg)| cfg.zdr)
            .map(|(name, _)| name.clone())
            .collect();

        let no_training_providers = config
            .providers
            .iter()
            .filter(|(_, cfg)| cfg.no_training)
            .map(|(name, _)| name.clone())
            .collect();

        let provider_rpm = config
            .providers
            .iter()
            .filter_map(|(name, cfg)| cfg.requests_per_minute.map(|rpm| (name.clone(), rpm)))
            .collect();

        // A bad [persistence] config (e.g. an unwritable path, or an
        // unreachable Postgres database) is a soft failure -- the router
        // still starts and runs with in-memory-only usage/budget tracking,
        // matching how a misconfigured provider or client is
        // skipped-with-a-warning rather than refused at startup.
        let persistence = match config
            .persistence
            .as_ref()
            .and_then(resolve_persistence_target)
        {
            None => None,
            Some(target) => match Persistence::open(target).await {
                Ok(p) => Some(Arc::new(p)),
                Err(e) => {
                    tracing::warn!(
                        "failed to open persistence database, falling back to in-memory usage tracking: {e}"
                    );
                    None
                }
            },
        };
        let usage = match &persistence {
            Some(p) => match p.load_all().await {
                Ok(loaded) => loaded,
                Err(e) => {
                    tracing::warn!("failed to load persisted usage stats: {e}");
                    HashMap::new()
                }
            },
            None => HashMap::new(),
        };

        let client_budgets = client_budget::settings_from_clients(&config.clients);

        let webhook = config.webhook.as_ref().map(|cfg| {
            let auth_header = cfg.auth_header_env.as_ref().and_then(|var| {
                match std::env::var(var) {
                    Ok(v) if !v.is_empty() => Some(v),
                    _ => {
                        tracing::warn!(env_var = %var, "webhook auth_header_env is set but not resolvable; sending webhook requests with no Authorization header");
                        None
                    }
                }
            });
            Arc::new(WebhookNotifier::new(
                cfg.url.clone(),
                auth_header,
                cfg.timeout_secs,
            ))
        });

        // A guardrail with an invalid regex is a soft failure -- skipped
        // with a warning, same as a misconfigured provider or client,
        // rather than refusing to start the whole router over one bad
        // pattern.
        let guardrails = config
            .guardrails
            .iter()
            .filter_map(|cfg| match Guardrail::compile(cfg) {
                Ok(g) => Some(g),
                Err(e) => {
                    tracing::warn!(guardrail = %cfg.name, "invalid guardrail pattern, skipping: {e}");
                    None
                }
            })
            .collect();

        let presets = config
            .presets
            .iter()
            .map(|cfg| (cfg.name.clone(), cfg.clone()))
            .collect();

        let auto_routing = config.auto_routing.clone();

        // Same "skip with a warning" resilience as a misconfigured
        // provider: an unresolvable api_key_env disables moderation
        // rather than refusing to start the whole router over it.
        let moderation = config.moderation.as_ref().and_then(|cfg| {
            match std::env::var(&cfg.api_key_env) {
                Ok(key) if !key.is_empty() => {
                    Some(Arc::new(ModerationClient::new(cfg, key)))
                }
                _ => {
                    tracing::warn!(env_var = %cfg.api_key_env, "moderation.api_key_env is set but not resolvable; moderation stays disabled");
                    None
                }
            }
        });

        // Same "skip with a warning" resilience as moderation above.
        let web_search = config.web_search.as_ref().and_then(|cfg| {
            match std::env::var(&cfg.api_key_env) {
                Ok(key) if !key.is_empty() => Some(Arc::new(WebSearchClient::new(cfg, key))),
                _ => {
                    tracing::warn!(env_var = %cfg.api_key_env, "web_search.api_key_env is set but not resolvable; web search stays disabled");
                    None
                }
            }
        });

        // No external credential to resolve, unlike moderation/web_search
        // above -- this is purely an in-process cache, so there's nothing
        // to skip-with-a-warning over.
        let cache = config
            .cache
            .as_ref()
            .map(|cfg| Arc::new(RwLock::new(ResponseCache::new(cfg))));

        Self {
            providers,
            provider_kinds,
            routes,
            pricing: Arc::new(pricing),
            zdr_providers,
            no_training_providers,
            latency: RwLock::new(HashMap::new()),
            uptime: RwLock::new(HashMap::new()),
            throughput: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(usage)),
            persistence,
            generations: Arc::new(RwLock::new(GenerationCache::default())),
            metrics,
            provider_rpm,
            outbound_limiter: RateLimiter::new(),
            client_budgets: RwLock::new(client_budgets),
            client_spend: Mutex::new(HashMap::new()),
            webhook,
            guardrails,
            presets,
            auto_routing,
            moderation,
            web_search,
            cache,
        }
    }

    pub fn configured_providers(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    pub fn route_aliases(&self) -> impl Iterator<Item = &str> {
        self.routes.keys().map(String::as_str)
    }

    /// Rich metadata for every "provider/model" with a `[[pricing]]`
    /// entry, for `GET /v1/models` -- context length, pricing, and which
    /// request params that provider's adapter actually understands. A
    /// pricing entry for a provider this process couldn't configure (e.g.
    /// its API key env var was unset) is skipped, since `supported_params`
    /// depends on knowing that provider's `kind`.
    pub fn priced_models(&self) -> Vec<ModelInfo> {
        self.pricing
            .iter()
            .filter_map(|(id, rates)| {
                let provider = id.split('/').next()?;
                let kind = *self.provider_kinds.get(provider)?;
                Some(ModelInfo {
                    id: id.clone(),
                    object: "model",
                    owned_by: provider.to_string(),
                    context_length: rates.context_length,
                    pricing: Some(ModelPricing {
                        prompt: rates.prompt_ppm,
                        completion: rates.completion_ppm,
                        cache_read: rates.cache_read_ppm,
                        cache_write: rates.cache_write_ppm,
                    }),
                    supported_params: Some(
                        supported_params(kind)
                            .into_iter()
                            .map(String::from)
                            .collect(),
                    ),
                })
            })
            .collect()
    }

    /// Snapshot of this process's own observed latency/throughput/uptime
    /// EWMAs per "provider/model", for `GET /v1/providers/stats`. Only
    /// includes entries this process has actually dispatched to at least
    /// once -- a "provider/model" this process has never tried isn't
    /// listed at all, rather than listed with every figure absent.
    /// In-memory only, per-process: resets on restart, isn't a shared or
    /// global feed, and reflects only this process's own traffic even
    /// behind a load balancer.
    pub fn provider_stats(&self) -> HashMap<String, ProviderStats> {
        let latency = self.latency.read().unwrap();
        let throughput = self.throughput.read().unwrap();
        let uptime = self.uptime.read().unwrap();

        let keys: HashSet<&String> = latency
            .keys()
            .chain(throughput.keys())
            .chain(uptime.keys())
            .collect();

        keys.into_iter()
            .map(|key| {
                (
                    key.clone(),
                    ProviderStats {
                        latency_ms: latency.get(key).copied(),
                        throughput_tokens_per_sec: throughput.get(key).copied(),
                        uptime: uptime.get(key).copied(),
                    },
                )
            })
            .collect()
    }

    /// Snapshot of cumulative usage/cost per "provider/model", for
    /// `GET /v1/usage`. When `[persistence]` is configured this reads
    /// fresh from the shared database (reflecting every process's writes,
    /// not just this one's), falling back to this process's own in-memory
    /// view if that read fails. Without persistence, it's always the
    /// in-memory view.
    pub async fn usage_snapshot(&self) -> HashMap<String, UsageStats> {
        if let Some(persistence) = &self.persistence {
            match persistence.snapshot().await {
                Ok(snapshot) => return snapshot,
                Err(e) => {
                    tracing::warn!("failed to read usage snapshot from persistence, falling back to in-memory view: {e}");
                }
            }
        }
        self.usage.read().unwrap().clone()
    }

    /// `Ok(())` if this router can actually serve traffic right now, for
    /// `GET /ready`. Without `[persistence]` configured there's nothing
    /// external to check, so this is always `Ok`; with it configured, a
    /// trivial round trip confirms the database is actually reachable
    /// rather than just having been reachable at startup.
    pub async fn check_readiness(&self) -> Result<(), String> {
        match &self.persistence {
            Some(persistence) => persistence.ping().await.map_err(|e| e.to_string()),
            None => Ok(()),
        }
    }

    /// Records one completed request's token/cost breakdown for later
    /// `GET /v1/generation?id=` lookup.
    fn record_generation(&self, record: GenerationRecord) {
        self.generations.write().unwrap().insert(record);
    }

    /// Looks up a single completed request's token/cost breakdown by its
    /// response `id`, for `GET /v1/generation?id=`. `None` if this
    /// process never served that id, or served it long enough ago to have
    /// been evicted from the bounded cache (see `GENERATION_CACHE_CAPACITY`).
    pub fn generation(&self, id: &str) -> Option<GenerationRecord> {
        self.generations.read().unwrap().get(id)
    }

    /// `Ok(())` if `client_name` has no configured `budget_usd`, or
    /// hasn't yet reached it for the current `budget_period`.
    /// `Err(ClientBudgetExceeded)` if it has. When `[persistence]` is
    /// configured this reads fresh from the shared database (so a
    /// client's budget is enforced consistently across every process/host
    /// sharing it, not just this one); without persistence it's this
    /// process's own in-memory view, same as latency/throughput tracking.
    pub async fn check_client_budget(&self, client_name: &str) -> Result<(), ClientBudgetExceeded> {
        let Some(setting) = self
            .client_budgets
            .read()
            .unwrap()
            .get(client_name)
            .copied()
        else {
            return Ok(());
        };
        let spent_usd = self.spent_usd_for(client_name, &setting).await;

        if spent_usd >= setting.budget_usd {
            Err(ClientBudgetExceeded {
                spent_usd,
                budget_usd: setting.budget_usd,
            })
        } else {
            Ok(())
        }
    }

    /// Shared by `check_client_budget` and `client_spend_status`: reads
    /// `client_name`'s tracked spend for `setting`'s current period,
    /// treating a rolled-over or unreadable value as unspent.
    async fn spent_usd_for(&self, client_name: &str, setting: &ClientBudgetSetting) -> f64 {
        let current_key = client_budget::period_key_at(setting.period, client_budget::now_unix());

        if let Some(persistence) = &self.persistence {
            match persistence.client_spend(client_name).await {
                Ok(Some((period_key, spent_usd))) if period_key == current_key => spent_usd,
                Ok(_) => 0.0,
                Err(e) => {
                    tracing::warn!("failed to read client spend from persistence, treating as unspent for this check: {e}");
                    0.0
                }
            }
        } else {
            let mut spend = self.client_spend.lock().unwrap();
            let state = spend.entry(client_name.to_string()).or_default();
            client_budget::roll_period_if_needed(state, current_key);
            state.spent_usd
        }
    }

    /// `client_name`'s live spend against its configured budget, for the
    /// admin API (`GET /v1/admin/clients`). `None` if `client_name` has no
    /// configured `budget_usd` -- there's nothing to report.
    pub async fn client_spend_status(&self, client_name: &str) -> Option<ClientSpendStatus> {
        let setting = *self.client_budgets.read().unwrap().get(client_name)?;
        let spent_usd = self.spent_usd_for(client_name, &setting).await;
        Some(ClientSpendStatus {
            spent_usd,
            budget_usd: setting.budget_usd,
            period: setting.period,
        })
    }

    /// Resets `client_name`'s tracked spend to zero for the current
    /// `budget_period`, for the admin API's manual budget reset
    /// (`POST /v1/admin/clients/{name}/reset-spend`). Returns `false` (a
    /// no-op) for a client with no configured budget -- there's nothing to
    /// reset.
    pub fn reset_client_spend(&self, client_name: &str) -> bool {
        let Some(setting) = self
            .client_budgets
            .read()
            .unwrap()
            .get(client_name)
            .copied()
        else {
            return false;
        };
        let current_key = client_budget::period_key_at(setting.period, client_budget::now_unix());

        if let Some(persistence) = &self.persistence {
            persistence.reset_client_spend(client_name, current_key);
        } else {
            let mut spend = self.client_spend.lock().unwrap();
            spend.insert(
                client_name.to_string(),
                SpendState {
                    period_key: current_key,
                    spent_usd: 0.0,
                },
            );
        }
        if let Some(webhook) = &self.webhook {
            webhook.notify_budget_reset(client_name, setting.budget_usd, setting.period);
        }
        true
    }

    /// Adds `cost_usd` to `client_name`'s tracked spend for the current
    /// `budget_period`. A no-op for clients with no configured budget —
    /// there's nothing to track against. Never blocks the caller on I/O
    /// when `[persistence]` is configured, the same as `record_usage`.
    ///
    /// If a `[webhook]` is configured and this call looks like it just
    /// pushed `client_name` from under its budget to at-or-over it, fires
    /// a `budget_exceeded` event (delivery itself never blocks the
    /// caller). Under `[persistence]`, "just crossed" is a best-effort,
    /// eventually-consistent read-back rather than an atomic
    /// check-and-set, so concurrent requests to the same client near the
    /// boundary could both fire (or, rarely, neither) -- same class of
    /// caveat as the spend total itself already carries.
    pub fn record_client_spend(&self, client_name: &str, cost_usd: f64) {
        let Some(setting) = self
            .client_budgets
            .read()
            .unwrap()
            .get(client_name)
            .copied()
        else {
            return;
        };
        let current_key = client_budget::period_key_at(setting.period, client_budget::now_unix());

        if let Some(persistence) = &self.persistence {
            persistence.record_client_spend(client_name, current_key, cost_usd);
            if let Some(webhook) = self.webhook.clone() {
                let persistence = persistence.clone();
                let client_name = client_name.to_string();
                tokio::spawn(async move {
                    if let Ok(Some((period_key, spent_usd))) =
                        persistence.client_spend(&client_name).await
                    {
                        let before = spent_usd - cost_usd;
                        if period_key == current_key
                            && before < setting.budget_usd
                            && spent_usd >= setting.budget_usd
                        {
                            webhook.notify_budget_exceeded(
                                &client_name,
                                spent_usd,
                                setting.budget_usd,
                                setting.period,
                            );
                        }
                    }
                });
            }
        } else {
            let mut spend = self.client_spend.lock().unwrap();
            let state = spend.entry(client_name.to_string()).or_default();
            client_budget::roll_period_if_needed(state, current_key);
            let before = state.spent_usd;
            state.spent_usd += cost_usd;
            let after = state.spent_usd;
            drop(spend);
            if before < setting.budget_usd && after >= setting.budget_usd {
                if let Some(webhook) = &self.webhook {
                    webhook.notify_budget_exceeded(
                        client_name,
                        after,
                        setting.budget_usd,
                        setting.period,
                    );
                }
            }
        }
    }

    /// Adds, updates, or clears `client_name`'s budget setting, for the
    /// admin API's runtime client provisioning (`POST`/`PATCH
    /// /v1/admin/clients`). `Some((budget_usd, period))` adds it if new or
    /// overwrites it if it already existed; `None` clears it, making the
    /// client unrestricted -- same as a `[[clients]]` entry with no
    /// `budget_usd` set. Doesn't touch tracked spend either way, so
    /// re-adding a budget after clearing it picks up wherever the client's
    /// spend already was.
    pub fn set_client_budget(&self, client_name: &str, budget: Option<(f64, BudgetPeriod)>) {
        let mut budgets = self.client_budgets.write().unwrap();
        match budget {
            Some((budget_usd, period)) => {
                budgets.insert(
                    client_name.to_string(),
                    ClientBudgetSetting { budget_usd, period },
                );
            }
            None => {
                budgets.remove(client_name);
            }
        }
    }

    /// Forgets `client_name` entirely, for the admin API's runtime client
    /// deletion (`DELETE /v1/admin/clients/{name}`) -- drops its budget
    /// setting and in-memory spend state. Persisted spend rows (if
    /// `[persistence]` is configured) are left alone, matching
    /// `reset_client_spend`'s existing behavior of never deleting rows;
    /// they simply go unread once nothing references this client by name
    /// again.
    pub fn remove_client(&self, client_name: &str) {
        self.client_budgets.write().unwrap().remove(client_name);
        self.client_spend.lock().unwrap().remove(client_name);
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

    /// Record a request rejected by the HTTP layer's own per-client spend
    /// budget (`[[clients]].budget_usd`), so it shows up in `GET /metrics`
    /// alongside every other counter this router tracks.
    pub fn record_client_budget_rejection(&self, client_name: &str) {
        self.metrics.record_client_budget_rejection(client_name);
    }

    /// Resolve a client-supplied `model` string into an ordered chain of
    /// (provider, model) pairs. With `fallbacks` non-empty, the chain is an
    /// ad-hoc `model` followed by each of `fallbacks`, entirely bypassing
    /// `[[routes]]` alias lookup -- a client-supplied chain for this one
    /// request rather than an operator-predefined one. Otherwise, the chain
    /// is either a configured alias's fallback chain, or a single
    /// "provider/model" entry.
    fn resolve_chain(
        &self,
        model: &str,
        fallbacks: Option<&[String]>,
    ) -> Result<Vec<(String, String)>, RouterError> {
        let entries: Vec<String> = match fallbacks {
            Some(models) if !models.is_empty() => std::iter::once(model.to_string())
                .chain(models.iter().cloned())
                .collect(),
            _ => match self.routes.get(model) {
                Some(chain) => chain.clone(),
                None => vec![model.to_string()],
            },
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
    /// `provider.data_collection`/`provider.sort` constraints to a resolved
    /// chain, in that order: filter, then sort.
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
        if prefs.data_collection == Some(true) {
            chain.retain(|(provider, _)| self.no_training_providers.contains(provider));
        }
        if let Some(max_price) = prefs.max_price {
            chain.retain(|(provider, model)| {
                self.pricing
                    .get(&format!("{provider}/{model}"))
                    .is_some_and(|rates| rates.prompt_ppm <= max_price)
            });
        }
        if chain.is_empty() {
            return Err(RouterError::NoEligibleProvider(model.to_string()));
        }

        match prefs.sort.as_deref() {
            Some("price") => chain.sort_by(|a, b| {
                let price_of = |entry: &(String, String)| {
                    self.pricing
                        .get(&format!("{}/{}", entry.0, entry.1))
                        .map(|rates| rates.prompt_ppm)
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
            // Descending: higher observed success rate is better, and an
            // unobserved entry (0.0) sorts last -- same convention as
            // throughput, not an optimistic "assume healthy" default.
            Some("uptime") => chain.sort_by(|a, b| {
                ewma_lookup(&self.uptime, &b.0, &b.1, 0.0).total_cmp(&ewma_lookup(
                    &self.uptime,
                    &a.0,
                    &a.1,
                    0.0,
                ))
            }),
            _ => {}
        }

        Ok(chain)
    }

    /// Runs every configured `[[guardrails]]` entry against `req`'s
    /// message text, in config order -- redacting matches in place for a
    /// `"redact"` guardrail, or failing the whole request for a
    /// `"block"` guardrail that matches anywhere. A no-op when no
    /// guardrails are configured. Intended to run before dispatch, on the
    /// caller's own mutable copy of the request (this router never
    /// mutates the caller's original), so a redaction actually reaches
    /// whichever provider ends up serving it.
    pub fn apply_guardrails(&self, req: &mut ChatRequest) -> Result<(), RouterError> {
        guardrails::apply(&self.guardrails, req).map_err(RouterError::GuardrailBlocked)
    }

    /// Checks `req`'s message text against `[moderation]`'s configured
    /// backend, if any -- a no-op when moderation isn't configured (or its
    /// `api_key_env` didn't resolve at startup). Intended to run after
    /// `apply_guardrails`, so a guardrail's own redaction is what gets
    /// checked, not the raw input. A flagged result blocks the request
    /// with the triggering category names. A failure to reach the
    /// moderation backend itself (network error, non-2xx, bad body) fails
    /// *open* -- the request is allowed through and the failure only
    /// logged, the same resilience-over-strictness this router already
    /// gives every other auxiliary system (an unreachable webhook, an
    /// unreachable persistence backend, an invalid guardrail regex at
    /// startup): a moderation-backend outage shouldn't take down chat
    /// completions entirely.
    pub async fn apply_moderation(&self, req: &ChatRequest) -> Result<(), RouterError> {
        let Some(moderation) = &self.moderation else {
            return Ok(());
        };
        match moderation.check(req).await {
            Ok(()) => Ok(()),
            Err(ModerationError::Flagged(categories)) => {
                for category in &categories {
                    self.metrics.record_moderation_blocked(category);
                }
                Err(RouterError::ModerationFlagged(categories))
            }
            Err(ModerationError::RequestFailed(msg)) => {
                tracing::warn!("moderation check failed, allowing request through: {msg}");
                Ok(())
            }
        }
    }

    /// If `req.web_search` is `true` and `[web_search]` is configured,
    /// searches using the latest `user`-role message's text as the query
    /// and prepends the results as context onto that same message,
    /// mutating the caller's owned copy in place -- same "mutate before
    /// dispatch" pattern `apply_guardrails`'s redaction uses. A no-op when
    /// `web_search` isn't requested, isn't configured, there's no
    /// user-message text to search for, or the search comes back with
    /// zero results. A search-backend failure (network error, non-2xx,
    /// bad body) never blocks or errors the request either -- only logged
    /// and counted, the request proceeds unmodified.
    pub async fn apply_web_search(&self, req: &mut ChatRequest) {
        if req.web_search != Some(true) {
            return;
        }
        let Some(web_search) = &self.web_search else {
            return;
        };
        let Some(query) = web_search::last_user_query(req) else {
            return;
        };
        match web_search.search(&query).await {
            Ok(results) if !results.is_empty() => {
                self.metrics.record_web_search("results");
                let prefix = web_search::format_results(&results);
                web_search::prepend_to_last_user_message(req, &prefix);
            }
            Ok(_) => {
                self.metrics.record_web_search("no_results");
            }
            Err(msg) => {
                tracing::warn!("web search failed, continuing without results: {msg}");
                self.metrics.record_web_search("error");
            }
        }
    }

    /// If `req.preset` is set, looks up that `[[presets]]` entry and
    /// folds its defaults into `req` (see `presets::apply` for exactly
    /// what that means per field). A no-op when `req.preset` is unset;
    /// `Err(RouterError::UnknownPreset)` when it names a preset this
    /// router has no config entry for.
    pub fn apply_preset(&self, req: &mut ChatRequest) -> Result<(), RouterError> {
        let Some(name) = req.preset.clone() else {
            return Ok(());
        };
        let preset = self
            .presets
            .get(&name)
            .ok_or_else(|| RouterError::UnknownPreset(name.clone()))?;
        presets::apply(preset, req);
        Ok(())
    }

    /// When `req.provider.require_parameters` is `true`, drops every
    /// candidate whose provider kind doesn't actually give an effect to
    /// every field `req` sets (see `active_params`/`supported_params`) --
    /// a no-op otherwise, run after `apply_preferences` since it needs
    /// the request itself, not just `provider.*` prefs, to know what's
    /// "active." Candidates for a provider this process never resolved a
    /// `kind` for (already excluded from the chain by this point, since
    /// only configured providers ever make it into `resolve_chain`) would
    /// be dropped too, as a defensive default.
    fn filter_by_required_parameters(
        &self,
        model: &str,
        mut chain: Vec<(String, String)>,
        req: &ChatRequest,
    ) -> Result<Vec<(String, String)>, RouterError> {
        let require = req
            .provider
            .as_ref()
            .and_then(|p| p.require_parameters)
            .unwrap_or(false);
        if !require {
            return Ok(chain);
        }

        let active = active_params(req);
        chain.retain(|(provider, _)| {
            self.provider_kinds.get(provider).is_some_and(|kind| {
                let supported = supported_params(*kind);
                active.iter().all(|p| supported.contains(p))
            })
        });
        if chain.is_empty() {
            return Err(RouterError::NoEligibleProvider(model.to_string()));
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
            .map(|_| ())
            .map_err(|status| ProviderError::RateLimited {
                retry_after_secs: Some(status.retry_after_secs.ceil() as u64),
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
        let persistence = self.persistence.clone();
        let pricing = self.pricing.clone();
        let metrics = self.metrics.clone();
        let generations = self.generations.clone();

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
                        let cost = record_usage(&usage_map, persistence.as_deref(), &pricing, &provider_name, &model_name, &usage);
                        metrics.record_tokens_and_cost(&provider_name, &model_name, usage.prompt_tokens, usage.completion_tokens, cost);
                        chunk.cost_usd = cost;
                        generations.write().unwrap().insert(GenerationRecord {
                            id: chunk.id.clone(),
                            model: chunk.model.clone(),
                            created: chunk.created,
                            prompt_tokens: usage.prompt_tokens as u64,
                            completion_tokens: usage.completion_tokens as u64,
                            total_tokens: usage.total_tokens as u64,
                            cost_usd: cost,
                        });
                    }
                }
            }
            item
        });
        Box::pin(instrumented)
    }

    /// Resolves `model: "auto"` to a concrete `[auto_routing]` tier's
    /// model string (a "provider/model" or a `[[routes]]` alias, exactly
    /// like `req.model` itself) before the rest of chain resolution runs.
    /// Every other `req.model` value, or `"auto"` when `[auto_routing]`
    /// isn't configured at all, passes through unchanged -- an
    /// unconfigured `"auto"` then resolves exactly like any other
    /// unrecognized alias (a `400`), not a special error.
    fn resolve_target_model(&self, req: &ChatRequest) -> String {
        if req.model == "auto" {
            if let Some(config) = &self.auto_routing {
                return auto_routing::resolve_tier(config, req);
            }
        }
        req.model.clone()
    }

    /// The caller-supplied BYOK key for `provider_name`, if
    /// `req.provider.byok` set one (see
    /// `rp_core::ProviderPreferences::byok`). `None` falls through to the
    /// operator's own `[providers.X].api_key_env`-resolved key, same as
    /// before this field existed.
    fn byok_key_for<'a>(&self, req: &'a ChatRequest, provider_name: &str) -> Option<&'a str> {
        req.provider
            .as_ref()
            .and_then(|p| p.byok.as_ref())
            .and_then(|m| m.get(provider_name))
            .map(|s| s.as_str())
    }

    /// `req`'s `[cache]` key, if caching is configured and this isn't a
    /// streaming request -- response caching only wraps `dispatch`, not
    /// `dispatch_stream` (see `cache` module docs). `None` from either
    /// condition means the caller should skip the cache entirely, same
    /// as if `[cache]` were never configured.
    fn cache_key_for(&self, req: &ChatRequest) -> Option<u64> {
        if req.is_streaming() {
            return None;
        }
        self.cache.as_ref().map(|_| ResponseCache::key_for(req))
    }

    /// Looks up `key` and records a `cache_lookups_total` hit/miss --
    /// always called through `cache_key_for` first, so `self.cache` is
    /// known to be `Some` whenever this runs with a real `key`, but it's
    /// still guarded here for safety against a future call site that
    /// forgets to check.
    fn cache_get(&self, key: u64) -> Option<ChatResponse> {
        let cache = self.cache.as_ref()?;
        let resp = cache.write().unwrap().get(key);
        self.metrics
            .record_cache_lookup(if resp.is_some() { "hit" } else { "miss" });
        resp
    }

    fn cache_insert(&self, key: u64, resp: ChatResponse) {
        if let Some(cache) = &self.cache {
            cache.write().unwrap().insert(key, resp);
        }
    }

    pub async fn dispatch(&self, req: &ChatRequest) -> Result<ChatResponse, RouterError> {
        let cache_key = self.cache_key_for(req);
        if let Some(key) = cache_key {
            if let Some(resp) = self.cache_get(key) {
                return Ok(resp);
            }
        }

        let target_model = self.resolve_target_model(req);
        let chain = self.resolve_chain(&target_model, req.models.as_deref())?;
        let chain = self.apply_preferences(&target_model, chain, req.provider.as_ref())?;
        let chain = self.filter_by_required_parameters(&target_model, chain, req)?;
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

            let truncated_req =
                maybe_apply_middle_out(req, provider_name, model_name, &self.pricing);
            let req_to_send = truncated_req.as_ref().unwrap_or(req);
            let api_key_override = self.byok_key_for(req, provider_name);

            let started_at = Instant::now();
            match provider
                .chat(req_to_send, model_name, api_key_override)
                .await
            {
                Ok(mut resp) => {
                    let elapsed_secs = started_at.elapsed().as_secs_f64();
                    ewma_record(
                        &self.latency,
                        format!("{provider_name}/{model_name}"),
                        elapsed_secs * 1000.0,
                    );
                    ewma_record(&self.uptime, format!("{provider_name}/{model_name}"), 1.0);
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
                            self.persistence.as_deref(),
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
                        self.record_generation(GenerationRecord {
                            id: resp.id.clone(),
                            model: resp.model.clone(),
                            created: resp.created,
                            prompt_tokens: usage.prompt_tokens as u64,
                            completion_tokens: usage.completion_tokens as u64,
                            total_tokens: usage.total_tokens as u64,
                            cost_usd: cost,
                        });
                    }

                    if let Some(key) = cache_key {
                        self.cache_insert(key, resp.clone());
                    }
                    return Ok(resp);
                }
                Err(e) if e.is_retryable() => {
                    tracing::warn!(provider = %provider_name, model = %model_name, "provider failed, falling back: {e}");
                    ewma_record(&self.uptime, format!("{provider_name}/{model_name}"), 0.0);
                    self.metrics
                        .record_attempt(provider_name, model_name, "retryable_error");
                    last_err = Some(RouterError::Provider(e));
                }
                Err(e) => {
                    ewma_record(&self.uptime, format!("{provider_name}/{model_name}"), 0.0);
                    self.metrics
                        .record_attempt(provider_name, model_name, "error");
                    return Err(RouterError::Provider(e));
                }
            }
        }

        Err(last_err.unwrap_or_else(|| RouterError::InvalidModel(target_model.clone())))
    }

    pub async fn dispatch_stream(&self, req: &ChatRequest) -> Result<ChatStream, RouterError> {
        let target_model = self.resolve_target_model(req);
        let chain = self.resolve_chain(&target_model, req.models.as_deref())?;
        let chain = self.apply_preferences(&target_model, chain, req.provider.as_ref())?;
        let chain = self.filter_by_required_parameters(&target_model, chain, req)?;
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

            let truncated_req =
                maybe_apply_middle_out(req, provider_name, model_name, &self.pricing);
            let req_to_send = truncated_req.as_ref().unwrap_or(req);
            let api_key_override = self.byok_key_for(req, provider_name);

            let started_at = Instant::now();
            match provider
                .chat_stream(req_to_send, model_name, api_key_override)
                .await
            {
                Ok(stream) => {
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    ewma_record(
                        &self.latency,
                        format!("{provider_name}/{model_name}"),
                        elapsed_ms,
                    );
                    ewma_record(&self.uptime, format!("{provider_name}/{model_name}"), 1.0);
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
                    ewma_record(&self.uptime, format!("{provider_name}/{model_name}"), 0.0);
                    self.metrics
                        .record_attempt(provider_name, model_name, "retryable_error");
                    last_err = Some(RouterError::Provider(e));
                }
                Err(e) => {
                    ewma_record(&self.uptime, format!("{provider_name}/{model_name}"), 0.0);
                    self.metrics
                        .record_attempt(provider_name, model_name, "error");
                    return Err(RouterError::Provider(e));
                }
            }
        }

        Err(last_err.unwrap_or_else(|| RouterError::InvalidModel(target_model.clone())))
    }

    /// Resolves `req.model` to a provider chain and dispatches, falling
    /// back on a retryable error exactly like `dispatch` -- but without
    /// `dispatch`'s cost/latency/throughput/generation-cache bookkeeping,
    /// none of which has an established meaning for an embeddings call
    /// yet (no completion tokens, no `[[pricing]]` entry shape for a
    /// prompt-only request). Every candidate's outcome is still counted
    /// via the same `dispatch_attempts_total` metric as `dispatch`, so a
    /// failing/misconfigured provider is still observable.
    pub async fn embeddings(
        &self,
        req: &EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, RouterError> {
        let chain = self.resolve_chain(&req.model, None)?;
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

            match provider.embeddings(req, model_name, None).await {
                Ok(resp) => {
                    self.metrics
                        .record_attempt(provider_name, model_name, "success");
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
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures::stream;
    use rp_core::{
        ChatChunk, ChatMessage, ChatMessageDelta, Choice, ChunkChoice, EmbeddingData,
        EmbeddingsUsage, MessageContent, Role,
    };
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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
            provider_kinds: HashMap::new(),
            routes: routes
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.into_iter().map(String::from).collect()))
                .collect(),
            pricing: Arc::new(
                pricing
                    .into_iter()
                    .map(|(k, p, c)| {
                        (
                            k.to_string(),
                            PriceRates {
                                prompt_ppm: p,
                                completion_ppm: c,
                                cache_read_ppm: p,
                                cache_write_ppm: p,
                                context_length: None,
                            },
                        )
                    })
                    .collect(),
            ),
            zdr_providers: zdr_providers.into_iter().map(String::from).collect(),
            no_training_providers: HashSet::new(),
            latency: RwLock::new(HashMap::new()),
            uptime: RwLock::new(HashMap::new()),
            throughput: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(HashMap::new())),
            persistence: None,
            generations: Arc::new(RwLock::new(GenerationCache::default())),
            metrics: Metrics::new(),
            provider_rpm: provider_rpm
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            outbound_limiter: RateLimiter::new(),
            client_budgets: RwLock::new(HashMap::new()),
            client_spend: Mutex::new(HashMap::new()),
            webhook: None,
            guardrails: Vec::new(),
            presets: HashMap::new(),
            auto_routing: None,
            moderation: None,
            web_search: None,
            cache: None,
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
            models: None,
            preset: None,
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
            response_format: None,
            reasoning: None,
            top_k: None,
            min_p: None,
            top_a: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            logit_bias: None,
            seed: None,
            transforms: None,
            logprobs: None,
            top_logprobs: None,
            web_search: None,
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
            _api_key_override: Option<&str>,
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
                        logprobs: None,
                    }],
                    usage: Some(Usage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                        cached_tokens: None,
                        cache_creation_tokens: None,
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
            _api_key_override: Option<&str>,
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
                                reasoning: None,
                            },
                            finish_reason: Some("stop".to_string()),
                            logprobs: None,
                        }],
                        usage: Some(Usage {
                            prompt_tokens: 1,
                            completion_tokens: 1,
                            total_tokens: 2,
                            cached_tokens: None,
                            cache_creation_tokens: None,
                        }),
                        cost_usd: None,
                    };
                    Ok(Box::pin(stream::once(async { Ok(chunk) })))
                }
                _ => Err(self.canned_error()),
            }
        }

        async fn embeddings(
            &self,
            req: &EmbeddingsRequest,
            model: &str,
            _api_key_override: Option<&str>,
        ) -> Result<EmbeddingsResponse, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.behavior {
                MockBehavior::Succeed => Ok(EmbeddingsResponse {
                    object: "list",
                    data: req
                        .input
                        .as_slice()
                        .into_iter()
                        .enumerate()
                        .map(|(index, _)| EmbeddingData {
                            object: "embedding",
                            embedding: vec![0.1, 0.2],
                            index,
                        })
                        .collect(),
                    model: format!("{}/{model}", self.name),
                    usage: Some(EmbeddingsUsage {
                        prompt_tokens: 1,
                        total_tokens: 1,
                    }),
                }),
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
            _api_key_override: Option<&str>,
        ) -> Result<ChatResponse, ProviderError> {
            unreachable!("ScriptedStreamProvider only implements chat_stream")
        }

        async fn chat_stream(
            &self,
            _req: &ChatRequest,
            _model: &str,
            _api_key_override: Option<&str>,
        ) -> Result<ChatStream, ProviderError> {
            let chunks = self.chunks.clone();
            Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
        }

        async fn embeddings(
            &self,
            _req: &EmbeddingsRequest,
            _model: &str,
            _api_key_override: Option<&str>,
        ) -> Result<EmbeddingsResponse, ProviderError> {
            unreachable!("ScriptedStreamProvider only implements chat_stream")
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
                    reasoning: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                cached_tokens: None,
                cache_creation_tokens: None,
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

    /// A `Provider` that records the `api_key_override` it was called
    /// with, so BYOK plumbing (`Router::byok_key_for` actually reaching
    /// the `Provider::chat`/`chat_stream` call) can be asserted without a
    /// real HTTP call.
    struct KeyCapturingProvider {
        name: String,
        seen_key: Mutex<Option<String>>,
    }

    #[async_trait]
    impl Provider for KeyCapturingProvider {
        fn name(&self) -> &str {
            &self.name
        }

        async fn chat(
            &self,
            _req: &ChatRequest,
            model: &str,
            api_key_override: Option<&str>,
        ) -> Result<ChatResponse, ProviderError> {
            *self.seen_key.lock().unwrap() = api_key_override.map(str::to_string);
            Ok(ChatResponse {
                id: "test-id".to_string(),
                object: "chat.completion",
                created: 0,
                model: format!("{}/{model}", self.name),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage::assistant("ok"),
                    finish_reason: Some("stop".to_string()),
                    logprobs: None,
                }],
                usage: None,
                cost_usd: None,
            })
        }

        async fn chat_stream(
            &self,
            _req: &ChatRequest,
            _model: &str,
            api_key_override: Option<&str>,
        ) -> Result<ChatStream, ProviderError> {
            *self.seen_key.lock().unwrap() = api_key_override.map(str::to_string);
            Ok(Box::pin(stream::empty::<Result<ChatChunk, ProviderError>>()))
        }

        async fn embeddings(
            &self,
            _req: &EmbeddingsRequest,
            model: &str,
            api_key_override: Option<&str>,
        ) -> Result<EmbeddingsResponse, ProviderError> {
            *self.seen_key.lock().unwrap() = api_key_override.map(str::to_string);
            Ok(EmbeddingsResponse {
                object: "list",
                data: vec![EmbeddingData {
                    object: "embedding",
                    embedding: vec![0.1],
                    index: 0,
                }],
                model: format!("{}/{model}", self.name),
                usage: None,
            })
        }
    }

    // --- from_config ---------------------------------------------------------

    #[tokio::test]
    async fn from_config_duplicate_route_alias_keeps_only_the_last_entry() {
        // Config::routes is a plain Vec (TOML happily accepts two [[routes]]
        // blocks with the same alias), but Router::from_config folds them
        // into a HashMap keyed by alias -- the last one parsed silently
        // wins rather than merging or erroring.
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[routes]]
            alias = "smart"
            chain = ["a/m1"]

            [[routes]]
            alias = "smart"
            chain = ["b/m2"]
            "#,
        )
        .unwrap();
        let router = Router::from_config(&config).await;

        assert_eq!(router.route_aliases().collect::<Vec<_>>(), vec!["smart"]);
        assert_eq!(
            router.resolve_chain("smart", None).unwrap(),
            chain(&[("b", "m2")])
        );
    }

    fn unique_temp_db_path(label: &str) -> std::path::PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rp_router_from_config_persistence_test_{label}_{}.db",
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[tokio::test]
    async fn from_config_has_no_persistence_when_the_section_is_absent() {
        let config = Config::from_toml_str("providers = {}").unwrap();
        let router = Router::from_config(&config).await;
        assert!(router.persistence.is_none());
    }

    #[tokio::test]
    async fn from_config_opens_persistence_when_configured() {
        let path = unique_temp_db_path("opens");
        let config = Config::from_toml_str(&format!(
            "providers = {{}}\n\n[persistence]\nsqlite_path = {:?}\n",
            path.to_str().unwrap()
        ))
        .unwrap();

        let router = Router::from_config(&config).await;

        assert!(router.persistence.is_some());
    }

    #[tokio::test]
    async fn from_config_falls_back_to_in_memory_when_persistence_path_is_invalid() {
        // The parent directory doesn't exist, so Persistence::open fails --
        // from_config must not panic or refuse to start, just skip it.
        let bad_path = std::env::temp_dir()
            .join("rp_router_from_config_test_nonexistent_dir")
            .join("usage.db");
        let config = Config::from_toml_str(&format!(
            "providers = {{}}\n\n[persistence]\nsqlite_path = {:?}\n",
            bad_path.to_str().unwrap()
        ))
        .unwrap();

        let router = Router::from_config(&config).await;

        assert!(router.persistence.is_none());
    }

    #[tokio::test]
    async fn from_config_falls_back_to_in_memory_when_persistence_backend_is_postgres_but_env_var_is_unset(
    ) {
        let unset_var = "RP_ROUTER_TEST_UNSET_POSTGRES_URL_VAR";
        std::env::remove_var(unset_var);
        let config = Config::from_toml_str(&format!(
            "providers = {{}}\n\n[persistence]\nbackend = \"postgres\"\npostgres_url_env = {unset_var:?}\n"
        ))
        .unwrap();

        let router = Router::from_config(&config).await;

        assert!(router.persistence.is_none());
    }

    #[tokio::test]
    async fn from_config_falls_back_to_in_memory_when_persistence_backend_is_sqlite_but_path_is_unset(
    ) {
        let config = Config::from_toml_str("providers = {}\n\n[persistence]\n").unwrap();
        let router = Router::from_config(&config).await;
        assert!(router.persistence.is_none());
    }

    #[tokio::test]
    async fn from_config_reads_client_budget_settings() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            budget_usd = 10.0
            budget_period = "monthly"

            [[clients]]
            name = "globex"
            api_key_env = "GLOBEX_KEY"
            requests_per_minute = 60
            "#,
        )
        .unwrap();
        let router = Router::from_config(&config).await;

        assert!(router.check_client_budget("globex").await.is_ok());
        router.record_client_spend("acme", 100.0);
        assert!(router.check_client_budget("acme").await.is_err());
        // "globex" has no configured budget, so it stays unrestricted
        // regardless of what "acme" (a completely different client) spent.
        assert!(router.check_client_budget("globex").await.is_ok());
    }

    // --- client_spend_status / reset_client_spend (admin API) --------------------

    fn router_with_budgeted_client(budget_usd: f64, period: BudgetPeriod) -> Router {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.client_budgets.write().unwrap().insert(
            "acme".to_string(),
            ClientBudgetSetting { budget_usd, period },
        );
        router
    }

    #[tokio::test]
    async fn client_spend_status_is_none_for_a_client_with_no_configured_budget() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        assert_eq!(router.client_spend_status("acme").await, None);
    }

    #[tokio::test]
    async fn client_spend_status_reports_zero_spend_before_any_requests() {
        let router = router_with_budgeted_client(10.0, BudgetPeriod::Total);
        let status = router.client_spend_status("acme").await.unwrap();
        assert_eq!(status.spent_usd, 0.0);
        assert_eq!(status.budget_usd, 10.0);
        assert_eq!(status.period, BudgetPeriod::Total);
    }

    #[tokio::test]
    async fn client_spend_status_reflects_recorded_spend() {
        let router = router_with_budgeted_client(10.0, BudgetPeriod::Total);
        router.record_client_spend("acme", 4.0);
        let status = router.client_spend_status("acme").await.unwrap();
        assert_eq!(status.spent_usd, 4.0);
    }

    #[tokio::test]
    async fn reset_client_spend_is_a_no_op_for_a_client_with_no_configured_budget() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        assert!(!router.reset_client_spend("acme"));
    }

    #[tokio::test]
    async fn reset_client_spend_zeroes_a_budgeted_clients_spend() {
        let router = router_with_budgeted_client(10.0, BudgetPeriod::Total);
        router.record_client_spend("acme", 8.0);
        assert!(router.check_client_budget("acme").await.is_ok());
        router.record_client_spend("acme", 5.0);
        assert!(router.check_client_budget("acme").await.is_err());

        assert!(router.reset_client_spend("acme"));

        assert!(router.check_client_budget("acme").await.is_ok());
        assert_eq!(
            router.client_spend_status("acme").await.unwrap().spent_usd,
            0.0
        );
    }

    // --- webhook -----------------------------------------------------------

    fn router_with_budgeted_client_and_webhook(
        budget_usd: f64,
        webhook_url: String,
        auth_header: Option<String>,
    ) -> Router {
        let router = router_with_budgeted_client(budget_usd, BudgetPeriod::Total);
        Router {
            webhook: Some(Arc::new(WebhookNotifier::new(webhook_url, auth_header, 5))),
            ..router
        }
    }

    #[tokio::test]
    async fn record_client_spend_fires_budget_exceeded_on_the_crossing_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .and(body_json(json!({
                "event": "budget_exceeded",
                "client": "acme",
                "spent_usd": 12.0,
                "budget_usd": 10.0,
                "period": "total",
            })))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let router = router_with_budgeted_client_and_webhook(10.0, server.uri(), None);
        router.record_client_spend("acme", 8.0);
        router.record_client_spend("acme", 4.0);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        server.verify().await;
    }

    #[tokio::test]
    async fn record_client_spend_does_not_refire_once_already_over_budget() {
        // `.expect(1)` on the mock means a second delivery would fail
        // verification -- proving the second, already-over-budget request
        // doesn't fire another event.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let router = router_with_budgeted_client_and_webhook(10.0, server.uri(), None);
        router.record_client_spend("acme", 12.0);
        router.record_client_spend("acme", 3.0);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        server.verify().await;
    }

    #[tokio::test]
    async fn record_client_spend_sends_no_webhook_when_the_budget_is_not_crossed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let router = router_with_budgeted_client_and_webhook(10.0, server.uri(), None);
        router.record_client_spend("acme", 4.0);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        server.verify().await;
    }

    #[tokio::test]
    async fn record_client_spend_sends_the_configured_authorization_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .and(header("authorization", "Bearer s3cret"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let router = router_with_budgeted_client_and_webhook(
            10.0,
            server.uri(),
            Some("Bearer s3cret".to_string()),
        );
        router.record_client_spend("acme", 12.0);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        server.verify().await;
    }

    #[tokio::test]
    async fn reset_client_spend_fires_budget_reset() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .and(body_json(json!({
                "event": "budget_reset",
                "client": "acme",
                "budget_usd": 10.0,
                "period": "total",
            })))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let router = router_with_budgeted_client_and_webhook(10.0, server.uri(), None);
        assert!(router.reset_client_spend("acme"));
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        server.verify().await;
    }

    #[tokio::test]
    async fn reset_client_spend_sends_no_webhook_for_a_client_with_no_configured_budget() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let router = Router {
            webhook: Some(Arc::new(WebhookNotifier::new(server.uri(), None, 5))),
            ..router
        };
        assert!(!router.reset_client_spend("acme"));
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        server.verify().await;
    }

    #[tokio::test]
    async fn client_budget_is_persisted_and_shared_across_two_router_instances_sqlite() {
        // Same shape as the usage-sharing test below, but for client spend
        // budgets: router_a records spend against "acme"; router_b, a
        // separate Router pointed at the same SQLite file, must see it via
        // check_client_budget() -- proving the budget check itself (not
        // just the underlying Persistence API) reads through to the
        // shared backend rather than router_a's in-memory state.
        let path = unique_temp_db_path("client_budget_shared");
        let setting = ClientBudgetSetting {
            budget_usd: 10.0,
            period: BudgetPeriod::Total,
        };

        let mut router_a = test_router(vec![], vec![], vec![], vec![], vec![]);
        router_a.persistence = Some(Arc::new(
            Persistence::open(PersistenceTarget::Sqlite(path.clone()))
                .await
                .unwrap(),
        ));
        router_a
            .client_budgets
            .write()
            .unwrap()
            .insert("acme".to_string(), setting);
        router_a.record_client_spend("acme", 12.0);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let mut router_b = test_router(vec![], vec![], vec![], vec![], vec![]);
        router_b.persistence = Some(Arc::new(
            Persistence::open(PersistenceTarget::Sqlite(path))
                .await
                .unwrap(),
        ));
        router_b
            .client_budgets
            .write()
            .unwrap()
            .insert("acme".to_string(), setting);

        assert!(router_b.check_client_budget("acme").await.is_err());
    }

    fn test_postgres_url() -> Option<String> {
        std::env::var("TEST_POSTGRES_URL").ok()
    }

    #[tokio::test]
    async fn client_budget_is_persisted_and_shared_across_two_router_instances_postgres() {
        let Some(url) = test_postgres_url() else {
            eprintln!("skipping: TEST_POSTGRES_URL not set");
            return;
        };
        // Unlike the SQLite test's per-test temp file, the Postgres test
        // database persists across `cargo test` invocations, so the
        // client name needs to be unique across runs too, not just within
        // this one.
        let client_name = format!("router_test_client_budget_{}", std::process::id());
        let setting = ClientBudgetSetting {
            budget_usd: 10.0,
            period: BudgetPeriod::Total,
        };

        let mut router_a = test_router(vec![], vec![], vec![], vec![], vec![]);
        router_a.persistence = Some(Arc::new(
            Persistence::open(PersistenceTarget::Postgres {
                url: url.clone(),
                tls: PostgresTlsMode::Disable,
            })
            .await
            .unwrap(),
        ));
        router_a
            .client_budgets
            .write()
            .unwrap()
            .insert(client_name.clone(), setting);
        router_a.record_client_spend(&client_name, 12.0);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let mut router_b = test_router(vec![], vec![], vec![], vec![], vec![]);
        router_b.persistence = Some(Arc::new(
            Persistence::open(PersistenceTarget::Postgres {
                url,
                tls: PostgresTlsMode::Disable,
            })
            .await
            .unwrap(),
        ));
        router_b
            .client_budgets
            .write()
            .unwrap()
            .insert(client_name.clone(), setting);

        assert!(router_b.check_client_budget(&client_name).await.is_err());
    }

    #[tokio::test]
    async fn two_router_instances_sharing_a_persistence_file_see_each_others_usage() {
        // Simulates two processes (or a restart): router_a dispatches and
        // persists a request; router_b, an entirely separate Router that
        // never dispatched anything itself, is pointed at the same SQLite
        // file and must see router_a's usage via usage_snapshot().
        let path = unique_temp_db_path("shared");

        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls,
        });
        let mut router_a = test_router(
            vec![("anthropic", mock)],
            vec![],
            vec![("anthropic/m1", 2.0, 4.0)],
            vec![],
            vec![],
        );
        router_a.persistence = Some(Arc::new(
            Persistence::open(PersistenceTarget::Sqlite(path.clone()))
                .await
                .expect("persistence should open"),
        ));

        router_a
            .dispatch(&test_request("anthropic/m1"))
            .await
            .expect("dispatch should succeed");
        // The write goes through a background thread inside Persistence;
        // give it a moment to land before a second handle reads it back.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let router_b = test_router(vec![], vec![], vec![], vec![], vec![]);
        let router_b = Router {
            persistence: Some(Arc::new(
                Persistence::open(PersistenceTarget::Sqlite(path))
                    .await
                    .expect("persistence should reopen"),
            )),
            ..router_b
        };

        let snapshot = router_b.usage_snapshot().await;
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(stats.requests, 1);
        assert_eq!(stats.prompt_tokens, 1);
        assert_eq!(stats.completion_tokens, 1);
        assert!((stats.cost_usd - 6.0 / 1_000_000.0).abs() < 1e-12);
    }

    // --- check_readiness -------------------------------------------------------

    #[tokio::test]
    async fn check_readiness_is_ok_when_no_persistence_is_configured() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        assert!(router.check_readiness().await.is_ok());
    }

    #[tokio::test]
    async fn check_readiness_is_ok_against_a_reachable_persistence_file() {
        let path = unique_temp_db_path("readiness_ok");
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let router = Router {
            persistence: Some(Arc::new(
                Persistence::open(PersistenceTarget::Sqlite(path))
                    .await
                    .expect("persistence should open"),
            )),
            ..router
        };

        assert!(router.check_readiness().await.is_ok());
    }

    // --- supported_params ----------------------------------------------------

    #[test]
    fn supported_params_lists_top_k_as_common_but_gates_the_rest_by_kind() {
        let anthropic = supported_params(ProviderKind::Anthropic);
        let gemini = supported_params(ProviderKind::Gemini);
        let openai = supported_params(ProviderKind::Openai);

        for params in [&anthropic, &gemini, &openai] {
            assert!(params.contains(&"top_k"));
        }
        assert!(anthropic.contains(&"cache_control"));
        assert!(!gemini.contains(&"cache_control"));
        assert!(!openai.contains(&"cache_control"));

        assert!(gemini.contains(&"frequency_penalty"));
        assert!(!anthropic.contains(&"frequency_penalty"));

        assert!(openai.contains(&"logit_bias"));
        assert!(!anthropic.contains(&"logit_bias"));
        assert!(!gemini.contains(&"logit_bias"));

        assert!(openai.contains(&"logprobs"));
        assert!(!anthropic.contains(&"logprobs"));
        assert!(!gemini.contains(&"logprobs"));
    }

    // --- priced_models ---------------------------------------------------

    #[test]
    fn priced_models_reports_context_length_pricing_and_supported_params() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router
            .provider_kinds
            .insert("anthropic".to_string(), ProviderKind::Anthropic);
        router.pricing = Arc::new(HashMap::from([(
            "anthropic/m1".to_string(),
            PriceRates {
                prompt_ppm: 3.0,
                completion_ppm: 15.0,
                cache_read_ppm: 0.3,
                cache_write_ppm: 3.75,
                context_length: Some(200_000),
            },
        )]));

        let models = router.priced_models();
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!(model.id, "anthropic/m1");
        assert_eq!(model.owned_by, "anthropic");
        assert_eq!(model.context_length, Some(200_000));
        let pricing = model.pricing.expect("pricing should be present");
        assert_eq!(pricing.prompt, 3.0);
        assert_eq!(pricing.completion, 15.0);
        assert_eq!(pricing.cache_read, 0.3);
        assert_eq!(pricing.cache_write, 3.75);
        let params = model
            .supported_params
            .as_ref()
            .expect("supported_params should be present");
        assert!(params.iter().any(|p| p == "cache_control"));
    }

    #[test]
    fn priced_models_skips_an_entry_for_a_provider_that_was_never_configured() {
        // A `[[pricing]]` entry can reference a provider whose API key env
        // var wasn't set at startup (so it never made it into
        // `provider_kinds`) -- skipped rather than reported with a
        // guessed/missing `supported_params`.
        let router = test_router(
            vec![],
            vec![],
            vec![("anthropic/m1", 3.0, 15.0)],
            vec![],
            vec![],
        );
        assert!(router.priced_models().is_empty());
    }

    // --- provider_stats ------------------------------------------------------

    #[test]
    fn provider_stats_is_empty_with_no_observations() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        assert!(router.provider_stats().is_empty());
    }

    #[test]
    fn provider_stats_reports_only_observed_figures_per_key() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        // "anthropic/m1" has all three observed; "openai/m2" only latency.
        ewma_record(&router.latency, "anthropic/m1".to_string(), 500.0);
        ewma_record(&router.throughput, "anthropic/m1".to_string(), 42.0);
        ewma_record(&router.uptime, "anthropic/m1".to_string(), 1.0);
        ewma_record(&router.latency, "openai/m2".to_string(), 100.0);

        let stats = router.provider_stats();
        assert_eq!(stats.len(), 2);

        let anthropic = &stats["anthropic/m1"];
        assert_eq!(anthropic.latency_ms, Some(500.0));
        assert_eq!(anthropic.throughput_tokens_per_sec, Some(42.0));
        assert_eq!(anthropic.uptime, Some(1.0));

        let openai = &stats["openai/m2"];
        assert_eq!(openai.latency_ms, Some(100.0));
        assert_eq!(openai.throughput_tokens_per_sec, None);
        assert_eq!(openai.uptime, None);
    }

    // --- generation / GenerationCache -----------------------------------------

    fn generation_record(id: &str) -> GenerationRecord {
        GenerationRecord {
            id: id.to_string(),
            model: "anthropic/m1".to_string(),
            created: 1_700_000_000,
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cost_usd: Some(0.001),
        }
    }

    #[test]
    fn generation_is_none_for_an_unknown_id() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        assert!(router.generation("chatcmpl-unknown").is_none());
    }

    #[test]
    fn generation_returns_a_previously_recorded_record() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.record_generation(generation_record("chatcmpl-abc"));

        let record = router.generation("chatcmpl-abc").expect("should be found");
        assert_eq!(record.model, "anthropic/m1");
        assert_eq!(record.prompt_tokens, 10);
        assert_eq!(record.completion_tokens, 5);
        assert_eq!(record.total_tokens, 15);
        assert_eq!(record.cost_usd, Some(0.001));
    }

    #[test]
    fn generation_cache_evicts_the_oldest_entry_once_over_capacity() {
        let mut cache = GenerationCache::default();
        for i in 0..GENERATION_CACHE_CAPACITY {
            cache.insert(generation_record(&format!("id-{i}")));
        }
        assert!(cache.get("id-0").is_some());

        // One more insert past capacity must evict the oldest ("id-0").
        cache.insert(generation_record("id-overflow"));
        assert!(cache.get("id-0").is_none());
        assert!(cache.get("id-1").is_some());
        assert!(cache.get("id-overflow").is_some());
    }

    #[test]
    fn generation_cache_reinserting_an_existing_id_does_not_evict() {
        let mut cache = GenerationCache::default();
        cache.insert(generation_record("id-a"));
        cache.insert(generation_record("id-b"));
        // Re-inserting an already-present id must overwrite in place, not
        // consume another slot toward the eviction count.
        cache.insert(generation_record("id-a"));
        assert!(cache.get("id-a").is_some());
        assert!(cache.get("id-b").is_some());
    }

    // --- apply_moderation ------------------------------------------------------

    fn moderation_config(base_url: &str) -> ModerationConfig {
        ModerationConfig {
            api_key_env: "UNUSED".to_string(),
            base_url: base_url.to_string(),
            model: "omni-moderation-latest".to_string(),
            timeout_secs: 5,
        }
    }

    #[tokio::test]
    async fn apply_moderation_is_a_noop_when_unconfigured() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let req = test_request("anthropic/m1");
        assert!(router.apply_moderation(&req).await.is_ok());
    }

    #[tokio::test]
    async fn apply_moderation_blocks_and_records_a_metric_per_flagged_category() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{
                    "flagged": true,
                    "categories": {"violence": true, "hate": false}
                }]
            })))
            .mount(&server)
            .await;

        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.moderation = Some(Arc::new(ModerationClient::new(
            &moderation_config(&server.uri()),
            "test-key".to_string(),
        )));

        let req = test_request("anthropic/m1");
        let err = router.apply_moderation(&req).await.unwrap_err();
        assert!(
            matches!(err, RouterError::ModerationFlagged(categories) if categories == vec!["violence".to_string()])
        );

        let metrics = router.render_prometheus_metrics();
        assert!(metrics.contains("rusty_provider_moderation_blocked_total"));
        assert!(metrics.contains(r#"category="violence""#));
    }

    #[tokio::test]
    async fn apply_moderation_allows_the_request_through_when_the_backend_is_unreachable() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        // Port 0 never accepts connections -- a real, unrecoverable
        // network failure, not just a non-2xx response.
        router.moderation = Some(Arc::new(ModerationClient::new(
            &moderation_config("http://127.0.0.1:0"),
            "test-key".to_string(),
        )));

        let req = test_request("anthropic/m1");
        assert!(
            router.apply_moderation(&req).await.is_ok(),
            "a moderation-backend outage must fail open, not block the request"
        );
    }

    // --- apply_web_search ------------------------------------------------------

    fn web_search_config(base_url: &str) -> WebSearchConfig {
        WebSearchConfig {
            api_key_env: "UNUSED".to_string(),
            base_url: base_url.to_string(),
            max_results: 5,
            timeout_secs: 5,
        }
    }

    fn request_wanting_web_search(text: &str) -> ChatRequest {
        let mut req = test_request("anthropic/m1");
        req.messages = vec![ChatMessage::user(text)];
        req.web_search = Some(true);
        req
    }

    #[tokio::test]
    async fn apply_web_search_is_a_noop_when_unconfigured() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let mut req = request_wanting_web_search("what's new in Rust");
        let before = req.messages[0].content.clone();
        router.apply_web_search(&mut req).await;
        assert_eq!(req.messages[0].content, before);
    }

    #[tokio::test]
    async fn apply_web_search_is_a_noop_when_not_requested() {
        let server = MockServer::start().await;
        // No mock mounted -- a call here would fail the test, proving
        // apply_web_search never even reaches the backend when
        // req.web_search isn't set.
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.web_search = Some(Arc::new(WebSearchClient::new(
            &web_search_config(&server.uri()),
            "test-key".to_string(),
        )));

        let mut req = test_request("anthropic/m1");
        req.messages = vec![ChatMessage::user("what's new in Rust")];
        let before = req.messages[0].content.clone();
        router.apply_web_search(&mut req).await;
        assert_eq!(req.messages[0].content, before);
    }

    #[tokio::test]
    async fn apply_web_search_prepends_results_to_the_last_user_message_and_records_a_metric() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/res/v1/web/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "web": {
                    "results": [
                        {"title": "Rust 1.80", "url": "https://blog.rust-lang.org", "description": "Release notes"}
                    ]
                }
            })))
            .mount(&server)
            .await;

        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.web_search = Some(Arc::new(WebSearchClient::new(
            &web_search_config(&format!("{}/res/v1/web/search", server.uri())),
            "test-key".to_string(),
        )));

        let mut req = request_wanting_web_search("what's new in Rust");
        router.apply_web_search(&mut req).await;

        match &req.messages[0].content {
            Some(MessageContent::Text(text)) => {
                assert!(text.contains("Rust 1.80"));
                assert!(text.contains("https://blog.rust-lang.org"));
                assert!(text.ends_with("what's new in Rust"));
            }
            other => panic!("expected Text content, got {other:?}"),
        }

        let metrics = router.render_prometheus_metrics();
        assert!(metrics.contains("rusty_provider_web_search_total"));
        assert!(metrics.contains(r#"outcome="results""#));
    }

    #[tokio::test]
    async fn apply_web_search_leaves_the_message_unchanged_when_there_are_no_results() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&server)
            .await;

        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.web_search = Some(Arc::new(WebSearchClient::new(
            &web_search_config(&server.uri()),
            "test-key".to_string(),
        )));

        let mut req = request_wanting_web_search("an obscure query");
        let before = req.messages[0].content.clone();
        router.apply_web_search(&mut req).await;
        assert_eq!(req.messages[0].content, before);

        let metrics = router.render_prometheus_metrics();
        assert!(metrics.contains(r#"outcome="no_results""#));
    }

    #[tokio::test]
    async fn apply_web_search_leaves_the_request_unmodified_when_the_backend_is_unreachable() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        // Port 0 never accepts connections -- a real, unrecoverable
        // network failure, not just a non-2xx response.
        router.web_search = Some(Arc::new(WebSearchClient::new(
            &web_search_config("http://127.0.0.1:0"),
            "test-key".to_string(),
        )));

        let mut req = request_wanting_web_search("what's new in Rust");
        let before = req.messages[0].content.clone();
        router.apply_web_search(&mut req).await;
        assert_eq!(
            req.messages[0].content, before,
            "a web-search-backend outage must leave the request unmodified, not error"
        );

        let metrics = router.render_prometheus_metrics();
        assert!(metrics.contains(r#"outcome="error""#));
    }

    // --- apply_preset --------------------------------------------------------

    fn preset_config(name: &str) -> PresetConfig {
        PresetConfig {
            name: name.to_string(),
            model: None,
            system_prompt: None,
            provider: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            top_k: None,
            min_p: None,
            top_a: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            logit_bias: None,
            seed: None,
        }
    }

    #[test]
    fn apply_preset_is_a_noop_when_unset() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let mut req = test_request("smart");
        req.preset = None;
        router.apply_preset(&mut req).unwrap();
        assert_eq!(req.model, "smart");
    }

    #[test]
    fn apply_preset_errors_for_an_unknown_name() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let mut req = test_request("smart");
        req.preset = Some("nope".to_string());
        let err = router.apply_preset(&mut req).unwrap_err();
        assert!(matches!(err, RouterError::UnknownPreset(name) if name == "nope"));
    }

    #[test]
    fn apply_preset_overrides_the_model_when_configured() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let mut cfg = preset_config("support-bot");
        cfg.model = Some("anthropic/claude-sonnet-5".to_string());
        router.presets.insert("support-bot".to_string(), cfg);

        let mut req = test_request("smart");
        req.preset = Some("support-bot".to_string());
        router.apply_preset(&mut req).unwrap();
        assert_eq!(req.model, "anthropic/claude-sonnet-5");
    }

    // --- resolve_target_model ("auto") ----------------------------------------

    fn auto_routing_config() -> AutoRoutingConfig {
        AutoRoutingConfig {
            simple_model: "openai/gpt-4o-mini".to_string(),
            medium_model: "smart".to_string(),
            complex_model: "anthropic/claude-opus-4-8".to_string(),
            simple_max_score: 20,
            medium_max_score: 80,
        }
    }

    #[test]
    fn resolve_target_model_passes_through_a_non_auto_model_unchanged() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let req = test_request("anthropic/claude-sonnet-5");
        assert_eq!(
            router.resolve_target_model(&req),
            "anthropic/claude-sonnet-5"
        );
    }

    #[test]
    fn resolve_target_model_passes_auto_through_unchanged_when_unconfigured() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let req = test_request("auto");
        assert_eq!(router.resolve_target_model(&req), "auto");
    }

    #[test]
    fn resolve_target_model_resolves_auto_to_the_simple_tier_for_a_short_prompt() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.auto_routing = Some(auto_routing_config());
        let req = test_request("auto");
        assert_eq!(router.resolve_target_model(&req), "openai/gpt-4o-mini");
    }

    #[test]
    fn resolve_target_model_resolves_auto_to_the_complex_tier_for_a_long_prompt() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.auto_routing = Some(auto_routing_config());
        let mut req = test_request("auto");
        req.messages = vec![ChatMessage::user("word ".repeat(1000))];
        assert_eq!(
            router.resolve_target_model(&req),
            "anthropic/claude-opus-4-8"
        );
    }

    // --- resolve_chain -----------------------------------------------------

    #[test]
    fn resolve_chain_direct_model_string() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let result = router
            .resolve_chain("anthropic/claude-sonnet-5", None)
            .unwrap();
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
        let result = router.resolve_chain("smart", None).unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
    }

    #[test]
    fn resolve_chain_rejects_model_without_a_slash() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let err = router.resolve_chain("not-a-valid-model", None).unwrap_err();
        assert!(matches!(err, RouterError::InvalidModel(_)));
    }

    #[test]
    fn resolve_chain_with_fallbacks_builds_an_ad_hoc_chain_from_model_plus_fallbacks() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let fallbacks = vec!["openai/m2".to_string(), "gemini/m3".to_string()];
        let result = router
            .resolve_chain("anthropic/m1", Some(&fallbacks))
            .unwrap();
        assert_eq!(
            result,
            chain(&[("anthropic", "m1"), ("openai", "m2"), ("gemini", "m3")])
        );
    }

    #[test]
    fn resolve_chain_with_fallbacks_bypasses_route_alias_lookup_for_model() {
        // "smart" is a configured alias, but a non-empty `fallbacks` takes
        // over entirely -- it's treated as a literal "provider/model", not
        // resolved through `[[routes]]`.
        let router = test_router(
            vec![],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            vec![],
        );
        let fallbacks = vec!["gemini/m3".to_string()];
        let err = router.resolve_chain("smart", Some(&fallbacks)).unwrap_err();
        assert!(matches!(err, RouterError::InvalidModel(_)));
    }

    #[test]
    fn resolve_chain_with_empty_fallbacks_falls_back_to_route_alias_lookup() {
        let router = test_router(
            vec![],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            vec![],
        );
        let empty: Vec<String> = vec![];
        let result = router.resolve_chain("smart", Some(&empty)).unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
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
    fn apply_preferences_ignore_alone_can_empty_the_chain_and_errors() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            ignore: Some(vec!["anthropic".to_string()]),
            ..Default::default()
        };
        let err = router
            .apply_preferences("smart", chain(&[("anthropic", "m1")]), Some(&prefs))
            .unwrap_err();
        assert!(matches!(err, RouterError::NoEligibleProvider(_)));
    }

    #[test]
    fn apply_preferences_only_and_ignore_naming_the_same_provider_empties_the_chain() {
        // "openai" survives `only` but is then dropped by `ignore` --
        // ignore is applied after only, so it wins even when they name the
        // exact same provider.
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            only: Some(vec!["openai".to_string()]),
            ignore: Some(vec!["openai".to_string()]),
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
    fn apply_preferences_only_and_ignore_combine_as_independent_successive_filters() {
        // `only` keeps {anthropic, openai}; `ignore` then separately drops
        // "openai". "gemini" was never eligible in the first place.
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            only: Some(vec!["anthropic".to_string(), "openai".to_string()]),
            ignore: Some(vec!["openai".to_string()]),
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
    fn apply_preferences_ignore_sort_price_applies_the_filter_before_sorting() {
        // "gemini" is the cheapest of the three but is dropped by `ignore`
        // before sort:"price" ever sees it.
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
            ignore: Some(vec!["gemini".to_string()]),
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
    fn apply_preferences_max_price_drops_candidates_priced_above_the_ceiling() {
        let router = test_router(
            vec![],
            vec![],
            vec![("anthropic/m1", 3.0, 15.0), ("gemini/m3", 0.1, 0.4)],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            max_price: Some(1.0),
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
    fn apply_preferences_max_price_drops_candidates_with_no_configured_price() {
        // Without a price on record the router can't verify the candidate
        // is under the ceiling, so an unpriced entry is dropped too rather
        // than let through unchecked.
        let router = test_router(
            vec![],
            vec![],
            vec![("gemini/m3", 0.1, 0.4)],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            max_price: Some(1.0),
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
    fn apply_preferences_max_price_below_every_candidate_errors() {
        let router = test_router(
            vec![],
            vec![],
            vec![("anthropic/m1", 3.0, 15.0)],
            vec![],
            vec![],
        );
        let prefs = ProviderPreferences {
            max_price: Some(1.0),
            ..Default::default()
        };
        let err = router
            .apply_preferences("smart", chain(&[("anthropic", "m1")]), Some(&prefs))
            .unwrap_err();
        assert!(matches!(err, RouterError::NoEligibleProvider(_)));
    }

    #[test]
    fn apply_preferences_max_price_applies_before_sort_price() {
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
            max_price: Some(1.0),
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
        assert_eq!(result, chain(&[("gemini", "m3"), ("openai", "m2")]));
    }

    #[test]
    fn apply_preferences_ignore_sort_latency_applies_the_filter_before_sorting() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut latency = router.latency.write().unwrap();
            latency.insert("anthropic/m1".to_string(), 2000.0);
            latency.insert("openai/m2".to_string(), 500.0);
            // Fastest of all three, but dropped by `ignore` first.
            latency.insert("gemini/m3".to_string(), 10.0);
        }
        let prefs = ProviderPreferences {
            ignore: Some(vec!["gemini".to_string()]),
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
    fn apply_preferences_ignore_sort_throughput_applies_the_filter_before_sorting() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut throughput = router.throughput.write().unwrap();
            throughput.insert("anthropic/m1".to_string(), 20.0);
            throughput.insert("openai/m2".to_string(), 80.0);
            // Fastest of all three, but dropped by `ignore` first.
            throughput.insert("gemini/m3".to_string(), 500.0);
        }
        let prefs = ProviderPreferences {
            ignore: Some(vec!["gemini".to_string()]),
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

    #[test]
    fn apply_preferences_only_and_ignore_together_then_sort_by_price() {
        // `only` narrows to {anthropic, openai, gemini}; `ignore` then
        // drops "gemini" (the cheapest) before price sort runs on the rest.
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
            only: Some(vec![
                "anthropic".to_string(),
                "openai".to_string(),
                "gemini".to_string(),
            ]),
            ignore: Some(vec!["gemini".to_string()]),
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
    fn apply_preferences_zdr_filters_before_latency_sort() {
        // The fastest candidate ("gemini") isn't ZDR-flagged and must be
        // dropped before latency sorting ever sees it, not merely sorted
        // last among the full chain.
        let router = test_router(vec![], vec![], vec![], vec!["anthropic", "openai"], vec![]);
        {
            let mut latency = router.latency.write().unwrap();
            latency.insert("anthropic/m1".to_string(), 2000.0);
            latency.insert("openai/m2".to_string(), 500.0);
            latency.insert("gemini/m3".to_string(), 10.0);
        }
        let prefs = ProviderPreferences {
            zdr: Some(true),
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
    fn apply_preferences_zdr_and_latency_sort_unobserved_survivor_sorts_last() {
        // Both "anthropic" and "openai" are ZDR-flagged and survive the
        // filter; only "anthropic" has an observed latency. The unobserved
        // ZDR survivor must still sort last among the *filtered* set, not
        // be compared against the non-ZDR "gemini" that was dropped.
        let router = test_router(vec![], vec![], vec![], vec!["anthropic", "openai"], vec![]);
        router
            .latency
            .write()
            .unwrap()
            .insert("anthropic/m1".to_string(), 500.0);
        let prefs = ProviderPreferences {
            zdr: Some(true),
            sort: Some("latency".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("openai", "m2"), ("anthropic", "m1"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
    }

    #[test]
    fn apply_preferences_zdr_filters_before_throughput_sort() {
        // "gemini" has the highest throughput of the three but isn't
        // ZDR-flagged and must be dropped before throughput sorting ever
        // sees it, not merely sorted last among the full chain.
        let router = test_router(vec![], vec![], vec![], vec!["anthropic", "openai"], vec![]);
        {
            let mut throughput = router.throughput.write().unwrap();
            throughput.insert("anthropic/m1".to_string(), 20.0);
            throughput.insert("openai/m2".to_string(), 80.0);
            throughput.insert("gemini/m3".to_string(), 500.0);
        }
        let prefs = ProviderPreferences {
            zdr: Some(true),
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

    #[test]
    fn apply_preferences_zdr_and_throughput_sort_unobserved_survivor_sorts_last() {
        // Both "anthropic" and "openai" are ZDR-flagged and survive the
        // filter; only "anthropic" has an observed throughput. The
        // unobserved ZDR survivor must still sort last among the
        // *filtered* set, not be compared against the non-ZDR "gemini"
        // that was dropped.
        let router = test_router(vec![], vec![], vec![], vec!["anthropic", "openai"], vec![]);
        router
            .throughput
            .write()
            .unwrap()
            .insert("anthropic/m1".to_string(), 50.0);
        let prefs = ProviderPreferences {
            zdr: Some(true),
            sort: Some("throughput".to_string()),
            ..Default::default()
        };
        let result = router
            .apply_preferences(
                "smart",
                chain(&[("openai", "m2"), ("anthropic", "m1"), ("gemini", "m3")]),
                Some(&prefs),
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
    }

    #[test]
    fn apply_preferences_data_collection_filters_to_flagged_providers_only() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.no_training_providers = HashSet::from(["anthropic".to_string()]);
        let prefs = ProviderPreferences {
            data_collection: Some(true),
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
    fn apply_preferences_data_collection_false_is_a_no_op() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.no_training_providers = HashSet::from(["anthropic".to_string()]);
        let prefs = ProviderPreferences {
            data_collection: Some(false),
            ..Default::default()
        };
        let input = chain(&[("anthropic", "m1"), ("openai", "m2")]);
        let result = router
            .apply_preferences("smart", input.clone(), Some(&prefs))
            .unwrap();
        assert_eq!(
            result, input,
            "data_collection: false must not filter out non-no_training providers"
        );
    }

    #[test]
    fn apply_preferences_data_collection_unset_is_a_no_op() {
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.no_training_providers = HashSet::from(["anthropic".to_string()]);
        // `data_collection` left unset within an otherwise-present
        // preferences object, as opposed to `prefs` being `None` entirely.
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
    fn apply_preferences_data_collection_with_no_flagged_providers_empties_the_chain_and_errors() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        let prefs = ProviderPreferences {
            data_collection: Some(true),
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
    fn apply_preferences_data_collection_is_independent_of_zdr() {
        // "anthropic" is zdr-flagged but not no_training-flagged; "openai"
        // is the reverse. Requiring both must drop everything, since no
        // single provider satisfies both axes here.
        let mut router = test_router(vec![], vec![], vec![], vec!["anthropic"], vec![]);
        router.no_training_providers = HashSet::from(["openai".to_string()]);
        let prefs = ProviderPreferences {
            zdr: Some(true),
            data_collection: Some(true),
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
    fn apply_preferences_data_collection_combines_with_zdr_when_both_satisfied() {
        // "anthropic" satisfies both axes and must survive; "openai"
        // satisfies neither.
        let mut router = test_router(vec![], vec![], vec![], vec!["anthropic"], vec![]);
        router.no_training_providers = HashSet::from(["anthropic".to_string()]);
        let prefs = ProviderPreferences {
            zdr: Some(true),
            data_collection: Some(true),
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
    fn apply_preferences_data_collection_combines_with_only_filter() {
        // "openai" passes `only` but isn't no_training-flagged, so it must
        // still be dropped -- the two filters are independent, not either/or.
        let mut router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router.no_training_providers = HashSet::from(["anthropic".to_string()]);
        let prefs = ProviderPreferences {
            only: Some(vec!["anthropic".to_string(), "openai".to_string()]),
            data_collection: Some(true),
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
    fn apply_preferences_data_collection_filters_before_price_sort() {
        // The cheapest candidate ("gemini") isn't no_training-flagged and
        // must be dropped before price sorting ever sees it, not merely
        // sorted last.
        let mut router = test_router(
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
        router.no_training_providers =
            HashSet::from(["anthropic".to_string(), "openai".to_string()]);
        let prefs = ProviderPreferences {
            data_collection: Some(true),
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

    // --- uptime sort -----------------------------------------------------

    #[test]
    fn apply_preferences_sorts_descending_by_uptime() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        {
            let mut uptime = router.uptime.write().unwrap();
            uptime.insert("anthropic/m1".to_string(), 0.5);
            uptime.insert("openai/m2".to_string(), 0.95);
        }
        let prefs = ProviderPreferences {
            sort: Some("uptime".to_string()),
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
    fn apply_preferences_unobserved_uptime_sorts_last() {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        router
            .uptime
            .write()
            .unwrap()
            .insert("anthropic/m1".to_string(), 0.8);
        // "openai/m2" has no observed uptime -- despite being first in the
        // chain, it should sort after the entry with a real observation,
        // not be assumed healthy.
        let prefs = ProviderPreferences {
            sort: Some("uptime".to_string()),
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

    #[tokio::test]
    async fn dispatch_records_uptime_of_one_on_success() {
        let calls = Arc::new(AtomicUsize::new(0));
        let succeeding = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls,
        });
        let router = test_router(
            vec![("anthropic", succeeding)],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        router
            .dispatch(&test_request("anthropic/claude-sonnet-5"))
            .await
            .expect("dispatch should succeed");
        assert_eq!(
            *router
                .uptime
                .read()
                .unwrap()
                .get("anthropic/claude-sonnet-5")
                .unwrap(),
            1.0
        );
    }

    #[tokio::test]
    async fn dispatch_records_uptime_of_zero_on_retryable_failure_and_falls_through() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let failing = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::FailRetryable,
            calls: calls_a,
        });
        let succeeding = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls_b,
        });
        let router = test_router(
            vec![("anthropic", failing), ("openai", succeeding)],
            vec![("smart", vec!["anthropic/m1", "openai/m2"])],
            vec![],
            vec![],
            vec![],
        );
        router
            .dispatch(&test_request("smart"))
            .await
            .expect("should fall through to openai");
        let uptime = router.uptime.read().unwrap();
        assert_eq!(*uptime.get("anthropic/m1").unwrap(), 0.0);
        assert_eq!(*uptime.get("openai/m2").unwrap(), 1.0);
    }

    // --- middle-out --------------------------------------------------------

    fn msg(role_text: &str, chars: usize) -> ChatMessage {
        let text = "x".repeat(chars);
        if role_text == "system" {
            ChatMessage::system(text)
        } else {
            ChatMessage::user(text)
        }
    }

    #[test]
    fn apply_middle_out_is_a_no_op_within_budget() {
        let messages = vec![msg("system", 40), msg("user", 40), msg("user", 40)];
        let result = apply_middle_out(&messages, 1000);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn apply_middle_out_keeps_first_and_last_and_drops_the_middle() {
        // Each message is ~10 estimated tokens (40 chars / 4). A budget of
        // 15 forces dropping the middle two, leaving just system + last.
        let messages = vec![
            msg("system", 40),
            msg("user", 40),
            msg("user", 40),
            msg("user", 40),
        ];
        let result = apply_middle_out(&messages, 15);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, messages[0].content);
        assert_eq!(result[1].content, messages[3].content);
    }

    #[test]
    fn apply_middle_out_never_drops_below_two_messages() {
        // Even an impossibly small budget leaves the first and last
        // message in place rather than dropping everything.
        let messages = vec![msg("system", 40), msg("user", 40), msg("user", 40)];
        let result = apply_middle_out(&messages, 0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn apply_middle_out_drops_oldest_middle_message_first() {
        let messages = vec![
            msg("system", 40),
            msg("user", 40), // oldest middle -- should go first
            msg("user", 40), // newest middle -- should survive longer
            msg("user", 40),
        ];
        // Budget for 3 messages' worth (~30 tokens) -- only one middle
        // message needs to be dropped.
        let result = apply_middle_out(&messages, 30);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, messages[0].content);
        assert_eq!(result[1].content, messages[2].content);
        assert_eq!(result[2].content, messages[3].content);
    }

    #[test]
    fn maybe_apply_middle_out_is_none_when_transform_not_requested() {
        let mut req = test_request("anthropic/claude-sonnet-5");
        req.messages = vec![msg("system", 4000), msg("user", 4000)];
        let mut pricing = HashMap::new();
        pricing.insert(
            "anthropic/claude-sonnet-5".to_string(),
            PriceRates {
                prompt_ppm: 3.0,
                completion_ppm: 15.0,
                cache_read_ppm: 3.0,
                cache_write_ppm: 3.0,
                context_length: Some(100),
            },
        );
        assert!(maybe_apply_middle_out(&req, "anthropic", "claude-sonnet-5", &pricing).is_none());
    }

    #[test]
    fn maybe_apply_middle_out_is_none_without_a_known_context_length() {
        let mut req = test_request("anthropic/claude-sonnet-5");
        req.transforms = Some(vec!["middle-out".to_string()]);
        req.messages = vec![msg("system", 4000), msg("user", 4000)];
        // No [[pricing]] entry at all for this candidate.
        let pricing = HashMap::new();
        assert!(maybe_apply_middle_out(&req, "anthropic", "claude-sonnet-5", &pricing).is_none());
    }

    #[test]
    fn maybe_apply_middle_out_is_none_when_already_within_budget() {
        let mut req = test_request("anthropic/claude-sonnet-5");
        req.transforms = Some(vec!["middle-out".to_string()]);
        req.max_tokens = Some(100);
        req.messages = vec![msg("system", 40), msg("user", 40)];
        let mut pricing = HashMap::new();
        pricing.insert(
            "anthropic/claude-sonnet-5".to_string(),
            PriceRates {
                prompt_ppm: 3.0,
                completion_ppm: 15.0,
                cache_read_ppm: 3.0,
                cache_write_ppm: 3.0,
                context_length: Some(200_000),
            },
        );
        assert!(maybe_apply_middle_out(&req, "anthropic", "claude-sonnet-5", &pricing).is_none());
    }

    #[test]
    fn maybe_apply_middle_out_truncates_an_over_length_request() {
        let mut req = test_request("anthropic/claude-sonnet-5");
        req.transforms = Some(vec!["middle-out".to_string()]);
        req.max_tokens = Some(100);
        req.messages = vec![
            msg("system", 40),
            msg("user", 4000),
            msg("user", 4000),
            msg("user", 40),
        ];
        let mut pricing = HashMap::new();
        pricing.insert(
            "anthropic/claude-sonnet-5".to_string(),
            PriceRates {
                prompt_ppm: 3.0,
                completion_ppm: 15.0,
                cache_read_ppm: 3.0,
                cache_write_ppm: 3.0,
                // Budget after reserving max_tokens is small relative to
                // the ~2000-estimated-token middle messages.
                context_length: Some(300),
            },
        );
        let truncated = maybe_apply_middle_out(&req, "anthropic", "claude-sonnet-5", &pricing)
            .expect("should truncate");
        assert!(truncated.messages.len() < req.messages.len());
        assert_eq!(truncated.messages[0].content, req.messages[0].content);
        assert_eq!(
            truncated.messages.last().unwrap().content,
            req.messages.last().unwrap().content
        );
    }

    #[tokio::test]
    async fn dispatch_sends_a_middle_out_truncated_request_body() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let mut router = test_router(vec![("anthropic", mock)], vec![], vec![], vec![], vec![]);
        router.pricing = Arc::new(HashMap::from([(
            "anthropic/claude-sonnet-5".to_string(),
            PriceRates {
                prompt_ppm: 3.0,
                completion_ppm: 15.0,
                cache_read_ppm: 3.0,
                cache_write_ppm: 3.0,
                context_length: Some(300),
            },
        )]));
        let mut req = test_request("anthropic/claude-sonnet-5");
        req.transforms = Some(vec!["middle-out".to_string()]);
        req.max_tokens = Some(100);
        req.messages = vec![
            msg("system", 40),
            msg("user", 4000),
            msg("user", 4000),
            msg("user", 40),
        ];

        router
            .dispatch(&req)
            .await
            .expect("dispatch should succeed even though the request was truncated");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    // --- active_params / filter_by_required_parameters ------------------------

    fn router_with_provider_kinds(kinds: Vec<(&str, ProviderKind)>) -> Router {
        let router = test_router(vec![], vec![], vec![], vec![], vec![]);
        Router {
            provider_kinds: kinds.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            ..router
        }
    }

    #[test]
    fn active_params_is_empty_for_a_bare_request() {
        let req = test_request("anthropic/claude-sonnet-5");
        assert!(active_params(&req).is_empty());
    }

    #[test]
    fn active_params_lists_every_set_field() {
        let mut req = test_request("anthropic/claude-sonnet-5");
        req.tools = Some(vec![]);
        req.tool_choice = Some(json!("auto"));
        req.response_format = Some(rp_core::ResponseFormat::JsonObject);
        req.top_k = Some(40);
        req.min_p = Some(0.1);
        req.top_a = Some(0.1);
        req.frequency_penalty = Some(0.1);
        req.presence_penalty = Some(0.1);
        req.repetition_penalty = Some(1.1);
        req.logit_bias = Some(HashMap::from([("1".to_string(), -1.0)]));
        req.seed = Some(1);
        req.logprobs = Some(true);
        let params = active_params(&req);
        for expected in [
            "tools",
            "tool_choice",
            "response_format",
            "top_k",
            "min_p",
            "top_a",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "logit_bias",
            "seed",
            "logprobs",
        ] {
            assert!(params.contains(&expected), "missing {expected}");
        }
        assert!(!params.contains(&"reasoning"));
        assert!(!params.contains(&"cache_control"));
    }

    #[test]
    fn active_params_detects_cache_control_on_any_message() {
        let mut req = test_request("anthropic/claude-sonnet-5");
        req.messages[0].cache_control = Some(rp_core::CacheControl::Ephemeral);
        assert_eq!(active_params(&req), vec!["cache_control"]);
    }

    #[test]
    fn filter_by_required_parameters_is_a_no_op_when_unset() {
        let router = router_with_provider_kinds(vec![("anthropic", ProviderKind::Anthropic)]);
        let mut req = test_request("smart");
        req.response_format = Some(rp_core::ResponseFormat::JsonObject);
        let result = router
            .filter_by_required_parameters(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                &req,
            )
            .unwrap();
        assert_eq!(result, chain(&[("anthropic", "m1"), ("openai", "m2")]));
    }

    #[test]
    fn filter_by_required_parameters_drops_a_candidate_missing_an_active_field() {
        // logit_bias is native only to OpenAI-compatible; Anthropic has no
        // field for it at all.
        let router = router_with_provider_kinds(vec![
            ("anthropic", ProviderKind::Anthropic),
            ("openai", ProviderKind::Openai),
        ]);
        let mut req = test_request("smart");
        req.logit_bias = Some(HashMap::from([("1".to_string(), -1.0)]));
        req.provider = Some(ProviderPreferences {
            require_parameters: Some(true),
            ..Default::default()
        });
        let result = router
            .filter_by_required_parameters(
                "smart",
                chain(&[("anthropic", "m1"), ("openai", "m2")]),
                &req,
            )
            .unwrap();
        assert_eq!(result, chain(&[("openai", "m2")]));
    }

    #[test]
    fn filter_by_required_parameters_keeps_a_candidate_supporting_every_active_field() {
        let router = router_with_provider_kinds(vec![("gemini", ProviderKind::Gemini)]);
        let mut req = test_request("smart");
        req.top_k = Some(40);
        req.seed = Some(1);
        req.provider = Some(ProviderPreferences {
            require_parameters: Some(true),
            ..Default::default()
        });
        let result = router
            .filter_by_required_parameters("smart", chain(&[("gemini", "m1")]), &req)
            .unwrap();
        assert_eq!(result, chain(&[("gemini", "m1")]));
    }

    #[test]
    fn filter_by_required_parameters_errors_when_nothing_survives() {
        let router = router_with_provider_kinds(vec![("anthropic", ProviderKind::Anthropic)]);
        let mut req = test_request("smart");
        req.logit_bias = Some(HashMap::from([("1".to_string(), -1.0)]));
        req.provider = Some(ProviderPreferences {
            require_parameters: Some(true),
            ..Default::default()
        });
        let err = router
            .filter_by_required_parameters("smart", chain(&[("anthropic", "m1")]), &req)
            .unwrap_err();
        assert!(matches!(err, RouterError::NoEligibleProvider(_)));
    }

    #[test]
    fn filter_by_required_parameters_drops_a_candidate_with_no_resolved_kind() {
        // "anthropic" survived apply_preferences (e.g. from a route alias
        // config), but this process never actually configured it, so
        // there's no kind on record -- dropped defensively rather than
        // assumed compatible.
        let router = router_with_provider_kinds(vec![]);
        let mut req = test_request("smart");
        req.top_k = Some(40);
        req.provider = Some(ProviderPreferences {
            require_parameters: Some(true),
            ..Default::default()
        });
        let err = router
            .filter_by_required_parameters("smart", chain(&[("anthropic", "m1")]), &req)
            .unwrap_err();
        assert!(matches!(err, RouterError::NoEligibleProvider(_)));
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
            cached_tokens: None,
            cache_creation_tokens: None,
        }
    }

    /// Flat pricing with no cache discount/premium -- cache reads and
    /// writes both cost the same as a normal prompt token, matching what
    /// an operator gets by leaving `cache_read_per_million`/
    /// `cache_write_per_million` unset in `[[pricing]]`.
    fn flat_rates(prompt_ppm: f64, completion_ppm: f64) -> PriceRates {
        PriceRates {
            prompt_ppm,
            completion_ppm,
            cache_read_ppm: prompt_ppm,
            cache_write_ppm: prompt_ppm,
            context_length: None,
        }
    }

    #[test]
    fn record_usage_computes_cost_from_per_million_token_pricing() {
        let usage_map = RwLock::new(HashMap::new());
        let mut pricing = HashMap::new();
        // $2/1M prompt tokens, $10/1M completion tokens.
        pricing.insert("anthropic/m1".to_string(), flat_rates(2.0, 10.0));

        let cost = record_usage(
            &usage_map,
            None,
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

        let cost = record_usage(
            &usage_map,
            None,
            &pricing,
            "anthropic",
            "m1",
            &usage(100, 50),
        );

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
        pricing.insert("anthropic/m1".to_string(), flat_rates(2.0, 10.0));

        let cost = record_usage(&usage_map, None, &pricing, "anthropic", "m1", &usage(0, 0));

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
        pricing.insert("anthropic/m1".to_string(), flat_rates(1.0, 1.0));

        record_usage(
            &usage_map,
            None,
            &pricing,
            "anthropic",
            "m1",
            &usage(100, 50),
        );
        record_usage(
            &usage_map,
            None,
            &pricing,
            "anthropic",
            "m1",
            &usage(200, 100),
        );

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
        pricing.insert("anthropic/m1".to_string(), flat_rates(1.0, 1.0));
        pricing.insert("openai/m2".to_string(), flat_rates(5.0, 5.0));

        record_usage(
            &usage_map,
            None,
            &pricing,
            "anthropic",
            "m1",
            &usage(100, 0),
        );
        record_usage(&usage_map, None, &pricing, "openai", "m2", &usage(200, 0));

        let stats = usage_map.read().unwrap();
        assert_eq!(stats["anthropic/m1"].requests, 1);
        assert_eq!(stats["anthropic/m1"].prompt_tokens, 100);
        assert_eq!(stats["openai/m2"].requests, 1);
        assert_eq!(stats["openai/m2"].prompt_tokens, 200);
    }

    // --- PriceRates::from(&PricingEntry) ----------------------------------------

    #[test]
    fn price_rates_from_pricing_entry_defaults_cache_rates_to_prompt_price() {
        let entry = PricingEntry {
            model: "anthropic/m1".to_string(),
            prompt_per_million: 3.0,
            completion_per_million: 15.0,
            cache_read_per_million: None,
            cache_write_per_million: None,
            context_length: None,
        };
        let rates = PriceRates::from(&entry);
        assert_eq!(rates.cache_read_ppm, 3.0);
        assert_eq!(rates.cache_write_ppm, 3.0);
    }

    #[test]
    fn price_rates_from_pricing_entry_honors_explicit_cache_rates() {
        let entry = PricingEntry {
            model: "anthropic/m1".to_string(),
            prompt_per_million: 3.0,
            completion_per_million: 15.0,
            cache_read_per_million: Some(0.3),
            cache_write_per_million: Some(3.75),
            context_length: None,
        };
        let rates = PriceRates::from(&entry);
        assert_eq!(rates.cache_read_ppm, 0.3);
        assert_eq!(rates.cache_write_ppm, 3.75);
    }

    // --- record_usage: cache-aware cost -----------------------------------------

    fn usage_with_cache(
        prompt_tokens: u32,
        completion_tokens: u32,
        cached: u32,
        cache_creation: u32,
    ) -> Usage {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cached_tokens: (cached > 0).then_some(cached),
            cache_creation_tokens: (cache_creation > 0).then_some(cache_creation),
        }
    }

    #[test]
    fn record_usage_prices_cached_tokens_at_the_discounted_rate() {
        let usage_map = RwLock::new(HashMap::new());
        let mut pricing = HashMap::new();
        pricing.insert(
            "anthropic/m1".to_string(),
            PriceRates {
                prompt_ppm: 3.0,
                completion_ppm: 15.0,
                cache_read_ppm: 0.3,
                cache_write_ppm: 3.75,
                context_length: None,
            },
        );

        // 1000 total prompt tokens: 200 fresh, 800 cached (read), 0 written.
        let cost = record_usage(
            &usage_map,
            None,
            &pricing,
            "anthropic",
            "m1",
            &usage_with_cache(1000, 100, 800, 0),
        );

        let expected = (200.0 * 3.0 + 800.0 * 0.3 + 100.0 * 15.0) / 1_000_000.0;
        assert!((cost.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn record_usage_prices_cache_write_tokens_at_the_premium_rate() {
        let usage_map = RwLock::new(HashMap::new());
        let mut pricing = HashMap::new();
        pricing.insert(
            "anthropic/m1".to_string(),
            PriceRates {
                prompt_ppm: 3.0,
                completion_ppm: 15.0,
                cache_read_ppm: 0.3,
                cache_write_ppm: 3.75,
                context_length: None,
            },
        );

        // 1000 total prompt tokens: 500 fresh, 0 read, 500 written.
        let cost = record_usage(
            &usage_map,
            None,
            &pricing,
            "anthropic",
            "m1",
            &usage_with_cache(1000, 0, 0, 500),
        );

        let expected = (500.0 * 3.0 + 500.0 * 3.75) / 1_000_000.0;
        assert!((cost.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn record_usage_with_no_cache_tokens_matches_flat_prompt_pricing() {
        let usage_map = RwLock::new(HashMap::new());
        let mut pricing = HashMap::new();
        pricing.insert("anthropic/m1".to_string(), flat_rates(3.0, 15.0));

        let cost = record_usage(
            &usage_map,
            None,
            &pricing,
            "anthropic",
            "m1",
            &usage_with_cache(1000, 100, 0, 0),
        );

        let expected = (1000.0 * 3.0 + 100.0 * 15.0) / 1_000_000.0;
        assert!((cost.unwrap() - expected).abs() < 1e-12);
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

        let snapshot = router.usage_snapshot().await;
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
        let snapshot = router.usage_snapshot().await;
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

        let snapshot = router.usage_snapshot().await;
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(stats.requests, 3);
        assert_eq!(stats.prompt_tokens, 3);
        assert_eq!(stats.completion_tokens, 3);
    }

    // --- cache ---------------------------------------------------------------

    fn router_with_cache(
        providers: Vec<(&str, Arc<dyn Provider>)>,
        ttl_secs: u64,
        max_entries: usize,
    ) -> Router {
        let router = test_router(providers, vec![], vec![], vec![], vec![]);
        Router {
            cache: Some(Arc::new(RwLock::new(ResponseCache::new(&CacheConfig {
                ttl_secs,
                max_entries,
            })))),
            ..router
        }
    }

    #[tokio::test]
    async fn dispatch_a_second_identical_request_is_served_from_cache() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = router_with_cache(vec![("anthropic", mock)], 60, 10);
        let req = test_request("anthropic/m1");

        let first = router.dispatch(&req).await.expect("should succeed");
        let second = router.dispatch(&req).await.expect("should succeed");

        assert_eq!(first.id, second.id);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the provider should only be called once -- the second dispatch should hit cache"
        );
    }

    #[tokio::test]
    async fn dispatch_a_different_request_is_not_served_from_cache() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = router_with_cache(vec![("anthropic", mock)], 60, 10);

        router
            .dispatch(&test_request("anthropic/m1"))
            .await
            .unwrap();
        router
            .dispatch(&test_request("anthropic/m2"))
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn dispatch_without_cache_configured_always_calls_the_provider() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        // test_router's default -- no [cache] configured at all.
        let router = test_router(vec![("anthropic", mock)], vec![], vec![], vec![], vec![]);
        let req = test_request("anthropic/m1");

        router.dispatch(&req).await.unwrap();
        router.dispatch(&req).await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn dispatch_bypasses_cache_for_a_streaming_request() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = router_with_cache(vec![("anthropic", mock)], 60, 10);
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);

        // dispatch (not dispatch_stream) still runs to completion even
        // for a `stream: true` request -- this only proves cache_key_for
        // itself returns None for one, not that dispatch refuses it.
        router.dispatch(&req).await.unwrap();
        router.dispatch(&req).await.unwrap();

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "a streaming request should never be served from or written to cache"
        );
    }

    #[tokio::test]
    async fn dispatch_a_cache_hit_does_not_double_count_usage_snapshot() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "anthropic".to_string(),
            behavior: MockBehavior::Succeed,
            calls,
        });
        let router = router_with_cache(vec![("anthropic", mock)], 60, 10);
        let req = test_request("anthropic/m1");

        router.dispatch(&req).await.unwrap();
        router.dispatch(&req).await.unwrap();
        router.dispatch(&req).await.unwrap();

        // Only the first (cache-miss) dispatch actually recorded usage --
        // a cache hit replays the stored response without re-running any
        // of dispatch's usage/cost/latency/throughput bookkeeping, so
        // this doesn't triple-count a single generation's tokens/cost.
        let snapshot = router.usage_snapshot().await;
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(stats.requests, 1);
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

        let snapshot = router.usage_snapshot().await;
        let stats = &snapshot["anthropic/m1"];
        assert_eq!(stats.requests, 1);
        assert!((stats.cost_usd - expected_cost).abs() < 1e-12);
    }

    // --- BYOK (provider.byok) -------------------------------------------------

    #[tokio::test]
    async fn dispatch_passes_the_matching_byok_key_to_the_provider() {
        let provider = Arc::new(KeyCapturingProvider {
            name: "anthropic".to_string(),
            seen_key: Mutex::new(None),
        });
        let router = test_router(
            vec![("anthropic", provider.clone())],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.provider = Some(rp_core::ProviderPreferences {
            byok: Some(HashMap::from([(
                "anthropic".to_string(),
                "sk-byok-key".to_string(),
            )])),
            ..Default::default()
        });

        router
            .dispatch(&req)
            .await
            .expect("dispatch should succeed");

        assert_eq!(
            provider.seen_key.lock().unwrap().as_deref(),
            Some("sk-byok-key")
        );
    }

    #[tokio::test]
    async fn dispatch_passes_none_when_byok_has_no_entry_for_the_dispatched_provider() {
        let provider = Arc::new(KeyCapturingProvider {
            name: "anthropic".to_string(),
            seen_key: Mutex::new(None),
        });
        let router = test_router(
            vec![("anthropic", provider.clone())],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.provider = Some(rp_core::ProviderPreferences {
            byok: Some(HashMap::from([(
                "openai".to_string(),
                "sk-someone-elses-key".to_string(),
            )])),
            ..Default::default()
        });

        router
            .dispatch(&req)
            .await
            .expect("dispatch should succeed");

        assert_eq!(*provider.seen_key.lock().unwrap(), None);
    }

    #[tokio::test]
    async fn dispatch_passes_none_when_byok_is_unset() {
        let provider = Arc::new(KeyCapturingProvider {
            name: "anthropic".to_string(),
            seen_key: Mutex::new(None),
        });
        let router = test_router(
            vec![("anthropic", provider.clone())],
            vec![],
            vec![],
            vec![],
            vec![],
        );

        router
            .dispatch(&test_request("anthropic/m1"))
            .await
            .expect("dispatch should succeed");

        assert_eq!(*provider.seen_key.lock().unwrap(), None);
    }

    #[tokio::test]
    async fn dispatch_stream_passes_the_matching_byok_key_to_the_provider() {
        let provider = Arc::new(KeyCapturingProvider {
            name: "anthropic".to_string(),
            seen_key: Mutex::new(None),
        });
        let router = test_router(
            vec![("anthropic", provider.clone())],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.stream = Some(true);
        req.provider = Some(rp_core::ProviderPreferences {
            byok: Some(HashMap::from([(
                "anthropic".to_string(),
                "sk-byok-key".to_string(),
            )])),
            ..Default::default()
        });

        let _stream = router
            .dispatch_stream(&req)
            .await
            .expect("dispatch_stream should succeed");

        assert_eq!(
            provider.seen_key.lock().unwrap().as_deref(),
            Some("sk-byok-key")
        );
    }

    // --- embeddings ------------------------------------------------------------

    fn embeddings_request(model: &str, text: &str) -> EmbeddingsRequest {
        EmbeddingsRequest {
            model: model.to_string(),
            input: rp_core::EmbeddingsInput::Single(text.to_string()),
            encoding_format: None,
            dimensions: None,
        }
    }

    #[tokio::test]
    async fn embeddings_returns_success_from_the_first_provider() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = test_router(vec![("openai", mock)], vec![], vec![], vec![], vec![]);

        let resp = router
            .embeddings(&embeddings_request("openai/text-embedding-3-small", "hi"))
            .await
            .expect("embeddings should succeed");

        assert_eq!(resp.model, "openai/text-embedding-3-small");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn embeddings_resolves_a_configured_route_alias_like_dispatch_does() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = test_router(
            vec![("openai", mock)],
            vec![("embed", vec!["openai/text-embedding-3-small"])],
            vec![],
            vec![],
            vec![],
        );

        let resp = router
            .embeddings(&embeddings_request("embed", "hi"))
            .await
            .expect("embeddings should succeed");

        assert_eq!(resp.model, "openai/text-embedding-3-small");
    }

    #[tokio::test]
    async fn embeddings_falls_back_to_the_next_candidate_on_a_retryable_error() {
        // Anthropic has no embeddings API -- UnsupportedFeature is
        // retryable, so a chain naming it alongside a real embeddings
        // provider should fall through rather than failing outright.
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let unsupported = Arc::new(MockProvider {
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
            vec![("anthropic", unsupported), ("openai", succeeding)],
            vec![(
                "embed",
                vec!["anthropic/claude-sonnet-5", "openai/text-embedding-3-small"],
            )],
            vec![],
            vec![],
            vec![],
        );

        let resp = router
            .embeddings(&embeddings_request("embed", "hi"))
            .await
            .expect("should fall through to openai");

        assert_eq!(resp.model, "openai/text-embedding-3-small");
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn embeddings_aborts_immediately_on_a_fatal_error() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let failing = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::FailFatal,
            calls: calls_a.clone(),
        });
        let never_called = Arc::new(MockProvider {
            name: "gemini".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls_b.clone(),
        });
        let router = test_router(
            vec![("openai", failing), ("gemini", never_called)],
            vec![(
                "embed",
                vec!["openai/text-embedding-3-small", "gemini/text-embedding-004"],
            )],
            vec![],
            vec![],
            vec![],
        );

        let err = router
            .embeddings(&embeddings_request("embed", "hi"))
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::InvalidRequest(_))
        ));
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn embeddings_skips_a_chain_entry_with_no_registered_provider() {
        let calls = Arc::new(AtomicUsize::new(0));
        let configured = Arc::new(MockProvider {
            name: "openai".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls.clone(),
        });
        let router = test_router(
            vec![("openai", configured)],
            vec![(
                "embed",
                vec!["anthropic/m1", "openai/text-embedding-3-small"],
            )],
            vec![],
            vec![],
            vec![],
        );

        let resp = router
            .embeddings(&embeddings_request("embed", "hi"))
            .await
            .expect("should fall through to the configured provider");

        assert_eq!(resp.model, "openai/text-embedding-3-small");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
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
            !router.usage_snapshot().await.contains_key("anthropic/m1"),
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

        let snapshot = router.usage_snapshot().await;
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
        assert!(!router.usage_snapshot().await.contains_key("anthropic/m1"));
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
    async fn dispatch_falls_back_across_an_ad_hoc_models_list_with_no_configured_route() {
        // No `[[routes]]` alias at all -- the chain comes entirely from
        // `model` + `req.models`, proving the ad-hoc list is honored even
        // when the operator never predefined a fallback chain for it.
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
            vec![],
            vec![],
            vec![],
            vec![],
        );
        let mut req = test_request("anthropic/m1");
        req.models = Some(vec!["openai/m2".to_string()]);

        let resp = router
            .dispatch(&req)
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

    // --- retry/fallback chain exhaustion ----------------------------------------

    #[test]
    fn dispatch_chain_resolution_returns_invalid_model_for_an_alias_with_an_empty_chain() {
        // A route alias configured with no entries at all -- resolve_chain
        // succeeds with an empty Vec (nothing to reject syntactically), and
        // with no request-side `provider` preferences to short-circuit on,
        // apply_preferences passes the empty chain straight through. Only
        // dispatch's loop-then-fallback-to-InvalidModel actually catches it.
        let router = test_router(vec![], vec![("smart", vec![])], vec![], vec![], vec![]);
        let chain = router.resolve_chain("smart", None).unwrap();
        assert!(chain.is_empty());
        let chain = router.apply_preferences("smart", chain, None).unwrap();
        assert!(chain.is_empty());
    }

    #[tokio::test]
    async fn dispatch_returns_invalid_model_when_route_alias_chain_is_empty() {
        let router = test_router(vec![], vec![("smart", vec![])], vec![], vec![], vec![]);

        let err = router.dispatch(&test_request("smart")).await.unwrap_err();

        assert!(matches!(err, RouterError::InvalidModel(m) if m == "smart"));
    }

    #[tokio::test]
    async fn dispatch_exhausts_a_longer_chain_trying_every_candidate_before_failing() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let calls_c = Arc::new(AtomicUsize::new(0));
        let providers = vec![
            (
                "a",
                Arc::new(MockProvider {
                    name: "a".to_string(),
                    behavior: MockBehavior::FailRetryable,
                    calls: calls_a.clone(),
                }) as Arc<dyn Provider>,
            ),
            (
                "b",
                Arc::new(MockProvider {
                    name: "b".to_string(),
                    behavior: MockBehavior::FailRetryable,
                    calls: calls_b.clone(),
                }) as Arc<dyn Provider>,
            ),
            (
                "c",
                Arc::new(MockProvider {
                    name: "c".to_string(),
                    behavior: MockBehavior::FailRetryable,
                    calls: calls_c.clone(),
                }) as Arc<dyn Provider>,
            ),
        ];
        let router = test_router(
            providers,
            vec![("smart", vec!["a/m1", "b/m2", "c/m3"])],
            vec![],
            vec![],
            vec![],
        );

        let err = router.dispatch(&test_request("smart")).await.unwrap_err();

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::Upstream { .. })
        ));
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 1);
        assert_eq!(
            calls_c.load(Ordering::SeqCst),
            1,
            "every candidate in the chain must be tried before giving up"
        );
    }

    #[tokio::test]
    async fn dispatch_final_error_reflects_the_last_candidate_even_when_its_kind_differs() {
        // "a" is configured and fails retryably; "b" isn't registered at
        // all. The chain still runs to exhaustion and returns whichever
        // error came last, regardless of whether it's a ProviderError or a
        // router-level ProviderNotConfigured.
        let calls_a = Arc::new(AtomicUsize::new(0));
        let a = Arc::new(MockProvider {
            name: "a".to_string(),
            behavior: MockBehavior::FailRetryable,
            calls: calls_a.clone(),
        });
        let router = test_router(
            vec![("a", a)],
            vec![("smart", vec!["a/m1", "b/m2"])],
            vec![],
            vec![],
            vec![],
        );

        let err = router.dispatch(&test_request("smart")).await.unwrap_err();

        assert!(matches!(err, RouterError::ProviderNotConfigured(p) if p == "b"));
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_stops_on_a_fatal_error_without_trying_remaining_candidates() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let calls_c = Arc::new(AtomicUsize::new(0));
        let a = Arc::new(MockProvider {
            name: "a".to_string(),
            behavior: MockBehavior::FailRetryable,
            calls: calls_a.clone(),
        });
        let b = Arc::new(MockProvider {
            name: "b".to_string(),
            behavior: MockBehavior::FailFatal,
            calls: calls_b.clone(),
        });
        let c = Arc::new(MockProvider {
            name: "c".to_string(),
            behavior: MockBehavior::Succeed,
            calls: calls_c.clone(),
        });
        let router = test_router(
            vec![("a", a), ("b", b), ("c", c)],
            vec![("smart", vec!["a/m1", "b/m2", "c/m3"])],
            vec![],
            vec![],
            vec![],
        );

        let err = router.dispatch(&test_request("smart")).await.unwrap_err();

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::InvalidRequest(_))
        ));
        assert_eq!(
            calls_a.load(Ordering::SeqCst),
            1,
            "a is tried and falls back"
        );
        assert_eq!(
            calls_b.load(Ordering::SeqCst),
            1,
            "b's fatal error stops the chain"
        );
        assert_eq!(
            calls_c.load(Ordering::SeqCst),
            0,
            "c is never reached once b fails fatally"
        );
    }

    #[tokio::test]
    async fn dispatch_stream_falls_back_to_next_candidate_on_retryable_error() {
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
        let mut req = test_request("smart");
        req.stream = Some(true);

        let mut stream = router
            .dispatch_stream(&req)
            .await
            .expect("should fall through to openai");
        let chunk = stream.next().await.unwrap().unwrap();

        assert_eq!(chunk.model, "openai/m2");
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dispatch_stream_aborts_immediately_on_fatal_error() {
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
        let mut req = test_request("smart");
        req.stream = Some(true);

        let err = match router.dispatch_stream(&req).await {
            Ok(_) => panic!("expected a fatal error"),
            Err(e) => e,
        };

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::InvalidRequest(_))
        ));
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn dispatch_stream_returns_last_error_when_every_candidate_fails() {
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
        let mut req = test_request("smart");
        req.stream = Some(true);

        let err = match router.dispatch_stream(&req).await {
            Ok(_) => panic!("expected every candidate to fail"),
            Err(e) => e,
        };

        assert!(matches!(
            err,
            RouterError::Provider(ProviderError::Upstream { .. })
        ));
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
