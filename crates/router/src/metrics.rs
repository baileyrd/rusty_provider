use prometheus::{
    CounterVec, Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
    TextEncoder,
};

const PROVIDER_MODEL_LABELS: [&str; 2] = ["provider", "model"];

/// LLM response times commonly range from well under a second (short
/// completions) to tens of seconds (long non-streaming generations),
/// outside Prometheus's default 10ms-10s bucket range.
const LATENCY_BUCKETS_SECONDS: &[f64] = &[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0];

/// Typical completion throughput observed across providers, in tokens/sec.
const THROUGHPUT_BUCKETS_TPS: &[f64] = &[5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0];

/// Prometheus counters/histograms/gauges for this router's own dispatch
/// activity, rendered as `GET /metrics`. Every field is cheap to clone
/// (the `prometheus` crate's metric handles are themselves `Arc`-backed),
/// so `Metrics` as a whole derives `Clone` and can be captured into the
/// streaming-response instrumentation closure the same way `pricing`/
/// `throughput`/`usage` already are.
#[derive(Clone)]
pub struct Metrics {
    registry: Registry,
    dispatch_attempts_total: IntCounterVec,
    prompt_tokens_total: IntCounterVec,
    completion_tokens_total: IntCounterVec,
    cost_usd_total: CounterVec,
    response_latency_seconds: HistogramVec,
    throughput_tokens_per_second: HistogramVec,
    provider_configured: IntGaugeVec,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let dispatch_attempts_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_dispatch_attempts_total",
                "Dispatch attempts per provider/model, labeled by outcome (success, retryable_error, error, not_configured).",
            ),
            &["provider", "model", "outcome"],
        )
        .expect("valid metric definition");

        let prompt_tokens_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_prompt_tokens_total",
                "Cumulative prompt tokens sent, per provider/model.",
            ),
            &PROVIDER_MODEL_LABELS,
        )
        .expect("valid metric definition");

        let completion_tokens_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_completion_tokens_total",
                "Cumulative completion tokens received, per provider/model.",
            ),
            &PROVIDER_MODEL_LABELS,
        )
        .expect("valid metric definition");

        let cost_usd_total = CounterVec::new(
            Opts::new(
                "rusty_provider_cost_usd_total",
                "Cumulative estimated USD cost, per provider/model. Only accumulates for models with a [[pricing]] config entry.",
            ),
            &PROVIDER_MODEL_LABELS,
        )
        .expect("valid metric definition");

        let response_latency_seconds = HistogramVec::new(
            HistogramOpts::new(
                "rusty_provider_response_latency_seconds",
                "Response latency: full round-trip for non-streaming requests, time-to-first-byte for streaming ones.",
            )
            .buckets(LATENCY_BUCKETS_SECONDS.to_vec()),
            &PROVIDER_MODEL_LABELS,
        )
        .expect("valid metric definition");

        let throughput_tokens_per_second = HistogramVec::new(
            HistogramOpts::new(
                "rusty_provider_throughput_tokens_per_second",
                "Observed completion token generation rate per response.",
            )
            .buckets(THROUGHPUT_BUCKETS_TPS.to_vec()),
            &PROVIDER_MODEL_LABELS,
        )
        .expect("valid metric definition");

        let provider_configured = IntGaugeVec::new(
            Opts::new(
                "rusty_provider_provider_configured",
                "1 if this provider's API key env var resolved at startup, 0 otherwise.",
            ),
            &["provider"],
        )
        .expect("valid metric definition");

        registry
            .register(Box::new(dispatch_attempts_total.clone()))
            .expect("metric name is unique");
        registry
            .register(Box::new(prompt_tokens_total.clone()))
            .expect("metric name is unique");
        registry
            .register(Box::new(completion_tokens_total.clone()))
            .expect("metric name is unique");
        registry
            .register(Box::new(cost_usd_total.clone()))
            .expect("metric name is unique");
        registry
            .register(Box::new(response_latency_seconds.clone()))
            .expect("metric name is unique");
        registry
            .register(Box::new(throughput_tokens_per_second.clone()))
            .expect("metric name is unique");
        registry
            .register(Box::new(provider_configured.clone()))
            .expect("metric name is unique");

        Self {
            registry,
            dispatch_attempts_total,
            prompt_tokens_total,
            completion_tokens_total,
            cost_usd_total,
            response_latency_seconds,
            throughput_tokens_per_second,
            provider_configured,
        }
    }

    pub fn set_provider_configured(&self, provider: &str, configured: bool) {
        self.provider_configured
            .with_label_values(&[provider])
            .set(configured as i64);
    }

    pub fn record_attempt(&self, provider: &str, model: &str, outcome: &str) {
        self.dispatch_attempts_total
            .with_label_values(&[provider, model, outcome])
            .inc();
    }

    pub fn observe_latency_seconds(&self, provider: &str, model: &str, seconds: f64) {
        self.response_latency_seconds
            .with_label_values(&[provider, model])
            .observe(seconds);
    }

    pub fn observe_throughput_tps(&self, provider: &str, model: &str, tokens_per_sec: f64) {
        self.throughput_tokens_per_second
            .with_label_values(&[provider, model])
            .observe(tokens_per_sec);
    }

    pub fn record_tokens_and_cost(
        &self,
        provider: &str,
        model: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
        cost_usd: Option<f64>,
    ) {
        self.prompt_tokens_total
            .with_label_values(&[provider, model])
            .inc_by(prompt_tokens as u64);
        self.completion_tokens_total
            .with_label_values(&[provider, model])
            .inc_by(completion_tokens as u64);
        if let Some(cost) = cost_usd {
            self.cost_usd_total
                .with_label_values(&[provider, model])
                .inc_by(cost);
        }
    }

    /// Render every registered metric in the Prometheus text exposition
    /// format, for `GET /metrics`.
    pub fn render(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buf = Vec::new();
        encoder
            .encode(&metric_families, &mut buf)
            .expect("prometheus text encoding into an in-memory buffer is infallible");
        String::from_utf8(buf).expect("prometheus text encoder always emits valid UTF-8")
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
