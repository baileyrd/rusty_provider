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
    inbound_rate_limit_rejections_total: IntCounterVec,
    client_budget_rejections_total: IntCounterVec,
    moderation_blocked_total: IntCounterVec,
    web_search_total: IntCounterVec,
    cache_lookups_total: IntCounterVec,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let dispatch_attempts_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_dispatch_attempts_total",
                "Dispatch attempts per provider/model, labeled by outcome (success, retryable_error, error, not_configured, rate_limited).",
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

        let inbound_rate_limit_rejections_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_inbound_rate_limit_rejections_total",
                "Requests to this router's own API rejected for exceeding a per-client or per-IP rate limit, labeled by the resolved caller identity (\"client:<name>\" or \"ip:<addr>\").",
            ),
            &["identity"],
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
        registry
            .register(Box::new(inbound_rate_limit_rejections_total.clone()))
            .expect("metric name is unique");

        let client_budget_rejections_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_client_budget_rejections_total",
                "Requests rejected because the calling client has exceeded its configured budget_usd, labeled by client name.",
            ),
            &["client"],
        )
        .expect("valid metric definition");

        registry
            .register(Box::new(client_budget_rejections_total.clone()))
            .expect("metric name is unique");

        let moderation_blocked_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_moderation_blocked_total",
                "Requests blocked by [moderation], labeled by the flagged category (\"hate\", \"violence\", etc. -- whatever the moderation backend reports).",
            ),
            &["category"],
        )
        .expect("valid metric definition");

        registry
            .register(Box::new(moderation_blocked_total.clone()))
            .expect("metric name is unique");

        let web_search_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_web_search_total",
                "Requests that triggered [web_search], labeled by outcome (\"results\", \"no_results\", \"error\").",
            ),
            &["outcome"],
        )
        .expect("valid metric definition");

        registry
            .register(Box::new(web_search_total.clone()))
            .expect("metric name is unique");

        let cache_lookups_total = IntCounterVec::new(
            Opts::new(
                "rusty_provider_cache_lookups_total",
                "Non-streaming dispatch requests checked against [cache], labeled by outcome (\"hit\", \"miss\").",
            ),
            &["outcome"],
        )
        .expect("valid metric definition");

        registry
            .register(Box::new(cache_lookups_total.clone()))
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
            inbound_rate_limit_rejections_total,
            client_budget_rejections_total,
            moderation_blocked_total,
            web_search_total,
            cache_lookups_total,
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

    pub fn record_inbound_rate_limit_rejection(&self, identity: &str) {
        self.inbound_rate_limit_rejections_total
            .with_label_values(&[identity])
            .inc();
    }

    pub fn record_client_budget_rejection(&self, client_name: &str) {
        self.client_budget_rejections_total
            .with_label_values(&[client_name])
            .inc();
    }

    pub fn record_moderation_blocked(&self, category: &str) {
        self.moderation_blocked_total
            .with_label_values(&[category])
            .inc();
    }

    pub fn record_web_search(&self, outcome: &str) {
        self.web_search_total.with_label_values(&[outcome]).inc();
    }

    pub fn record_cache_lookup(&self, outcome: &str) {
        self.cache_lookups_total.with_label_values(&[outcome]).inc();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Finds the rendered line for `metric_name` whose labels include every
    /// string in `must_contain`, and parses its trailing value. Matching on
    /// individual label substrings (rather than a full exact line) avoids
    /// depending on the prometheus crate's label-ordering in the text
    /// exposition format.
    fn metric_value(rendered: &str, metric_name: &str, must_contain: &[&str]) -> f64 {
        rendered
            .lines()
            .find(|line| {
                line.starts_with(metric_name) && must_contain.iter().all(|s| line.contains(s))
            })
            .unwrap_or_else(|| {
                panic!(
                    "no line for metric {metric_name} containing {must_contain:?} in:\n{rendered}"
                )
            })
            .rsplit(' ')
            .next()
            .unwrap()
            .parse()
            .unwrap_or_else(|e| panic!("failed to parse metric value: {e}"))
    }

    #[test]
    fn record_attempt_increments_the_labeled_counter() {
        let metrics = Metrics::new();
        metrics.record_attempt("anthropic", "m1", "success");
        metrics.record_attempt("anthropic", "m1", "success");

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_dispatch_attempts_total",
                &[
                    "provider=\"anthropic\"",
                    "model=\"m1\"",
                    "outcome=\"success\""
                ],
            ),
            2.0
        );
    }

    #[test]
    fn record_attempt_keeps_different_outcomes_independent() {
        let metrics = Metrics::new();
        metrics.record_attempt("anthropic", "m1", "success");
        metrics.record_attempt("anthropic", "m1", "retryable_error");
        metrics.record_attempt("anthropic", "m1", "retryable_error");

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_dispatch_attempts_total",
                &["outcome=\"success\""],
            ),
            1.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_dispatch_attempts_total",
                &["outcome=\"retryable_error\""],
            ),
            2.0
        );
    }

    #[test]
    fn observe_latency_seconds_populates_the_histogram_count_and_sum() {
        let metrics = Metrics::new();
        metrics.observe_latency_seconds("anthropic", "m1", 1.5);
        metrics.observe_latency_seconds("anthropic", "m1", 2.5);

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_response_latency_seconds_count",
                &["provider=\"anthropic\"", "model=\"m1\""],
            ),
            2.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_response_latency_seconds_sum",
                &["provider=\"anthropic\"", "model=\"m1\""],
            ),
            4.0
        );
    }

    #[test]
    fn observe_throughput_tps_populates_the_histogram_count_and_sum() {
        let metrics = Metrics::new();
        metrics.observe_throughput_tps("anthropic", "m1", 10.0);

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_throughput_tokens_per_second_count",
                &["provider=\"anthropic\"", "model=\"m1\""],
            ),
            1.0
        );
    }

    #[test]
    fn record_tokens_and_cost_increments_prompt_completion_and_cost() {
        let metrics = Metrics::new();
        metrics.record_tokens_and_cost("anthropic", "m1", 100, 50, Some(0.5));
        metrics.record_tokens_and_cost("anthropic", "m1", 200, 25, Some(0.25));

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_prompt_tokens_total",
                &["provider=\"anthropic\"", "model=\"m1\""],
            ),
            300.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_completion_tokens_total",
                &["provider=\"anthropic\"", "model=\"m1\""],
            ),
            75.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_cost_usd_total",
                &["provider=\"anthropic\"", "model=\"m1\""],
            ),
            0.75
        );
    }

    #[test]
    fn record_tokens_and_cost_with_no_cost_still_records_tokens_but_not_cost() {
        let metrics = Metrics::new();
        metrics.record_tokens_and_cost("anthropic", "m1", 100, 50, None);

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_prompt_tokens_total",
                &["provider=\"anthropic\"", "model=\"m1\""],
            ),
            100.0
        );
        assert!(
            !rendered.contains("rusty_provider_cost_usd_total"),
            "an unpriced request should never touch the cost counter"
        );
    }

    #[test]
    fn set_provider_configured_reflects_the_latest_call_not_an_accumulation() {
        let metrics = Metrics::new();
        metrics.set_provider_configured("openai", true);
        assert_eq!(
            metric_value(
                &metrics.render(),
                "rusty_provider_provider_configured",
                &["provider=\"openai\""],
            ),
            1.0
        );

        metrics.set_provider_configured("openai", false);
        assert_eq!(
            metric_value(
                &metrics.render(),
                "rusty_provider_provider_configured",
                &["provider=\"openai\""],
            ),
            0.0,
            "a gauge overwrites its previous value rather than accumulating"
        );
    }

    #[test]
    fn record_inbound_rate_limit_rejection_increments_by_identity() {
        let metrics = Metrics::new();
        metrics.record_inbound_rate_limit_rejection("client:acme");
        metrics.record_inbound_rate_limit_rejection("client:acme");
        metrics.record_inbound_rate_limit_rejection("ip:127.0.0.1");

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_inbound_rate_limit_rejections_total",
                &["identity=\"client:acme\""],
            ),
            2.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_inbound_rate_limit_rejections_total",
                &["identity=\"ip:127.0.0.1\""],
            ),
            1.0
        );
    }

    #[test]
    fn record_client_budget_rejection_increments_by_client_name() {
        let metrics = Metrics::new();
        metrics.record_client_budget_rejection("acme");
        metrics.record_client_budget_rejection("acme");
        metrics.record_client_budget_rejection("globex");

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_client_budget_rejections_total",
                &["client=\"acme\""],
            ),
            2.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_client_budget_rejections_total",
                &["client=\"globex\""],
            ),
            1.0
        );
    }

    #[test]
    fn record_moderation_blocked_increments_by_category() {
        let metrics = Metrics::new();
        metrics.record_moderation_blocked("violence");
        metrics.record_moderation_blocked("violence");
        metrics.record_moderation_blocked("hate");

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_moderation_blocked_total",
                &["category=\"violence\""],
            ),
            2.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_moderation_blocked_total",
                &["category=\"hate\""],
            ),
            1.0
        );
    }

    #[test]
    fn record_web_search_increments_by_outcome() {
        let metrics = Metrics::new();
        metrics.record_web_search("results");
        metrics.record_web_search("results");
        metrics.record_web_search("error");

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_web_search_total",
                &["outcome=\"results\""],
            ),
            2.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_web_search_total",
                &["outcome=\"error\""],
            ),
            1.0
        );
    }

    #[test]
    fn record_cache_lookup_increments_by_outcome() {
        let metrics = Metrics::new();
        metrics.record_cache_lookup("hit");
        metrics.record_cache_lookup("hit");
        metrics.record_cache_lookup("miss");

        let rendered = metrics.render();
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_cache_lookups_total",
                &["outcome=\"hit\""],
            ),
            2.0
        );
        assert_eq!(
            metric_value(
                &rendered,
                "rusty_provider_cache_lookups_total",
                &["outcome=\"miss\""],
            ),
            1.0
        );
    }

    #[test]
    fn render_on_a_fresh_registry_does_not_panic_and_has_no_sample_lines() {
        let metrics = Metrics::new();
        let rendered = metrics.render();
        assert!(
            !rendered.lines().any(|l| !l.starts_with('#')),
            "no metric has been recorded yet, so there should be no sample lines: {rendered}"
        );
    }
}
