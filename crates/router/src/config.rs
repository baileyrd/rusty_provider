use std::collections::HashMap;

use rp_core::ProviderPreferences;
use serde::{Deserialize, Serialize};

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// If set, the env var holding a bearer token clients must present to
    /// this router. Leave unset to run with no auth (e.g. behind your own
    /// gateway). Any key from `[[clients]]` below also authenticates,
    /// independent of this field.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Requests-per-minute limit applied to any caller not matched to a
    /// `[[clients]]` entry, bucketed by source IP address. Unset means no
    /// limit for such callers.
    #[serde(default)]
    pub default_rate_limit_rpm: Option<u32>,
    /// If set, the env var holding a bearer token that unlocks the admin
    /// API (`/v1/admin/*`) -- listing configured clients' live spend and
    /// resetting a client's budget. Deliberately separate from
    /// `api_key_env`/`[[clients]]` keys: those grant access to
    /// `/v1/chat/completions`, not to every other client's spend data or
    /// the ability to reset it. Leaving this unset disables the admin API
    /// entirely (`404`, as if the routes didn't exist) rather than falling
    /// open.
    #[serde(default)]
    pub admin_key_env: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key_env: None,
            default_rate_limit_rpm: None,
            admin_key_env: None,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// Any backend speaking the OpenAI `/chat/completions` wire format:
    /// OpenAI itself, Groq, Together AI, Fireworks, etc.
    Openai,
    Anthropic,
    Gemini,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub base_url: String,
    /// Name of the environment variable holding the API key for this
    /// provider (not the key itself — keeps secrets out of the config file).
    pub api_key_env: String,
    /// Whether the operator has a Zero Data Retention agreement with this
    /// provider. Self-declared — the router trusts this flag and never
    /// verifies it against the provider itself. Only consulted for
    /// requests that set `"provider": {"zdr": true}`.
    #[serde(default)]
    pub zdr: bool,
    /// Whether the operator has confirmed this provider does not use
    /// submitted data to train its models. Self-declared, same trust model
    /// as `zdr` — but a distinct axis from it: `zdr` is about retention
    /// (does the provider keep your data at all), this is about training
    /// (if they keep it, do they learn from it). A provider can be one
    /// without the other. Only consulted for requests that set
    /// `"provider": {"data_collection": true}`.
    #[serde(default)]
    pub no_training: bool,
    /// Self-imposed outbound rate limit for this provider (requests per
    /// minute), so this router doesn't exceed the provider's own limits
    /// and get 429'd/banned. Unset means no self-imposed limit — only
    /// the provider's real limits apply. Enforced per-provider (not
    /// per-model), since real-world provider rate limits are account-wide.
    #[serde(default)]
    pub requests_per_minute: Option<u32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RouteAlias {
    pub alias: String,
    /// Ordered "provider/model" fallback chain, e.g.
    /// ["anthropic/claude-sonnet-5", "openai/gpt-4o"].
    pub chain: Vec<String>,
}

/// Prompt/completion token pricing for one "provider/model" entry, used
/// only for `provider.sort: "price"` requests — this is operator-supplied
/// static data, not a live pricing feed, so keep it current by hand.
#[derive(Debug, Deserialize, Clone)]
pub struct PricingEntry {
    /// "provider/model", matching a `[[routes]]` chain entry.
    pub model: String,
    pub prompt_per_million: f64,
    #[serde(default)]
    pub completion_per_million: f64,
    /// Price for prompt tokens served from a cache (Anthropic's
    /// `cache_read_input_tokens`, OpenAI's `prompt_tokens_details.cached_tokens`,
    /// Gemini's `cachedContentTokenCount`) — typically a steep discount off
    /// `prompt_per_million`. Defaults to `prompt_per_million` (i.e. no
    /// assumed discount) when unset, so leaving it out never *under*counts
    /// cost relative to not tracking caching at all.
    #[serde(default)]
    pub cache_read_per_million: Option<f64>,
    /// Price for prompt tokens newly written into a cache (Anthropic's
    /// `cache_creation_input_tokens` only — OpenAI/Gemini bill a cache
    /// write the same as a normal prompt token, no separate rate needed).
    /// Defaults to `prompt_per_million` when unset, same rationale as
    /// `cache_read_per_million`.
    #[serde(default)]
    pub cache_write_per_million: Option<f64>,
    /// This model's context window, in tokens -- purely informational,
    /// surfaced at `GET /v1/models` for clients that want to pick a model
    /// by capacity. Not enforced anywhere: an over-length request still
    /// goes to the provider and fails (or gets silently truncated) exactly
    /// as it would without this field set.
    #[serde(default)]
    pub context_length: Option<u32>,
}

/// A named inbound caller, identified by its own API key, with its own
/// rate limit — independent of (and in addition to) `server.api_key_env`.
/// Presenting this key both authenticates the request and buckets it
/// under `name` rather than the source-IP fallback.
#[derive(Debug, Deserialize, Clone)]
pub struct ClientConfig {
    pub name: String,
    /// Name of the environment variable holding this client's API key
    /// (not the key itself — keeps secrets out of the config file).
    pub api_key_env: String,
    pub requests_per_minute: u32,
    /// If set, this client is cut off (`402`) once its tracked spend for
    /// the current `budget_period` reaches this many US dollars. Spend is
    /// tracked from the same `cost_usd` this router already computes for
    /// `GET /v1/usage`, so it's only as accurate as `[[pricing]]` is —
    /// requests to an unpriced model don't count against the budget.
    /// Unset means unrestricted, same as omitting the field entirely.
    #[serde(default)]
    pub budget_usd: Option<f64>,
    /// How `budget_usd` resets. Meaningless (but harmless) without
    /// `budget_usd` set.
    #[serde(default)]
    pub budget_period: BudgetPeriod,
    /// Groups this client under a named organization, for admin-API
    /// scoping (see `role`) and the `GET /v1/admin/organizations` rollup.
    /// Purely a label otherwise -- it has no effect on chat completions.
    /// Unset is its own bucket ("no organization"), distinct from every
    /// named one.
    #[serde(default)]
    pub organization: Option<String>,
    /// Sub-groups this client within `organization`, for the
    /// `GET /v1/admin/organizations` rollup only -- never consulted for
    /// authorization, which scopes by `organization` alone.
    #[serde(default)]
    pub workspace: Option<String>,
    /// Whether this client's own API key can also authenticate to
    /// `/v1/admin/*`, in addition to `server.admin_key_env`. An admin-role
    /// client's access is scoped to clients sharing its `organization`
    /// (matching only other clients with the identical `organization`,
    /// including the shared "unset" bucket) -- unlike `server.admin_key_env`,
    /// which always sees every client regardless of organization.
    /// `Member` (the default) grants no admin access, same as before this
    /// field existed.
    #[serde(default)]
    pub role: ClientRole,
}

/// Whether a client's own API key also unlocks `/v1/admin/*`, and if so,
/// how broadly.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ClientRole {
    /// Chat-completions access only.
    #[default]
    Member,
    /// Also grants admin API access, scoped to this client's own
    /// `organization`.
    Admin,
}

/// How a client's `budget_usd` cap resets.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BudgetPeriod {
    /// Never resets — a lifetime cap on this client's total tracked spend.
    #[default]
    Total,
    /// Resets to zero at the start of each calendar day (UTC midnight).
    Daily,
    /// Resets to zero every 7 days, counted from the Unix epoch
    /// (1970-01-01T00:00:00Z, a Thursday) -- not aligned to any particular
    /// weekday like a calendar Monday/Sunday week. A fixed 7-day cadence
    /// rather than calendar-week alignment, since the latter would need an
    /// operator-configurable "which day does the week start on" that
    /// nothing else here has a reason to need.
    Weekly,
    /// Resets to zero at the start of each calendar month (server wall
    /// clock, UTC).
    Monthly,
}

/// Which database backend `[persistence]` uses.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PersistenceBackend {
    /// A single local file. Fine for multiple processes on one host or a
    /// shared local volume, not for processes spread across machines.
    #[default]
    Sqlite,
    /// A shared Postgres database, reachable over the network — the way
    /// to get usage/spend tracking consistent across multiple hosts, not
    /// just multiple processes on one.
    Postgres,
}

/// Whether the Postgres backend encrypts its connection.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PostgresTlsMode {
    /// Plaintext connection. Fine on a trusted network or an already-
    /// encrypted tunnel (e.g. a Unix socket, a VPN, `stunnel`); never use
    /// this across an untrusted network.
    #[default]
    Disable,
    /// Require TLS, verifying the server's certificate against the host's
    /// native trust store (the same roots `reqwest` trusts for outbound
    /// provider calls). The connection is refused if TLS can't be
    /// negotiated or the certificate doesn't validate.
    Require,
}

/// Durable storage for cumulative usage/cost stats and client spend
/// budgets, so they survive a restart and stay consistent across every
/// router process pointed at the same backend. Omit this section entirely
/// to keep the original in-memory-only behavior, which resets on every
/// restart and is never shared across processes.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct PersistenceConfig {
    #[serde(default)]
    pub backend: PersistenceBackend,
    /// Path to a SQLite database file. Created (along with its tables) on
    /// first use if it doesn't already exist. Required when `backend` is
    /// `"sqlite"` (the default); ignored otherwise.
    #[serde(default)]
    pub sqlite_path: Option<String>,
    /// Name of the environment variable holding a Postgres connection
    /// string (e.g. `postgres://user:pass@host/dbname`), kept out of the
    /// config file the same way provider/client API keys are. Required
    /// when `backend` is `"postgres"`; ignored otherwise.
    #[serde(default)]
    pub postgres_url_env: Option<String>,
    /// Whether the Postgres connection is encrypted. Ignored when
    /// `backend` is `"sqlite"`.
    #[serde(default)]
    pub postgres_tls: PostgresTlsMode,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub routes: Vec<RouteAlias>,
    #[serde(default)]
    pub pricing: Vec<PricingEntry>,
    #[serde(default)]
    pub clients: Vec<ClientConfig>,
    #[serde(default)]
    pub persistence: Option<PersistenceConfig>,
    #[serde(default)]
    pub webhook: Option<WebhookConfig>,
    #[serde(default)]
    pub guardrails: Vec<GuardrailConfig>,
    #[serde(default)]
    pub presets: Vec<PresetConfig>,
    #[serde(default)]
    pub auto_routing: Option<AutoRoutingConfig>,
    #[serde(default)]
    pub moderation: Option<ModerationConfig>,
}

/// Configures `model: "auto"` -- a heuristic (not ML) complexity-based
/// router, roughly mirroring OpenRouter's `openrouter/auto`. Each of the
/// three tier fields is a "provider/model" string or a `[[routes]]`
/// alias, exactly like `ChatRequest.model` itself, so a tier can point at
/// a whole fallback chain rather than one fixed model. Absent entirely
/// means `model: "auto"` isn't special-cased at all -- it's resolved the
/// same as any other unrecognized alias (a `400`).
#[derive(Debug, Deserialize, Clone)]
pub struct AutoRoutingConfig {
    /// Used for requests scoring at or below `simple_max_score`.
    pub simple_model: String,
    /// Used for requests scoring above `simple_max_score` but at or below
    /// `medium_max_score`.
    pub medium_model: String,
    /// Used for requests scoring above `medium_max_score`.
    pub complex_model: String,
    /// Upper (inclusive) complexity-score bound for `simple_model`. The
    /// score itself has no fixed unit -- it's an estimated-token count
    /// plus heuristic bonuses (see `estimate_complexity`) -- so these
    /// thresholds are necessarily something to tune empirically against
    /// your own traffic, not a universal constant.
    #[serde(default = "default_simple_max_score")]
    pub simple_max_score: u32,
    /// Upper (inclusive) complexity-score bound for `medium_model`.
    #[serde(default = "default_medium_max_score")]
    pub medium_max_score: u32,
}

fn default_simple_max_score() -> u32 {
    200
}

fn default_medium_max_score() -> u32 {
    800
}

/// A named, reusable bundle of request defaults (`[[presets]]`), applied
/// when a request sets `"preset": "<name>"`. Every field here is a
/// per-field *default* -- whatever the request itself already set always
/// wins, so a preset only ever fills in what the caller left unset --
/// except `model`, which overrides the request outright when set, since
/// centralizing model selection is the point of a preset.
#[derive(Debug, Deserialize, Clone)]
pub struct PresetConfig {
    /// The slug clients reference via `"preset": "<name>"`.
    pub name: String,
    #[serde(default)]
    pub model: Option<String>,
    /// Prepended as a new `role = "system"` message, but only if the
    /// request has no system message of its own -- never appended
    /// alongside or merged with one the caller already provided.
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub provider: Option<ProviderPreferences>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub top_a: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    pub seed: Option<i64>,
}

/// A regex-based content guardrail, checked against every request's
/// message text before dispatch -- OpenRouter's org-level guardrails,
/// scoped globally here since rusty has no workspace/org concept to
/// scope these to individually (see the deferred organizations/
/// workspaces/roles item).
#[derive(Debug, Deserialize, Clone)]
pub struct GuardrailConfig {
    /// Identifies this guardrail in a block error message and log line.
    pub name: String,
    /// A regex (the `regex` crate's syntax) checked against each
    /// message's plain text content.
    pub pattern: String,
    pub action: GuardrailAction,
    /// Replacement text for a `"redact"` guardrail's matches. Ignored
    /// for `"block"`.
    #[serde(default = "default_guardrail_replacement")]
    pub replacement: String,
}

fn default_guardrail_replacement() -> String {
    "[redacted]".to_string()
}

/// What a matching [`GuardrailConfig`] does to the request.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GuardrailAction {
    /// Reject the request with `400` if the pattern matches anywhere in
    /// its message text.
    Block,
    /// Replace every matched substring with `replacement` before the
    /// request is dispatched to any provider.
    Redact,
}

/// Outbound webhook fired on budget-exceeded/reset events -- a proactive
/// push notification on top of the `402` a client already sees on its next
/// request and the `client_budget_rejections_total` Prometheus counter, so
/// an operator can wire up alerting without polling either.
#[derive(Debug, Deserialize, Clone)]
pub struct WebhookConfig {
    /// URL this router POSTs a JSON event payload to.
    pub url: String,
    /// Name of the environment variable holding the exact value to send
    /// as this POST's `Authorization` header (e.g. `"Bearer <token>"`),
    /// so the receiver can verify the request came from this router.
    /// Unset means no `Authorization` header is sent.
    #[serde(default)]
    pub auth_header_env: Option<String>,
}

fn default_moderation_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

fn default_moderation_model() -> String {
    "omni-moderation-latest".to_string()
}

/// Checks every request's message text against an external moderation
/// endpoint before it's ever dispatched to a provider, blocking anything
/// flagged. Only OpenAI's `/moderations` endpoint (or a compatible one --
/// `base_url` is configurable) is supported; Anthropic and Gemini don't
/// expose a public moderation API of their own. This is a different axis
/// from `[[guardrails]]`: guardrails are operator-authored regex patterns
/// (PII, specific keywords), this is a third-party classifier judging
/// broad policy categories (hate, violence, self-harm, etc.) the operator
/// doesn't have to enumerate by hand. Runs after guardrails, so a
/// guardrail's own redaction is what gets checked, not the raw input.
#[derive(Debug, Deserialize, Clone)]
pub struct ModerationConfig {
    /// Name of the environment variable holding the API key for the
    /// moderation backend (not the key itself).
    pub api_key_env: String,
    #[serde(default = "default_moderation_base_url")]
    pub base_url: String,
    #[serde(default = "default_moderation_model")]
    pub model: String,
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.as_ref().display()))?;
        Self::from_toml_str(&raw)
            .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.as_ref().display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A unique path under the OS temp dir, so parallel tests never race on
    /// the same file.
    fn unique_temp_path(label: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        std::env::temp_dir().join(format!(
            "rp_router_config_test_{label}_{}.toml",
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    // --- server section ------------------------------------------------------

    #[test]
    fn server_section_defaults_when_absent() {
        let config = Config::from_toml_str("providers = {}").unwrap();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.api_key_env, None);
        assert_eq!(config.server.default_rate_limit_rpm, None);
        assert_eq!(config.server.admin_key_env, None);
    }

    #[test]
    fn server_section_overrides_are_honored() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [server]
            host = "127.0.0.1"
            port = 9000
            api_key_env = "RP_API_KEY"
            default_rate_limit_rpm = 60
            admin_key_env = "RP_ADMIN_KEY"
            "#,
        )
        .unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.api_key_env.as_deref(), Some("RP_API_KEY"));
        assert_eq!(config.server.default_rate_limit_rpm, Some(60));
        assert_eq!(config.server.admin_key_env.as_deref(), Some("RP_ADMIN_KEY"));
    }

    // --- providers -------------------------------------------------------------

    #[test]
    fn providers_defaults_to_empty_when_absent() {
        let config = Config::from_toml_str("").unwrap();
        assert!(config.providers.is_empty());
    }

    #[test]
    fn provider_kind_accepts_all_three_documented_values() {
        let config = Config::from_toml_str(
            r#"
            [providers.a]
            kind = "openai"
            base_url = "https://a"
            api_key_env = "A"

            [providers.b]
            kind = "anthropic"
            base_url = "https://b"
            api_key_env = "B"

            [providers.c]
            kind = "gemini"
            base_url = "https://c"
            api_key_env = "C"
            "#,
        )
        .unwrap();
        assert_eq!(config.providers["a"].kind, ProviderKind::Openai);
        assert_eq!(config.providers["b"].kind, ProviderKind::Anthropic);
        assert_eq!(config.providers["c"].kind, ProviderKind::Gemini);
    }

    #[test]
    fn provider_kind_rejects_unknown_value() {
        let err = Config::from_toml_str(
            r#"
            [providers.a]
            kind = "mistral"
            base_url = "https://a"
            api_key_env = "A"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("kind"));
    }

    #[test]
    fn provider_config_requires_base_url_and_api_key_env() {
        let missing_base_url = Config::from_toml_str(
            r#"
            [providers.a]
            kind = "openai"
            api_key_env = "A"
            "#,
        );
        assert!(missing_base_url.is_err());

        let missing_api_key_env = Config::from_toml_str(
            r#"
            [providers.a]
            kind = "openai"
            base_url = "https://a"
            "#,
        );
        assert!(missing_api_key_env.is_err());
    }

    #[test]
    fn provider_config_zdr_and_requests_per_minute_default_when_absent() {
        let config = Config::from_toml_str(
            r#"
            [providers.a]
            kind = "openai"
            base_url = "https://a"
            api_key_env = "A"
            "#,
        )
        .unwrap();
        let provider = &config.providers["a"];
        assert!(!provider.zdr);
        assert!(!provider.no_training);
        assert_eq!(provider.requests_per_minute, None);
    }

    #[test]
    fn provider_config_zdr_and_requests_per_minute_are_honored_when_set() {
        let config = Config::from_toml_str(
            r#"
            [providers.a]
            kind = "openai"
            base_url = "https://a"
            api_key_env = "A"
            zdr = true
            no_training = true
            requests_per_minute = 500
            "#,
        )
        .unwrap();
        let provider = &config.providers["a"];
        assert!(provider.zdr);
        assert!(provider.no_training);
        assert_eq!(provider.requests_per_minute, Some(500));
    }

    // --- routes/pricing/clients default to empty --------------------------------

    #[test]
    fn routes_pricing_and_clients_default_to_empty_when_absent() {
        let config = Config::from_toml_str("").unwrap();
        assert!(config.routes.is_empty());
        assert!(config.pricing.is_empty());
        assert!(config.clients.is_empty());
    }

    #[test]
    fn route_alias_requires_alias_and_chain() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[routes]]
            alias = "smart"
            chain = ["anthropic/claude-sonnet-5", "openai/gpt-4o"]
            "#,
        )
        .unwrap();
        assert_eq!(config.routes.len(), 1);
        assert_eq!(config.routes[0].alias, "smart");
        assert_eq!(
            config.routes[0].chain,
            vec!["anthropic/claude-sonnet-5", "openai/gpt-4o"]
        );

        let missing_chain = Config::from_toml_str(
            r#"
            providers = {}

            [[routes]]
            alias = "smart"
            "#,
        );
        assert!(missing_chain.is_err());
    }

    #[test]
    fn pricing_entry_completion_per_million_defaults_to_zero() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[pricing]]
            model = "openai/gpt-4o"
            prompt_per_million = 2.5
            "#,
        )
        .unwrap();
        assert_eq!(config.pricing[0].completion_per_million, 0.0);

        let missing_prompt_price = Config::from_toml_str(
            r#"
            providers = {}

            [[pricing]]
            model = "openai/gpt-4o"
            "#,
        );
        assert!(missing_prompt_price.is_err());
    }

    #[test]
    fn pricing_entry_cache_rates_default_to_unset() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[pricing]]
            model = "anthropic/claude-sonnet-5"
            prompt_per_million = 3.0
            completion_per_million = 15.0
            "#,
        )
        .unwrap();
        assert_eq!(config.pricing[0].cache_read_per_million, None);
        assert_eq!(config.pricing[0].cache_write_per_million, None);
    }

    #[test]
    fn pricing_entry_cache_rates_are_honored_when_set() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[pricing]]
            model = "anthropic/claude-sonnet-5"
            prompt_per_million = 3.0
            completion_per_million = 15.0
            cache_read_per_million = 0.3
            cache_write_per_million = 3.75
            "#,
        )
        .unwrap();
        assert_eq!(config.pricing[0].cache_read_per_million, Some(0.3));
        assert_eq!(config.pricing[0].cache_write_per_million, Some(3.75));
    }

    #[test]
    fn client_config_requires_every_field() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            "#,
        )
        .unwrap();
        assert_eq!(config.clients.len(), 1);
        assert_eq!(config.clients[0].name, "acme");
        assert_eq!(config.clients[0].api_key_env, "ACME_KEY");
        assert_eq!(config.clients[0].requests_per_minute, 60);

        let missing_rpm = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            "#,
        );
        assert!(missing_rpm.is_err());
    }

    #[test]
    fn client_budget_defaults_to_unset_with_total_period() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            "#,
        )
        .unwrap();
        assert_eq!(config.clients[0].budget_usd, None);
        assert_eq!(config.clients[0].budget_period, BudgetPeriod::Total);
    }

    #[test]
    fn client_budget_usd_and_period_are_honored_when_set() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            budget_usd = 25.0
            budget_period = "monthly"
            "#,
        )
        .unwrap();
        assert_eq!(config.clients[0].budget_usd, Some(25.0));
        assert_eq!(config.clients[0].budget_period, BudgetPeriod::Monthly);
    }

    #[test]
    fn client_budget_period_accepts_daily_and_weekly() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            budget_period = "daily"

            [[clients]]
            name = "beta"
            api_key_env = "BETA_KEY"
            requests_per_minute = 60
            budget_period = "weekly"
            "#,
        )
        .unwrap();
        assert_eq!(config.clients[0].budget_period, BudgetPeriod::Daily);
        assert_eq!(config.clients[1].budget_period, BudgetPeriod::Weekly);
    }

    #[test]
    fn client_budget_period_rejects_unknown_value() {
        let err = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            budget_period = "yearly"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("budget_period"));
    }

    #[test]
    fn client_organization_workspace_and_role_default_to_absent_and_member() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            "#,
        )
        .unwrap();
        assert_eq!(config.clients[0].organization, None);
        assert_eq!(config.clients[0].workspace, None);
        assert_eq!(config.clients[0].role, ClientRole::Member);
    }

    #[test]
    fn client_organization_workspace_and_role_are_honored_when_set() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            organization = "acme-corp"
            workspace = "prod"
            role = "admin"
            "#,
        )
        .unwrap();
        assert_eq!(
            config.clients[0].organization,
            Some("acme-corp".to_string())
        );
        assert_eq!(config.clients[0].workspace, Some("prod".to_string()));
        assert_eq!(config.clients[0].role, ClientRole::Admin);
    }

    #[test]
    fn client_role_rejects_unknown_value() {
        let err = Config::from_toml_str(
            r#"
            providers = {}

            [[clients]]
            name = "acme"
            api_key_env = "ACME_KEY"
            requests_per_minute = 60
            role = "owner"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("role"));
    }

    // --- guardrails ----------------------------------------------------------------

    #[test]
    fn guardrails_defaults_to_empty_when_absent() {
        let config = Config::from_toml_str("providers = {}").unwrap();
        assert!(config.guardrails.is_empty());
    }

    #[test]
    fn guardrail_parses_a_block_entry() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[guardrails]]
            name = "no-ssn"
            pattern = '\d{3}-\d{2}-\d{4}'
            action = "block"
            "#,
        )
        .unwrap();
        assert_eq!(config.guardrails.len(), 1);
        let guardrail = &config.guardrails[0];
        assert_eq!(guardrail.name, "no-ssn");
        assert_eq!(guardrail.pattern, r"\d{3}-\d{2}-\d{4}");
        assert_eq!(guardrail.action, GuardrailAction::Block);
        assert_eq!(guardrail.replacement, "[redacted]");
    }

    #[test]
    fn guardrail_parses_a_redact_entry_with_a_custom_replacement() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[guardrails]]
            name = "no-email"
            pattern = '\S+@\S+'
            action = "redact"
            replacement = "<email>"
            "#,
        )
        .unwrap();
        let guardrail = &config.guardrails[0];
        assert_eq!(guardrail.action, GuardrailAction::Redact);
        assert_eq!(guardrail.replacement, "<email>");
    }

    #[test]
    fn guardrail_rejects_an_unknown_action() {
        let err = Config::from_toml_str(
            r#"
            providers = {}

            [[guardrails]]
            name = "bogus"
            pattern = "x"
            action = "quarantine"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("action"));
    }

    // --- presets ---------------------------------------------------------------

    #[test]
    fn presets_defaults_to_empty_when_absent() {
        let config = Config::from_toml_str("providers = {}").unwrap();
        assert!(config.presets.is_empty());
    }

    #[test]
    fn preset_parses_model_system_prompt_and_sampling_params() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[presets]]
            name = "support-bot"
            model = "smart"
            system_prompt = "You are a support agent."
            temperature = 0.2
            max_tokens = 500
            "#,
        )
        .unwrap();
        assert_eq!(config.presets.len(), 1);
        let preset = &config.presets[0];
        assert_eq!(preset.name, "support-bot");
        assert_eq!(preset.model, Some("smart".to_string()));
        assert_eq!(
            preset.system_prompt,
            Some("You are a support agent.".to_string())
        );
        assert_eq!(preset.temperature, Some(0.2));
        assert_eq!(preset.max_tokens, Some(500));
    }

    #[test]
    fn preset_parses_provider_prefs() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[presets]]
            name = "cheap"

            [presets.provider]
            sort = "price"
            "#,
        )
        .unwrap();
        let preset = &config.presets[0];
        assert_eq!(
            preset.provider.as_ref().unwrap().sort,
            Some("price".to_string())
        );
    }

    #[test]
    fn preset_only_requires_a_name() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [[presets]]
            name = "minimal"
            "#,
        )
        .unwrap();
        let preset = &config.presets[0];
        assert_eq!(preset.model, None);
        assert_eq!(preset.system_prompt, None);
        assert!(preset.provider.is_none());
    }

    // --- auto_routing ------------------------------------------------------------

    #[test]
    fn auto_routing_defaults_to_absent() {
        let config = Config::from_toml_str("providers = {}").unwrap();
        assert!(config.auto_routing.is_none());
    }

    #[test]
    fn auto_routing_parses_the_three_tiers_and_default_thresholds() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [auto_routing]
            simple_model = "openai/gpt-4o-mini"
            medium_model = "smart"
            complex_model = "anthropic/claude-opus-4-8"
            "#,
        )
        .unwrap();
        let auto_routing = config.auto_routing.unwrap();
        assert_eq!(auto_routing.simple_model, "openai/gpt-4o-mini");
        assert_eq!(auto_routing.medium_model, "smart");
        assert_eq!(auto_routing.complex_model, "anthropic/claude-opus-4-8");
        assert_eq!(auto_routing.simple_max_score, 200);
        assert_eq!(auto_routing.medium_max_score, 800);
    }

    #[test]
    fn auto_routing_honors_explicit_thresholds() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [auto_routing]
            simple_model = "fast"
            medium_model = "smart"
            complex_model = "anthropic/claude-opus-4-8"
            simple_max_score = 50
            medium_max_score = 300
            "#,
        )
        .unwrap();
        let auto_routing = config.auto_routing.unwrap();
        assert_eq!(auto_routing.simple_max_score, 50);
        assert_eq!(auto_routing.medium_max_score, 300);
    }

    // --- moderation ----------------------------------------------------------------

    #[test]
    fn moderation_defaults_to_absent() {
        let config = Config::from_toml_str("providers = {}").unwrap();
        assert!(config.moderation.is_none());
    }

    #[test]
    fn moderation_parses_default_base_url_and_model() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [moderation]
            api_key_env = "OPENAI_API_KEY"
            "#,
        )
        .unwrap();
        let moderation = config.moderation.unwrap();
        assert_eq!(moderation.api_key_env, "OPENAI_API_KEY");
        assert_eq!(moderation.base_url, "https://api.openai.com/v1");
        assert_eq!(moderation.model, "omni-moderation-latest");
    }

    #[test]
    fn moderation_honors_explicit_base_url_and_model() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [moderation]
            api_key_env = "OPENAI_API_KEY"
            base_url = "http://localhost:9999/v1"
            model = "text-moderation-stable"
            "#,
        )
        .unwrap();
        let moderation = config.moderation.unwrap();
        assert_eq!(moderation.base_url, "http://localhost:9999/v1");
        assert_eq!(moderation.model, "text-moderation-stable");
    }

    // --- persistence backend -----------------------------------------------------

    #[test]
    fn persistence_backend_defaults_to_sqlite_when_absent() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [persistence]
            sqlite_path = "usage.db"
            "#,
        )
        .unwrap();
        let persistence = config.persistence.unwrap();
        assert_eq!(persistence.backend, PersistenceBackend::Sqlite);
        assert_eq!(persistence.sqlite_path.as_deref(), Some("usage.db"));
        assert_eq!(persistence.postgres_url_env, None);
    }

    #[test]
    fn persistence_postgres_backend_is_honored_when_set() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [persistence]
            backend = "postgres"
            postgres_url_env = "DATABASE_URL"
            "#,
        )
        .unwrap();
        let persistence = config.persistence.unwrap();
        assert_eq!(persistence.backend, PersistenceBackend::Postgres);
        assert_eq!(
            persistence.postgres_url_env.as_deref(),
            Some("DATABASE_URL")
        );
        assert_eq!(persistence.sqlite_path, None);
    }

    #[test]
    fn persistence_backend_rejects_unknown_value() {
        let err = Config::from_toml_str(
            r#"
            providers = {}

            [persistence]
            backend = "mysql"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("backend"));
    }

    #[test]
    fn postgres_tls_defaults_to_disable() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [persistence]
            backend = "postgres"
            postgres_url_env = "DATABASE_URL"
            "#,
        )
        .unwrap();
        assert_eq!(
            config.persistence.unwrap().postgres_tls,
            PostgresTlsMode::Disable
        );
    }

    #[test]
    fn postgres_tls_require_is_honored_when_set() {
        let config = Config::from_toml_str(
            r#"
            providers = {}

            [persistence]
            backend = "postgres"
            postgres_url_env = "DATABASE_URL"
            postgres_tls = "require"
            "#,
        )
        .unwrap();
        assert_eq!(
            config.persistence.unwrap().postgres_tls,
            PostgresTlsMode::Require
        );
    }

    #[test]
    fn postgres_tls_rejects_unknown_value() {
        let err = Config::from_toml_str(
            r#"
            providers = {}

            [persistence]
            backend = "postgres"
            postgres_url_env = "DATABASE_URL"
            postgres_tls = "verify-full"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("postgres_tls"));
    }

    // --- malformed input / from_file --------------------------------------------

    #[test]
    fn malformed_toml_syntax_is_a_parse_error_not_a_panic() {
        let err = Config::from_toml_str("providers = { [[[").unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn from_file_on_a_missing_path_reports_the_path_in_the_error() {
        let path = unique_temp_path("missing");
        let err = Config::from_file(&path).unwrap_err();
        assert!(err.to_string().contains(&path.display().to_string()));
    }

    #[test]
    fn from_file_on_invalid_toml_reports_the_path_in_the_error() {
        let path = unique_temp_path("invalid");
        std::fs::write(&path, "not valid toml [[[").unwrap();
        let err = Config::from_file(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.to_string().contains(&path.display().to_string()));
    }

    #[test]
    fn from_file_round_trips_a_well_formed_config() {
        let path = unique_temp_path("valid");
        std::fs::write(
            &path,
            r#"
            [providers.openai]
            kind = "openai"
            base_url = "https://api.openai.com/v1"
            api_key_env = "OPENAI_API_KEY"
            "#,
        )
        .unwrap();
        let config = Config::from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(config.providers["openai"].kind, ProviderKind::Openai);
    }

    // --- the shipped example config ---------------------------------------------

    #[test]
    fn config_example_toml_parses_and_matches_documented_shape() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config.example.toml");
        let config = Config::from_file(path).expect("config.example.toml should parse");

        let expected_providers = [
            ("openai", ProviderKind::Openai),
            ("anthropic", ProviderKind::Anthropic),
            ("gemini", ProviderKind::Gemini),
            ("groq", ProviderKind::Openai),
            ("together", ProviderKind::Openai),
            ("fireworks", ProviderKind::Openai),
        ];
        assert_eq!(config.providers.len(), expected_providers.len());
        for (name, kind) in expected_providers {
            let provider = config
                .providers
                .get(name)
                .unwrap_or_else(|| panic!("missing provider {name}"));
            assert_eq!(provider.kind, kind);
            assert!(!provider.zdr, "zdr is commented out in the example");
        }

        let aliases: Vec<&str> = config.routes.iter().map(|r| r.alias.as_str()).collect();
        assert_eq!(aliases, vec!["smart", "fast"]);
        assert_eq!(
            config.routes[0].chain,
            vec![
                "anthropic/claude-sonnet-5",
                "openai/gpt-4o",
                "gemini/gemini-2.0-flash",
            ]
        );

        assert_eq!(config.pricing.len(), 4);
        assert!(config
            .pricing
            .iter()
            .any(|p| p.model == "anthropic/claude-sonnet-5" && p.prompt_per_million == 3.0));
        let anthropic_pricing = config
            .pricing
            .iter()
            .find(|p| p.model == "anthropic/claude-sonnet-5")
            .unwrap();
        assert_eq!(anthropic_pricing.cache_read_per_million, Some(0.3));
        assert_eq!(anthropic_pricing.cache_write_per_million, Some(3.75));

        // Every [[clients]] entry in the example is commented out.
        assert!(config.clients.is_empty());
    }
}
