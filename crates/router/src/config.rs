use std::collections::HashMap;

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
}

/// How a client's `budget_usd` cap resets.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BudgetPeriod {
    /// Never resets — a lifetime cap on this client's total tracked spend.
    #[default]
    Total,
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
            requests_per_minute = 500
            "#,
        )
        .unwrap();
        let provider = &config.providers["a"];
        assert!(provider.zdr);
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
