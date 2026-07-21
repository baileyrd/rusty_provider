use std::collections::HashMap;

use serde::Deserialize;

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
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key_env: None,
            default_rate_limit_rpm: None,
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
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub routes: Vec<RouteAlias>,
    #[serde(default)]
    pub pricing: Vec<PricingEntry>,
    #[serde(default)]
    pub clients: Vec<ClientConfig>,
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
