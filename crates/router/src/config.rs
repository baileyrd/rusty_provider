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
    /// gateway).
    #[serde(default)]
    pub api_key_env: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key_env: None,
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
}

#[derive(Debug, Deserialize, Clone)]
pub struct RouteAlias {
    pub alias: String,
    /// Ordered "provider/model" fallback chain, e.g.
    /// ["anthropic/claude-sonnet-5", "openai/gpt-4o"].
    pub chain: Vec<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub routes: Vec<RouteAlias>,
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
