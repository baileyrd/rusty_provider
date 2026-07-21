use std::collections::HashMap;
use std::sync::Arc;

use rp_core::RateLimiter;
use rp_router::{ClientConfig, Router};

#[derive(Clone)]
pub struct AppState {
    /// Owns per-client spend budget tracking (`Router::check_client_budget`/
    /// `record_client_spend`) in addition to dispatch -- there's no
    /// separate client-budget type in this crate anymore, since sharing
    /// state with `[persistence]` requires living alongside it in
    /// `rp-router`.
    pub router: Arc<Router>,
    /// Bearer token clients must present to this router's own API, if
    /// `server.api_key_env` was set in config and the env var resolved.
    /// Any key in `client_keys` below also authenticates, independent of
    /// this field.
    pub api_key: Option<String>,
    /// Resolved API key string -> (client name, requests-per-minute).
    /// Presenting one of these keys both authenticates the request and
    /// buckets its rate limit under the client's name instead of the
    /// source-IP fallback.
    pub client_keys: Arc<HashMap<String, (String, u32)>>,
    /// Requests-per-minute limit for callers not matched to `client_keys`,
    /// bucketed by source IP. `None` means no limit for such callers.
    pub default_rate_limit_rpm: Option<u32>,
    pub rate_limiter: Arc<RateLimiter>,
    /// Every configured `[[clients]]` entry, for the admin API
    /// (`GET /v1/admin/clients`) to enumerate -- `client_keys` above is
    /// keyed by API key (for authenticating inbound requests), not by
    /// name, so it can't be listed the other way around.
    pub clients: Arc<Vec<ClientConfig>>,
    /// Bearer token that unlocks `/v1/admin/*`, if `server.admin_key_env`
    /// was set in config and the env var resolved. `None` disables the
    /// admin API entirely, independent of `api_key`/`client_keys` above.
    pub admin_key: Option<String>,
}
