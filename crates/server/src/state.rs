use std::collections::HashMap;
use std::sync::Arc;

use rp_core::RateLimiter;
use rp_router::Router;

use crate::budget::ClientBudgets;

#[derive(Clone)]
pub struct AppState {
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
    /// Per-client spend tracking against each `[[clients]].budget_usd`, if
    /// configured. A client with no configured budget is unrestricted.
    pub client_budgets: Arc<ClientBudgets>,
}
