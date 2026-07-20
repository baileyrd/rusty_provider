use std::sync::Arc;

use rp_router::Router;

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<Router>,
    /// Bearer token clients must present to this router's own API, if
    /// `server.api_key_env` was set in config and the env var resolved.
    pub api_key: Option<String>,
}
