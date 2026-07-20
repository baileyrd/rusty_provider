mod errors;
mod routes;
mod state;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router as AxumRouter;
use rp_router::{Config, Router as ProviderRouter};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rp_server=info".parse()?),
        )
        .init();

    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.toml".to_string());
    let config = Config::from_file(&config_path)
        .map_err(|e| anyhow::anyhow!("{e}\n\nSee config.example.toml for a starting point."))?;

    let router = Arc::new(ProviderRouter::from_config(&config));
    let configured: Vec<&str> = router.configured_providers().collect();
    if configured.is_empty() {
        tracing::warn!("no providers configured (check that their api_key_env vars are set) — every request will fail");
    } else {
        tracing::info!(providers = ?configured, "providers ready");
    }

    let api_key = config
        .server
        .api_key_env
        .as_ref()
        .and_then(|var| std::env::var(var).ok());
    if api_key.is_none() && config.server.api_key_env.is_some() {
        tracing::warn!(
            "server.api_key_env is set in config but the env var isn't — running with no auth"
        );
    }

    let state = AppState { router, api_key };

    let app = AxumRouter::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "rusty_provider listening");
    axum::serve(listener, app).await?;

    Ok(())
}
