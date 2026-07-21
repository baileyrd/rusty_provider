use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use rp_core::RateLimiter;
use rp_router::{Config, Router as ProviderRouter};
use rp_server::build_app;
use rp_server::state::AppState;

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

    let router = Arc::new(ProviderRouter::from_config(&config).await);
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

    let mut client_keys = HashMap::new();
    for client in &config.clients {
        match std::env::var(&client.api_key_env) {
            Ok(k) if !k.is_empty() => {
                client_keys.insert(k, (client.name.clone(), client.requests_per_minute));
            }
            _ => {
                tracing::warn!(client = %client.name, env_var = %client.api_key_env, "skipping client: API key env var not set");
            }
        }
    }
    if !client_keys.is_empty() {
        tracing::info!(clients = ?config.clients.iter().map(|c| &c.name).collect::<Vec<_>>(), "named clients ready");
    }

    let state = AppState {
        router,
        api_key,
        client_keys: Arc::new(client_keys),
        default_rate_limit_rpm: config.server.default_rate_limit_rpm,
        rate_limiter: Arc::new(RateLimiter::new()),
    };

    let app = build_app(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "rusty_provider listening");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}
