//! `rusty-provider-acp` -- the router as an ACP coding agent.
//!
//! Editors launch this as a subprocess and speak JSON-RPC on its stdin and
//! stdout, so **stdout belongs to the protocol**: logs go to stderr, and
//! nothing else may ever print to stdout.

use std::sync::Arc;

use rp_acp::{serve, Agent};
use rp_router::{Config, Router};

/// Used when `[acp].system_prompt` isn't set. Deliberately describes the
/// tools by the contract they actually have (client-mediated, permission
/// gated) rather than as generic filesystem access, since that's what
/// keeps a model from inventing shell tricks to get around them.
const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a coding agent working inside the user's editor. You help with software \
engineering tasks in their open workspace.

Working rules:
- Read before you change. Never edit a file whose current contents you haven't seen.
- Prefer edit_file for changes to existing files; use write_file only for new files \
or a full rewrite.
- Paths may be given relative to the workspace root.
- Writes and commands ask the user for permission, and they may say no. If they \
reject something, stop and ask how they'd like to proceed instead of trying \
another way around it.
- Use execute_command for builds, tests, linters and version control, and read the \
output before deciding what to do next.
- For a task with several steps, call update_plan first and keep it current as you go.
- When you're done, say briefly what you changed. If something failed or you left \
part of the task undone, say so plainly.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        // stdout is the protocol channel -- writing logs there would
        // corrupt the JSON-RPC stream.
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("rp_acp=info".parse()?),
        )
        .init();

    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.toml".to_string());
    let config = Config::from_file(&config_path)
        .map_err(|e| anyhow::anyhow!("{e}\n\nSee config.example.toml for a starting point."))?;

    let acp = config.acp.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "no [acp] section in {config_path} -- the ACP agent needs at least a model \
             to drive, e.g.\n\n[acp]\nmodel = \"anthropic/claude-sonnet-5\"\n"
        )
    })?;

    let router = Arc::new(Router::from_config(&config).await);
    let configured: Vec<&str> = router.configured_providers().collect();
    if configured.is_empty() {
        anyhow::bail!(
            "no providers configured (check that their api_key_env vars are set) -- \
             every prompt would fail"
        );
    }
    tracing::info!(providers = ?configured, model = %acp.model, "acp agent ready");

    let turn_config = rp_acp::turn::TurnConfig {
        model: acp.model,
        max_turn_requests: acp.max_turn_requests,
        system_prompt: acp
            .system_prompt
            .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string()),
        max_tokens: acp.max_tokens,
        temperature: acp.temperature,
        client_name: acp.client_name,
    };

    serve(tokio::io::stdin(), tokio::io::stdout(), |connection| {
        Arc::new(Agent::new(connection, router, turn_config))
    })
    .await?;

    tracing::info!("client disconnected");
    Ok(())
}
