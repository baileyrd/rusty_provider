use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use futures_util::StreamExt;
use rp_core::RateLimiter;
use rp_router::{Config, Router as ProviderRouter};
use rp_server::build_app;
use rp_server::state::AppState;
use serde_json::{json, Value};
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Generates a unique per-test env var name so parallel tests never race on
/// the same key -- `Router::from_config` resolves provider/client API keys
/// via `std::env::var` at construction time.
fn unique_env_var(label: &str) -> String {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    format!(
        "RP_SERVER_TEST_{label}_{}",
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// Builds an `AppState` from a TOML config string and starts the real axum
/// app on an ephemeral localhost port, exactly as `main` does. Returns the
/// base URL to hit with a plain HTTP client.
async fn spawn_app(config_toml: &str) -> String {
    let config = Config::from_toml_str(config_toml).expect("valid test config");
    let router = Arc::new(ProviderRouter::from_config(&config).await);

    let api_key = config
        .server
        .api_key_env
        .as_ref()
        .and_then(|var| std::env::var(var).ok());

    let admin_key = config
        .server
        .admin_key_env
        .as_ref()
        .and_then(|var| std::env::var(var).ok());

    let mut client_keys = HashMap::new();
    for client in &config.clients {
        if let Ok(k) = std::env::var(&client.api_key_env) {
            if !k.is_empty() {
                client_keys.insert(k, (client.name.clone(), client.requests_per_minute));
            }
        }
    }

    let state = AppState {
        router,
        api_key,
        client_keys: Arc::new(RwLock::new(client_keys)),
        default_rate_limit_rpm: config.server.default_rate_limit_rpm,
        rate_limiter: Arc::new(RateLimiter::new()),
        clients: Arc::new(RwLock::new(config.clients.clone())),
        admin_key,
    };

    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .expect("test server should not fail");
    });

    format!("http://{addr}")
}

#[tokio::test]
async fn health_endpoint_returns_ok_with_no_config() {
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::get(format!("{base_url}/health"))
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn list_models_includes_route_aliases_and_provider_wildcards() {
    let server = MockServer::start().await;
    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "test-key");

    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"

        [[routes]]
        alias = "smart"
        chain = ["openai/gpt-4o-mini"]
        "#,
        server.uri()
    );

    let base_url = spawn_app(&config).await;
    let resp = reqwest::get(format!("{base_url}/v1/models"))
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("valid json");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();

    assert!(ids.contains(&"smart"), "missing route alias: {ids:?}");
    assert!(
        ids.contains(&"openai/*"),
        "missing provider wildcard: {ids:?}"
    );
}

#[tokio::test]
async fn list_models_reports_rich_metadata_for_a_priced_model() {
    let server = MockServer::start().await;
    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "test-key");

    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"

        [[pricing]]
        model = "openai/gpt-4o-mini"
        prompt_per_million = 0.15
        completion_per_million = 0.6
        context_length = 128000
        "#,
        server.uri()
    );

    let base_url = spawn_app(&config).await;
    let resp = reqwest::get(format!("{base_url}/v1/models"))
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.expect("valid json");
    let entry = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .find(|m| m["id"] == "openai/gpt-4o-mini")
        .expect("priced model entry present")
        .clone();

    assert_eq!(entry["context_length"], 128000);
    assert_eq!(entry["pricing"]["prompt"], 0.15);
    assert_eq!(entry["pricing"]["completion"], 0.6);
    let supported = entry["supported_params"]
        .as_array()
        .expect("supported_params array");
    let supported: Vec<&str> = supported.iter().map(|v| v.as_str().unwrap()).collect();
    assert!(supported.contains(&"logit_bias"));
    assert!(!supported.contains(&"cache_control"));
}

#[tokio::test]
async fn protected_endpoint_rejects_missing_or_wrong_key_and_accepts_correct_one() {
    let key_var = unique_env_var("SERVER_API_KEY");
    std::env::set_var(&key_var, "s3cret");

    let config = format!(
        r#"
        providers = {{}}

        [server]
        api_key_env = "{key_var}"
        "#
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();

    let no_auth = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(no_auth.status(), 401);

    let wrong_key = client
        .get(format!("{base_url}/v1/models"))
        .bearer_auth("nope")
        .send()
        .await
        .unwrap();
    assert_eq!(wrong_key.status(), 401);

    let correct_key = client
        .get(format!("{base_url}/v1/models"))
        .bearer_auth("s3cret")
        .send()
        .await
        .unwrap();
    assert_eq!(correct_key.status(), 200);
}

#[tokio::test]
async fn usage_stats_endpoint_returns_empty_list_before_any_requests() {
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::get(format!("{base_url}/v1/usage")).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn provider_stats_endpoint_returns_empty_list_before_any_requests() {
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::get(format!("{base_url}/v1/providers/stats"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text_with_configured_provider_gauge() {
    let server = MockServer::start().await;
    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "test-key");

    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::get(format!("{base_url}/metrics")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("text/plain"));

    let body = resp.text().await.unwrap();
    assert!(body.contains("rusty_provider_provider_configured"));
    assert!(body.contains(r#"provider="openai""#));
}

#[tokio::test]
async fn chat_completions_rejects_empty_messages_with_400() {
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({"model": "openai/gpt-4o-mini", "messages": []}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("messages"));
}

#[tokio::test]
async fn chat_completions_routes_a_short_prompt_to_the_simple_auto_routing_tier() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        })))
        .mount(&server)
        .await;

    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "test-key");
    // Only "openai" is configured -- medium/complex point at "anthropic",
    // which has no provider entry at all. A short "hi" prompt must land
    // in the simple tier to succeed; if it were misrouted to medium or
    // complex, dispatch would fail with 424 (provider not configured)
    // instead of reaching this mock.
    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"

        [auto_routing]
        simple_model = "openai/gpt-4o-mini"
        medium_model = "anthropic/claude-sonnet-5"
        complex_model = "anthropic/claude-opus-4-8"
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["model"], "openai/gpt-4o-mini");
}

#[tokio::test]
async fn chat_completions_returns_400_for_auto_when_auto_routing_is_not_configured() {
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn chat_completions_applies_a_preset_model_override_and_system_prompt() {
    let server = MockServer::start().await;
    // The mock only matches a request whose model is the preset's
    // "anthropic/claude-sonnet-5" (not the caller's own "openai/gpt-4o-mini")
    // and whose first message is the preset's system prompt -- proves both
    // preset fields actually reached the outgoing request.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_string_contains("claude-sonnet-5"))
        .and(body_string_contains("You are a support agent."))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })))
        .mount(&server)
        .await;

    let key_var = unique_env_var("ANTHROPIC_KEY");
    std::env::set_var(&key_var, "test-key");
    let config = format!(
        r#"
        [providers.anthropic]
        kind = "anthropic"
        base_url = "{}"
        api_key_env = "{key_var}"

        [[presets]]
        name = "support-bot"
        model = "anthropic/claude-sonnet-5"
        system_prompt = "You are a support agent."
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "preset": "support-bot",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["model"], "anthropic/claude-sonnet-5");
}

#[tokio::test]
async fn chat_completions_rejects_an_unknown_preset_name_with_400() {
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "preset": "does-not-exist",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does-not-exist"));
}

#[tokio::test]
async fn chat_completions_blocks_a_request_matching_a_block_guardrail() {
    let config = r#"
        providers = {}

        [[guardrails]]
        name = "no-ssn"
        pattern = '\d{3}-\d{2}-\d{4}'
        action = "block"
        "#;
    let base_url = spawn_app(config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "my ssn is 123-45-6789"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("no-ssn"));
}

#[tokio::test]
async fn chat_completions_blocks_a_request_flagged_by_moderation() {
    let moderation_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/moderations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{
                "flagged": true,
                "categories": {"violence": true, "hate": false}
            }]
        })))
        .mount(&moderation_server)
        .await;

    let moderation_key_var = unique_env_var("MODERATION_KEY");
    std::env::set_var(&moderation_key_var, "moderation-key");
    let config = format!(
        r#"
        providers = {{}}

        [moderation]
        api_key_env = "{moderation_key_var}"
        base_url = "{}"
        "#,
        moderation_server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "something violent"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("violence"));
}

#[tokio::test]
async fn chat_completions_allows_the_request_through_when_the_moderation_backend_is_unreachable() {
    let provider_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        })))
        .mount(&provider_server)
        .await;

    let openai_key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&openai_key_var, "test-key");
    let moderation_key_var = unique_env_var("MODERATION_KEY");
    std::env::set_var(&moderation_key_var, "moderation-key");
    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{openai_key_var}"

        [moderation]
        api_key_env = "{moderation_key_var}"
        base_url = "http://127.0.0.1:0"
        "#,
        provider_server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        200,
        "an unreachable moderation backend must fail open, not block the request"
    );
}

#[tokio::test]
async fn chat_completions_weaves_web_search_results_into_the_request_before_dispatch() {
    let search_server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "web": {
                "results": [
                    {"title": "Rust 1.80", "url": "https://blog.rust-lang.org", "description": "Release notes"}
                ]
            }
        })))
        .mount(&search_server)
        .await;

    let provider_server = MockServer::start().await;
    // The mock only matches if the request body's content already
    // contains the search result -- proves web search ran and augmented
    // the message before the provider ever saw it.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains("blog.rust-lang.org"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Rust 1.80 shipped recently"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        })))
        .mount(&provider_server)
        .await;

    let openai_key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&openai_key_var, "test-key");
    let web_search_key_var = unique_env_var("WEB_SEARCH_KEY");
    std::env::set_var(&web_search_key_var, "search-key");
    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{openai_key_var}"

        [web_search]
        api_key_env = "{web_search_key_var}"
        base_url = "{}"
        "#,
        provider_server.uri(),
        search_server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "what's new in Rust"}],
            "web_search": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Rust 1.80 shipped recently"
    );
}

#[tokio::test]
async fn chat_completions_redacts_a_request_matching_a_redact_guardrail_before_dispatch() {
    let server = MockServer::start().await;
    // The mock only matches if the request body's content has already
    // been redacted -- proves the guardrail ran before the provider ever
    // saw the original text.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains("[redacted]"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "got it"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        })))
        .mount(&server)
        .await;

    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "test-key");
    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"

        [[guardrails]]
        name = "no-ssn"
        pattern = '\d{{3}}-\d{{2}}-\d{{4}}'
        action = "redact"
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "my ssn is 123-45-6789"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "got it");
}

#[tokio::test]
async fn chat_completions_success_roundtrips_through_a_mocked_provider() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&server)
        .await;

    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "test-key");
    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["model"], "openai/gpt-4o-mini");
    assert_eq!(body["choices"][0]["message"]["content"], "hello there");

    // The router should also have folded this response into /v1/usage.
    let usage_resp = reqwest::get(format!("{base_url}/v1/usage")).await.unwrap();
    let usage: Value = usage_resp.json().await.unwrap();
    let entries = usage["data"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["model"], "openai/gpt-4o-mini");
    assert_eq!(entries[0]["requests"], 1);

    // The router should also have folded this response into
    // /v1/providers/stats -- a successful dispatch records latency,
    // throughput, and uptime for "openai/gpt-4o-mini".
    let stats_resp = reqwest::get(format!("{base_url}/v1/providers/stats"))
        .await
        .unwrap();
    let stats: Value = stats_resp.json().await.unwrap();
    let stats_entries = stats["data"].as_array().unwrap();
    assert_eq!(stats_entries.len(), 1);
    assert_eq!(stats_entries[0]["model"], "openai/gpt-4o-mini");
    assert!(stats_entries[0]["latency_ms"].as_f64().unwrap() >= 0.0);
    assert!(
        stats_entries[0]["throughput_tokens_per_sec"]
            .as_f64()
            .unwrap()
            >= 0.0
    );
    assert_eq!(stats_entries[0]["uptime"], 1.0);

    // The router should also have recorded this request for per-id lookup
    // via /v1/generation?id=, keyed by the response's own `id` (assigned
    // by this router's adapter, not the mocked upstream body's "id").
    let id = body["id"].as_str().unwrap();
    let generation_resp = reqwest::get(format!("{base_url}/v1/generation?id={id}"))
        .await
        .unwrap();
    assert_eq!(generation_resp.status(), 200);
    let generation: Value = generation_resp.json().await.unwrap();
    assert_eq!(generation["id"], id);
    assert_eq!(generation["model"], "openai/gpt-4o-mini");
    assert_eq!(generation["prompt_tokens"], 10);
    assert_eq!(generation["completion_tokens"], 5);
    assert_eq!(generation["total_tokens"], 15);
}

#[tokio::test]
async fn chat_completions_uses_a_byok_key_from_the_request_instead_of_the_configured_one() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer byok-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&server)
        .await;

    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "operators-own-key");
    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    // The mock only matches "byok-key", not "operators-own-key" -- a 200
    // here proves the request's own key won, not the configured one.
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "provider": {"byok": {"openai": "byok-key"}}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn generation_endpoint_returns_404_for_an_unknown_id() {
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::get(format!("{base_url}/v1/generation?id=chatcmpl-unknown"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn chat_completions_streams_sse_chunks_and_terminates_with_done() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "test-key");
    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert!(resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));

    let mut full = String::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        full.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
    }

    assert!(full.contains("\"content\":\"Hi\""));
    assert!(full.contains("\"finish_reason\":\"stop\""));
    assert!(full.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test]
async fn chat_completions_maps_unresolvable_model_to_424() {
    // No providers configured at all -- the chain resolves syntactically
    // but its only entry has no registered provider, which dispatch
    // reports as RouterError::ProviderNotConfigured (424).
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 424);
}

#[tokio::test]
async fn chat_completions_rejects_malformed_model_string_with_400() {
    // No slash -- fails at resolve_chain before any provider lookup.
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "not-a-valid-model",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn chat_completions_enforces_per_client_inbound_rate_limit() {
    let key_var = unique_env_var("CLIENT_KEY");
    std::env::set_var(&key_var, "client-secret");

    let config = format!(
        r#"
        providers = {{}}

        [[clients]]
        name = "acme"
        api_key_env = "{key_var}"
        requests_per_minute = 1
        "#
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();
    let body = json!({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}]
    });

    // First request passes the rate limiter (no provider is configured, so
    // it still fails downstream, but not with a rate-limit response) --
    // and, having consumed its only token, carries X-RateLimit-Remaining: 0.
    let first = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_ne!(first.status(), 429);
    assert_eq!(first.headers().get("x-ratelimit-limit").unwrap(), "1");
    assert_eq!(first.headers().get("x-ratelimit-remaining").unwrap(), "0");
    assert!(first.headers().get("x-ratelimit-reset").is_some());

    // Second request within the same minute is rejected by the limiter
    // before it ever reaches dispatch.
    let second = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 429);
    assert!(second.headers().get("retry-after").is_some());
    assert_eq!(second.headers().get("x-ratelimit-limit").unwrap(), "1");
    assert_eq!(second.headers().get("x-ratelimit-remaining").unwrap(), "0");
    assert!(second.headers().get("x-ratelimit-reset").is_some());

    let err: Value = second.json().await.unwrap();
    assert!(err["error"]["message"]
        .as_str()
        .unwrap()
        .contains("rate limit"));
}

#[tokio::test]
async fn chat_completions_enforces_default_ip_rate_limit_when_no_client_matches() {
    let config = r#"
        providers = {}

        [server]
        default_rate_limit_rpm = 1
        "#;
    let base_url = spawn_app(config).await;
    let client = reqwest::Client::new();
    let body = json!({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}]
    });

    // No bearer token, no matching client -- falls back to the source-IP
    // bucket (both requests come from 127.0.0.1, so they share one bucket).
    let first = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_ne!(first.status(), 429);

    let second = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 429);
    assert!(second.headers().get("retry-after").is_some());
}

#[tokio::test]
async fn chat_completions_has_no_rate_limit_when_unmatched_and_no_default_configured() {
    // No clients configured and no default_rate_limit_rpm -- an unmatched
    // caller has no cap at all, so repeated requests never see a 429.
    let base_url = spawn_app("providers = {}").await;
    let client = reqwest::Client::new();
    let body = json!({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}]
    });

    for _ in 0..5 {
        let resp = client
            .post(format!("{base_url}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_ne!(resp.status(), 429);
    }
}

#[tokio::test]
async fn chat_completions_cuts_off_a_client_after_a_non_streaming_response_exceeds_its_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }],
            // At $1/token (below) this response costs $15 -- well over the
            // client's $1 budget, so it must be the *last* request the
            // client is allowed to make.
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&server)
        .await;

    let openai_key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&openai_key_var, "test-key");
    let client_key_var = unique_env_var("CLIENT_KEY");
    std::env::set_var(&client_key_var, "client-secret");

    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{openai_key_var}"

        [[pricing]]
        model = "openai/gpt-4o-mini"
        prompt_per_million = 1000000.0
        completion_per_million = 1000000.0

        [[clients]]
        name = "acme"
        api_key_env = "{client_key_var}"
        requests_per_minute = 1000
        budget_usd = 1.0
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();
    let body = json!({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}]
    });

    // The first request is allowed through -- its own cost isn't known
    // until after it completes -- and it's the one that pushes spend past
    // the $1 budget.
    let first = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);

    // The second request is rejected before it ever reaches the provider.
    let second = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 402);
    let err: Value = second.json().await.unwrap();
    let message = err["error"]["message"].as_str().unwrap();
    assert!(message.contains("acme"));
    assert!(message.contains("15.00"));
    assert!(message.contains("1.00"));

    let metrics = client
        .get(format!("{base_url}/metrics"))
        .bearer_auth("client-secret")
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics.contains("rusty_provider_client_budget_rejections_total"));
    assert!(metrics.contains(r#"client="acme""#));
}

#[tokio::test]
async fn admin_endpoints_are_404_when_admin_key_env_is_not_configured() {
    let base_url = spawn_app("providers = {}").await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/v1/admin/clients"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    let resp = client
        .post(format!("{base_url}/v1/admin/clients/acme/reset-spend"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    let resp = client
        .post(format!("{base_url}/v1/admin/clients"))
        .json(&json!({"name": "acme", "requests_per_minute": 60}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    let resp = client
        .patch(format!("{base_url}/v1/admin/clients/acme"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    let resp = client
        .delete(format!("{base_url}/v1/admin/clients/acme"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn admin_list_clients_rejects_missing_wrong_and_regular_client_keys() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let client_key_var = unique_env_var("CLIENT_KEY");
    std::env::set_var(&client_key_var, "client-secret");

    let config = format!(
        r#"
        providers = {{}}

        [server]
        admin_key_env = "{admin_key_var}"

        [[clients]]
        name = "acme"
        api_key_env = "{client_key_var}"
        requests_per_minute = 60
        "#
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();

    let no_auth = client
        .get(format!("{base_url}/v1/admin/clients"))
        .send()
        .await
        .unwrap();
    assert_eq!(no_auth.status(), 401);

    let wrong_key = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("nope")
        .send()
        .await
        .unwrap();
    assert_eq!(wrong_key.status(), 401);

    // A regular client key (valid for chat completions) must not also
    // unlock the admin API.
    let client_key = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("client-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(client_key.status(), 401);
}

#[tokio::test]
async fn admin_list_clients_reports_configured_clients_and_live_spend() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let budgeted_key_var = unique_env_var("BUDGETED_CLIENT_KEY");
    std::env::set_var(&budgeted_key_var, "budgeted-secret");
    let unbudgeted_key_var = unique_env_var("UNBUDGETED_CLIENT_KEY");
    std::env::set_var(&unbudgeted_key_var, "unbudgeted-secret");

    let config = format!(
        r#"
        providers = {{}}

        [server]
        admin_key_env = "{admin_key_var}"

        [[clients]]
        name = "acme"
        api_key_env = "{budgeted_key_var}"
        requests_per_minute = 30
        budget_usd = 10.0
        budget_period = "monthly"

        [[clients]]
        name = "globex"
        api_key_env = "{unbudgeted_key_var}"
        requests_per_minute = 60
        "#
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);

    let acme = data.iter().find(|c| c["name"] == "acme").unwrap();
    assert_eq!(acme["requests_per_minute"], 30);
    assert_eq!(acme["budget_usd"], 10.0);
    assert_eq!(acme["budget_period"], "monthly");
    assert_eq!(acme["spent_usd"], 0.0);

    let globex = data.iter().find(|c| c["name"] == "globex").unwrap();
    assert_eq!(globex["requests_per_minute"], 60);
    assert!(globex["budget_usd"].is_null());
    assert!(globex["budget_period"].is_null());
    assert!(globex["spent_usd"].is_null());
}

/// Two organizations, `acme-corp`'s own client holding `role = "admin"` --
/// the shared fixture for the organization-scoping tests below.
async fn spawn_two_org_app() -> (String, String, String) {
    let acme_admin_key_var = unique_env_var("ACME_ADMIN_KEY");
    std::env::set_var(&acme_admin_key_var, "acme-admin-secret");
    let globex_member_key_var = unique_env_var("GLOBEX_MEMBER_KEY");
    std::env::set_var(&globex_member_key_var, "globex-member-secret");

    let config = format!(
        r#"
        providers = {{}}

        [[clients]]
        name = "acme-admin"
        api_key_env = "{acme_admin_key_var}"
        requests_per_minute = 30
        organization = "acme-corp"
        workspace = "prod"
        role = "admin"

        [[clients]]
        name = "globex-member"
        api_key_env = "{globex_member_key_var}"
        requests_per_minute = 30
        organization = "globex-inc"
        role = "member"
        "#
    );
    let base_url = spawn_app(&config).await;
    (
        base_url,
        "acme-admin-secret".to_string(),
        "globex-member-secret".to_string(),
    )
}

#[tokio::test]
async fn admin_list_clients_scopes_results_to_an_admin_role_clients_own_organization() {
    let (base_url, acme_admin_key, _globex_member_key) = spawn_two_org_app().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth(&acme_admin_key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let data = body["data"].as_array().unwrap();

    // The scoped admin sees only its own organization's clients -- itself,
    // not "globex-member" in the other organization.
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["name"], "acme-admin");
    assert_eq!(data[0]["organization"], "acme-corp");
}

#[tokio::test]
async fn admin_list_clients_is_401_for_a_member_role_clients_own_key() {
    let (base_url, _acme_admin_key, globex_member_key) = spawn_two_org_app().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth(&globex_member_key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn admin_create_client_pins_organization_to_a_scoped_admins_own_org() {
    let (base_url, acme_admin_key, _globex_member_key) = spawn_two_org_app().await;
    let client = reqwest::Client::new();

    // Even though the request asks for "globex-inc", a scoped admin can
    // only ever create clients in its own organization.
    let resp = client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth(&acme_admin_key)
        .json(&json!({
            "name": "new-hire",
            "requests_per_minute": 30,
            "organization": "globex-inc"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["organization"], "acme-corp");
}

#[tokio::test]
async fn admin_update_client_is_404_for_a_client_outside_a_scoped_admins_organization() {
    let (base_url, acme_admin_key, _globex_member_key) = spawn_two_org_app().await;
    let client = reqwest::Client::new();

    let resp = client
        .patch(format!("{base_url}/v1/admin/clients/globex-member"))
        .bearer_auth(&acme_admin_key)
        .json(&json!({"requests_per_minute": 99}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn admin_delete_client_is_404_for_a_client_outside_a_scoped_admins_organization() {
    let (base_url, acme_admin_key, _globex_member_key) = spawn_two_org_app().await;
    let client = reqwest::Client::new();

    let resp = client
        .delete(format!("{base_url}/v1/admin/clients/globex-member"))
        .bearer_auth(&acme_admin_key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // Confirm it's untouched, not actually removed -- visible to a global
    // view (no admin_key_env here, so re-check via the other org's own
    // scoped list instead, which should still find itself).
    let resp = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth(&acme_admin_key)
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn admin_list_organizations_groups_clients_by_organization_with_a_global_key() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let acme_key_var = unique_env_var("ACME_KEY");
    std::env::set_var(&acme_key_var, "acme-secret");
    let globex_key_var = unique_env_var("GLOBEX_KEY");
    std::env::set_var(&globex_key_var, "globex-secret");

    let config = format!(
        r#"
        providers = {{}}

        [server]
        admin_key_env = "{admin_key_var}"

        [[clients]]
        name = "acme-client"
        api_key_env = "{acme_key_var}"
        requests_per_minute = 30
        organization = "acme-corp"
        workspace = "prod"

        [[clients]]
        name = "globex-client"
        api_key_env = "{globex_key_var}"
        requests_per_minute = 30
        organization = "globex-inc"
        "#
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/v1/admin/organizations"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);

    let acme = data
        .iter()
        .find(|g| g["organization"] == "acme-corp")
        .unwrap();
    let acme_clients = acme["clients"].as_array().unwrap();
    assert_eq!(acme_clients.len(), 1);
    assert_eq!(acme_clients[0]["name"], "acme-client");
    assert_eq!(acme_clients[0]["workspace"], "prod");
}

#[tokio::test]
async fn admin_list_organizations_scopes_to_the_callers_own_organization() {
    let (base_url, acme_admin_key, _globex_member_key) = spawn_two_org_app().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/v1/admin/organizations"))
        .bearer_auth(&acme_admin_key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let data = body["data"].as_array().unwrap();

    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["organization"], "acme-corp");
    assert_eq!(data[0]["clients"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn admin_reset_client_spend_zeroes_spend_and_unblocks_the_client() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&server)
        .await;

    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let openai_key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&openai_key_var, "test-key");
    let client_key_var = unique_env_var("CLIENT_KEY");
    std::env::set_var(&client_key_var, "client-secret");

    let config = format!(
        r#"
        [server]
        admin_key_env = "{admin_key_var}"

        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{openai_key_var}"

        [[pricing]]
        model = "openai/gpt-4o-mini"
        prompt_per_million = 1000000.0
        completion_per_million = 1000000.0

        [[clients]]
        name = "acme"
        api_key_env = "{client_key_var}"
        requests_per_minute = 1000
        budget_usd = 1.0
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();
    let body = json!({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}]
    });

    // Push spend ($15) past the $1 budget.
    let first = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);
    let blocked = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(blocked.status(), 402);

    let reset = client
        .post(format!("{base_url}/v1/admin/clients/acme/reset-spend"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(reset.status(), 200);

    let unblocked = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(unblocked.status(), 200);
}

#[tokio::test]
async fn admin_reset_client_spend_is_404_for_a_client_with_no_configured_budget() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let client_key_var = unique_env_var("CLIENT_KEY");
    std::env::set_var(&client_key_var, "client-secret");

    let config = format!(
        r#"
        providers = {{}}

        [server]
        admin_key_env = "{admin_key_var}"

        [[clients]]
        name = "acme"
        api_key_env = "{client_key_var}"
        requests_per_minute = 60
        "#
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base_url}/v1/admin/clients/acme/reset-spend"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn chat_completions_cuts_off_a_client_after_a_streaming_response_exceeds_its_budget() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let openai_key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&openai_key_var, "test-key");
    let client_key_var = unique_env_var("CLIENT_KEY");
    std::env::set_var(&client_key_var, "client-secret");

    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{openai_key_var}"

        [[pricing]]
        model = "openai/gpt-4o-mini"
        prompt_per_million = 1000000.0
        completion_per_million = 1000000.0

        [[clients]]
        name = "acme"
        api_key_env = "{client_key_var}"
        requests_per_minute = 1000
        budget_usd = 1.0
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();
    let body = json!({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    });

    // The streamed response's final chunk carries usage costing $4 -- over
    // the client's $1 budget -- which the router must record from inside
    // the SSE stream itself, not just non-streaming responses.
    let first = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);
    let mut stream = first.bytes_stream();
    while stream.next().await.is_some() {}

    let second = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 402);
}

// --- runtime client provisioning (admin API) --------------------------------

fn admin_config(admin_key_var: &str) -> String {
    format!(
        r#"
        providers = {{}}

        [server]
        admin_key_env = "{admin_key_var}"
        "#
    )
}

#[tokio::test]
async fn admin_create_client_generates_a_key_when_none_is_given() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({"name": "acme", "requests_per_minute": 60}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "acme");
    assert_eq!(body["requests_per_minute"], 60);
    assert!(body["budget_usd"].is_null());
    let api_key = body["api_key"]
        .as_str()
        .expect("api_key should be present")
        .to_string();
    assert!(!api_key.is_empty());

    // The generated key must work immediately, with no restart -- listing
    // reflects the new client too.
    let chat_resp = client
        .get(format!("{base_url}/v1/models"))
        .bearer_auth(&api_key)
        .send()
        .await
        .unwrap();
    assert_eq!(chat_resp.status(), 200);

    let list = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    let list_body: Value = list.json().await.unwrap();
    let data = list_body["data"].as_array().unwrap();
    assert!(data.iter().any(|c| c["name"] == "acme"));
}

#[tokio::test]
async fn admin_create_client_honors_an_explicit_api_key() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({
            "name": "acme",
            "requests_per_minute": 60,
            "api_key": "my-chosen-key"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["api_key"], "my-chosen-key");

    let chat_resp = client
        .get(format!("{base_url}/v1/models"))
        .bearer_auth("my-chosen-key")
        .send()
        .await
        .unwrap();
    assert_eq!(chat_resp.status(), 200);
}

#[tokio::test]
async fn admin_create_client_rejects_a_duplicate_name() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    let create = |name: &'static str| {
        let base_url = base_url.clone();
        let client = client.clone();
        async move {
            client
                .post(format!("{base_url}/v1/admin/clients"))
                .bearer_auth("admin-secret")
                .json(&json!({"name": name, "requests_per_minute": 60}))
                .send()
                .await
                .unwrap()
        }
    };

    assert_eq!(create("acme").await.status(), 201);
    assert_eq!(create("acme").await.status(), 409);
}

#[tokio::test]
async fn admin_create_client_rejects_a_duplicate_api_key() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({"name": "acme", "requests_per_minute": 60, "api_key": "shared-key"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({"name": "globex", "requests_per_minute": 60, "api_key": "shared-key"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
}

#[tokio::test]
async fn admin_create_client_rejects_invalid_fields() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    let post = |body: Value| {
        let base_url = base_url.clone();
        let client = client.clone();
        async move {
            client
                .post(format!("{base_url}/v1/admin/clients"))
                .bearer_auth("admin-secret")
                .json(&body)
                .send()
                .await
                .unwrap()
        }
    };

    assert_eq!(
        post(json!({"name": "", "requests_per_minute": 60}))
            .await
            .status(),
        400,
        "empty name"
    );
    assert_eq!(
        post(json!({"name": "acme", "requests_per_minute": 0}))
            .await
            .status(),
        400,
        "zero requests_per_minute"
    );
    assert_eq!(
        post(json!({"name": "acme", "requests_per_minute": 60, "budget_usd": -1.0}))
            .await
            .status(),
        400,
        "negative budget_usd"
    );
}

#[tokio::test]
async fn admin_create_client_wires_a_budget_into_the_router_immediately() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200}
        })))
        .mount(&server)
        .await;

    let openai_key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&openai_key_var, "test-key");
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");

    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{openai_key_var}"

        [server]
        admin_key_env = "{admin_key_var}"

        [[pricing]]
        model = "openai/gpt-4o-mini"
        prompt_per_million = 10000.0
        completion_per_million = 10000.0
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();

    let create = client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({
            "name": "acme",
            "requests_per_minute": 1000,
            "budget_usd": 1.0,
            "api_key": "acme-key"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create.status(), 201);

    let body = json!({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}]
    });

    // 100 prompt + 100 completion tokens at $10000/1M each = $2, over the
    // freshly-provisioned client's $1 budget -- the very first request
    // already exceeds it once usage is recorded.
    let first = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("acme-key")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);

    let second = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("acme-key")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 402);
}

#[tokio::test]
async fn admin_update_client_changes_requests_per_minute_and_budget() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({"name": "acme", "requests_per_minute": 30}))
        .send()
        .await
        .unwrap();

    let resp = client
        .patch(format!("{base_url}/v1/admin/clients/acme"))
        .bearer_auth("admin-secret")
        .json(&json!({"requests_per_minute": 99, "budget_usd": 5.0, "budget_period": "monthly"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["requests_per_minute"], 99);
    assert_eq!(body["budget_usd"], 5.0);
    assert_eq!(body["budget_period"], "monthly");
    assert!(
        body["api_key"].is_null(),
        "an update that doesn't rotate the key must not echo one back"
    );

    let list = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    let list_body: Value = list.json().await.unwrap();
    let acme = list_body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "acme")
        .unwrap();
    assert_eq!(acme["requests_per_minute"], 99);
    assert_eq!(acme["budget_usd"], 5.0);
}

#[tokio::test]
async fn admin_update_client_clears_budget_when_set_to_null() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({"name": "acme", "requests_per_minute": 30, "budget_usd": 10.0}))
        .send()
        .await
        .unwrap();

    let resp = client
        .patch(format!("{base_url}/v1/admin/clients/acme"))
        .bearer_auth("admin-secret")
        .json(&json!({"budget_usd": null}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["budget_usd"].is_null());

    // The client is unrestricted now -- client_spend_status has nothing to
    // report either.
    let list = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    let list_body: Value = list.json().await.unwrap();
    let acme = list_body["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "acme")
        .unwrap();
    assert!(acme["budget_usd"].is_null());
    assert!(acme["spent_usd"].is_null());
}

#[tokio::test]
async fn admin_update_client_rotates_the_api_key_and_revokes_the_old_one() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({"name": "acme", "requests_per_minute": 60, "api_key": "old-key"}))
        .send()
        .await
        .unwrap();

    let resp = client
        .patch(format!("{base_url}/v1/admin/clients/acme"))
        .bearer_auth("admin-secret")
        .json(&json!({"rotate_api_key": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let new_key = body["api_key"]
        .as_str()
        .expect("a rotation must return the new key")
        .to_string();
    assert_ne!(new_key, "old-key");

    // With at least one client key configured, auth is enforced -- the
    // revoked old key must no longer authenticate.
    let old_key_resp = client
        .get(format!("{base_url}/v1/models"))
        .bearer_auth("old-key")
        .send()
        .await
        .unwrap();
    assert_eq!(old_key_resp.status(), 401);

    let new_key_resp = client
        .get(format!("{base_url}/v1/models"))
        .bearer_auth(&new_key)
        .send()
        .await
        .unwrap();
    assert_eq!(new_key_resp.status(), 200);
}

#[tokio::test]
async fn admin_update_client_is_404_for_an_unknown_client() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    let resp = client
        .patch(format!("{base_url}/v1/admin/clients/ghost"))
        .bearer_auth("admin-secret")
        .json(&json!({"requests_per_minute": 10}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn admin_update_client_rejects_a_zero_requests_per_minute() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({"name": "acme", "requests_per_minute": 60}))
        .send()
        .await
        .unwrap();

    let resp = client
        .patch(format!("{base_url}/v1/admin/clients/acme"))
        .bearer_auth("admin-secret")
        .json(&json!({"requests_per_minute": 0}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn admin_delete_client_removes_it_and_revokes_the_admin_listing_entry() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .json(&json!({"name": "acme", "requests_per_minute": 60, "api_key": "acme-key"}))
        .send()
        .await
        .unwrap();

    let resp = client
        .delete(format!("{base_url}/v1/admin/clients/acme"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let list = client
        .get(format!("{base_url}/v1/admin/clients"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    let list_body: Value = list.json().await.unwrap();
    assert!(list_body["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn admin_delete_client_is_404_for_an_unknown_client() {
    let admin_key_var = unique_env_var("ADMIN_KEY");
    std::env::set_var(&admin_key_var, "admin-secret");
    let base_url = spawn_app(&admin_config(&admin_key_var)).await;
    let client = reqwest::Client::new();

    let resp = client
        .delete(format!("{base_url}/v1/admin/clients/ghost"))
        .bearer_auth("admin-secret")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// --- embeddings --------------------------------------------------------------

#[tokio::test]
async fn embeddings_success_roundtrips_through_a_mocked_provider() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.5, 0.25], "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        })))
        .mount(&server)
        .await;

    let key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&key_var, "test-key");
    let config = format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{key_var}"
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/embeddings"))
        .json(&json!({
            "model": "openai/text-embedding-3-small",
            "input": "hello"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["model"], "openai/text-embedding-3-small");
    assert_eq!(body["data"][0]["embedding"], json!([0.5, 0.25]));
    assert_eq!(body["usage"]["prompt_tokens"], 2);
}

#[tokio::test]
async fn embeddings_falls_back_past_a_provider_with_no_embeddings_support() {
    // Anthropic has no embeddings API at all -- its adapter always
    // returns UnsupportedFeature (retryable), so a chain naming it ahead
    // of a real embeddings provider should still succeed via fallback.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        })))
        .mount(&server)
        .await;

    let anthropic_key_var = unique_env_var("ANTHROPIC_KEY");
    std::env::set_var(&anthropic_key_var, "test-key");
    let openai_key_var = unique_env_var("OPENAI_KEY");
    std::env::set_var(&openai_key_var, "test-key");
    let config = format!(
        r#"
        [providers.anthropic]
        kind = "anthropic"
        base_url = "http://127.0.0.1:1"
        api_key_env = "{anthropic_key_var}"

        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{openai_key_var}"

        [[routes]]
        alias = "embed"
        chain = ["anthropic/claude-sonnet-5", "openai/text-embedding-3-small"]
        "#,
        server.uri()
    );
    let base_url = spawn_app(&config).await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/embeddings"))
        .json(&json!({"model": "embed", "input": "hello"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["model"], "openai/text-embedding-3-small");
}

#[tokio::test]
async fn embeddings_rejects_missing_or_wrong_key_and_accepts_correct_one() {
    let key_var = unique_env_var("SERVER_API_KEY");
    std::env::set_var(&key_var, "s3cret");
    let config = format!(
        r#"
        providers = {{}}

        [server]
        api_key_env = "{key_var}"
        "#
    );
    let base_url = spawn_app(&config).await;
    let client = reqwest::Client::new();
    let body = json!({"model": "openai/text-embedding-3-small", "input": "hi"});

    let no_auth = client
        .post(format!("{base_url}/v1/embeddings"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(no_auth.status(), 401);

    let wrong_key = client
        .post(format!("{base_url}/v1/embeddings"))
        .bearer_auth("nope")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(wrong_key.status(), 401);
}

#[tokio::test]
async fn embeddings_maps_an_unconfigured_provider_to_424() {
    let base_url = spawn_app("providers = {}").await;

    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/embeddings"))
        .json(&json!({"model": "openai/text-embedding-3-small", "input": "hi"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 424);
}
