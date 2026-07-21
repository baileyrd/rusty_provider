use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use rp_core::RateLimiter;
use rp_router::{Config, Router as ProviderRouter};
use rp_server::build_app;
use rp_server::state::AppState;
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
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
        client_keys: Arc::new(client_keys),
        default_rate_limit_rpm: config.server.default_rate_limit_rpm,
        rate_limiter: Arc::new(RateLimiter::new()),
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
    // it still fails downstream, but not with a rate-limit response).
    let first = client
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth("client-secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_ne!(first.status(), 429);

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
