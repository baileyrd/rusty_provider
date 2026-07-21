use std::convert::Infallible;
use std::net::SocketAddr;

use axum::extract::{ConnectInfo, State};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::{stream, StreamExt};
use rp_core::{ChatRequest, ModelInfo};
use rp_router::UsageStats;
use serde::Serialize;
use serde_json::json;

use crate::errors::{json_error, json_error_with_retry_after, router_error_response};
use crate::state::AppState;

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

/// Accepts either the legacy shared `server.api_key_env` token or any
/// configured `[[clients]]` key. Auth is skipped entirely if neither is
/// configured.
pub fn check_auth(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    if state.api_key.is_none() && state.client_keys.is_empty() {
        return None;
    }

    let Some(token) = bearer_token(headers) else {
        return Some(json_error(401, "missing or invalid API key"));
    };

    let legacy_ok = state.api_key.as_deref() == Some(token);
    let client_ok = state.client_keys.contains_key(token);

    if legacy_ok || client_ok {
        None
    } else {
        Some(json_error(401, "missing or invalid API key"))
    }
}

/// Resolve which rate-limit bucket a request falls into: the named client
/// its bearer token matches, or (if `server.default_rate_limit_rpm` is
/// set) a bucket keyed by source IP. Returns `None` if no limit applies —
/// an unmatched caller with no configured default has no cap.
///
/// The source IP is the raw TCP peer address; behind a reverse proxy this
/// is the proxy's address, not the real client's, unless you run
/// rusty_provider with the proxy's connection preserved end-to-end (this
/// router does not parse `X-Forwarded-For`, since trusting it without a
/// configured list of trusted proxies would let any caller spoof their
/// bucket).
fn resolve_rate_limit(
    state: &AppState,
    headers: &HeaderMap,
    addr: SocketAddr,
) -> Option<(String, u32)> {
    if let Some(token) = bearer_token(headers) {
        if let Some((name, rpm)) = state.client_keys.get(token) {
            return Some((format!("client:{name}"), *rpm));
        }
    }
    state
        .default_rate_limit_rpm
        .map(|rpm| (format!("ip:{}", addr.ip()), rpm))
}

fn rate_limited_response(state: &AppState, identity: &str, retry_after_secs: f64) -> Response {
    state.router.record_inbound_rate_limit_rejection(identity);
    let secs = retry_after_secs.ceil().max(1.0) as u64;
    json_error_with_retry_after(
        429,
        &format!("rate limit exceeded, retry after {secs}s"),
        Some(secs),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::http::HeaderValue;
    use rp_core::RateLimiter;
    use rp_router::{Config, Router};

    fn test_state(
        client_keys: Vec<(&str, &str, u32)>,
        default_rate_limit_rpm: Option<u32>,
    ) -> AppState {
        let router = Arc::new(Router::from_config(
            &Config::from_toml_str("providers = {}").unwrap(),
        ));
        let client_keys = client_keys
            .into_iter()
            .map(|(key, name, rpm)| (key.to_string(), (name.to_string(), rpm)))
            .collect::<HashMap<_, _>>();
        AppState {
            router,
            api_key: None,
            client_keys: Arc::new(client_keys),
            default_rate_limit_rpm,
            rate_limiter: Arc::new(RateLimiter::new()),
        }
    }

    fn bearer_headers(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        headers
    }

    fn addr() -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], 54321))
    }

    // --- resolve_rate_limit ----------------------------------------------------

    #[test]
    fn resolve_rate_limit_is_none_with_no_client_match_and_no_default() {
        let state = test_state(vec![], None);
        let result = resolve_rate_limit(&state, &HeaderMap::new(), addr());
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_rate_limit_falls_back_to_ip_bucket_when_default_is_configured() {
        let state = test_state(vec![], Some(60));
        let result = resolve_rate_limit(&state, &HeaderMap::new(), addr());
        assert_eq!(result, Some(("ip:127.0.0.1".to_string(), 60)));
    }

    #[test]
    fn resolve_rate_limit_uses_client_bucket_when_bearer_token_matches() {
        let state = test_state(vec![("secret-key", "acme", 30)], None);
        let result = resolve_rate_limit(&state, &bearer_headers("secret-key"), addr());
        assert_eq!(result, Some(("client:acme".to_string(), 30)));
    }

    #[test]
    fn resolve_rate_limit_prefers_client_bucket_over_ip_fallback() {
        let state = test_state(vec![("secret-key", "acme", 30)], Some(60));
        let result = resolve_rate_limit(&state, &bearer_headers("secret-key"), addr());
        assert_eq!(
            result,
            Some(("client:acme".to_string(), 30)),
            "a matched client key must win over the IP-bucket default"
        );
    }

    #[test]
    fn resolve_rate_limit_falls_back_to_ip_when_bearer_present_but_unmatched() {
        let state = test_state(vec![("secret-key", "acme", 30)], Some(60));
        let result = resolve_rate_limit(&state, &bearer_headers("wrong-key"), addr());
        assert_eq!(result, Some(("ip:127.0.0.1".to_string(), 60)));
    }

    #[test]
    fn resolve_rate_limit_is_none_when_bearer_unmatched_and_no_default() {
        let state = test_state(vec![("secret-key", "acme", 30)], None);
        let result = resolve_rate_limit(&state, &bearer_headers("wrong-key"), addr());
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_rate_limit_ip_bucket_key_reflects_the_caller_address() {
        let state = test_state(vec![], Some(60));
        let other_addr = SocketAddr::from(([10, 0, 0, 5], 8080));
        let result = resolve_rate_limit(&state, &HeaderMap::new(), other_addr);
        assert_eq!(result, Some(("ip:10.0.0.5".to_string(), 60)));
    }

    // --- rate_limited_response ---------------------------------------------------

    #[test]
    fn rate_limited_response_returns_429_with_a_rounded_up_retry_after_header() {
        let state = test_state(vec![], None);
        let resp = rate_limited_response(&state, "ip:127.0.0.1", 0.2);
        assert_eq!(resp.status(), 429);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            &HeaderValue::from_static("1"),
            "0.2s should round up to a minimum of 1s"
        );
    }

    #[test]
    fn rate_limited_response_retry_after_ceils_fractional_seconds() {
        let state = test_state(vec![], None);
        let resp = rate_limited_response(&state, "ip:127.0.0.1", 4.1);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            &HeaderValue::from_static("5")
        );
    }

    #[test]
    fn rate_limited_response_records_the_rejection_under_the_given_identity() {
        let state = test_state(vec![], None);
        rate_limited_response(&state, "client:acme", 1.0);
        let metrics = state.router.render_prometheus_metrics();
        assert!(metrics.contains("rusty_provider_inbound_rate_limit_rejections_total"));
        assert!(metrics.contains(r#"identity="client:acme""#));
    }

    #[tokio::test]
    async fn rate_limited_response_body_reports_the_rounded_retry_after_in_the_message() {
        let state = test_state(vec![], None);
        let resp = rate_limited_response(&state, "ip:127.0.0.1", 4.1);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], 429);
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("retry after 5s"));
    }
}

pub async fn health() -> &'static str {
    "ok"
}

pub async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(resp) = check_auth(&state, &headers) {
        return resp;
    }

    let data: Vec<ModelInfo> = state
        .router
        .route_aliases()
        .map(|alias| ModelInfo {
            id: alias.to_string(),
            object: "model",
            owned_by: "router-alias".to_string(),
        })
        .chain(state.router.configured_providers().map(|p| ModelInfo {
            id: format!("{p}/*"),
            object: "model",
            owned_by: p.to_string(),
        }))
        .collect();

    Json(json!({ "object": "list", "data": data })).into_response()
}

#[derive(Serialize)]
struct UsageEntry {
    model: String,
    #[serde(flatten)]
    stats: UsageStats,
}

pub async fn usage_stats(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(resp) = check_auth(&state, &headers) {
        return resp;
    }

    let data: Vec<UsageEntry> = state
        .router
        .usage_snapshot()
        .await
        .into_iter()
        .map(|(model, stats)| UsageEntry { model, stats })
        .collect();

    Json(json!({ "object": "list", "data": data })).into_response()
}

pub async fn metrics(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(resp) = check_auth(&state, &headers) {
        return resp;
    }

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        state.router.render_prometheus_metrics(),
    )
        .into_response()
}

pub async fn chat_completions(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if let Some(resp) = check_auth(&state, &headers) {
        return resp;
    }
    if let Some((identity, rpm)) = resolve_rate_limit(&state, &headers, addr) {
        if let Err(retry_after_secs) = state.rate_limiter.check(&identity, rpm) {
            return rate_limited_response(&state, &identity, retry_after_secs);
        }
    }
    if req.messages.is_empty() {
        return json_error(400, "\"messages\" must not be empty");
    }

    if req.is_streaming() {
        match state.router.dispatch_stream(&req).await {
            Ok(chunk_stream) => {
                let events = chunk_stream
                    .map(|item| {
                        let event = match item {
                            Ok(chunk) => Event::default()
                                .json_data(&chunk)
                                .unwrap_or_else(|_| Event::default().data("{}")),
                            Err(e) => Event::default()
                                .event("error")
                                .data(json!({"error": {"message": e.to_string()}}).to_string()),
                        };
                        Ok::<_, Infallible>(event)
                    })
                    .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

                Sse::new(events)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            }
            Err(e) => router_error_response(e),
        }
    } else {
        match state.router.dispatch(&req).await {
            Ok(resp) => Json(resp).into_response(),
            Err(e) => router_error_response(e),
        }
    }
}
