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
