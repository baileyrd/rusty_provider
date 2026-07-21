use std::convert::Infallible;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::{stream, StreamExt};
use rp_core::{ChatRequest, ModelInfo};
use rp_router::UsageStats;
use serde::Serialize;
use serde_json::json;

use crate::errors::{json_error, router_error_response};
use crate::state::AppState;

pub fn check_auth(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    let Some(expected) = &state.api_key else {
        return None;
    };

    let provided = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match provided {
        Some(token) if token == expected => None,
        _ => Some(json_error(401, "missing or invalid API key")),
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
        .into_iter()
        .map(|(model, stats)| UsageEntry { model, stats })
        .collect();

    Json(json!({ "object": "list", "data": data })).into_response()
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if let Some(resp) = check_auth(&state, &headers) {
        return resp;
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
