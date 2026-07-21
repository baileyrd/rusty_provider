use axum::http::header::RETRY_AFTER;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use rp_router::RouterError;
use serde_json::json;

pub fn router_error_response(err: RouterError) -> Response {
    let status = err.status_code();
    let retry_after_secs = err.retry_after_secs();
    json_error_with_retry_after(status, &err.to_string(), retry_after_secs)
}

pub fn json_error(status: u16, message: &str) -> Response {
    json_error_with_retry_after(status, message, None)
}

pub fn json_error_with_retry_after(
    status: u16,
    message: &str,
    retry_after_secs: Option<u64>,
) -> Response {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut resp = (
        status,
        Json(json!({ "error": { "message": message, "code": status.as_u16() } })),
    )
        .into_response();
    if let Some(secs) = retry_after_secs {
        if let Ok(value) = HeaderValue::from_str(&secs.to_string()) {
            resp.headers_mut().insert(RETRY_AFTER, value);
        }
    }
    resp
}
