use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rp_router::RouterError;
use serde_json::json;

pub fn router_error_response(err: RouterError) -> Response {
    let status = err.status_code();
    json_error(status, &err.to_string())
}

pub fn json_error(status: u16, message: &str) -> Response {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        Json(json!({ "error": { "message": message, "code": status.as_u16() } })),
    )
        .into_response()
}
