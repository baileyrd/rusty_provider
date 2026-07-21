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

#[cfg(test)]
mod tests {
    use super::*;
    use rp_core::ProviderError;

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn json_error_sets_status_and_omits_retry_after() {
        let resp = json_error(401, "missing or invalid API key");
        assert_eq!(resp.status(), 401);
        assert!(resp.headers().get(RETRY_AFTER).is_none());
    }

    #[tokio::test]
    async fn json_error_body_carries_the_message_and_matching_code() {
        let resp = json_error(400, "bad request");
        let body = body_json(resp).await;
        assert_eq!(body["error"]["message"], "bad request");
        assert_eq!(body["error"]["code"], 400);
    }

    #[test]
    fn json_error_with_retry_after_sets_the_header() {
        let resp = json_error_with_retry_after(429, "rate limited", Some(5));
        assert_eq!(resp.status(), 429);
        assert_eq!(resp.headers().get(RETRY_AFTER).unwrap(), "5");
    }

    #[test]
    fn json_error_with_retry_after_none_omits_the_header() {
        let resp = json_error_with_retry_after(500, "oops", None);
        assert!(resp.headers().get(RETRY_AFTER).is_none());
    }

    #[test]
    fn json_error_with_retry_after_falls_back_to_500_for_an_invalid_status_code() {
        let resp = json_error_with_retry_after(9999, "unmappable", None);
        assert_eq!(resp.status(), 500);
    }

    #[tokio::test]
    async fn json_error_body_code_reflects_the_fallback_status_not_the_invalid_input() {
        let resp = json_error_with_retry_after(9999, "unmappable", None);
        let body = body_json(resp).await;
        assert_eq!(body["error"]["code"], 500);
    }

    #[test]
    fn router_error_response_delegates_status_and_retry_after_from_router_error() {
        let err = RouterError::from(ProviderError::RateLimited {
            retry_after_secs: Some(7),
        });
        let resp = router_error_response(err);
        assert_eq!(resp.status(), 429);
        assert_eq!(resp.headers().get(RETRY_AFTER).unwrap(), "7");
    }

    #[test]
    fn router_error_response_has_no_retry_after_for_a_non_rate_limited_error() {
        let resp = router_error_response(RouterError::InvalidModel("bogus".to_string()));
        assert_eq!(resp.status(), 400);
        assert!(resp.headers().get(RETRY_AFTER).is_none());
    }

    #[tokio::test]
    async fn router_error_response_body_message_matches_the_error_display() {
        let err = RouterError::ProviderNotConfigured("openai".to_string());
        let expected_message = err.to_string();
        let resp = router_error_response(err);
        assert_eq!(resp.status(), 424);
        let body = body_json(resp).await;
        assert_eq!(body["error"]["message"], expected_message);
    }
}
