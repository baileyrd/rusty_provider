use std::time::Duration;

use rp_core::ProviderError;

/// Default total-request timeout for a provider's `reqwest::Client`, used
/// by every adapter's `new()`. Generous on purpose -- a non-streaming
/// completion from a large or reasoning-heavy model, or a long-running
/// stream, can legitimately take minutes; this only needs to bound the
/// failure mode of a connection that hangs forever, not shave time off
/// normal slow responses. Overridable per-provider via `with_timeout`
/// (wired to `[[providers]].timeout_secs` in `rp-router`).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Builds a `reqwest::Client` with `timeout` as its total per-request
/// timeout (covers connecting, sending, and reading the full response --
/// including a streamed one). Panics only if the underlying TLS backend
/// fails to initialize, which `reqwest::Client::new()` (used everywhere
/// before this) would also have panicked on.
pub fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("reqwest client should build with a timeout configured")
}

/// Turn a non-2xx reqwest response into a classified `ProviderError`,
/// consuming the body for the error message.
pub async fn map_error_response(resp: reqwest::Response) -> ProviderError {
    let status = resp.status();
    let retry_after_secs = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let body = resp.text().await.unwrap_or_default();
    let message = extract_error_message(&body).unwrap_or(body);

    match status.as_u16() {
        401 | 403 => ProviderError::Auth(message),
        400 | 404 | 422 => ProviderError::InvalidRequest(message),
        429 => ProviderError::RateLimited { retry_after_secs },
        s => ProviderError::Upstream { status: s, message },
    }
}

/// Best-effort extraction of a human-readable message from a provider's
/// JSON error body. Providers disagree on the exact shape, so this tries
/// the common spots and falls back to the raw body.
fn extract_error_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value
        .get("error")
        .and_then(|e| e.get("message").or(Some(e)))
        .and_then(|m| m.as_str().map(str::to_owned))
        .or_else(|| {
            value
                .get("message")
                .and_then(|m| m.as_str().map(str::to_owned))
        })
}

pub fn map_reqwest_error(err: reqwest::Error) -> ProviderError {
    if err.is_timeout() {
        ProviderError::Timeout
    } else {
        ProviderError::Network(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // --- build_client --------------------------------------------------------

    #[tokio::test]
    async fn build_client_times_out_a_request_that_outlasts_the_configured_timeout() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(200)))
            .mount(&server)
            .await;

        let client = build_client(Duration::from_millis(20));
        let err = client.get(server.uri()).send().await.unwrap_err();
        assert!(err.is_timeout());
        assert!(matches!(map_reqwest_error(err), ProviderError::Timeout));
    }

    #[tokio::test]
    async fn build_client_succeeds_within_the_configured_timeout() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let client = build_client(Duration::from_secs(5));
        let resp = client.get(server.uri()).send().await.unwrap();
        assert!(resp.status().is_success());
    }

    #[test]
    fn extract_error_message_from_nested_error_message_field() {
        assert_eq!(
            extract_error_message(r#"{"error":{"message":"bad request"}}"#),
            Some("bad request".to_string())
        );
    }

    #[test]
    fn extract_error_message_from_a_plain_string_error_field() {
        assert_eq!(
            extract_error_message(r#"{"error":"overloaded"}"#),
            Some("overloaded".to_string())
        );
    }

    #[test]
    fn extract_error_message_from_a_top_level_message_field() {
        assert_eq!(
            extract_error_message(r#"{"message":"invalid api key"}"#),
            Some("invalid api key".to_string())
        );
    }

    #[test]
    fn extract_error_message_prefers_error_over_top_level_message() {
        assert_eq!(
            extract_error_message(r#"{"error":{"message":"from error"},"message":"from top"}"#),
            Some("from error".to_string())
        );
    }

    #[test]
    fn extract_error_message_is_none_for_json_with_neither_recognized_shape() {
        assert_eq!(extract_error_message(r#"{"foo":"bar"}"#), None);
    }

    #[test]
    fn extract_error_message_is_none_for_non_json_body() {
        assert_eq!(extract_error_message("not json at all"), None);
    }

    #[test]
    fn extract_error_message_is_none_for_an_empty_body() {
        assert_eq!(extract_error_message(""), None);
    }
}
