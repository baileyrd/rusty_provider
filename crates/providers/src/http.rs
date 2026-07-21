use rp_core::ProviderError;

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
