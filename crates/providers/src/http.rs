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
