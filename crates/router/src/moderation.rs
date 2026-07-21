//! `[moderation]` -- checks a request's message text against an external
//! moderation endpoint (OpenAI's `/moderations`, or a compatible one)
//! before it's ever dispatched to a provider. A different axis from
//! `[[guardrails]]`: guardrails are operator-authored regex patterns,
//! this defers the actual judgment call (hate, violence, self-harm, etc.)
//! to a third-party classifier the operator doesn't have to enumerate
//! patterns for.

use std::collections::HashMap;
use std::time::Duration;

use rp_core::{ChatRequest, ContentPart, MessageContent};
use serde::Deserialize;

use crate::config::ModerationConfig;

/// The result of a moderation check that didn't pass.
pub(crate) enum ModerationError {
    /// The moderation backend flagged the content -- the category names
    /// it reported as triggered (e.g. `["violence", "hate"]`).
    Flagged(Vec<String>),
    /// The moderation backend itself couldn't be reached, or returned
    /// something this router couldn't parse. Callers are expected to
    /// treat this as fail-open (see `Router::apply_moderation`), not as
    /// equivalent to `Flagged`.
    RequestFailed(String),
}

pub(crate) struct ModerationClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

#[derive(Deserialize)]
struct ModerationResponse {
    results: Vec<ModerationResult>,
}

#[derive(Deserialize)]
struct ModerationResult {
    flagged: bool,
    categories: HashMap<String, bool>,
}

impl ModerationClient {
    pub(crate) fn new(config: &ModerationConfig, api_key: String) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(config.timeout_secs))
                .build()
                .expect("reqwest client should build with a timeout configured"),
            base_url: config.base_url.trim_end_matches('/').to_string(),
            api_key,
            model: config.model.clone(),
        }
    }

    /// Checks every plain-text part of every one of `req`'s messages in a
    /// single moderation call (joined with newlines, rather than one call
    /// per message -- fewer round trips). A request with no text content
    /// at all (e.g. image-only) is trivially `Ok` without a call, since
    /// there's nothing for a text classifier to judge.
    pub(crate) async fn check(&self, req: &ChatRequest) -> Result<(), ModerationError> {
        let text = collect_text(req);
        if text.trim().is_empty() {
            return Ok(());
        }

        let resp = self
            .client
            .post(format!("{}/moderations", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({ "model": self.model, "input": text }))
            .send()
            .await
            .map_err(|e| ModerationError::RequestFailed(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ModerationError::RequestFailed(format!(
                "moderation endpoint returned {}",
                resp.status()
            )));
        }

        let body: ModerationResponse = resp
            .json()
            .await
            .map_err(|e| ModerationError::RequestFailed(e.to_string()))?;

        let flagged_categories: Vec<String> = body
            .results
            .into_iter()
            .find(|r| r.flagged)
            .map(|r| {
                let mut names: Vec<String> = r
                    .categories
                    .into_iter()
                    .filter(|(_, flagged)| *flagged)
                    .map(|(name, _)| name)
                    .collect();
                names.sort();
                names
            })
            .unwrap_or_default();

        if flagged_categories.is_empty() {
            Ok(())
        } else {
            Err(ModerationError::Flagged(flagged_categories))
        }
    }
}

fn collect_text(req: &ChatRequest) -> String {
    let mut segments = Vec::new();
    for message in &req.messages {
        let Some(content) = &message.content else {
            continue;
        };
        match content {
            MessageContent::Text(text) => segments.push(text.as_str()),
            MessageContent::Parts(parts) => {
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        segments.push(text.as_str());
                    }
                }
            }
        }
    }
    segments.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(base_url: &str) -> ModerationConfig {
        ModerationConfig {
            api_key_env: "UNUSED".to_string(),
            base_url: base_url.to_string(),
            model: "omni-moderation-latest".to_string(),
            timeout_secs: 5,
        }
    }

    fn request_with_text(text: &str) -> ChatRequest {
        serde_json::from_value(serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": text}]
        }))
        .unwrap()
    }

    #[test]
    fn collect_text_joins_plain_text_messages_with_newlines() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "smart",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "how are you"}
            ]
        }))
        .unwrap();
        assert_eq!(collect_text(&req), "hello\nhi there\nhow are you");
    }

    #[test]
    fn collect_text_includes_text_parts_of_a_multimodal_message() {
        let mut req = request_with_text("placeholder");
        req.messages[0].content = Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "a caption".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: rp_core::ImageUrl {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                },
            },
        ]));
        assert_eq!(collect_text(&req), "a caption");
    }

    #[tokio::test]
    async fn check_is_ok_when_nothing_is_flagged() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{"flagged": false, "categories": {"violence": false, "hate": false}}]
            })))
            .mount(&server)
            .await;

        let client = ModerationClient::new(&config(&server.uri()), "test-key".to_string());
        assert!(client
            .check(&request_with_text("hello there"))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn check_returns_flagged_categories_when_the_backend_flags_it() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "flagged": true,
                    "categories": {"violence": true, "hate": false, "self-harm": true}
                }]
            })))
            .mount(&server)
            .await;

        let client = ModerationClient::new(&config(&server.uri()), "test-key".to_string());
        let err = client
            .check(&request_with_text("something bad"))
            .await
            .unwrap_err();
        match err {
            ModerationError::Flagged(categories) => {
                assert_eq!(
                    categories,
                    vec!["self-harm".to_string(), "violence".to_string()]
                );
            }
            ModerationError::RequestFailed(msg) => panic!("expected Flagged, got {msg}"),
        }
    }

    #[tokio::test]
    async fn check_is_a_noop_with_no_text_content_at_all() {
        // No mock mounted -- a call would 404 the mock server and fail the
        // test, proving the client never even makes a request here.
        let server = MockServer::start().await;
        let client = ModerationClient::new(&config(&server.uri()), "test-key".to_string());
        let mut req = request_with_text("placeholder");
        req.messages[0].content = None;
        assert!(client.check(&req).await.is_ok());
    }

    #[tokio::test]
    async fn check_fails_with_requestfailed_on_a_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let client = ModerationClient::new(&config(&server.uri()), "test-key".to_string());
        let err = client.check(&request_with_text("hello")).await.unwrap_err();
        assert!(matches!(err, ModerationError::RequestFailed(_)));
    }

    #[tokio::test]
    async fn check_fails_with_requestfailed_when_the_endpoint_is_unreachable() {
        // Port 0 never accepts connections -- proves a network-level
        // failure (not just a non-2xx) also maps to RequestFailed.
        let client = ModerationClient::new(&config("http://127.0.0.1:0"), "test-key".to_string());
        let err = client.check(&request_with_text("hello")).await.unwrap_err();
        assert!(matches!(err, ModerationError::RequestFailed(_)));
    }

    #[tokio::test]
    async fn check_fails_with_requestfailed_when_the_backend_outlasts_timeout_secs() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_millis(1200))
                    .set_body_json(
                        serde_json::json!({"results": [{"flagged": false, "categories": {}}]}),
                    ),
            )
            .mount(&server)
            .await;

        let mut cfg = config(&server.uri());
        cfg.timeout_secs = 1;
        let client = ModerationClient::new(&cfg, "test-key".to_string());
        let err = client.check(&request_with_text("hello")).await.unwrap_err();
        assert!(matches!(err, ModerationError::RequestFailed(_)));
    }
}
