//! `[web_search]` -- `"web_search": true` on a request triggers a live
//! web search (via Brave Search's API, or a compatible one) whose top
//! results are woven into the request as extra context before dispatch,
//! so the model can ground its answer in information beyond its training
//! data. Loosely mirrors OpenRouter's `:online` model suffix / `web`
//! plugin, scoped down to one search backend and a plain-text injection
//! into the last user message instead of a structured citations/
//! annotations response field.

use rp_core::{ChatRequest, ContentPart, MessageContent, Role};
use serde::Deserialize;

use crate::config::WebSearchConfig;

pub(crate) struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub(crate) struct WebSearchClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    max_results: u32,
}

#[derive(Deserialize)]
struct BraveSearchResponse {
    #[serde(default)]
    web: Option<BraveWebResults>,
}

#[derive(Deserialize)]
struct BraveWebResults {
    #[serde(default)]
    results: Vec<BraveResult>,
}

#[derive(Deserialize)]
struct BraveResult {
    title: String,
    url: String,
    #[serde(default)]
    description: String,
}

impl WebSearchClient {
    pub(crate) fn new(config: &WebSearchConfig, api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: config.base_url.clone(),
            api_key,
            max_results: config.max_results,
        }
    }

    /// Runs `query` against the configured search backend. `Err` covers
    /// both a network-level failure and a non-2xx/unparseable response --
    /// callers are expected to treat this as fail-open (see
    /// `Router::apply_web_search`), never as a reason to block the
    /// request.
    pub(crate) async fn search(&self, query: &str) -> Result<Vec<SearchResult>, String> {
        let resp = self
            .client
            .get(&self.base_url)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .query(&[("q", query), ("count", &self.max_results.to_string())])
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!("web search endpoint returned {}", resp.status()));
        }

        let body: BraveSearchResponse = resp.json().await.map_err(|e| e.to_string())?;

        Ok(body
            .web
            .map(|w| w.results)
            .unwrap_or_default()
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.description,
            })
            .collect())
    }
}

/// The latest `Role::User` message's plain text, to use as the search
/// query. `None` when there's no user message at all, or its text (after
/// trimming) is empty -- e.g. an image-only turn, which has nothing for a
/// text search to work from.
pub(crate) fn last_user_query(req: &ChatRequest) -> Option<String> {
    let text = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .and_then(|m| m.content.as_ref())
        .map(MessageContent::as_plain_text)?;
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Formats `results` as a numbered, human-readable block (title, snippet,
/// URL) meant to be prepended as plain-text context ahead of the user's
/// own question -- not a structured field any adapter has to understand,
/// so it round-trips through every provider unchanged.
pub(crate) fn format_results(results: &[SearchResult]) -> String {
    let mut out = String::from("[Web search results]\n");
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} — {} ({})\n",
            i + 1,
            r.title,
            r.snippet,
            r.url
        ));
    }
    out.push('\n');
    out
}

/// Prepends `prefix` to the latest `Role::User` message's text, in place
/// -- same "mutate the caller's owned copy before dispatch" pattern
/// `guardrails::apply`'s redaction uses. A no-op if there's no user
/// message at all (shouldn't happen if `last_user_query` already found
/// one, but this is defensive rather than assuming the caller re-checks).
pub(crate) fn prepend_to_last_user_message(req: &mut ChatRequest, prefix: &str) {
    let Some(message) = req.messages.iter_mut().rev().find(|m| m.role == Role::User) else {
        return;
    };
    match &mut message.content {
        Some(MessageContent::Text(text)) => {
            *text = format!("{prefix}{text}");
        }
        Some(MessageContent::Parts(parts)) => {
            if let Some(ContentPart::Text { text }) = parts
                .iter_mut()
                .find(|p| matches!(p, ContentPart::Text { .. }))
            {
                *text = format!("{prefix}{text}");
            } else {
                parts.insert(
                    0,
                    ContentPart::Text {
                        text: prefix.to_string(),
                    },
                );
            }
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(base_url: &str) -> WebSearchConfig {
        WebSearchConfig {
            api_key_env: "UNUSED".to_string(),
            base_url: base_url.to_string(),
            max_results: 5,
        }
    }

    fn request_with_text(text: &str) -> ChatRequest {
        serde_json::from_value(serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": text}]
        }))
        .unwrap()
    }

    // --- last_user_query ---------------------------------------------------------

    #[test]
    fn last_user_query_returns_the_latest_user_messages_text() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "smart",
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": "second question"}
            ]
        }))
        .unwrap();
        assert_eq!(last_user_query(&req).as_deref(), Some("second question"));
    }

    #[test]
    fn last_user_query_is_none_with_no_user_message() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "smart",
            "messages": [{"role": "system", "content": "be helpful"}]
        }))
        .unwrap();
        assert_eq!(last_user_query(&req), None);
    }

    #[test]
    fn last_user_query_is_none_when_the_user_messages_text_is_blank() {
        let req = request_with_text("   ");
        assert_eq!(last_user_query(&req), None);
    }

    // --- format_results ------------------------------------------------------------

    #[test]
    fn format_results_numbers_each_entry_with_title_snippet_and_url() {
        let results = vec![
            SearchResult {
                title: "Rust".to_string(),
                url: "https://rust-lang.org".to_string(),
                snippet: "A systems language".to_string(),
            },
            SearchResult {
                title: "Tokio".to_string(),
                url: "https://tokio.rs".to_string(),
                snippet: "An async runtime".to_string(),
            },
        ];
        let formatted = format_results(&results);
        assert!(formatted.contains("1. Rust — A systems language (https://rust-lang.org)"));
        assert!(formatted.contains("2. Tokio — An async runtime (https://tokio.rs)"));
    }

    // --- prepend_to_last_user_message ------------------------------------------

    #[test]
    fn prepend_to_last_user_message_prefixes_plain_text_content() {
        let mut req = request_with_text("what's the weather");
        prepend_to_last_user_message(&mut req, "[context]\n");
        match &req.messages[0].content {
            Some(MessageContent::Text(text)) => {
                assert_eq!(text, "[context]\nwhat's the weather");
            }
            other => panic!("expected Text content, got {other:?}"),
        }
    }

    #[test]
    fn prepend_to_last_user_message_prefixes_the_text_part_of_a_multimodal_message() {
        let mut req = request_with_text("placeholder");
        req.messages[0].content = Some(MessageContent::Parts(vec![ContentPart::Text {
            text: "what's in this photo".to_string(),
        }]));
        prepend_to_last_user_message(&mut req, "[context]\n");
        match &req.messages[0].content {
            Some(MessageContent::Parts(parts)) => match &parts[0] {
                ContentPart::Text { text } => {
                    assert_eq!(text, "[context]\nwhat's in this photo");
                }
                other => panic!("expected Text part, got {other:?}"),
            },
            other => panic!("expected Parts content, got {other:?}"),
        }
    }

    #[test]
    fn prepend_to_last_user_message_targets_the_last_user_turn_not_the_first() {
        let mut req: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "smart",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"}
            ]
        }))
        .unwrap();
        prepend_to_last_user_message(&mut req, "[context]\n");
        assert_eq!(req.messages[0].content, Some(MessageContent::text("first")));
        assert_eq!(
            req.messages[2].content,
            Some(MessageContent::text("[context]\nsecond"))
        );
    }

    // --- WebSearchClient::search -------------------------------------------------

    #[tokio::test]
    async fn search_parses_brave_style_results() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(header("X-Subscription-Token", "test-key"))
            .and(query_param("q", "rust programming language"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "web": {
                    "results": [
                        {"title": "Rust", "url": "https://rust-lang.org", "description": "A systems language"}
                    ]
                }
            })))
            .mount(&server)
            .await;

        let client = WebSearchClient::new(&config(&server.uri()), "test-key".to_string());
        let results = client.search("rust programming language").await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Rust");
        assert_eq!(results[0].url, "https://rust-lang.org");
        assert_eq!(results[0].snippet, "A systems language");
    }

    #[tokio::test]
    async fn search_returns_an_empty_vec_when_the_backend_reports_no_web_results() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;

        let client = WebSearchClient::new(&config(&server.uri()), "test-key".to_string());
        let results = client.search("obscure query").await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_fails_on_a_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let client = WebSearchClient::new(&config(&server.uri()), "test-key".to_string());
        assert!(client.search("hello").await.is_err());
    }

    #[tokio::test]
    async fn search_fails_when_the_endpoint_is_unreachable() {
        // Port 0 never accepts connections -- a real network-level
        // failure, not just a non-2xx response.
        let client = WebSearchClient::new(&config("http://127.0.0.1:0"), "test-key".to_string());
        assert!(client.search("hello").await.is_err());
    }
}
