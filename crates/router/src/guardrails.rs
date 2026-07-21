//! Regex-based content guardrails (`[[guardrails]]`): block or redact a
//! request's message text before it's ever dispatched to a provider.
//! OpenRouter scopes its equivalent per workspace; rusty has no
//! workspace/org concept (see the deferred organizations/workspaces/
//! roles backlog item), so these are scoped globally instead.

use regex::Regex;
use rp_core::{ChatRequest, ContentPart, MessageContent};

use crate::config::{GuardrailAction, GuardrailConfig};

/// A guardrail with its pattern compiled once at startup, rather than
/// re-parsing the regex on every request.
pub struct Guardrail {
    name: String,
    action: GuardrailAction,
    replacement: String,
    regex: Regex,
}

impl Guardrail {
    pub fn compile(cfg: &GuardrailConfig) -> Result<Self, regex::Error> {
        Ok(Self {
            name: cfg.name.clone(),
            action: cfg.action,
            replacement: cfg.replacement.clone(),
            regex: Regex::new(&cfg.pattern)?,
        })
    }
}

/// Apply every compiled guardrail to `req`'s message text, in config
/// order. A `"redact"` guardrail rewrites matched substrings in place, so
/// a later guardrail (and eventually the provider) only ever sees the
/// redacted text. A `"block"` guardrail that matches anywhere fails the
/// whole request immediately with that guardrail's name -- any redaction
/// already applied by earlier guardrails is irrelevant at that point,
/// since the request never goes out.
///
/// Only plain text is scanned: a message's `Text` content, or `Text`
/// parts within a multimodal `Parts` array. Image/audio/file parts are
/// untouched -- a regex has nothing meaningful to check there.
pub fn apply(guardrails: &[Guardrail], req: &mut ChatRequest) -> Result<(), String> {
    for guardrail in guardrails {
        for message in &mut req.messages {
            let Some(content) = &mut message.content else {
                continue;
            };
            match guardrail.action {
                GuardrailAction::Block => {
                    if content_matches(&guardrail.regex, content) {
                        return Err(guardrail.name.clone());
                    }
                }
                GuardrailAction::Redact => {
                    redact_content(&guardrail.regex, &guardrail.replacement, content);
                }
            }
        }
    }
    Ok(())
}

fn content_matches(regex: &Regex, content: &MessageContent) -> bool {
    match content {
        MessageContent::Text(text) => regex.is_match(text),
        MessageContent::Parts(parts) => parts.iter().any(|part| match part {
            ContentPart::Text { text } => regex.is_match(text),
            _ => false,
        }),
    }
}

fn redact_content(regex: &Regex, replacement: &str, content: &mut MessageContent) {
    match content {
        MessageContent::Text(text) => {
            if regex.is_match(text) {
                *text = regex.replace_all(text, replacement).into_owned();
            }
        }
        MessageContent::Parts(parts) => {
            for part in parts {
                if let ContentPart::Text { text } = part {
                    if regex.is_match(text) {
                        *text = regex.replace_all(text, replacement).into_owned();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guardrail(name: &str, pattern: &str, action: GuardrailAction) -> Guardrail {
        Guardrail::compile(&GuardrailConfig {
            name: name.to_string(),
            pattern: pattern.to_string(),
            action,
            replacement: "[redacted]".to_string(),
        })
        .unwrap()
    }

    /// A minimal single-user-turn request built via JSON deserialization,
    /// so this test module doesn't need to hand-construct every field of
    /// `ChatRequest` (it has no `Default`).
    fn request_with_text(text: &str) -> ChatRequest {
        serde_json::from_value(serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": text}]
        }))
        .unwrap()
    }

    #[test]
    fn compile_rejects_an_invalid_pattern() {
        let err = Guardrail::compile(&GuardrailConfig {
            name: "bad".to_string(),
            pattern: "(unclosed".to_string(),
            action: GuardrailAction::Block,
            replacement: String::new(),
        });
        assert!(err.is_err());
    }

    #[test]
    fn block_matches_plain_text_content() {
        let guardrails = vec![guardrail(
            "no-ssn",
            r"\d{3}-\d{2}-\d{4}",
            GuardrailAction::Block,
        )];
        let mut req = request_with_text("my ssn is 123-45-6789");
        let err = apply(&guardrails, &mut req).unwrap_err();
        assert_eq!(err, "no-ssn");
    }

    #[test]
    fn block_is_a_noop_when_nothing_matches() {
        let guardrails = vec![guardrail(
            "no-ssn",
            r"\d{3}-\d{2}-\d{4}",
            GuardrailAction::Block,
        )];
        let mut req = request_with_text("hello there");
        assert!(apply(&guardrails, &mut req).is_ok());
    }

    #[test]
    fn redact_replaces_matches_in_place() {
        let guardrails = vec![guardrail(
            "no-ssn",
            r"\d{3}-\d{2}-\d{4}",
            GuardrailAction::Redact,
        )];
        let mut req = request_with_text("my ssn is 123-45-6789, ok?");
        apply(&guardrails, &mut req).unwrap();
        match &req.messages[0].content {
            Some(MessageContent::Text(text)) => {
                assert_eq!(text, "my ssn is [redacted], ok?");
            }
            other => panic!("expected Text content, got {other:?}"),
        }
    }

    #[test]
    fn redact_uses_the_configured_replacement() {
        let guardrail = Guardrail::compile(&GuardrailConfig {
            name: "no-email".to_string(),
            pattern: r"\S+@\S+".to_string(),
            action: GuardrailAction::Redact,
            replacement: "<email>".to_string(),
        })
        .unwrap();
        let mut req = request_with_text("contact me at a@b.com");
        apply(&[guardrail], &mut req).unwrap();
        match &req.messages[0].content {
            Some(MessageContent::Text(text)) => assert_eq!(text, "contact me at <email>"),
            other => panic!("expected Text content, got {other:?}"),
        }
    }

    #[test]
    fn redact_covers_text_parts_within_a_multimodal_message() {
        let guardrails = vec![guardrail(
            "no-ssn",
            r"\d{3}-\d{2}-\d{4}",
            GuardrailAction::Redact,
        )];
        let mut req = request_with_text("placeholder");
        req.messages[0].content = Some(MessageContent::Parts(vec![ContentPart::Text {
            text: "ssn: 123-45-6789".to_string(),
        }]));
        apply(&guardrails, &mut req).unwrap();
        match &req.messages[0].content {
            Some(MessageContent::Parts(parts)) => match &parts[0] {
                ContentPart::Text { text } => assert_eq!(text, "ssn: [redacted]"),
                other => panic!("expected Text part, got {other:?}"),
            },
            other => panic!("expected Parts content, got {other:?}"),
        }
    }

    #[test]
    fn block_checks_text_parts_within_a_multimodal_message() {
        let guardrails = vec![guardrail(
            "no-ssn",
            r"\d{3}-\d{2}-\d{4}",
            GuardrailAction::Block,
        )];
        let mut req = request_with_text("placeholder");
        req.messages[0].content = Some(MessageContent::Parts(vec![ContentPart::Text {
            text: "ssn: 123-45-6789".to_string(),
        }]));
        let err = apply(&guardrails, &mut req).unwrap_err();
        assert_eq!(err, "no-ssn");
    }

    #[test]
    fn multiple_guardrails_apply_in_config_order() {
        let guardrails = vec![
            guardrail("redact-ssn", r"\d{3}-\d{2}-\d{4}", GuardrailAction::Redact),
            guardrail("block-redacted", r"\[redacted\]", GuardrailAction::Block),
        ];
        // The first guardrail redacts the ssn to "[redacted]"; the second
        // then blocks on seeing that literal marker -- proves later
        // guardrails see the mutations earlier ones already made.
        let mut req = request_with_text("ssn: 123-45-6789");
        let err = apply(&guardrails, &mut req).unwrap_err();
        assert_eq!(err, "block-redacted");
    }

    #[test]
    fn a_later_block_guardrail_still_runs_after_an_earlier_redact() {
        let guardrails = vec![
            guardrail("redact-ssn", r"\d{3}-\d{2}-\d{4}", GuardrailAction::Redact),
            guardrail("block-profanity", r"badword", GuardrailAction::Block),
        ];
        let mut req = request_with_text("ssn: 123-45-6789, also badword");
        let err = apply(&guardrails, &mut req).unwrap_err();
        assert_eq!(err, "block-profanity");
    }

    #[test]
    fn no_guardrails_configured_is_always_a_noop() {
        let mut req = request_with_text("anything goes, 123-45-6789 included");
        assert!(apply(&[], &mut req).is_ok());
    }
}
