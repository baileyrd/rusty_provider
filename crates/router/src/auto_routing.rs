//! `model: "auto"` -- a heuristic complexity-based router, roughly
//! mirroring OpenRouter's `openrouter/auto`. There's no ML classifier
//! here (and no tokenizer -- same `chars / 4` estimate the rest of this
//! router already uses for context-budget math): `estimate_complexity`
//! is a small, deterministic, explainable scoring function over signals
//! that correlate with harder tasks, and `[auto_routing]` maps score
//! ranges to three fixed tiers. This is an approximation tuned by the
//! operator's own thresholds, not a guarantee of picking the "right"
//! model for any given prompt.

use rp_core::{ChatRequest, MessageContent, ResponseFormat};

use crate::config::AutoRoutingConfig;
use crate::estimate_tokens;

/// Points added for a request that uses tools, requests reasoning, or
/// constrains output to a JSON schema -- each correlates with a harder
/// task than a plain chat turn, independent of prompt length.
const TOOLS_BONUS: u32 = 50;
const REASONING_BONUS: u32 = 50;
const JSON_SCHEMA_BONUS: u32 = 30;
const CODE_BONUS: u32 = 50;
/// Points added per message beyond the first -- a longer back-and-forth
/// tends to carry more context/nuance than a single-turn question, even
/// at the same total token count.
const PER_EXTRA_MESSAGE_BONUS: u32 = 10;

/// A deterministic, explainable complexity score for `req` -- estimated
/// prompt tokens (summed across every message, the same per-message
/// estimate `transforms: ["middle-out"]` uses) plus flat bonuses for
/// signals that tend to mean a harder task: multi-turn context, code in
/// the conversation, tool use, requested reasoning, or a JSON-schema
/// output constraint. The score has no fixed unit or upper bound --
/// `[auto_routing]`'s thresholds are what give it meaning.
pub fn estimate_complexity(req: &ChatRequest) -> u32 {
    let mut score: u32 = req.messages.iter().map(|m| estimate_tokens(m) as u32).sum();

    if req.messages.len() > 1 {
        score += (req.messages.len() as u32 - 1) * PER_EXTRA_MESSAGE_BONUS;
    }
    if req
        .messages
        .iter()
        .any(|m| message_contains_code(&m.content))
    {
        score += CODE_BONUS;
    }
    if req.tools.is_some() {
        score += TOOLS_BONUS;
    }
    if req.reasoning.is_some() {
        score += REASONING_BONUS;
    }
    if matches!(req.response_format, Some(ResponseFormat::JsonSchema { .. })) {
        score += JSON_SCHEMA_BONUS;
    }

    score
}

fn message_contains_code(content: &Option<MessageContent>) -> bool {
    content
        .as_ref()
        .is_some_and(|c| c.as_plain_text().contains("```"))
}

/// Multiplier applied to both configured thresholds before bucketing,
/// from `provider.auto_bias`. `"cost"` raises both thresholds, so a
/// request has to score higher before it escalates into a pricier tier
/// (stays on `simple_model`/`medium_model` longer); `"quality"` lowers
/// them, escalating into `medium_model`/`complex_model` sooner. Anything
/// else, including unset, is `"balanced"` (no change) -- an unrecognized
/// value is never an error here, since this is a soft routing hint, not
/// a structural request field.
fn bias_multiplier(auto_bias: Option<&str>) -> f64 {
    match auto_bias {
        Some("cost") => 2.0,
        Some("quality") => 0.5,
        _ => 1.0,
    }
}

/// Picks which of `config`'s three tiers `req` belongs in, applying
/// `req.provider.auto_bias` (if set) to the configured thresholds first.
/// Returns that tier's configured model string -- a "provider/model" or
/// a `[[routes]]` alias, exactly like `ChatRequest.model` itself, for
/// the caller to resolve the same way it would any other `model` value.
pub fn resolve_tier(config: &AutoRoutingConfig, req: &ChatRequest) -> String {
    let score = estimate_complexity(req);
    let bias = bias_multiplier(req.provider.as_ref().and_then(|p| p.auto_bias.as_deref()));
    let simple_max = (config.simple_max_score as f64 * bias) as u32;
    let medium_max = (config.medium_max_score as f64 * bias) as u32;

    if score <= simple_max {
        config.simple_model.clone()
    } else if score <= medium_max {
        config.medium_model.clone()
    } else {
        config.complex_model.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AutoRoutingConfig {
        AutoRoutingConfig {
            simple_model: "openai/gpt-4o-mini".to_string(),
            medium_model: "smart".to_string(),
            complex_model: "anthropic/claude-opus-4-8".to_string(),
            simple_max_score: 20,
            medium_max_score: 80,
        }
    }

    fn request_with_text(text: &str) -> ChatRequest {
        serde_json::from_value(serde_json::json!({
            "model": "auto",
            "messages": [{"role": "user", "content": text}]
        }))
        .unwrap()
    }

    // --- estimate_complexity ---------------------------------------------------

    #[test]
    fn longer_prompts_score_higher() {
        let short = estimate_complexity(&request_with_text("hi"));
        let long = estimate_complexity(&request_with_text(&"word ".repeat(200)));
        assert!(long > short);
    }

    #[test]
    fn multi_turn_conversations_score_higher_than_single_turn_at_equal_length() {
        let mut multi = request_with_text("hello");
        multi.messages.push(rp_core::ChatMessage::assistant("hi"));
        multi.messages.push(rp_core::ChatMessage::user("hello"));
        let single = request_with_text("hellohello");
        // Both have roughly the same total text, but multi-turn adds a
        // per-extra-message bonus the single-turn request doesn't get.
        assert!(estimate_complexity(&multi) > estimate_complexity(&single));
    }

    #[test]
    fn code_in_the_conversation_adds_a_bonus() {
        let with_code = request_with_text("explain this: ```fn main() {}```");
        let without = request_with_text("explain this: fn main length similar text");
        assert!(estimate_complexity(&with_code) >= estimate_complexity(&without));
    }

    #[test]
    fn tools_add_a_flat_bonus() {
        let mut with_tools = request_with_text("hi");
        with_tools.tools = Some(vec![rp_core::Tool {
            kind: "function".to_string(),
            function: rp_core::FunctionDef {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
            },
        }]);
        let without = request_with_text("hi");
        assert_eq!(
            estimate_complexity(&with_tools) - estimate_complexity(&without),
            TOOLS_BONUS
        );
    }

    #[test]
    fn reasoning_adds_a_flat_bonus() {
        let mut with_reasoning = request_with_text("hi");
        with_reasoning.reasoning = Some(rp_core::ReasoningConfig {
            effort: None,
            max_tokens: None,
            exclude: None,
        });
        let without = request_with_text("hi");
        assert_eq!(
            estimate_complexity(&with_reasoning) - estimate_complexity(&without),
            REASONING_BONUS
        );
    }

    // --- resolve_tier ------------------------------------------------------------

    #[test]
    fn a_short_plain_request_resolves_to_the_simple_tier() {
        let req = request_with_text("hi");
        assert_eq!(resolve_tier(&config(), &req), "openai/gpt-4o-mini");
    }

    #[test]
    fn a_request_with_tools_resolves_to_a_higher_tier_than_plain_text_of_the_same_length() {
        let mut req = request_with_text("hi");
        req.tools = Some(vec![rp_core::Tool {
            kind: "function".to_string(),
            function: rp_core::FunctionDef {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
            },
        }]);
        // TOOLS_BONUS (50) alone pushes a 2-token prompt past
        // simple_max_score (20) into the medium tier.
        assert_eq!(resolve_tier(&config(), &req), "smart");
    }

    #[test]
    fn a_very_long_request_resolves_to_the_complex_tier() {
        let req = request_with_text(&"word ".repeat(1000));
        assert_eq!(resolve_tier(&config(), &req), "anthropic/claude-opus-4-8");
    }

    #[test]
    fn cost_bias_raises_thresholds_keeping_requests_on_cheaper_tiers_longer() {
        // At balanced bias, a ~24-token prompt clears simple_max_score
        // (20) into "medium". Doubling the thresholds under a cost bias
        // pulls it back under the (now 40) simple threshold.
        let req_balanced = request_with_text(&"word ".repeat(24));
        assert_eq!(resolve_tier(&config(), &req_balanced), "smart");

        let mut req_cost = request_with_text(&"word ".repeat(24));
        req_cost.provider = Some(rp_core::ProviderPreferences {
            auto_bias: Some("cost".to_string()),
            ..Default::default()
        });
        assert_eq!(resolve_tier(&config(), &req_cost), "openai/gpt-4o-mini");
    }

    #[test]
    fn quality_bias_lowers_thresholds_escalating_to_pricier_tiers_sooner() {
        // At balanced bias, a ~15-token prompt stays under
        // simple_max_score (20) in "simple". Halving the thresholds
        // under a quality bias (down to 10) pushes it into "medium".
        let req_balanced = request_with_text(&"word ".repeat(15));
        assert_eq!(resolve_tier(&config(), &req_balanced), "openai/gpt-4o-mini");

        let mut req_quality = request_with_text(&"word ".repeat(15));
        req_quality.provider = Some(rp_core::ProviderPreferences {
            auto_bias: Some("quality".to_string()),
            ..Default::default()
        });
        assert_eq!(resolve_tier(&config(), &req_quality), "smart");
    }

    #[test]
    fn an_unrecognized_bias_value_behaves_like_balanced() {
        let mut req = request_with_text(&"word ".repeat(15));
        let balanced = resolve_tier(&config(), &req);
        req.provider = Some(rp_core::ProviderPreferences {
            auto_bias: Some("ludicrous-speed".to_string()),
            ..Default::default()
        });
        assert_eq!(resolve_tier(&config(), &req), balanced);
    }
}
