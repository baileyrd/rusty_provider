/// Picks a reasoning/thinking-token budget from `max_tokens` directly if
/// set, else derives one from `effort` as a fraction of the request's own
/// `max_tokens` (high ~80%, medium ~50%, low ~20%, mirroring the fractions
/// OpenAI-ecosystem tooling generally uses for this conversion) -- or, if
/// the request left `max_tokens` unset too, a flat per-effort default.
/// Shared between Gemini's `thinkingBudget` and Anthropic's
/// `budget_tokens`, which both need this exact heuristic.
pub fn effort_thinking_budget(effort: Option<&str>, max_tokens: Option<u32>) -> u32 {
    match max_tokens {
        Some(max_tokens) => {
            let fraction = match effort {
                Some("high") => 0.8,
                Some("low") => 0.2,
                _ => 0.5,
            };
            ((max_tokens as f64) * fraction).round() as u32
        }
        None => match effort {
            Some("high") => 24_576,
            Some("low") => 1_024,
            _ => 8_192,
        },
    }
}

pub fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Cheap unique-enough id for responses/chunks; avoids pulling in a UUID
/// dependency just for this.
pub fn gen_id(prefix: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}-{nanos:x}")
}
