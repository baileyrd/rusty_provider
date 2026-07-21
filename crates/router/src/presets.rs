//! Named, reusable request-default bundles (`[[presets]]`): a request
//! that sets `"preset": "<name>"` gets that preset's `model`/`provider`/
//! `system_prompt`/sampling-param defaults folded in before dispatch --
//! pairing naturally with `[[routes]]` aliases, since a preset's `model`
//! can itself be a route alias.

use rp_core::{ChatMessage, ChatRequest, Role};

use crate::config::PresetConfig;

/// Applies `preset` to `req` in place:
/// - `model`, if set, overrides `req.model` outright (centralizing model
///   selection is the point of a preset, unlike every other field here).
/// - `system_prompt`, if set, is prepended as a new system message --
///   but only if `req` doesn't already have one of its own.
/// - `provider`, if set and `req.provider` is unset, becomes `req`'s
///   provider preferences wholesale (no per-field merge between the two
///   `ProviderPreferences` -- the request's, if present at all, wins
///   entirely).
/// - every sampling-param field is filled in only where `req` left the
///   corresponding field unset -- a per-field default, not an
///   all-or-nothing bundle.
pub fn apply(preset: &PresetConfig, req: &mut ChatRequest) {
    if let Some(model) = &preset.model {
        req.model = model.clone();
    }

    if let Some(system_prompt) = &preset.system_prompt {
        let has_system_message = req.messages.iter().any(|m| m.role == Role::System);
        if !has_system_message {
            req.messages.insert(0, ChatMessage::system(system_prompt));
        }
    }

    if req.provider.is_none() {
        req.provider = preset.provider.clone();
    }

    macro_rules! fill_default {
        ($field:ident) => {
            if req.$field.is_none() {
                req.$field = preset.$field.clone();
            }
        };
    }
    fill_default!(temperature);
    fill_default!(top_p);
    fill_default!(max_tokens);
    fill_default!(stop);
    fill_default!(top_k);
    fill_default!(min_p);
    fill_default!(top_a);
    fill_default!(frequency_penalty);
    fill_default!(presence_penalty);
    fill_default!(repetition_penalty);
    fill_default!(logit_bias);
    fill_default!(seed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rp_core::ProviderPreferences;

    fn preset(name: &str) -> PresetConfig {
        PresetConfig {
            name: name.to_string(),
            model: None,
            system_prompt: None,
            provider: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            top_k: None,
            min_p: None,
            top_a: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            logit_bias: None,
            seed: None,
        }
    }

    fn request() -> ChatRequest {
        serde_json::from_value(serde_json::json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    #[test]
    fn model_override_replaces_the_request_model() {
        let mut p = preset("support-bot");
        p.model = Some("smart".to_string());
        let mut req = request();
        apply(&p, &mut req);
        assert_eq!(req.model, "smart");
    }

    #[test]
    fn no_model_in_preset_leaves_the_request_model_untouched() {
        let p = preset("support-bot");
        let mut req = request();
        apply(&p, &mut req);
        assert_eq!(req.model, "openai/gpt-4o-mini");
    }

    #[test]
    fn system_prompt_is_prepended_when_the_request_has_none() {
        let mut p = preset("support-bot");
        p.system_prompt = Some("You are a support agent.".to_string());
        let mut req = request();
        apply(&p, &mut req);
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, Role::System);
        assert_eq!(req.messages[1].role, Role::User);
    }

    #[test]
    fn system_prompt_is_not_added_when_the_request_already_has_one() {
        let mut p = preset("support-bot");
        p.system_prompt = Some("preset prompt".to_string());
        let mut req = request();
        req.messages
            .insert(0, ChatMessage::system("caller's own system prompt"));
        apply(&p, &mut req);
        assert_eq!(req.messages.len(), 2);
        match &req.messages[0].content {
            Some(rp_core::MessageContent::Text(text)) => {
                assert_eq!(text, "caller's own system prompt");
            }
            other => panic!("expected Text content, got {other:?}"),
        }
    }

    #[test]
    fn provider_prefs_fill_in_only_when_the_request_has_none() {
        let mut p = preset("support-bot");
        p.provider = Some(ProviderPreferences {
            only: Some(vec!["anthropic".to_string()]),
            ..Default::default()
        });
        let mut req = request();
        apply(&p, &mut req);
        assert_eq!(
            req.provider.unwrap().only,
            Some(vec!["anthropic".to_string()])
        );
    }

    #[test]
    fn provider_prefs_are_untouched_when_the_request_already_sets_them() {
        let mut p = preset("support-bot");
        p.provider = Some(ProviderPreferences {
            only: Some(vec!["anthropic".to_string()]),
            ..Default::default()
        });
        let mut req = request();
        req.provider = Some(ProviderPreferences {
            only: Some(vec!["openai".to_string()]),
            ..Default::default()
        });
        apply(&p, &mut req);
        assert_eq!(req.provider.unwrap().only, Some(vec!["openai".to_string()]));
    }

    #[test]
    fn sampling_params_fill_in_only_unset_fields() {
        let mut p = preset("support-bot");
        p.temperature = Some(0.2);
        p.max_tokens = Some(500);
        let mut req = request();
        req.temperature = Some(0.9); // caller's own value must survive
        apply(&p, &mut req);
        assert_eq!(req.temperature, Some(0.9));
        assert_eq!(req.max_tokens, Some(500));
    }
}
