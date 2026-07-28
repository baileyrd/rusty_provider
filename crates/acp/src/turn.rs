//! One prompt turn: the loop that takes the user's message, asks the
//! model, runs whatever tools it calls, and asks again until the model
//! stops calling tools.
//!
//! Every model request goes through [`rp_router::Router`] and the same
//! pre-dispatch stages the HTTP server applies (preset, web search,
//! guardrails, moderation), so an ACP session inherits fallback chains,
//! budgets and usage accounting rather than re-implementing them.

use std::collections::BTreeMap;

use futures_util::StreamExt;
use serde_json::Value;

use rp_core::{
    ChatMessage, ChatRequest, ContentPart, FileData, ImageUrl, InputAudio, MessageContent, Role,
    ToolCall,
};
use rp_router::{Router, RouterError};

use crate::jsonrpc::{Connection, RpcError};
use crate::schema::{
    method, ContentBlock, Cost, EmbeddedResource, SessionNotification, SessionUpdate, StopReason,
    ToolCallStatus,
};
use crate::session::Session;
use crate::tools::{self, ToolContext, ToolError};

/// The model-side settings for a turn, from the `[acp]` config section.
#[derive(Debug, Clone)]
pub struct TurnConfig {
    /// A "provider/model" string or a `[[routes]]` alias, exactly like
    /// `ChatRequest.model`.
    pub model: String,
    /// Ceiling on model requests in a single turn. Without it, a model
    /// that keeps calling tools never returns control to the user.
    pub max_turn_requests: u32,
    pub system_prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// Attributes this session's spend to a configured `[[clients]]`
    /// entry, so an ACP session counts against the same budget as that
    /// client's HTTP traffic.
    pub client_name: Option<String>,
}

/// Runs a prompt turn to completion.
///
/// The `Err` case is reserved for failures the *client* has to see as a
/// failed `session/prompt` (a misconfigured model, an unreachable
/// provider). Anything the model can recover from -- a tool that failed,
/// a file that didn't exist -- is fed back into the loop instead.
pub async fn run(
    connection: &Connection,
    router: &Router,
    session: &Session,
    config: &TurnConfig,
    prompt: Vec<ContentBlock>,
) -> Result<StopReason, RpcError> {
    session.begin_turn();
    session
        .state()
        .await
        .history
        .push(prompt_to_message(prompt));

    if let Some(name) = &config.client_name {
        if let Err(exceeded) = router.check_client_budget(name).await {
            return Err(RpcError::internal(format!(
                "client budget exceeded: ${:.2} spent of a ${:.2} budget",
                exceeded.spent_usd, exceeded.budget_usd
            )));
        }
    }

    let tool_definitions = tools::definitions(&session.client_capabilities);

    for _ in 0..config.max_turn_requests {
        if session.is_cancelled() {
            return Ok(StopReason::Cancelled);
        }

        let mut request = ChatRequest {
            model: config.model.clone(),
            messages: with_system_prompt(config, &session.state().await.history),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            stream: Some(true),
            tools: (!tool_definitions.is_empty()).then(|| tool_definitions.clone()),
            user: config.client_name.clone(),
            ..Default::default()
        };

        // The same pre-dispatch pipeline `POST /v1/chat/completions`
        // runs, in the same order.
        if let Err(e) = router.apply_preset(&mut request) {
            return Err(RpcError::internal(e));
        }
        router.apply_web_search(&mut request).await;
        if let Err(e) = router.apply_guardrails(&mut request) {
            return refusal_or_error(connection, session, e);
        }
        if let Err(e) = router.apply_moderation(&request).await {
            return refusal_or_error(connection, session, e);
        }

        let completion =
            match stream_completion(connection, router, session, config, &request).await {
                Ok(completion) => completion,
                Err(StreamFailure::Cancelled) => return Ok(StopReason::Cancelled),
                Err(StreamFailure::Router(e)) => return Err(RpcError::internal(e)),
            };

        session
            .state()
            .await
            .history
            .push(completion.as_assistant_message());

        if completion.tool_calls.is_empty() {
            return Ok(match completion.finish_reason.as_deref() {
                Some("length") => StopReason::MaxTokens,
                Some("content_filter") => StopReason::Refusal,
                _ => StopReason::EndTurn,
            });
        }

        for call in &completion.tool_calls {
            match run_tool_call(connection, session, call).await {
                Ok(result) => session.state().await.history.push(result),
                Err(ToolError::Cancelled) => return Ok(StopReason::Cancelled),
                Err(ToolError::Transport(e)) => {
                    return Err(RpcError::internal(format!("client connection lost: {e}")))
                }
            }
        }
    }

    Ok(StopReason::MaxTurnRequests)
}

/// A guardrail or moderation block is a refusal the user should see, not
/// a protocol error -- so tell them why, then end the turn cleanly.
/// Anything else is a real failure.
fn refusal_or_error(
    connection: &Connection,
    session: &Session,
    error: RouterError,
) -> Result<StopReason, RpcError> {
    match error {
        RouterError::GuardrailBlocked(_) | RouterError::ModerationFlagged(_) => {
            notify(
                connection,
                session,
                SessionUpdate::AgentMessageChunk {
                    content: ContentBlock::text(error.to_string()),
                },
            );
            Ok(StopReason::Refusal)
        }
        other => Err(RpcError::internal(other)),
    }
}

/// What one model request produced.
#[derive(Default)]
struct Completion {
    text: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
    finish_reason: Option<String>,
}

impl Completion {
    fn as_assistant_message(&self) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: (!self.text.is_empty()).then(|| MessageContent::text(self.text.clone())),
            name: None,
            tool_calls: (!self.tool_calls.is_empty()).then(|| self.tool_calls.clone()),
            tool_call_id: None,
            reasoning: (!self.reasoning.is_empty()).then(|| self.reasoning.clone()),
            cache_control: None,
        }
    }
}

enum StreamFailure {
    Cancelled,
    Router(RouterError),
}

/// Dispatches one request and streams it back to the client as it
/// arrives, accumulating the pieces the loop needs.
async fn stream_completion(
    connection: &Connection,
    router: &Router,
    session: &Session,
    config: &TurnConfig,
    request: &ChatRequest,
) -> Result<Completion, StreamFailure> {
    let mut stream = router
        .dispatch_stream(request)
        .await
        .map_err(StreamFailure::Router)?;

    let mut completion = Completion::default();
    // Tool calls arrive as fragments keyed by index: the id and name once,
    // the arguments as accumulating string pieces.
    let mut partial_calls: BTreeMap<u32, PartialToolCall> = BTreeMap::new();

    while let Some(chunk) = stream.next().await {
        if session.is_cancelled() {
            return Err(StreamFailure::Cancelled);
        }

        let chunk = match chunk {
            Ok(chunk) => chunk,
            // Mid-stream provider failures can't fall back to another
            // candidate -- bytes are already on the wire -- so surface it.
            Err(e) => return Err(StreamFailure::Router(RouterError::Provider(e))),
        };

        for choice in &chunk.choices {
            if let Some(text) = &choice.delta.content {
                if !text.is_empty() {
                    completion.text.push_str(text);
                    notify(
                        connection,
                        session,
                        SessionUpdate::AgentMessageChunk {
                            content: ContentBlock::text(text.clone()),
                        },
                    );
                }
            }
            if let Some(reasoning) = &choice.delta.reasoning {
                if !reasoning.is_empty() {
                    completion.reasoning.push_str(reasoning);
                    notify(
                        connection,
                        session,
                        SessionUpdate::AgentThoughtChunk {
                            content: ContentBlock::text(reasoning.clone()),
                        },
                    );
                }
            }
            for delta in choice.delta.tool_calls.iter().flatten() {
                let partial = partial_calls.entry(delta.index).or_default();
                if let Some(id) = &delta.id {
                    partial.id = id.clone();
                }
                if let Some(function) = &delta.function {
                    if let Some(name) = &function.name {
                        partial.name.push_str(name);
                    }
                    if let Some(arguments) = &function.arguments {
                        partial.arguments.push_str(arguments);
                    }
                }
            }
            if let Some(reason) = &choice.finish_reason {
                completion.finish_reason = Some(reason.clone());
            }
        }

        if let Some(usage) = &chunk.usage {
            if let (Some(name), Some(cost)) = (&config.client_name, chunk.cost_usd) {
                router.record_client_spend(name, cost);
            }
            report_usage(
                connection,
                router,
                session,
                &chunk.model,
                usage,
                chunk.cost_usd,
            );
        }
    }

    completion.tool_calls = partial_calls
        .into_values()
        .filter(|partial| !partial.name.is_empty())
        .enumerate()
        .map(|(index, partial)| {
            ToolCall::function(
                // Some providers stream a nameless, id-less first
                // fragment; synthesize an id rather than emit an empty
                // one, which would break tool-result correlation.
                if partial.id.is_empty() {
                    format!("call_{index}")
                } else {
                    partial.id
                },
                partial.name,
                partial.arguments,
            )
        })
        .collect();

    Ok(completion)
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Reports context-window use, which clients render as a gauge. `size` is
/// required by the protocol and only known for models with a
/// `[[pricing]]` entry carrying `context_length`, so this stays silent
/// rather than guessing.
fn report_usage(
    connection: &Connection,
    router: &Router,
    session: &Session,
    model: &str,
    usage: &rp_core::Usage,
    cost_usd: Option<f64>,
) {
    let Some(size) = router
        .priced_models()
        .into_iter()
        .find(|info| info.id == model)
        .and_then(|info| info.context_length)
    else {
        return;
    };

    notify(
        connection,
        session,
        SessionUpdate::UsageUpdate {
            used: u64::from(usage.total_tokens),
            size: u64::from(size),
            cost: cost_usd.map(|amount| Cost {
                amount,
                currency: "USD".to_string(),
            }),
        },
    );
}

/// Reports a tool call to the client, runs it, reports the outcome, and
/// returns the message to feed back to the model.
async fn run_tool_call(
    connection: &Connection,
    session: &Session,
    call: &ToolCall,
) -> Result<ChatMessage, ToolError> {
    let arguments: Value = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
    let presentation = tools::presentation(&call.function.name, &arguments, session);

    notify(
        connection,
        session,
        SessionUpdate::ToolCall {
            tool_call_id: call.id.clone(),
            title: presentation.title.clone(),
            kind: presentation.kind,
            // Anything needing consent starts pending; the rest is
            // already under way by the time the client renders this.
            status: if tools::requires_permission(&call.function.name) {
                ToolCallStatus::Pending
            } else {
                ToolCallStatus::InProgress
            },
            content: Vec::new(),
            locations: presentation.locations.clone(),
            raw_input: (!arguments.is_null()).then(|| arguments.clone()),
        },
    );

    let context = ToolContext {
        connection,
        session,
        tool_call_id: call.id.clone(),
    };
    let outcome = tools::execute(
        &context,
        &call.function.name,
        &call.function.arguments,
        &presentation,
    )
    .await?;

    notify(
        connection,
        session,
        SessionUpdate::ToolCallUpdate {
            tool_call_id: call.id.clone(),
            status: Some(if outcome.failed {
                ToolCallStatus::Failed
            } else {
                ToolCallStatus::Completed
            }),
            title: None,
            content: (!outcome.content.is_empty()).then_some(outcome.content),
            raw_output: outcome.raw_output,
        },
    );

    Ok(ChatMessage {
        role: Role::Tool,
        content: Some(MessageContent::text(outcome.result)),
        name: None,
        tool_calls: None,
        tool_call_id: Some(call.id.clone()),
        reasoning: None,
        cache_control: None,
    })
}

fn notify(connection: &Connection, session: &Session, update: SessionUpdate) {
    let notification = SessionNotification {
        session_id: session.id.clone(),
        update,
    };
    if let Err(e) = connection.notify(method::SESSION_UPDATE, &notification) {
        tracing::debug!("dropping session update: {e}");
    }
}

/// The system prompt is prepended per request rather than stored in the
/// history, so a config change takes effect on the next turn of an
/// existing session.
fn with_system_prompt(config: &TurnConfig, history: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(history.len() + 1);
    messages.push(ChatMessage::system(config.system_prompt.clone()));
    messages.extend(history.iter().cloned());
    messages
}

/// Flattens the client's prompt blocks into one user message.
fn prompt_to_message(prompt: Vec<ContentBlock>) -> ChatMessage {
    let parts: Vec<ContentPart> = prompt.into_iter().map(content_block_to_part).collect();

    // Collapse the common all-text case back to a plain string, which is
    // what every provider adapter handles best.
    let content = match parts.as_slice() {
        [ContentPart::Text { text }] => MessageContent::Text(text.clone()),
        _ => MessageContent::Parts(parts),
    };

    ChatMessage {
        role: Role::User,
        content: Some(content),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning: None,
        cache_control: None,
    }
}

fn content_block_to_part(block: ContentBlock) -> ContentPart {
    match block {
        ContentBlock::Text { text } => ContentPart::Text { text },
        ContentBlock::Image {
            data,
            mime_type,
            uri,
        } => ContentPart::ImageUrl {
            image_url: ImageUrl {
                // ACP sends image bytes base64-encoded; the provider
                // adapters take the OpenAI `data:` URI form. A bare `uri`
                // with no data is passed through as-is.
                url: if data.is_empty() {
                    uri.unwrap_or_default()
                } else {
                    format!("data:{mime_type};base64,{data}")
                },
                detail: None,
            },
        },
        ContentBlock::Audio { data, mime_type } => ContentPart::InputAudio {
            input_audio: InputAudio {
                data,
                format: mime_type
                    .rsplit('/')
                    .next()
                    .unwrap_or("wav")
                    .trim_start_matches("x-")
                    .to_string(),
            },
        },
        // The agent can't dereference an arbitrary URI, so a link becomes
        // a mention: the model can ask to read it with `read_file`.
        ContentBlock::ResourceLink { uri, name, .. } => ContentPart::Text {
            text: format!("[the user referenced {name} at {uri}]"),
        },
        ContentBlock::Resource { resource } => match resource {
            EmbeddedResource::Text { uri, text, .. } => ContentPart::Text {
                text: format!("<context uri=\"{uri}\">\n{text}\n</context>"),
            },
            // Binary context: hand it over as a file part rather than
            // dropping it, and let the adapter decide what it can do.
            EmbeddedResource::Blob {
                uri,
                blob,
                mime_type,
            } => ContentPart::File {
                file: FileData {
                    file_data: format!(
                        "data:{};base64,{blob}",
                        mime_type.unwrap_or_else(|| "application/octet-stream".to_string())
                    ),
                    filename: Some(uri),
                },
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_text_block_becomes_a_plain_string_message() {
        let message = prompt_to_message(vec![ContentBlock::text("hello")]);
        assert!(matches!(message.role, Role::User));
        match message.content.unwrap() {
            MessageContent::Text(text) => assert_eq!(text, "hello"),
            other => panic!("expected plain text, got {other:?}"),
        }
    }

    #[test]
    fn mixed_blocks_stay_a_parts_array() {
        let message = prompt_to_message(vec![
            ContentBlock::text("what is this"),
            ContentBlock::Image {
                data: "AAAA".into(),
                mime_type: "image/png".into(),
                uri: None,
            },
        ]);
        match message.content.unwrap() {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "data:image/png;base64,AAAA");
                    }
                    other => panic!("expected an image part, got {other:?}"),
                }
            }
            other => panic!("expected parts, got {other:?}"),
        }
    }

    #[test]
    fn embedded_text_resources_are_wrapped_so_the_model_can_tell_them_apart() {
        let message = prompt_to_message(vec![ContentBlock::Resource {
            resource: EmbeddedResource::Text {
                uri: "file:///repo/a.rs".into(),
                text: "fn main() {}".into(),
                mime_type: None,
            },
        }]);
        let MessageContent::Text(text) = message.content.unwrap() else {
            panic!("expected a single collapsed text part");
        };
        assert!(text.contains("file:///repo/a.rs"), "{text}");
        assert!(text.contains("fn main() {}"), "{text}");
    }

    #[test]
    fn resource_links_become_a_mention_rather_than_being_dropped() {
        let message = prompt_to_message(vec![ContentBlock::ResourceLink {
            uri: "file:///repo/b.rs".into(),
            name: "b.rs".into(),
            title: None,
            description: None,
            mime_type: None,
        }]);
        let MessageContent::Text(text) = message.content.unwrap() else {
            panic!("expected text");
        };
        assert!(text.contains("b.rs"), "{text}");
        assert!(text.contains("file:///repo/b.rs"), "{text}");
    }

    #[test]
    fn audio_mime_types_reduce_to_the_format_the_adapters_expect() {
        let message = prompt_to_message(vec![ContentBlock::Audio {
            data: "AAAA".into(),
            mime_type: "audio/x-wav".into(),
        }]);
        let MessageContent::Parts(parts) = message.content.unwrap() else {
            panic!("expected parts");
        };
        match &parts[0] {
            ContentPart::InputAudio { input_audio } => assert_eq!(input_audio.format, "wav"),
            other => panic!("expected audio, got {other:?}"),
        }
    }

    #[test]
    fn the_system_prompt_leads_every_request_without_entering_the_history() {
        let config = TurnConfig {
            model: "openai/gpt-4o".into(),
            max_turn_requests: 4,
            system_prompt: "be helpful".into(),
            max_tokens: None,
            temperature: None,
            client_name: None,
        };
        let history = vec![ChatMessage::user("hi")];
        let messages = with_system_prompt(&config, &history);
        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0].role, Role::System));
        assert!(matches!(messages[1].role, Role::User));
        // The stored history itself is untouched.
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn an_assistant_turn_carries_its_tool_calls_and_reasoning() {
        let completion = Completion {
            text: "on it".into(),
            reasoning: "thinking".into(),
            tool_calls: vec![ToolCall::function("call_1", "read_file", "{}")],
            finish_reason: Some("tool_calls".into()),
        };
        let message = completion.as_assistant_message();
        assert!(matches!(message.role, Role::Assistant));
        assert_eq!(message.tool_calls.unwrap().len(), 1);
        assert_eq!(message.reasoning.unwrap(), "thinking");
    }

    #[test]
    fn an_empty_completion_carries_neither_content_nor_tool_calls() {
        let message = Completion::default().as_assistant_message();
        assert!(message.content.is_none());
        assert!(message.tool_calls.is_none());
        assert!(message.reasoning.is_none());
    }
}
