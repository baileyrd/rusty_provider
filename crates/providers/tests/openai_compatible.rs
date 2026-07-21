mod common;

use futures_util::StreamExt;
use rp_core::{Provider, ProviderError, ReasoningConfig};
use rp_providers::OpenAiCompatibleProvider;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn chat_success_parses_response_and_sends_correct_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer test-key"))
        .and(body_partial_json(
            json!({"model": "gpt-4o-mini", "stream": false}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let resp = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");

    // "provider/model", not the raw model the mock echoed back.
    assert_eq!(resp.model, "openai/gpt-4o-mini");
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(
        resp.choices[0].message.content,
        Some(rp_core::MessageContent::text("hello there"))
    );
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    let usage = resp.usage.expect("usage should be present");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
    assert!(
        resp.cost_usd.is_none(),
        "providers never set cost_usd themselves"
    );
}

#[tokio::test]
async fn chat_parses_tool_calls_in_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Boston\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::request_with_tool("gpt-4o-mini");
    let resp = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");

    let tool_calls = resp.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls should be present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_123");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Boston\"}");
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
}

#[tokio::test]
async fn chat_stream_parses_delta_chunks_and_stops_at_done() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let mut req = common::simple_request("gpt-4o-mini");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gpt-4o-mini")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    // The [DONE] sentinel ends the stream and isn't yielded as a chunk.
    assert_eq!(chunks.len(), 3);
    assert!(matches!(
        chunks[0].choices[0].delta.role,
        Some(rp_core::Role::Assistant)
    ));
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hi"));
    assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    assert_eq!(chunks[2].usage.as_ref().unwrap().completion_tokens, 2);
}

#[tokio::test]
async fn chat_stream_yields_a_decode_error_for_malformed_event_json() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {not valid json\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let mut req = common::simple_request("gpt-4o-mini");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gpt-4o-mini")
        .await
        .expect("chat_stream should succeed");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }

    assert_eq!(items.len(), 2);
    assert!(items[0].is_ok());
    match &items[1] {
        Err(ProviderError::Decode(_)) => {}
        other => panic!("expected a Decode error, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_stream_yields_no_chunks_for_an_immediately_done_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let mut req = common::simple_request("gpt-4o-mini");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gpt-4o-mini")
        .await
        .expect("chat_stream should succeed");
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn chat_maps_429_to_retryable_rate_limited() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "12")
                .set_body_json(json!({"error": {"message": "rate limit exceeded"}})),
        )
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let err = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect_err("should fail");

    assert!(err.is_retryable());
    match err {
        ProviderError::RateLimited { retry_after_secs } => assert_eq!(retry_after_secs, Some(12)),
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_forwards_tools_and_tool_choice_verbatim_in_request_body() {
    // Unlike Anthropic/Gemini, this adapter does no translation -- "tools"
    // and "tool_choice" are the OpenAI wire shape already, so they should
    // pass through byte-for-byte.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    }
                }
            }],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null, "tool_calls": []},
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let mut req = common::request_with_tool("gpt-4o-mini");
    req.tool_choice = Some(json!({"type": "function", "function": {"name": "get_weather"}}));

    provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_forwards_json_schema_response_format_verbatim_in_request_body() {
    // Matches the OpenAI wire shape already, so this adapter does no
    // translation -- straight passthrough.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_report",
                    "description": "A weather report for one city",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "temperature_f": {"type": "number"},
                        },
                        "required": ["city", "temperature_f"],
                    },
                    "strict": true,
                }
            },
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "{\"city\":\"Boston\",\"temperature_f\":72}"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::request_with_json_schema("gpt-4o-mini");

    provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_forwards_json_object_response_format_verbatim_in_request_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(
            json!({"response_format": {"type": "json_object"}}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "{}"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::request_with_json_object("gpt-4o-mini");

    provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_forwards_image_content_verbatim_in_request_body() {
    // Since this adapter's WireRequest holds `&[ChatMessage]` directly,
    // MessageContent's untagged serialization should pass a parts array
    // through byte-for-byte with no translation.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ],
            }],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "a hello image"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::request_with_image("gpt-4o-mini");
    provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_forwards_audio_content_verbatim_in_request_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's said in this clip?"},
                    {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
                ],
            }],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::request_with_audio("gpt-4o-mini");
    provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_stream_parses_tool_call_deltas() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Boston\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":8,\"total_tokens\":28}}\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let mut req = common::request_with_tool("gpt-4o-mini");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gpt-4o-mini")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    assert_eq!(chunks.len(), 4);
    let start_delta = chunks[0].choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(start_delta[0].id.as_deref(), Some("call_123"));
    assert_eq!(
        start_delta[0].function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );

    let mut assembled_args = String::new();
    for chunk in &chunks[1..3] {
        let delta = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assembled_args.push_str(
            delta
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref()
                .unwrap_or(""),
        );
    }
    let parsed: serde_json::Value = serde_json::from_str(&assembled_args).unwrap();
    assert_eq!(parsed["city"], "Boston");

    assert_eq!(
        chunks[3].choices[0].finish_reason.as_deref(),
        Some("tool_calls")
    );
}

#[tokio::test]
async fn chat_maps_400_to_non_retryable_invalid_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"error": {"message": "bad request"}})),
        )
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let err = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect_err("should fail");

    assert!(!err.is_retryable());
    assert!(matches!(err, ProviderError::InvalidRequest(_)));
}

#[tokio::test]
async fn chat_maps_401_and_403_to_non_retryable_auth_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_json(json!({"error": {"message": "invalid api key"}})),
        )
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let err = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect_err("should fail");

    assert!(!err.is_retryable());
    match err {
        ProviderError::Auth(message) => assert_eq!(message, "invalid api key"),
        other => panic!("expected Auth, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_maps_404_and_422_to_non_retryable_invalid_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(422)
                .set_body_json(json!({"error": {"message": "unprocessable entity"}})),
        )
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let err = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect_err("should fail");

    assert!(!err.is_retryable());
    assert!(matches!(err, ProviderError::InvalidRequest(_)));
}

#[tokio::test]
async fn chat_maps_an_unclassified_5xx_to_a_retryable_upstream_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(503)
                .set_body_json(json!({"error": {"message": "server overloaded"}})),
        )
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let err = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect_err("should fail");

    assert!(err.is_retryable());
    match err {
        ProviderError::Upstream { status, message } => {
            assert_eq!(status, 503);
            assert_eq!(message, "server overloaded");
        }
        other => panic!("expected Upstream, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_falls_back_to_the_raw_body_when_the_error_response_is_not_json() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_raw("upstream is on fire", "text/plain"))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let err = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect_err("should fail");

    match err {
        ProviderError::Upstream { message, .. } => {
            assert_eq!(message, "upstream is on fire")
        }
        other => panic!("expected Upstream, got {other:?}"),
    }
}

// --- reasoning ---------------------------------------------------------

#[tokio::test]
async fn chat_sends_reasoning_effort_and_parses_reasoning_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({"reasoning_effort": "high"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "42",
                    "reasoning_content": "Let me work this out."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::request_with_reasoning(
        "gpt-4o-mini",
        ReasoningConfig {
            effort: Some("high".to_string()),
            max_tokens: None,
            exclude: None,
        },
    );
    let resp = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");

    assert_eq!(
        resp.choices[0].message.reasoning.as_deref(),
        Some("Let me work this out.")
    );
}

#[tokio::test]
async fn chat_omits_reasoning_content_when_the_client_asked_to_exclude_it() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "42",
                    "reasoning_content": "secret reasoning"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::request_with_reasoning(
        "gpt-4o-mini",
        ReasoningConfig {
            effort: Some("high".to_string()),
            max_tokens: None,
            exclude: Some(true),
        },
    );
    let resp = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");

    assert_eq!(resp.choices[0].message.reasoning, None);
}

#[tokio::test]
async fn chat_stream_parses_reasoning_content_deltas() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Step 1. \"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Step 2.\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"42\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let mut req = common::request_with_reasoning(
        "gpt-4o-mini",
        ReasoningConfig {
            effort: Some("high".to_string()),
            max_tokens: None,
            exclude: None,
        },
    );
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gpt-4o-mini")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    let reasoning: String = chunks
        .iter()
        .filter_map(|c| c.choices[0].delta.reasoning.clone())
        .collect();
    assert_eq!(reasoning, "Step 1. Step 2.");

    let content: String = chunks
        .iter()
        .filter_map(|c| c.choices[0].delta.content.clone())
        .collect();
    assert_eq!(content, "42");
}

// --- prompt caching ----------------------------------------------------------

#[tokio::test]
async fn chat_parses_prompt_tokens_details_cached_tokens_into_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 20,
                "total_tokens": 1020,
                "prompt_tokens_details": {"cached_tokens": 900}
            }
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let resp = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");

    let usage = resp.usage.expect("usage should be present");
    // Already cache-inclusive on the wire -- no arithmetic needed.
    assert_eq!(usage.prompt_tokens, 1000);
    assert_eq!(usage.cached_tokens, Some(900));
    assert_eq!(usage.cache_creation_tokens, None);
}

#[tokio::test]
async fn chat_leaves_cached_tokens_none_without_prompt_tokens_details() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&server)
        .await;

    let provider = OpenAiCompatibleProvider::new("openai", server.uri(), "test-key");
    let req = common::simple_request("gpt-4o-mini");
    let resp = provider
        .chat(&req, "gpt-4o-mini")
        .await
        .expect("chat should succeed");

    assert_eq!(resp.usage.unwrap().cached_tokens, None);
}
