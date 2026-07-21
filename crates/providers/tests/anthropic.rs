mod common;

use futures_util::StreamExt;
use rp_core::ProviderError;
use rp_core::{ChatMessage, Provider};
use rp_providers::AnthropicProvider;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn chat_success_sends_correct_headers_and_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_partial_json(
            json!({"model": "claude-sonnet-5", "max_tokens": 64}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello there"}],
            "model": "claude-sonnet-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::simple_request("claude-sonnet-5");
    let resp = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");

    assert_eq!(resp.model, "anthropic/claude-sonnet-5");
    assert_eq!(
        resp.choices[0].message.content,
        Some(rp_core::MessageContent::text("hello there"))
    );
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    let usage = resp.usage.expect("usage should be present");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
}

#[tokio::test]
async fn chat_uses_default_max_tokens_when_request_leaves_it_unset() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({"max_tokens": 4096})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    req.max_tokens = None;

    // Anthropic requires max_tokens; the mock only matches if the adapter
    // filled in the documented default (4096) rather than omitting it.
    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_parses_tool_use_block_and_maps_finish_reason() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_01abc",
                "name": "get_weather",
                "input": {"city": "Boston"}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 8}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_tool("claude-sonnet-5");
    let resp = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");

    assert!(
        resp.choices[0].message.content.is_none(),
        "no text block means no content"
    );
    let tool_calls = resp.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls should be present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "toolu_01abc");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Boston");
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
}

#[tokio::test]
async fn chat_maps_tool_call_request_into_tool_use_and_tool_result_blocks() {
    let server = MockServer::start().await;
    // A prior assistant tool call plus its result, as a real multi-turn
    // conversation would send back. Confirms the request-side translation
    // (not just response parsing) produces Anthropic's block shape.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Boston"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "72F and sunny"}]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "It's 72F and sunny."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 30, "output_tokens": 10}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    req.messages = vec![
        ChatMessage::user("What's the weather?"),
        {
            let mut m = ChatMessage::assistant("");
            m.content = None;
            m.tool_calls = Some(vec![rp_core::ToolCall::function(
                "call_1",
                "get_weather",
                "{\"city\":\"Boston\"}",
            )]);
            m
        },
        {
            let mut m = ChatMessage::user("72F and sunny");
            m.tool_call_id = Some("call_1".to_string());
            m.role = rp_core::Role::Tool;
            m
        },
    ];

    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_image_content_as_a_base64_image_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                ]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "a hello image"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_image("claude-sonnet-5");
    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_tools_and_tool_choice_translated_to_wire_format() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_01abc",
                "name": "get_weather",
                "input": {"city": "Boston"}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 8}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_tool("claude-sonnet-5");
    req.tool_choice = Some(json!({"type": "function", "function": {"name": "get_weather"}}));

    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_stream_parses_text_deltas_and_final_usage() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    // message_stop carries no data we translate, so it yields nothing.
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hi"));
    assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    let usage = chunks[2]
        .usage
        .as_ref()
        .expect("final chunk should carry usage");
    assert_eq!(usage.prompt_tokens, 7);
    assert_eq!(usage.completion_tokens, 3);
}

#[tokio::test]
async fn chat_stream_parses_tool_call_deltas_by_index() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Boston\\\"}\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":6}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_tool("claude-sonnet-5");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    let start_delta = chunks[1].choices[0]
        .delta
        .tool_calls
        .as_ref()
        .expect("tool_calls delta");
    assert_eq!(start_delta[0].index, 1);
    assert_eq!(start_delta[0].id.as_deref(), Some("toolu_1"));
    assert_eq!(
        start_delta[0].function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );

    let mut assembled_args = String::new();
    for chunk in &chunks[2..4] {
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
        chunks[4].choices[0].finish_reason.as_deref(),
        Some("tool_calls")
    );
}

#[tokio::test]
async fn chat_stream_filters_out_ping_and_content_block_stop_events() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
        "data: {\"type\":\"ping\"}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    // "ping" and "content_block_stop" carry nothing we translate, so only
    // message_start and the text delta produce chunks.
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hi"));
}

#[tokio::test]
async fn chat_stream_filters_out_an_unrecognized_future_event_type() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "data: {\"type\":\"some_future_event_type\",\"foo\":\"bar\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse without erroring on the unknown type"));
    }

    assert_eq!(chunks.len(), 1);
}

#[tokio::test]
async fn chat_stream_maps_an_error_event_to_an_upstream_error() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }

    assert_eq!(items.len(), 2);
    assert!(items[0].is_ok());
    match &items[1] {
        Err(ProviderError::Upstream { status, message }) => {
            assert_eq!(*status, 500);
            assert_eq!(message, "Overloaded");
        }
        other => panic!("expected an Upstream error, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_stream_yields_a_decode_error_for_malformed_event_json() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "data: {not valid json\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
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
async fn chat_stream_yields_no_chunks_for_an_empty_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("", "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn chat_maps_429_to_retryable_rate_limited() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "20")
                .set_body_json(json!({"error": {"message": "rate limited"}})),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::simple_request("claude-sonnet-5");
    let err = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect_err("should fail");

    assert!(err.is_retryable());
    assert!(matches!(
        err,
        ProviderError::RateLimited {
            retry_after_secs: Some(20)
        }
    ));
}
