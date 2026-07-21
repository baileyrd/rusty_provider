mod common;

use futures_util::StreamExt;
use rp_core::ProviderError;
use rp_core::{ChatMessage, Provider, ReasoningConfig};
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
async fn chat_sends_json_schema_response_format_as_a_forced_tool_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "tools": [{
                "name": "weather_report",
                "description": "A weather report for one city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "temperature_f": {"type": "number"},
                    },
                    "required": ["city", "temperature_f"],
                }
            }],
            "tool_choice": {"type": "tool", "name": "weather_report"},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_01abc",
                "name": "weather_report",
                "input": {"city": "Boston", "temperature_f": 72}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 8}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_json_schema("claude-sonnet-5");
    let resp = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");

    // The forced tool call is unwrapped into plain JSON content, not
    // surfaced as a `tool_calls` entry the client would have to answer.
    assert!(resp.choices[0].message.tool_calls.is_none());
    let content = resp.choices[0]
        .message
        .content
        .as_ref()
        .expect("content should be present");
    let parsed: serde_json::Value =
        serde_json::from_str(&content.as_plain_text()).expect("content should be valid JSON");
    assert_eq!(parsed["city"], "Boston");
    assert_eq!(parsed["temperature_f"], 72);
    // Not "tool_calls" -- from the client's perspective this is a normal
    // completion with a JSON answer.
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn chat_rejects_json_object_response_format_without_contacting_the_server() {
    // No mock server is started -- if this somehow tried to make a real
    // HTTP call it would fail with a connection error, not
    // UnsupportedFeature, proving the check happens before any request.
    let provider = AnthropicProvider::new("http://127.0.0.1:1", "test-key");
    let req = common::request_with_json_object("claude-sonnet-5");
    let err = provider.chat(&req, "claude-sonnet-5").await.unwrap_err();
    assert!(matches!(err, ProviderError::UnsupportedFeature(_)));
    assert!(err.is_retryable());
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
async fn chat_sends_inline_file_content_as_a_base64_document_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "summarize this document"},
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "aGVsbG8="}},
                ]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "a summary"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_file("claude-sonnet-5");
    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_remote_file_content_as_a_url_document_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "summarize this document"},
                    {"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}},
                ]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "a summary"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 5}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_remote_file("claude-sonnet-5");
    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_rejects_audio_content_without_contacting_the_server() {
    // No Mock is registered on this server at all -- if the adapter tried
    // to make an HTTP request it would get a 404 from wiremock's default
    // "no matching mock" response, not this specific error, proving the
    // rejection happens before any request is sent.
    let server = MockServer::start().await;
    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_audio("claude-sonnet-5");

    let err = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect_err("Anthropic has no audio-input support");
    assert!(matches!(err, ProviderError::UnsupportedContent(_)));
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
async fn chat_forwards_top_k_but_has_no_field_for_the_rest_of_the_sampling_params() {
    // Anthropic's native API only has an equivalent for `top_k` among the
    // sampling params exercised by `request_with_sampling_params`; the
    // other 7 (min_p, top_a, frequency_penalty, presence_penalty,
    // repetition_penalty, logit_bias, seed) have no field on this
    // adapter's WireRequest at all, so there's nothing to serialize --
    // enforced at compile time rather than needing a runtime assertion.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({"top_k": 40})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_sampling_params("claude-sonnet-5");

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
async fn chat_stream_unwraps_a_forced_structured_output_tool_call_into_content_deltas() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"weather_report\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Boston\\\"}\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":6}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_json_schema("claude-sonnet-5");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    // No chunk should ever carry a `tool_calls` delta -- everything streams
    // as plain `content`, the same shape a JSON-mode answer takes on every
    // other provider.
    assert!(chunks
        .iter()
        .all(|c| c.choices[0].delta.tool_calls.is_none()));

    let assembled: String = chunks
        .iter()
        .filter_map(|c| c.choices[0].delta.content.clone())
        .collect();
    let parsed: serde_json::Value = serde_json::from_str(&assembled).unwrap();
    assert_eq!(parsed["city"], "Boston");

    let finish_reason = chunks
        .iter()
        .find_map(|c| c.choices[0].finish_reason.clone());
    assert_eq!(finish_reason.as_deref(), Some("stop"));
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

// --- reasoning / extended thinking ------------------------------------------

#[tokio::test]
async fn chat_sends_extended_thinking_config_derived_from_effort() {
    let server = MockServer::start().await;
    // effort "high" with no client `max_tokens` falls back to the flat
    // default (24_576); since that exceeds the request's own default
    // max_tokens (4096), max_tokens gets bumped so budget_tokens < max_tokens.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "thinking": {"type": "enabled", "budget_tokens": 24_576},
            "max_tokens": 28_672,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_reasoning(
        "claude-sonnet-5",
        ReasoningConfig {
            effort: Some("high".to_string()),
            max_tokens: None,
            exclude: None,
        },
    );
    req.max_tokens = None;
    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_clamps_thinking_budget_to_the_anthropic_minimum() {
    let server = MockServer::start().await;
    // An explicit `reasoning.max_tokens` below Anthropic's 1024 floor gets
    // clamped up; the request's own max_tokens (unset -> default 4096)
    // already exceeds it, so no bump is needed there.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "max_tokens": 4096,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_reasoning(
        "claude-sonnet-5",
        ReasoningConfig {
            effort: None,
            max_tokens: Some(10),
            exclude: None,
        },
    );
    req.max_tokens = None;
    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_bumps_max_tokens_when_thinking_budget_would_exceed_it() {
    let server = MockServer::start().await;
    // The client's own max_tokens (500) is below the requested thinking
    // budget (2000) -- Anthropic requires max_tokens > budget_tokens, so
    // it must get bumped rather than sent as-is.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "max_tokens": 6096,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_reasoning(
        "claude-sonnet-5",
        ReasoningConfig {
            effort: None,
            max_tokens: Some(2000),
            exclude: None,
        },
    );
    req.max_tokens = Some(500);
    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_parses_thinking_blocks_into_the_reasoning_field() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [
                {"type": "thinking", "thinking": "Let me work through this step by step."},
                {"type": "text", "text": "The answer is 42."}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_reasoning(
        "claude-sonnet-5",
        ReasoningConfig {
            effort: Some("low".to_string()),
            max_tokens: None,
            exclude: None,
        },
    );
    let resp = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");

    assert_eq!(
        resp.choices[0].message.reasoning.as_deref(),
        Some("Let me work through this step by step.")
    );
    assert_eq!(
        resp.choices[0].message.content,
        Some(rp_core::MessageContent::text("The answer is 42."))
    );
}

#[tokio::test]
async fn chat_omits_reasoning_when_the_client_asked_to_exclude_it() {
    let server = MockServer::start().await;
    // Anthropic has no server-side toggle to suppress `thinking` blocks --
    // it still sends one -- so `exclude` has to be enforced client-side.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [
                {"type": "thinking", "thinking": "secret reasoning"},
                {"type": "text", "text": "the answer"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::request_with_reasoning(
        "claude-sonnet-5",
        ReasoningConfig {
            effort: Some("low".to_string()),
            max_tokens: None,
            exclude: Some(true),
        },
    );
    let resp = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");

    assert_eq!(resp.choices[0].message.reasoning, None);
}

#[tokio::test]
async fn chat_stream_emits_thinking_delta_as_reasoning_deltas() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Step 1: \"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"add them up.\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc123\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"7\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":6}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_reasoning(
        "claude-sonnet-5",
        ReasoningConfig {
            effort: Some("low".to_string()),
            max_tokens: None,
            exclude: None,
        },
    );
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    // `signature_delta` carries no readable text and isn't surfaced.
    let reasoning: String = chunks
        .iter()
        .filter_map(|c| c.choices[0].delta.reasoning.clone())
        .collect();
    assert_eq!(reasoning, "Step 1: add them up.");

    let content: String = chunks
        .iter()
        .filter_map(|c| c.choices[0].delta.content.clone())
        .collect();
    assert_eq!(content, "7");
}

#[tokio::test]
async fn chat_stream_omits_reasoning_deltas_when_excluded() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"secret\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"7\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":6}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_reasoning(
        "claude-sonnet-5",
        ReasoningConfig {
            effort: Some("low".to_string()),
            max_tokens: None,
            exclude: Some(true),
        },
    );
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "claude-sonnet-5")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    assert!(chunks
        .iter()
        .all(|c| c.choices[0].delta.reasoning.is_none()));
}

// --- prompt caching ----------------------------------------------------------

#[tokio::test]
async fn chat_sends_system_as_a_cache_control_block_array_when_requested() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "system": [{
                "type": "text",
                "text": "You are a helpful assistant.",
                "cache_control": {"type": "ephemeral"},
            }],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("claude-sonnet-5");
    let mut system = ChatMessage::system("You are a helpful assistant.");
    system.cache_control = Some(rp_core::CacheControl::Ephemeral);
    req.messages = vec![system, ChatMessage::user("hi")];
    provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_parses_cache_creation_and_cache_read_tokens_into_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 1000,
            }
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::simple_request("claude-sonnet-5");
    let resp = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");

    let usage = resp.usage.expect("usage should be present");
    // Cache-inclusive total: 100 (fresh) + 500 (written) + 1000 (read).
    assert_eq!(usage.prompt_tokens, 1600);
    assert_eq!(usage.cached_tokens, Some(1000));
    assert_eq!(usage.cache_creation_tokens, Some(500));
}

#[tokio::test]
async fn chat_leaves_cache_usage_fields_none_without_any_caching() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 20}
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(server.uri(), "test-key");
    let req = common::simple_request("claude-sonnet-5");
    let resp = provider
        .chat(&req, "claude-sonnet-5")
        .await
        .expect("chat should succeed");

    let usage = resp.usage.expect("usage should be present");
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.cached_tokens, None);
    assert_eq!(usage.cache_creation_tokens, None);
}

#[tokio::test]
async fn chat_stream_reports_cache_tokens_from_message_start_in_the_final_usage() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":100,\"cache_creation_input_tokens\":500,\"cache_read_input_tokens\":1000}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":20}}\n\n",
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

    let usage = chunks
        .iter()
        .find_map(|c| c.usage.clone())
        .expect("final chunk should carry usage");
    assert_eq!(usage.prompt_tokens, 1600);
    assert_eq!(usage.cached_tokens, Some(1000));
    assert_eq!(usage.cache_creation_tokens, Some(500));
}
