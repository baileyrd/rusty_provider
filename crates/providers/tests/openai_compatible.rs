mod common;

use futures_util::StreamExt;
use rp_core::{Provider, ProviderError};
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
        resp.choices[0].message.content.as_deref(),
        Some("hello there")
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
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hi"));
    assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    assert_eq!(chunks[2].usage.as_ref().unwrap().completion_tokens, 2);
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
