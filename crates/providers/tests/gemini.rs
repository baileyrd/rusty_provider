mod common;

use futures_util::StreamExt;
use rp_core::{ChatMessage, Provider, ProviderError, Role, ToolCall};
use rp_providers::GeminiProvider;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn chat_success_sends_api_key_as_query_param() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(query_param("key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "hello there"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::simple_request("gemini-2.0-flash");
    let resp = provider
        .chat(&req, "gemini-2.0-flash")
        .await
        .expect("chat should succeed");

    assert_eq!(resp.model, "gemini/gemini-2.0-flash");
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
async fn chat_parses_function_call_and_overrides_finish_reason() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Boston"}}}],
                    "role": "model"
                },
                // Gemini reports STOP even for a function call turn -- the
                // adapter is responsible for overriding this to "tool_calls".
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8, "totalTokenCount": 28}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_tool("gemini-2.0-flash");
    let resp = provider
        .chat(&req, "gemini-2.0-flash")
        .await
        .expect("chat should succeed");

    let tool_calls = resp.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls should be present");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Boston");
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
}

#[tokio::test]
async fn chat_sends_tools_and_tool_choice_translated_to_function_declarations() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "tools": [{
                "functionDeclarations": [{
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    }
                }]
            }],
            "toolConfig": {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}
            },
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Boston"}}}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8, "totalTokenCount": 28}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_tool("gemini-2.0-flash");
    req.tool_choice = Some(json!({"type": "function", "function": {"name": "get_weather"}}));

    provider
        .chat(&req, "gemini-2.0-flash")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_stream_parses_function_call_delta() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"Boston\"}}}]},\"finishReason\":\"STOP\"}],",
        "\"usageMetadata\":{\"promptTokenCount\":12,\"candidatesTokenCount\":6,\"totalTokenCount\":18}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_tool("gemini-2.0-flash");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gemini-2.0-flash")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    assert_eq!(chunks.len(), 1);
    // Unlike Anthropic's index-based fragments, Gemini emits the whole call
    // -- name and complete arguments -- in a single delta.
    let tool_calls = chunks[0].choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    let args: serde_json::Value = serde_json::from_str(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(args["city"], "Boston");
    assert_eq!(
        chunks[0].choices[0].finish_reason.as_deref(),
        Some("tool_calls")
    );
}

#[tokio::test]
async fn chat_tracks_call_id_to_function_name_across_turns_for_tool_result() {
    let server = MockServer::start().await;
    // The Role::Tool message only carries a call id (OpenAI-shaped, like
    // every other adapter's input) but Gemini's functionResponse needs the
    // function *name* -- this asserts the adapter recovered "get_weather"
    // from the earlier assistant turn's tool_calls rather than sending it
    // blank.
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "contents": [
                {"role": "user", "parts": [{"text": "What's the weather?"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Boston"}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"content": "72F and sunny"}}}]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "It's 72F."}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 10, "totalTokenCount": 40}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("gemini-2.0-flash");
    req.messages = vec![
        ChatMessage::user("What's the weather?"),
        {
            let mut m = ChatMessage::assistant("");
            m.content = None;
            m.tool_calls = Some(vec![ToolCall::function(
                "call_1",
                "get_weather",
                "{\"city\":\"Boston\"}",
            )]);
            m
        },
        {
            let mut m = ChatMessage::user("72F and sunny");
            m.tool_call_id = Some("call_1".to_string());
            m.role = Role::Tool;
            m
        },
    ];

    provider
        .chat(&req, "gemini-2.0-flash")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_image_content_as_inline_data() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "contents": [
                {"role": "user", "parts": [
                    {"text": "what's in this image?"},
                    {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
                ]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "a hello image"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_image("gemini-2.0-flash");
    provider
        .chat(&req, "gemini-2.0-flash")
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_stream_parses_text_parts() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" there\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2,\"totalTokenCount\":7}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("gemini-2.0-flash");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gemini-2.0-flash")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse"));
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("Hi"));
    assert_eq!(
        chunks[1].choices[0].delta.content.as_deref(),
        Some(" there")
    );
    assert_eq!(chunks[1].choices[0].finish_reason.as_deref(), Some("stop"));
    assert_eq!(chunks[1].usage.as_ref().unwrap().completion_tokens, 2);
    // No usageMetadata on the first event -- it defaults to all-zero counts,
    // which the adapter treats as "no usage" rather than a real zero-token
    // observation.
    assert!(chunks[0].usage.is_none());
}

#[tokio::test]
async fn chat_stream_handles_a_candidate_with_no_parts_gracefully() {
    let server = MockServer::start().await;
    // No "candidates" key at all -- the adapter must not panic when there's
    // nothing to translate for an event.
    let sse_body = "data: {\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":0,\"totalTokenCount\":1}}\n\n";

    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("gemini-2.0-flash");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gemini-2.0-flash")
        .await
        .expect("chat_stream should succeed");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk should parse without a candidate"));
    }

    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].choices[0].delta.content.is_none());
    assert!(chunks[0].choices[0].delta.tool_calls.is_none());
    assert!(chunks[0].choices[0].finish_reason.is_none());
}

#[tokio::test]
async fn chat_stream_yields_a_decode_error_for_malformed_event_json() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
        "data: {not valid json\n\n",
    );

    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("gemini-2.0-flash");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gemini-2.0-flash")
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
        .and(path(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw("", "text/event-stream"))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::simple_request("gemini-2.0-flash");
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gemini-2.0-flash")
        .await
        .expect("chat_stream should succeed");
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn chat_maps_429_to_retryable_rate_limited() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(
            ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "rate limited"}})),
        )
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::simple_request("gemini-2.0-flash");
    let err = provider
        .chat(&req, "gemini-2.0-flash")
        .await
        .expect_err("should fail");

    assert!(err.is_retryable());
    assert!(matches!(err, ProviderError::RateLimited { .. }));
}
