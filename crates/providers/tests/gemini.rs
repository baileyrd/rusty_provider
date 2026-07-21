mod common;

use futures_util::StreamExt;
use rp_core::{
    ChatMessage, EmbeddingsInput, EmbeddingsRequest, Provider, ProviderError, ReasoningConfig,
    Role, ToolCall,
};
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
        .chat(&req, "gemini-2.0-flash", None)
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
async fn chat_uses_the_byok_override_key_instead_of_the_configured_one() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(query_param("key", "byok-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "hello there"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        })))
        .mount(&server)
        .await;

    // Constructed with "test-key" -- the mock only matches "byok-key", so a
    // successful response here proves the override won, not the configured
    // key wiremock would otherwise reject with a 404 (no matching mock).
    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::simple_request("gemini-2.0-flash");
    provider
        .chat(&req, "gemini-2.0-flash", Some("byok-key"))
        .await
        .expect("chat should succeed with the byok override key");
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
        .chat(&req, "gemini-2.0-flash", None)
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
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_json_schema_response_format_as_response_schema() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "temperature_f": {"type": "number"},
                    },
                    "required": ["city", "temperature_f"],
                },
            },
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "{\"city\":\"Boston\",\"temperature_f\":72}"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8, "totalTokenCount": 28}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_json_schema("gemini-2.0-flash");

    provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_json_object_response_format_with_no_schema() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "generationConfig": {"responseMimeType": "application/json"},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "{}"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_json_object("gemini-2.0-flash");

    let resp = provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
    assert_eq!(
        resp.choices[0].message.content,
        Some(rp_core::MessageContent::text("{}"))
    );
}

#[tokio::test]
async fn chat_forwards_its_native_sampling_params_camel_cased_but_has_no_field_for_the_rest() {
    // Gemini's native API has an equivalent for top_k, frequency_penalty,
    // presence_penalty, and seed among the sampling params exercised by
    // `request_with_sampling_params`; the other 4 (min_p, top_a,
    // repetition_penalty, logit_bias) have no field on this adapter's
    // GenerationConfig at all, so there's nothing to serialize --
    // enforced at compile time rather than needing a runtime assertion.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "generationConfig": {
                "topK": 40,
                "frequencyPenalty": 0.3,
                "presencePenalty": 0.4,
                "seed": 42,
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_sampling_params("gemini-2.0-flash");

    provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_has_no_wire_field_for_logprobs_and_never_returns_any() {
    // Gemini's generateContent API has no logprobs equivalent wired up in
    // this adapter -- no field on GenerationConfig to serialize
    // `logprobs`/`top_logprobs` into, and the response never populates
    // `logprobs`.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_logprobs("gemini-2.0-flash");

    let resp = provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
    assert!(resp.choices[0].logprobs.is_none());
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
        .chat_stream(&req, "gemini-2.0-flash", None)
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
        .chat(&req, "gemini-2.0-flash", None)
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
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_audio_content_as_inline_data() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "contents": [
                {"role": "user", "parts": [
                    {"text": "what's said in this clip?"},
                    {"inlineData": {"mimeType": "audio/wav", "data": "aGVsbG8="}},
                ]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 2, "totalTokenCount": 22}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_audio("gemini-2.0-flash");
    provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_inline_file_content_as_inline_data() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "contents": [
                {"role": "user", "parts": [
                    {"text": "summarize this document"},
                    {"inlineData": {"mimeType": "application/pdf", "data": "aGVsbG8="}},
                ]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "a summary"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_file("gemini-2.0-flash");
    provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sends_remote_file_content_as_file_data() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "contents": [
                {"role": "user", "parts": [
                    {"text": "summarize this document"},
                    {"fileData": {"mimeType": "application/pdf", "fileUri": "https://example.com/doc.pdf"}},
                ]},
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "a summary"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "totalTokenCount": 25}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_remote_file("gemini-2.0-flash");
    provider
        .chat(&req, "gemini-2.0-flash", None)
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
        .chat_stream(&req, "gemini-2.0-flash", None)
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
        .chat_stream(&req, "gemini-2.0-flash", None)
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
        .chat_stream(&req, "gemini-2.0-flash", None)
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
        .chat_stream(&req, "gemini-2.0-flash", None)
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
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect_err("should fail");

    assert!(err.is_retryable());
    assert!(matches!(err, ProviderError::RateLimited { .. }));
}

// --- reasoning / thinking config --------------------------------------------

#[tokio::test]
async fn chat_sends_thinking_config_derived_from_reasoning_effort() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "generationConfig": {
                "thinkingConfig": {"thinkingBudget": 8192, "includeThoughts": true}
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_reasoning(
        "gemini-2.0-flash",
        ReasoningConfig {
            effort: Some("medium".to_string()),
            max_tokens: None,
            exclude: None,
        },
    );
    req.max_tokens = None;
    provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_sets_include_thoughts_false_when_excluded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .and(body_partial_json(json!({
            "generationConfig": {
                "thinkingConfig": {"includeThoughts": false}
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_reasoning(
        "gemini-2.0-flash",
        ReasoningConfig {
            effort: None,
            max_tokens: None,
            exclude: Some(true),
        },
    );
    provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");
}

#[tokio::test]
async fn chat_parses_thought_marked_parts_into_the_reasoning_field() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me reason about this.", "thought": true},
                        {"text": "The answer is 42."}
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::request_with_reasoning(
        "gemini-2.0-flash",
        ReasoningConfig {
            effort: Some("low".to_string()),
            max_tokens: None,
            exclude: None,
        },
    );
    let resp = provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");

    assert_eq!(
        resp.choices[0].message.reasoning.as_deref(),
        Some("Let me reason about this.")
    );
    assert_eq!(
        resp.choices[0].message.content,
        Some(rp_core::MessageContent::text("The answer is 42."))
    );
}

#[tokio::test]
async fn chat_stream_emits_thought_parts_as_reasoning_deltas() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"thinking...\",\"thought\":true}],\"role\":\"model\"}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"42\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let mut req = common::request_with_reasoning(
        "gemini-2.0-flash",
        ReasoningConfig {
            effort: Some("low".to_string()),
            max_tokens: None,
            exclude: None,
        },
    );
    req.stream = Some(true);

    let mut stream = provider
        .chat_stream(&req, "gemini-2.0-flash", None)
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
    assert_eq!(reasoning, "thinking...");

    let content: String = chunks
        .iter()
        .filter_map(|c| c.choices[0].delta.content.clone())
        .collect();
    assert_eq!(content, "42");
}

// --- prompt caching ----------------------------------------------------------

#[tokio::test]
async fn chat_parses_cached_content_token_count_into_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 1000,
                "candidatesTokenCount": 20,
                "totalTokenCount": 1020,
                "cachedContentTokenCount": 800,
            }
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::simple_request("gemini-2.0-flash");
    let resp = provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");

    let usage = resp.usage.expect("usage should be present");
    // Already cache-inclusive on the wire -- no arithmetic needed, unlike Anthropic.
    assert_eq!(usage.prompt_tokens, 1000);
    assert_eq!(usage.cached_tokens, Some(800));
    assert_eq!(usage.cache_creation_tokens, None);
}

#[tokio::test]
async fn chat_leaves_cached_tokens_none_without_any_cache_hit() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "ok"}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = common::simple_request("gemini-2.0-flash");
    let resp = provider
        .chat(&req, "gemini-2.0-flash", None)
        .await
        .expect("chat should succeed");

    assert_eq!(resp.usage.unwrap().cached_tokens, None);
}

// --- embeddings ------------------------------------------------------------

fn embeddings_request(input: EmbeddingsInput) -> EmbeddingsRequest {
    EmbeddingsRequest {
        model: "gemini/text-embedding-004".to_string(),
        input,
        encoding_format: None,
        dimensions: None,
    }
}

#[tokio::test]
async fn embeddings_single_input_uses_batchembedcontents_with_one_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/text-embedding-004:batchEmbedContents"))
        .and(query_param("key", "test-key"))
        .and(body_partial_json(json!({
            "requests": [{
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": "hello"}]}
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [{"values": [0.5, 0.25]}]
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = embeddings_request(EmbeddingsInput::Single("hello".to_string()));
    let resp = provider
        .embeddings(&req, "text-embedding-004", None)
        .await
        .expect("embeddings should succeed");

    assert_eq!(resp.model, "gemini/text-embedding-004");
    assert_eq!(resp.data.len(), 1);
    assert_eq!(resp.data[0].embedding, vec![0.5, 0.25]);
    assert_eq!(resp.data[0].index, 0);
}

#[tokio::test]
async fn embeddings_batch_input_sends_one_request_per_text_and_preserves_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/text-embedding-004:batchEmbedContents"))
        .and(body_partial_json(json!({
            "requests": [
                {"model": "models/text-embedding-004", "content": {"parts": [{"text": "a"}]}},
                {"model": "models/text-embedding-004", "content": {"parts": [{"text": "b"}]}}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [{"values": [0.1]}, {"values": [0.2]}]
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = embeddings_request(EmbeddingsInput::Multiple(vec![
        "a".to_string(),
        "b".to_string(),
    ]));
    let resp = provider
        .embeddings(&req, "text-embedding-004", None)
        .await
        .expect("embeddings should succeed");

    assert_eq!(resp.data.len(), 2);
    assert_eq!(resp.data[0].index, 0);
    assert_eq!(resp.data[1].index, 1);
    assert_eq!(resp.data[1].embedding, vec![0.2]);
}

#[tokio::test]
async fn embeddings_response_reports_no_usage_at_all() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/text-embedding-004:batchEmbedContents"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [{"values": [0.1]}]
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = embeddings_request(EmbeddingsInput::Single("hello".to_string()));
    let resp = provider
        .embeddings(&req, "text-embedding-004", None)
        .await
        .expect("embeddings should succeed");

    assert_eq!(resp.usage, None);
}

#[tokio::test]
async fn embeddings_maps_a_non_success_status_to_a_classified_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/text-embedding-004:batchEmbedContents"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {"message": "bad request"}
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "test-key");
    let req = embeddings_request(EmbeddingsInput::Single("hello".to_string()));
    let err = provider
        .embeddings(&req, "text-embedding-004", None)
        .await
        .unwrap_err();

    assert!(matches!(err, ProviderError::InvalidRequest(_)));
}

#[tokio::test]
async fn embeddings_uses_the_byok_override_key_instead_of_the_configured_one() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/text-embedding-004:batchEmbedContents"))
        .and(query_param("key", "byok-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [{"values": [0.1]}]
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::new(server.uri(), "configured-key");
    let req = embeddings_request(EmbeddingsInput::Single("hello".to_string()));
    let resp = provider
        .embeddings(&req, "text-embedding-004", Some("byok-key"))
        .await
        .expect("embeddings should succeed");

    assert_eq!(resp.data.len(), 1);
}
