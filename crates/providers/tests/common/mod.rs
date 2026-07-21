use rp_core::{
    ChatMessage, ChatRequest, ContentPart, FileData, ImageUrl, InputAudio, JsonSchemaFormat,
    MessageContent, ReasoningConfig, ResponseFormat, Role, Tool,
};

/// A minimal single-user-turn request, for tests that don't care about the
/// rest of the request shape.
pub fn simple_request(model: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        models: None,
        messages: vec![ChatMessage::user("hi")],
        temperature: None,
        top_p: None,
        max_tokens: Some(64),
        stop: None,
        stream: None,
        user: None,
        tools: None,
        tool_choice: None,
        provider: None,
        response_format: None,
        reasoning: None,
        top_k: None,
        min_p: None,
        top_a: None,
        frequency_penalty: None,
        presence_penalty: None,
        repetition_penalty: None,
        logit_bias: None,
        seed: None,
        transforms: None,
    }
}

/// Same as [`simple_request`], but the user turn's content is a
/// multimodal parts array (text + a `data:` image), for tests exercising
/// image content translation.
pub fn request_with_image(model: &str) -> ChatRequest {
    let mut req = simple_request(model);
    req.messages = vec![ChatMessage {
        role: Role::User,
        content: Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "what's in this image?".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aGVsbG8=".to_string(),
                    detail: None,
                },
            },
        ])),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning: None,
        cache_control: None,
    }];
    req
}

/// Same as [`simple_request`], but the user turn's content is a
/// multimodal parts array (text + base64 audio), for tests exercising
/// audio content translation.
pub fn request_with_audio(model: &str) -> ChatRequest {
    let mut req = simple_request(model);
    req.messages = vec![ChatMessage {
        role: Role::User,
        content: Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "what's said in this clip?".to_string(),
            },
            ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: "aGVsbG8=".to_string(),
                    format: "wav".to_string(),
                },
            },
        ])),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning: None,
        cache_control: None,
    }];
    req
}

/// Same as [`simple_request`], but the user turn's content is a
/// multimodal parts array (text + a base64 `data:` PDF), for tests
/// exercising file content translation.
pub fn request_with_file(model: &str) -> ChatRequest {
    let mut req = simple_request(model);
    req.messages = vec![ChatMessage {
        role: Role::User,
        content: Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "summarize this document".to_string(),
            },
            ContentPart::File {
                file: FileData {
                    file_data: "data:application/pdf;base64,aGVsbG8=".to_string(),
                    filename: Some("doc.pdf".to_string()),
                },
            },
        ])),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning: None,
        cache_control: None,
    }];
    req
}

/// Same as [`request_with_file`], but the file is a remote URL rather than
/// inline base64 data, for tests exercising the other half of each
/// adapter's data-URI-vs-URL translation split.
pub fn request_with_remote_file(model: &str) -> ChatRequest {
    let mut req = simple_request(model);
    req.messages = vec![ChatMessage {
        role: Role::User,
        content: Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "summarize this document".to_string(),
            },
            ContentPart::File {
                file: FileData {
                    file_data: "https://example.com/doc.pdf".to_string(),
                    filename: None,
                },
            },
        ])),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning: None,
        cache_control: None,
    }];
    req
}

/// Same as [`simple_request`], but with a schema-constrained
/// `response_format`, for tests exercising structured-output translation.
pub fn request_with_json_schema(model: &str) -> ChatRequest {
    let mut req = simple_request(model);
    req.response_format = Some(ResponseFormat::JsonSchema {
        json_schema: JsonSchemaFormat {
            name: "weather_report".to_string(),
            description: Some("A weather report for one city".to_string()),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "temperature_f": {"type": "number"},
                },
                "required": ["city", "temperature_f"],
            }),
            strict: Some(true),
        },
    });
    req
}

/// Same as [`simple_request`], but with a schema-less `response_format`,
/// for tests exercising loose JSON-mode translation.
pub fn request_with_json_object(model: &str) -> ChatRequest {
    let mut req = simple_request(model);
    req.response_format = Some(ResponseFormat::JsonObject);
    req
}

/// Same as [`simple_request`], but with a `reasoning` config attached, for
/// tests exercising thinking-token request/response translation.
pub fn request_with_reasoning(model: &str, reasoning: ReasoningConfig) -> ChatRequest {
    let mut req = simple_request(model);
    req.reasoning = Some(reasoning);
    req
}

/// Same as [`simple_request`], but with every sampling-parameter field
/// set, for tests exercising per-provider passthrough/translation/silent-
/// ignore behavior.
pub fn request_with_sampling_params(model: &str) -> ChatRequest {
    let mut req = simple_request(model);
    req.top_k = Some(40);
    req.min_p = Some(0.05);
    req.top_a = Some(0.2);
    req.frequency_penalty = Some(0.3);
    req.presence_penalty = Some(0.4);
    req.repetition_penalty = Some(1.1);
    req.logit_bias = Some(std::collections::HashMap::from([(
        "1234".to_string(),
        -100.0,
    )]));
    req.seed = Some(42);
    req
}

/// Same as [`simple_request`], but with a tool attached, for tests
/// exercising tool-call request/response translation.
pub fn request_with_tool(model: &str) -> ChatRequest {
    let mut req = simple_request(model);
    req.tools = Some(vec![Tool {
        kind: "function".to_string(),
        function: rp_core::FunctionDef {
            name: "get_weather".to_string(),
            description: Some("Get the current weather for a city".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            })),
        },
    }]);
    req
}
