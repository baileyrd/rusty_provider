use rp_core::{
    ChatMessage, ChatRequest, ContentPart, ImageUrl, InputAudio, MessageContent, Role, Tool,
};

/// A minimal single-user-turn request, for tests that don't care about the
/// rest of the request shape.
pub fn simple_request(model: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
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
    }];
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
