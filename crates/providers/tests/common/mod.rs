use rp_core::{ChatMessage, ChatRequest, Tool};

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
