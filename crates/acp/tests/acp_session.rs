//! End-to-end ACP sessions: a scripted editor on one end of a pipe, the
//! real agent on the other, and a mock provider behind the router.
//!
//! These drive the actual wire protocol rather than calling the handler
//! directly, so they cover the parts unit tests can't -- that the agent
//! calls back to the client mid-turn, that tool results reach the next
//! model request, and that permission decisions are honoured.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream};
use wiremock::matchers::{method as http_method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use rp_acp::turn::TurnConfig;
use rp_acp::{serve, Agent};
use rp_router::{Config, Router};

const API_KEY_ENV: &str = "RP_ACP_TEST_KEY";

/// Serves a different scripted completion for each successive model
/// request, which is what makes a multi-step tool loop testable: the
/// first response asks for a tool, the next reacts to its result.
struct ScriptedModel {
    responses: Vec<String>,
    calls: AtomicUsize,
}

impl Respond for ScriptedModel {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let index = self
            .calls
            .fetch_add(1, Ordering::SeqCst)
            // A model that keeps being asked past the end of the script
            // just repeats its last answer.
            .min(self.responses.len() - 1);
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(self.responses[index].clone())
    }
}

/// Wraps chunks in the SSE framing the OpenAI-compatible adapter parses.
fn sse(chunks: &[Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn text_response(text: &str) -> String {
    sse(&[
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ])
}

fn tool_call_response(id: &str, name: &str, arguments: Value) -> String {
    sse(&[
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
            "index": 0,
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": arguments.to_string()},
        }]}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
    ])
}

async fn mock_provider(responses: Vec<String>) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(http_method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ScriptedModel {
            responses,
            calls: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;
    server
}

async fn router_for(server: &MockServer) -> Arc<Router> {
    std::env::set_var(API_KEY_ENV, "test-key");
    let config = Config::from_toml_str(&format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{API_KEY_ENV}"
        "#,
        server.uri()
    ))
    .expect("test config should parse");
    Arc::new(Router::from_config(&config).await)
}

fn turn_config(max_turn_requests: u32) -> TurnConfig {
    TurnConfig {
        model: "openai/gpt-4o".into(),
        max_turn_requests,
        system_prompt: "you are a test agent".into(),
        max_tokens: None,
        temperature: None,
        client_name: None,
    }
}

/// The editor side of the connection: sends requests, answers the agent's
/// callbacks from a script, and records every `session/update`.
struct TestClient {
    writer: tokio::io::WriteHalf<DuplexStream>,
    reader: tokio::io::Lines<BufReader<tokio::io::ReadHalf<DuplexStream>>>,
    next_id: i64,
    /// Every `session/update` notification seen so far.
    updates: Vec<Value>,
    /// Every request the *agent* made of us, in order.
    client_calls: Vec<Value>,
    /// Canned answers, keyed by method name.
    answers: Vec<(String, Value)>,
    /// When the agent calls this method, cancel the session before
    /// answering it -- how a user hitting "stop" mid-tool-call looks.
    cancel_when: Option<String>,
}

impl TestClient {
    fn new(stream: DuplexStream) -> Self {
        let (read_half, writer) = tokio::io::split(stream);
        Self {
            writer,
            reader: BufReader::new(read_half).lines(),
            next_id: 1,
            updates: Vec::new(),
            client_calls: Vec::new(),
            answers: Vec::new(),
            cancel_when: None,
        }
    }

    /// Registers the result to return for one of the agent's callbacks.
    fn answer(mut self, method: &str, result: Value) -> Self {
        self.answers.push((method.to_string(), result));
        self
    }

    fn cancel_when(mut self, method: &str) -> Self {
        self.cancel_when = Some(method.to_string());
        self
    }

    async fn send(&mut self, message: Value) {
        self.writer
            .write_all(format!("{message}\n").as_bytes())
            .await
            .expect("write to the agent");
    }

    /// Sends a request and pumps the connection until its response
    /// arrives, servicing whatever the agent asks for along the way.
    async fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await;

        loop {
            let line = tokio::time::timeout(Duration::from_secs(10), self.reader.next_line())
                .await
                .expect("the agent should answer within the timeout")
                .expect("read from the agent")
                .expect("the agent should not hang up mid-request");
            let message: Value = serde_json::from_str(&line).expect("agent sent valid JSON");

            match message.get("method").and_then(Value::as_str) {
                Some("session/update") => {
                    self.updates.push(message["params"]["update"].clone());
                }
                Some(agent_method) => {
                    let agent_method = agent_method.to_string();
                    self.client_calls.push(message.clone());

                    if self.cancel_when.as_deref() == Some(agent_method.as_str()) {
                        // Every client method carries the session id.
                        let session_id = message["params"]["sessionId"].clone();
                        self.send(json!({
                            "jsonrpc": "2.0",
                            "method": "session/cancel",
                            "params": {"sessionId": session_id},
                        }))
                        .await;
                    }

                    // Requests need an answer; notifications don't.
                    if let Some(request_id) = message.get("id").cloned() {
                        let result = self
                            .answers
                            .iter()
                            .find(|(name, _)| *name == agent_method)
                            .map(|(_, result)| result.clone())
                            .unwrap_or_else(|| {
                                panic!("test has no scripted answer for {agent_method}")
                            });
                        self.send(json!({"jsonrpc": "2.0", "id": request_id, "result": result}))
                            .await;
                    }
                }
                None if message.get("id") == Some(&json!(id)) => {
                    if let Some(error) = message.get("error") {
                        panic!("request {method} failed: {error}");
                    }
                    return message["result"].clone();
                }
                None => {}
            }
        }
    }

    fn updates_of(&self, kind: &str) -> Vec<&Value> {
        self.updates
            .iter()
            .filter(|update| update["sessionUpdate"] == json!(kind))
            .collect()
    }

    fn calls_to(&self, method: &str) -> Vec<&Value> {
        self.client_calls
            .iter()
            .filter(|call| call["method"] == json!(method))
            .collect()
    }

    /// Concatenates the streamed assistant text, as a user would see it.
    fn assistant_text(&self) -> String {
        self.updates_of("agent_message_chunk")
            .iter()
            .filter_map(|update| update["content"]["text"].as_str())
            .collect()
    }
}

/// Stands the agent up on one end of a pipe and hands back the other end.
fn connect(router: Arc<Router>, config: TurnConfig) -> (TestClient, tokio::task::JoinHandle<()>) {
    let (client_side, agent_side) = tokio::io::duplex(256 * 1024);
    let (agent_read, agent_write) = tokio::io::split(agent_side);

    let served = tokio::spawn(async move {
        let _ = serve(agent_read, agent_write, |connection| {
            Arc::new(Agent::new(connection, router, config))
        })
        .await;
    });

    (TestClient::new(client_side), served)
}

fn initialize_params(read: bool, write: bool, terminal: bool) -> Value {
    json!({
        "protocolVersion": 1,
        "clientCapabilities": {
            "fs": {"readTextFile": read, "writeTextFile": write},
            "terminal": terminal,
        },
    })
}

/// The tool names offered to the model on the nth model request.
async fn tools_offered(server: &MockServer, index: usize) -> Vec<String> {
    let requests = server.received_requests().await.expect("recorded requests");
    let body: Value = serde_json::from_slice(&requests[index].body).expect("valid request body");
    body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| tool["function"]["name"].as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn a_plain_answer_streams_back_and_ends_the_turn() {
    let server = mock_provider(vec![text_response("hello from the model")]).await;
    let (mut client, _served) = connect(router_for(&server).await, turn_config(8));

    let initialized = client
        .request("initialize", initialize_params(true, true, true))
        .await;
    assert_eq!(initialized["protocolVersion"], json!(1));

    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;
    let session_id = session["sessionId"].as_str().unwrap().to_string();

    let response = client
        .request(
            "session/prompt",
            json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}),
        )
        .await;

    assert_eq!(response["stopReason"], json!("end_turn"));
    assert_eq!(client.assistant_text(), "hello from the model");
}

#[tokio::test]
async fn the_model_reads_a_file_through_the_client_and_answers_from_it() {
    let server = mock_provider(vec![
        tool_call_response("call_1", "read_file", json!({"path": "src/main.rs"})),
        text_response("it prints nothing"),
    ])
    .await;
    let (client, _served) = connect(router_for(&server).await, turn_config(8));
    let mut client = client.answer("fs/read_text_file", json!({"content": "fn main() {}"}));

    client
        .request("initialize", initialize_params(true, true, true))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;
    let session_id = session["sessionId"].as_str().unwrap().to_string();

    let response = client
        .request(
            "session/prompt",
            json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "what does main do?"}]}),
        )
        .await;

    assert_eq!(response["stopReason"], json!("end_turn"));

    // The read went back to the client as an absolute path under cwd.
    let reads = client.calls_to("fs/read_text_file");
    assert_eq!(reads.len(), 1);
    assert_eq!(reads[0]["params"]["path"], json!("/repo/src/main.rs"));
    assert_eq!(reads[0]["params"]["sessionId"], json!(session_id));

    // The call was reported, then closed out as completed.
    let started = client.updates_of("tool_call");
    assert_eq!(started.len(), 1);
    assert_eq!(started[0]["title"], json!("Read src/main.rs"));
    assert_eq!(started[0]["kind"], json!("read"));
    assert_eq!(
        started[0]["locations"][0]["path"],
        json!("/repo/src/main.rs")
    );

    let finished = client.updates_of("tool_call_update");
    assert_eq!(finished.last().unwrap()["status"], json!("completed"));

    // And the file's contents reached the model's second request.
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let second: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let messages = second["messages"].as_array().unwrap();
    let tool_result = messages
        .iter()
        .find(|message| message["role"] == json!("tool"))
        .expect("the tool result should be fed back to the model");
    assert_eq!(tool_result["content"], json!("fn main() {}"));
    assert_eq!(tool_result["tool_call_id"], json!("call_1"));

    assert_eq!(client.assistant_text(), "it prints nothing");
}

#[tokio::test]
async fn a_write_is_applied_only_after_the_user_allows_it() {
    let server = mock_provider(vec![
        tool_call_response(
            "call_1",
            "write_file",
            json!({"path": "notes.md", "content": "# notes"}),
        ),
        text_response("written"),
    ])
    .await;
    let (client, _served) = connect(router_for(&server).await, turn_config(8));

    let mut client = client
        .answer("fs/read_text_file", json!({"content": "# old notes"}))
        .answer(
            "session/request_permission",
            json!({"outcome": {"outcome": "selected", "optionId": "allow_once"}}),
        )
        .answer("fs/write_text_file", json!({}));

    client
        .request("initialize", initialize_params(true, true, true))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;
    let session_id = session["sessionId"].as_str().unwrap().to_string();

    let response = client
        .request(
            "session/prompt",
            json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "write notes"}]}),
        )
        .await;
    assert_eq!(response["stopReason"], json!("end_turn"));

    // Permission was asked for before the write, and showed the diff.
    let prompts = client.calls_to("session/request_permission");
    assert_eq!(prompts.len(), 1);
    let tool_call = &prompts[0]["params"]["toolCall"];
    assert_eq!(tool_call["kind"], json!("edit"));
    assert_eq!(tool_call["status"], json!("pending"));
    assert_eq!(tool_call["content"][0]["type"], json!("diff"));
    assert_eq!(tool_call["content"][0]["oldText"], json!("# old notes"));
    assert_eq!(tool_call["content"][0]["newText"], json!("# notes"));

    let writes = client.calls_to("fs/write_text_file");
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0]["params"]["path"], json!("/repo/notes.md"));
    assert_eq!(writes[0]["params"]["content"], json!("# notes"));
}

#[tokio::test]
async fn a_rejected_write_never_reaches_the_client_and_the_model_is_told() {
    let server = mock_provider(vec![
        tool_call_response(
            "call_1",
            "write_file",
            json!({"path": "notes.md", "content": "# notes"}),
        ),
        text_response("understood, leaving it alone"),
    ])
    .await;
    let (client, _served) = connect(router_for(&server).await, turn_config(8));

    let mut client = client
        .answer("fs/read_text_file", json!({"content": "# old notes"}))
        .answer(
            "session/request_permission",
            json!({"outcome": {"outcome": "selected", "optionId": "reject_once"}}),
        );

    client
        .request("initialize", initialize_params(true, true, true))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;
    let session_id = session["sessionId"].as_str().unwrap().to_string();

    let response = client
        .request(
            "session/prompt",
            json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "write notes"}]}),
        )
        .await;

    // The turn continues -- a rejection is something the model reacts to.
    assert_eq!(response["stopReason"], json!("end_turn"));
    assert!(
        client.calls_to("fs/write_text_file").is_empty(),
        "a rejected write must never reach the client"
    );
    assert_eq!(
        client.updates_of("tool_call_update").last().unwrap()["status"],
        json!("failed")
    );

    let requests = server.received_requests().await.unwrap();
    let second: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let tool_result = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == json!("tool"))
        .expect("the rejection should be fed back to the model");
    assert!(
        tool_result["content"]
            .as_str()
            .unwrap()
            .contains("rejected"),
        "{tool_result}"
    );
}

#[tokio::test]
async fn the_model_is_only_offered_tools_the_client_can_service() {
    let server = mock_provider(vec![text_response("ok")]).await;
    let (mut client, _served) = connect(router_for(&server).await, turn_config(8));

    // A client that can read but not write, with no terminal.
    client
        .request("initialize", initialize_params(true, false, false))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;
    client
        .request(
            "session/prompt",
            json!({
                "sessionId": session["sessionId"],
                "prompt": [{"type": "text", "text": "hi"}],
            }),
        )
        .await;

    let offered = tools_offered(&server, 0).await;
    assert!(offered.contains(&"read_file".to_string()));
    assert!(offered.contains(&"update_plan".to_string()));
    assert!(!offered.contains(&"write_file".to_string()));
    assert!(!offered.contains(&"edit_file".to_string()));
    assert!(!offered.contains(&"execute_command".to_string()));
}

#[tokio::test]
async fn a_model_that_never_stops_calling_tools_is_cut_off() {
    // The script's single response is a tool call, and `ScriptedModel`
    // repeats its last answer -- so this model never finishes on its own.
    let server = mock_provider(vec![tool_call_response(
        "call_1",
        "read_file",
        json!({"path": "loop.rs"}),
    )])
    .await;
    let (client, _served) = connect(router_for(&server).await, turn_config(3));

    let mut client = client.answer("fs/read_text_file", json!({"content": "// again"}));

    client
        .request("initialize", initialize_params(true, true, true))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;

    let response = client
        .request(
            "session/prompt",
            json!({
                "sessionId": session["sessionId"],
                "prompt": [{"type": "text", "text": "go"}],
            }),
        )
        .await;

    assert_eq!(response["stopReason"], json!("max_turn_requests"));
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        3,
        "the cap should bound model requests, not just tool calls"
    );
}

#[tokio::test]
async fn the_plan_tool_surfaces_a_plan_to_the_client() {
    let server = mock_provider(vec![
        tool_call_response(
            "call_1",
            "update_plan",
            json!({"entries": [
                {"content": "read the code", "priority": "high", "status": "in_progress"},
                {"content": "fix the bug", "priority": "medium", "status": "pending"},
            ]}),
        ),
        text_response("starting"),
    ])
    .await;
    let (mut client, _served) = connect(router_for(&server).await, turn_config(8));

    client
        .request("initialize", initialize_params(true, true, true))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;
    client
        .request(
            "session/prompt",
            json!({
                "sessionId": session["sessionId"],
                "prompt": [{"type": "text", "text": "fix it"}],
            }),
        )
        .await;

    let plans = client.updates_of("plan");
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0]["entries"].as_array().unwrap().len(), 2);
    assert_eq!(plans[0]["entries"][0]["status"], json!("in_progress"));

    // A plan update needs no permission -- it changes nothing.
    assert!(client.calls_to("session/request_permission").is_empty());
}

#[tokio::test]
async fn a_command_runs_through_the_clients_terminal_and_is_released() {
    let server = mock_provider(vec![
        tool_call_response(
            "call_1",
            "execute_command",
            json!({"command": "cargo", "args": ["test"]}),
        ),
        text_response("tests pass"),
    ])
    .await;
    let (client, _served) = connect(router_for(&server).await, turn_config(8));

    let mut client = client
        .answer(
            "session/request_permission",
            json!({"outcome": {"outcome": "selected", "optionId": "allow_once"}}),
        )
        .answer("terminal/create", json!({"terminalId": "term_1"}))
        .answer("terminal/wait_for_exit", json!({"exitCode": 0}))
        .answer(
            "terminal/output",
            json!({"output": "test result: ok. 3 passed", "truncated": false}),
        )
        .answer("terminal/release", json!({}));

    client
        .request("initialize", initialize_params(true, true, true))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;
    let response = client
        .request(
            "session/prompt",
            json!({
                "sessionId": session["sessionId"],
                "prompt": [{"type": "text", "text": "run the tests"}],
            }),
        )
        .await;
    assert_eq!(response["stopReason"], json!("end_turn"));

    let created = client.calls_to("terminal/create");
    assert_eq!(created.len(), 1);
    assert_eq!(created[0]["params"]["command"], json!("cargo"));
    assert_eq!(created[0]["params"]["args"], json!(["test"]));
    assert_eq!(created[0]["params"]["cwd"], json!("/repo"));

    // The terminal is embedded live, then released once it's done.
    let embedded = client
        .updates_of("tool_call_update")
        .into_iter()
        .any(|update| update["content"][0]["terminalId"] == json!("term_1"));
    assert!(embedded, "the client should get a live terminal to render");
    assert_eq!(client.calls_to("terminal/release").len(), 1);

    // The model sees the exit status and the output.
    let requests = server.received_requests().await.unwrap();
    let second: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let tool_result = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == json!("tool"))
        .expect("the command result should reach the model");
    let content = tool_result["content"].as_str().unwrap();
    assert!(content.contains("exit code 0"), "{content}");
    assert!(content.contains("3 passed"), "{content}");
}

#[tokio::test]
async fn cancelling_mid_turn_ends_the_prompt_with_cancelled() {
    // The model would keep going forever; the cancel is what stops it.
    let server = mock_provider(vec![tool_call_response(
        "call_1",
        "read_file",
        json!({"path": "loop.rs"}),
    )])
    .await;
    let (client, _served) = connect(router_for(&server).await, turn_config(50));

    let mut client = client
        .answer("fs/read_text_file", json!({"content": "// again"}))
        .cancel_when("fs/read_text_file");

    client
        .request("initialize", initialize_params(true, true, true))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;

    let response = client
        .request(
            "session/prompt",
            json!({
                "sessionId": session["sessionId"],
                "prompt": [{"type": "text", "text": "go"}],
            }),
        )
        .await;

    // Cancellation is a stop reason, not an error response.
    assert_eq!(response["stopReason"], json!("cancelled"));
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        1,
        "the cancel should land before a second model request goes out"
    );
}

#[tokio::test]
async fn a_blocking_guardrail_ends_the_turn_as_a_refusal() {
    let server = mock_provider(vec![text_response("should never be reached")]).await;
    std::env::set_var(API_KEY_ENV, "test-key");
    let config = Config::from_toml_str(&format!(
        r#"
        [providers.openai]
        kind = "openai"
        base_url = "{}"
        api_key_env = "{API_KEY_ENV}"

        [[guardrails]]
        name = "no-secrets"
        pattern = "sk-live-[a-z0-9]+"
        action = "block"
        "#,
        server.uri()
    ))
    .expect("test config should parse");
    let router = Arc::new(Router::from_config(&config).await);
    let (mut client, _served) = connect(router, turn_config(8));

    client
        .request("initialize", initialize_params(true, true, true))
        .await;
    let session = client
        .request("session/new", json!({"cwd": "/repo", "mcpServers": []}))
        .await;

    let response = client
        .request(
            "session/prompt",
            json!({
                "sessionId": session["sessionId"],
                "prompt": [{"type": "text", "text": "here is my key sk-live-abc123"}],
            }),
        )
        .await;

    assert_eq!(response["stopReason"], json!("refusal"));
    // The user is told why, and no provider was ever called.
    assert!(
        client.assistant_text().contains("no-secrets"),
        "{}",
        client.assistant_text()
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn a_client_that_does_not_wait_for_initialize_is_still_answered_in_order() {
    // Nothing stops an editor from writing `initialize` and `session/new`
    // back to back. If the agent answered them concurrently, the session
    // could be created before initialization finished -- and get rejected
    // for it -- purely on scheduling luck.
    let server = mock_provider(vec![text_response("ok")]).await;
    let (mut client, _served) = connect(router_for(&server).await, turn_config(8));

    client
        .send(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": initialize_params(true, true, true),
        }))
        .await;
    client
        .send(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {"cwd": "/repo", "mcpServers": []},
        }))
        .await;

    let mut answers = Vec::new();
    while answers.len() < 2 {
        let line = tokio::time::timeout(Duration::from_secs(10), client.reader.next_line())
            .await
            .expect("the agent should answer")
            .unwrap()
            .unwrap();
        let message: Value = serde_json::from_str(&line).unwrap();
        if message.get("id").is_some() {
            answers.push(message);
        }
    }

    assert_eq!(answers[0]["id"], json!(1));
    assert_eq!(answers[0]["result"]["protocolVersion"], json!(1));
    assert_eq!(answers[1]["id"], json!(2));
    assert!(
        answers[1]["result"]["sessionId"].is_string(),
        "session/new should succeed, not race initialize: {}",
        answers[1]
    );
}

#[tokio::test]
async fn prompting_before_initialize_is_refused() {
    let server = mock_provider(vec![text_response("unreachable")]).await;
    let (mut client, _served) = connect(router_for(&server).await, turn_config(8));

    client
        .send(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": {"cwd": "/repo", "mcpServers": []},
        }))
        .await;

    let line = tokio::time::timeout(Duration::from_secs(10), client.reader.next_line())
        .await
        .expect("the agent should answer")
        .unwrap()
        .unwrap();
    let message: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(message["error"]["code"], json!(-32600));
}
