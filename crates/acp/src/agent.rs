//! The agent side of an ACP connection: capability negotiation, session
//! lifecycle, and dispatching `session/prompt` into a turn.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use serde_json::{json, Value};

use rp_router::Router;

use crate::jsonrpc::{Connection, Handler, RpcError};
use crate::schema::{
    error_code, method, AgentCapabilities, CancelNotification, ClientCapabilities,
    InitializeRequest, InitializeResponse, NewSessionRequest, NewSessionResponse,
    PromptCapabilities, PromptRequest, PromptResponse, PROTOCOL_VERSION,
};
use crate::session::SessionStore;
use crate::turn::{self, TurnConfig};

pub struct Agent {
    connection: Arc<Connection>,
    router: Arc<Router>,
    config: TurnConfig,
    sessions: SessionStore,
    /// Learned from `initialize` and then fixed for the connection's
    /// lifetime; it decides which tools the model is offered.
    client_capabilities: RwLock<ClientCapabilities>,
    initialized: AtomicBool,
}

impl Agent {
    pub fn new(connection: Arc<Connection>, router: Arc<Router>, config: TurnConfig) -> Self {
        Self {
            connection,
            router,
            config,
            sessions: SessionStore::new(),
            client_capabilities: RwLock::new(ClientCapabilities::default()),
            initialized: AtomicBool::new(false),
        }
    }

    fn initialize(&self, request: InitializeRequest) -> InitializeResponse {
        *self
            .client_capabilities
            .write()
            .expect("client capabilities poisoned") = request.client_capabilities;
        self.initialized.store(true, Ordering::SeqCst);

        InitializeResponse {
            // Negotiation is "answer with the highest version we both
            // support". We only speak v1, so that's what we answer with
            // regardless of what a newer client asked for; the client then
            // decides whether it can live with that.
            protocol_version: PROTOCOL_VERSION.min(request.protocol_version),
            agent_capabilities: AgentCapabilities {
                // No session persistence: sessions are process-scoped, so
                // there's nothing to replay after a restart.
                load_session: false,
                prompt_capabilities: PromptCapabilities {
                    image: true,
                    audio: true,
                    embedded_context: true,
                },
            },
            // The router's provider credentials come from the environment
            // it was started in, so there's nothing for the editor to
            // authenticate against.
            auth_methods: Vec::new(),
        }
    }

    fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, RpcError> {
        let cwd = PathBuf::from(&request.cwd);
        if !cwd.is_absolute() {
            return Err(RpcError::invalid_params(format!(
                "\"cwd\" must be an absolute path, got {:?}",
                request.cwd
            )));
        }
        if !request.mcp_servers.is_empty() {
            // We don't advertise mcpCapabilities, so this shouldn't
            // happen; carry on without them rather than failing the
            // session, but say so.
            tracing::warn!(
                count = request.mcp_servers.len(),
                "ignoring MCP servers: this agent doesn't advertise MCP support"
            );
        }

        let capabilities = self
            .client_capabilities
            .read()
            .expect("client capabilities poisoned")
            .clone();
        let session = self.sessions.create(cwd, capabilities);
        tracing::info!(session = %session.id, cwd = %request.cwd, "session started");

        Ok(NewSessionResponse {
            session_id: session.id.clone(),
        })
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, RpcError> {
        let Some(session) = self.sessions.get(&request.session_id) else {
            return Err(RpcError::new(
                error_code::INVALID_PARAMS,
                format!("unknown session {:?}", request.session_id),
            ));
        };

        let stop_reason = turn::run(
            &self.connection,
            &self.router,
            &session,
            &self.config,
            request.prompt,
        )
        .await?;

        tracing::info!(session = %session.id, ?stop_reason, "turn finished");
        Ok(PromptResponse { stop_reason })
    }

    fn cancel(&self, notification: CancelNotification) {
        match self.sessions.get(&notification.session_id) {
            // The turn observes this between steps and at every streamed
            // chunk, and answers the open `session/prompt` with
            // `stopReason: "cancelled"` -- which the protocol requires
            // rather than an error response.
            Some(session) => session.cancel(),
            None => tracing::debug!(
                session = %notification.session_id,
                "cancel for an unknown session, ignoring"
            ),
        }
    }
}

#[async_trait::async_trait]
impl Handler for Agent {
    async fn request(&self, method_name: &str, params: Value) -> Result<Value, RpcError> {
        // `initialize` is the only method valid before initialization;
        // everything else depends on the capabilities it establishes.
        if method_name != method::INITIALIZE && !self.initialized.load(Ordering::SeqCst) {
            return Err(RpcError::new(
                error_code::INVALID_REQUEST,
                format!("{method_name} was called before initialize"),
            ));
        }

        match method_name {
            method::INITIALIZE => {
                let request = parse(params)?;
                to_value(self.initialize(request))
            }
            method::SESSION_NEW => {
                let request = parse(params)?;
                to_value(self.new_session(request)?)
            }
            method::SESSION_PROMPT => {
                let request = parse(params)?;
                to_value(self.prompt(request).await?)
            }
            method::AUTHENTICATE => Err(RpcError::new(
                error_code::AUTH_REQUIRED,
                "this agent advertises no auth methods; provider credentials come from \
                 the environment it was launched in",
            )),
            method::SESSION_LOAD => Err(RpcError::new(
                error_code::METHOD_NOT_FOUND,
                "session/load is not supported (loadSession is not advertised); start a \
                 new session instead",
            )),
            other => Err(RpcError::method_not_found(other)),
        }
    }

    /// A prompt turn is the one method that must leave the dispatch loop:
    /// it runs for as long as the model and its tools take, and it calls
    /// back to the client throughout (`fs/*`, `terminal/*`,
    /// `session/request_permission`) while `session/cancel` may arrive.
    /// Everything else is short and answers in order.
    fn is_concurrent(&self, method_name: &str) -> bool {
        method_name == method::SESSION_PROMPT
    }

    async fn notification(&self, method_name: &str, params: Value) {
        match method_name {
            method::SESSION_CANCEL => match serde_json::from_value(params) {
                Ok(notification) => self.cancel(notification),
                Err(e) => tracing::warn!("ignoring malformed session/cancel: {e}"),
            },
            other => tracing::debug!("ignoring unknown notification {other}"),
        }
    }
}

fn parse<T: serde::de::DeserializeOwned>(params: Value) -> Result<T, RpcError> {
    // A method with no params is legal JSON-RPC; treat it as `{}` so
    // request types whose fields are all optional still parse.
    let params = if params.is_null() { json!({}) } else { params };
    serde_json::from_value(params).map_err(RpcError::invalid_params)
}

fn to_value<T: serde::Serialize>(value: T) -> Result<Value, RpcError> {
    serde_json::to_value(value).map_err(RpcError::internal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FileSystemCapabilities;

    fn initialize_request(read: bool, write: bool, terminal: bool) -> InitializeRequest {
        InitializeRequest {
            protocol_version: 1,
            client_capabilities: ClientCapabilities {
                fs: FileSystemCapabilities {
                    read_text_file: read,
                    write_text_file: write,
                },
                terminal,
            },
        }
    }

    /// An agent with no peer behind it -- enough to exercise everything
    /// that doesn't need the client to answer back.
    async fn agent() -> Agent {
        let (connection, _outbound) = Connection::detached();
        let config = TurnConfig {
            model: "openai/gpt-4o".into(),
            max_turn_requests: 4,
            system_prompt: "test".into(),
            max_tokens: None,
            temperature: None,
            client_name: None,
        };
        let router = Arc::new(Router::from_config(&rp_router::Config::default()).await);
        Agent::new(connection, router, config)
    }

    #[tokio::test]
    async fn initialize_answers_with_the_version_this_agent_speaks() {
        let agent = agent().await;
        let response = agent.initialize(initialize_request(true, true, true));
        assert_eq!(response.protocol_version, PROTOCOL_VERSION);
        assert!(!response.agent_capabilities.load_session);
        assert!(response.agent_capabilities.prompt_capabilities.image);
        assert!(response.auth_methods.is_empty());
    }

    #[tokio::test]
    async fn a_newer_clients_version_is_negotiated_down_to_ours() {
        let agent = agent().await;
        let mut request = initialize_request(true, true, true);
        request.protocol_version = 99;
        assert_eq!(agent.initialize(request).protocol_version, PROTOCOL_VERSION);
    }

    #[tokio::test]
    async fn an_older_clients_version_is_answered_in_kind() {
        let agent = agent().await;
        let mut request = initialize_request(true, true, true);
        request.protocol_version = 0;
        assert_eq!(agent.initialize(request).protocol_version, 0);
    }

    #[tokio::test]
    async fn methods_before_initialize_are_refused() {
        let agent = agent().await;
        let error = agent
            .request(
                method::SESSION_NEW,
                json!({"cwd": "/repo", "mcpServers": []}),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, error_code::INVALID_REQUEST);
        assert!(error.message.contains("before initialize"), "{error}");
    }

    #[tokio::test]
    async fn a_relative_cwd_is_rejected() {
        let agent = agent().await;
        agent.initialize(initialize_request(true, true, true));
        let error = agent
            .new_session(NewSessionRequest {
                cwd: "relative/path".into(),
                mcp_servers: Vec::new(),
            })
            .unwrap_err();
        assert_eq!(error.code, error_code::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn a_new_session_inherits_the_capabilities_from_initialize() {
        let agent = agent().await;
        agent.initialize(initialize_request(true, false, true));
        let response = agent
            .new_session(NewSessionRequest {
                cwd: "/repo".into(),
                mcp_servers: Vec::new(),
            })
            .unwrap();

        let session = agent.sessions.get(&response.session_id).unwrap();
        assert!(session.client_capabilities.fs.read_text_file);
        assert!(!session.client_capabilities.fs.write_text_file);
        assert!(session.client_capabilities.terminal);
    }

    #[tokio::test]
    async fn prompting_an_unknown_session_is_an_invalid_params_error() {
        let agent = agent().await;
        agent.initialize(initialize_request(true, true, true));
        let error = agent
            .prompt(PromptRequest {
                session_id: "sess_nope".into(),
                prompt: vec![crate::schema::ContentBlock::text("hi")],
            })
            .await
            .unwrap_err();
        assert_eq!(error.code, error_code::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn cancel_marks_the_session_and_tolerates_unknown_ids() {
        let agent = agent().await;
        agent.initialize(initialize_request(true, true, true));
        let response = agent
            .new_session(NewSessionRequest {
                cwd: "/repo".into(),
                mcp_servers: Vec::new(),
            })
            .unwrap();

        agent
            .notification(
                method::SESSION_CANCEL,
                json!({"sessionId": response.session_id}),
            )
            .await;
        assert!(agent
            .sessions
            .get(&response.session_id)
            .unwrap()
            .is_cancelled());

        // Must not panic.
        agent
            .notification(method::SESSION_CANCEL, json!({"sessionId": "sess_nope"}))
            .await;
    }

    #[tokio::test]
    async fn unsupported_methods_report_why_rather_than_just_failing() {
        let agent = agent().await;
        agent.initialize(initialize_request(true, true, true));

        let load = agent
            .request(method::SESSION_LOAD, json!({}))
            .await
            .unwrap_err();
        assert_eq!(load.code, error_code::METHOD_NOT_FOUND);

        let auth = agent
            .request(method::AUTHENTICATE, json!({"methodId": "x"}))
            .await
            .unwrap_err();
        assert_eq!(auth.code, error_code::AUTH_REQUIRED);

        let unknown = agent.request("session/fork", json!({})).await.unwrap_err();
        assert_eq!(unknown.code, error_code::METHOD_NOT_FOUND);
    }
}
