//! JSON-RPC 2.0 over a newline-delimited byte stream -- the transport ACP
//! uses when an editor spawns an agent as a subprocess and talks to it on
//! stdin/stdout.
//!
//! The structural decision worth calling out is how incoming messages are
//! dispatched, because ACP needs two opposite things at once.
//!
//! Ordering: a client may send `initialize` and `session/new` back to back
//! without waiting, and the second must not be answered first -- so by
//! default the read loop *awaits* each handler before reading the next
//! message.
//!
//! Concurrency: `session/prompt` stays open for a whole turn, during which
//! the agent issues its own requests back to the client
//! (`fs/read_text_file`, `session/request_permission`) and the client may
//! send `session/cancel`. Awaiting that inline would deadlock -- the
//! response it waits for can only arrive through the loop it's blocking.
//!
//! So the handler declares which methods leave the loop, via
//! [`Handler::is_concurrent`]. Those run as tracked tasks; everything else
//! stays ordered.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};

use crate::schema::error_code;

/// A JSON-RPC error object, as both sent and received.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    pub fn method_not_found(method: &str) -> Self {
        Self::new(
            error_code::METHOD_NOT_FOUND,
            format!("method not found: {method}"),
        )
    }

    pub fn invalid_params(detail: impl std::fmt::Display) -> Self {
        Self::new(
            error_code::INVALID_PARAMS,
            format!("invalid params: {detail}"),
        )
    }

    pub fn internal(detail: impl std::fmt::Display) -> Self {
        Self::new(error_code::INTERNAL_ERROR, detail.to_string())
    }

    fn to_value(&self) -> Value {
        let mut obj = json!({"code": self.code, "message": self.message});
        if let Some(data) = &self.data {
            obj["data"] = data.clone();
        }
        obj
    }

    fn from_value(value: &Value) -> Self {
        Self {
            code: value
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or(error_code::INTERNAL_ERROR as i64) as i32,
            message: value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
                .to_string(),
            data: value.get("data").cloned(),
        }
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code {})", self.message, self.code)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// The peer went away -- for a subprocess agent, the editor exited.
    #[error("connection closed")]
    Closed,
    #[error("{0}")]
    Rpc(RpcError),
    #[error("malformed message: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Handles messages arriving *from* the peer.
#[async_trait::async_trait]
pub trait Handler: Send + Sync + 'static {
    async fn request(&self, method: &str, params: Value) -> Result<Value, RpcError>;
    async fn notification(&self, method: &str, params: Value);

    /// Whether this method should be released from the dispatch loop to
    /// run concurrently with later messages, instead of being answered
    /// before the next message is read.
    ///
    /// This must be `true` for any method whose handler calls *back* to
    /// the peer or that runs for a long time -- `session/prompt` is both.
    /// Handling such a method inline would deadlock: its response can
    /// never arrive, because arriving is what the loop is blocked on.
    ///
    /// It must be `false` for everything else, so ordering is preserved:
    /// a client that sends `initialize` and `session/new` back to back
    /// without waiting must not have the second answered first.
    fn is_concurrent(&self, method: &str) -> bool {
        let _ = method;
        false
    }
}

/// The outgoing half of a connection: send requests and notifications to
/// the peer, and correlate responses back to their callers.
pub struct Connection {
    outgoing: mpsc::UnboundedSender<String>,
    next_id: AtomicI64,
    pending: Mutex<HashMap<i64, oneshot::Sender<Result<Value, RpcError>>>>,
    closed: AtomicBool,
}

impl Connection {
    fn new(outgoing: mpsc::UnboundedSender<String>) -> Self {
        Self {
            outgoing,
            next_id: AtomicI64::new(1),
            pending: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
        }
    }

    /// A connection with no peer on the other end: outbound messages land
    /// in the returned receiver instead of on a socket. Lets tests assert
    /// on exactly what the agent emits without standing up a transport.
    /// Outbound *requests* never complete here, so only use it for code
    /// paths that notify.
    pub fn detached() -> (Arc<Self>, mpsc::UnboundedReceiver<String>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Arc::new(Self::new(tx)), rx)
    }

    /// Sends a request and waits for the peer's response.
    pub async fn request<P, R>(&self, method: &str, params: &P) -> Result<R, TransportError>
    where
        P: Serialize,
        R: DeserializeOwned,
    {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .expect("pending map poisoned")
            .insert(id, tx);

        // Checked *after* inserting: `close` drains whatever is in the map,
        // so a request that raced it would otherwise wait on a response
        // that can never come.
        if self.closed.load(Ordering::SeqCst) {
            self.pending
                .lock()
                .expect("pending map poisoned")
                .remove(&id);
            return Err(TransportError::Closed);
        }

        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": serde_json::to_value(params)?,
        });
        if self.send(message).is_err() {
            self.pending
                .lock()
                .expect("pending map poisoned")
                .remove(&id);
            return Err(TransportError::Closed);
        }

        match rx.await {
            Ok(Ok(result)) => Ok(serde_json::from_value(result)?),
            Ok(Err(e)) => Err(TransportError::Rpc(e)),
            // The read loop dropped the sender, which only happens when
            // the connection itself is gone.
            Err(_) => Err(TransportError::Closed),
        }
    }

    /// Sends a notification -- fire-and-forget, no response expected.
    pub fn notify<P: Serialize>(&self, method: &str, params: &P) -> Result<(), TransportError> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": serde_json::to_value(params)?,
        });
        self.send(message).map_err(|_| TransportError::Closed)
    }

    fn send(&self, message: Value) -> Result<(), mpsc::error::SendError<String>> {
        self.outgoing.send(message.to_string())
    }

    fn complete(&self, id: i64, result: Result<Value, RpcError>) {
        let waiter = self
            .pending
            .lock()
            .expect("pending map poisoned")
            .remove(&id);
        match waiter {
            Some(tx) => {
                // The receiver is gone if the caller was cancelled while
                // waiting; that's expected, not an error.
                let _ = tx.send(result);
            }
            None => tracing::debug!(id, "response for an unknown request id, ignoring"),
        }
    }

    /// Marks the peer gone and fails every request still waiting on it.
    /// Without this, a turn blocked on `fs/read_text_file` when the editor
    /// exits would wait forever on a response nobody will send.
    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let waiting = std::mem::take(&mut *self.pending.lock().expect("pending map poisoned"));
        for (_, tx) in waiting {
            let _ = tx.send(Err(RpcError::new(
                error_code::INTERNAL_ERROR,
                "connection closed",
            )));
        }
    }
}

/// Runs the connection until the peer closes it (EOF on `reader`).
///
/// `build_handler` receives the `Connection` so the handler can call back
/// to the peer; the two are mutually referential, which is why the
/// handler is constructed here rather than passed in.
pub async fn serve<R, W, H, F>(reader: R, writer: W, build_handler: F) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
    H: Handler,
    F: FnOnce(Arc<Connection>) -> Arc<H>,
{
    let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel::<String>();
    let connection = Arc::new(Connection::new(outgoing_tx));
    let handler = build_handler(connection.clone());

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let writer_task = tokio::spawn(async move {
        let mut writer = writer;
        tokio::pin!(shutdown_rx);
        loop {
            let line = tokio::select! {
                line = outgoing_rx.recv() => match line {
                    Some(line) => line,
                    None => break,
                },
                _ = &mut shutdown_rx => {
                    // Shutdown only fires once every handler has finished,
                    // so whatever is still queued is a finished reply that
                    // hasn't been written yet. Drain it before hanging up.
                    while let Ok(line) = outgoing_rx.try_recv() {
                        if write_line(&mut writer, &line).await.is_err() {
                            break;
                        }
                    }
                    break;
                }
            };
            if write_line(&mut writer, &line).await.is_err() {
                break;
            }
        }
    });

    // Handlers run as their own tasks (see the module docs), so the read
    // loop tracks them here and the shutdown path below can wait for
    // their replies instead of cutting them off.
    let mut in_flight = tokio::task::JoinSet::new();

    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(e) => {
                tracing::warn!("discarding unparseable message: {e}");
                let _ = connection.send(json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": RpcError::new(error_code::PARSE_ERROR, e.to_string()).to_value(),
                }));
                continue;
            }
        };
        dispatch(&connection, &handler, &mut in_flight, message).await;
    }

    // The peer is gone. Fail anything waiting on it first, so a handler
    // blocked on a client callback unwinds instead of hanging the join
    // below; then let the handlers finish and their replies flush. A
    // client that pipelines requests and closes the pipe immediately is
    // ordinary usage, and it must still get its answers.
    connection.close();
    while in_flight.join_next().await.is_some() {}

    // Signalled explicitly rather than by dropping the sender: the
    // handler holds its own `Arc<Connection>`, so ours isn't the last.
    let _ = shutdown_tx.send(());
    drop(connection);
    let _ = writer_task.await;
    Ok(())
}

async fn write_line<W: AsyncWrite + Unpin>(writer: &mut W, line: &str) -> std::io::Result<()> {
    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

async fn dispatch<H: Handler>(
    connection: &Arc<Connection>,
    handler: &Arc<H>,
    in_flight: &mut tokio::task::JoinSet<()>,
    message: Value,
) {
    let id = message.get("id").cloned().filter(|id| !id.is_null());

    if let Some(method) = message
        .get("method")
        .and_then(Value::as_str)
        .map(str::to_string)
    {
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        match id {
            Some(id) => {
                let concurrent = handler.is_concurrent(&method);
                let handler = handler.clone();
                let connection = connection.clone();
                let answer = async move {
                    let response = match handler.request(&method, params).await {
                        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                        Err(e) => {
                            json!({"jsonrpc": "2.0", "id": id, "error": e.to_value()})
                        }
                    };
                    let _ = connection.send(response);
                };
                if concurrent {
                    in_flight.spawn(answer);
                } else {
                    answer.await;
                }
            }
            // Notifications are answer-less and cheap by construction
            // (`session/cancel` flips a flag), so they stay in order too.
            None => handler.notification(&method, params).await,
        }
        return;
    }

    // No method: this is a response to something we sent. Ids we issue are
    // always integers, so anything else is a response we can't match.
    match id.as_ref().and_then(Value::as_i64) {
        Some(id) => {
            let result = match message.get("error") {
                Some(error) => Err(RpcError::from_value(error)),
                None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
            };
            connection.complete(id, result);
        }
        None => tracing::warn!("ignoring message with neither a method nor a usable id"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use tokio::io::duplex;

    /// Echoes requests back and counts notifications, so tests can assert
    /// on what the transport delivered.
    struct TestHandler {
        connection: Arc<Connection>,
        notifications: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Handler for TestHandler {
        async fn request(&self, method: &str, params: Value) -> Result<Value, RpcError> {
            match method {
                "echo" => Ok(params),
                // Calls back to the peer *while* handling an inbound
                // request -- the interleaving `session/prompt` needs.
                "reenter" => {
                    let back: Value = self
                        .connection
                        .request("ping", &json!({}))
                        .await
                        .map_err(RpcError::internal)?;
                    Ok(back)
                }
                other => Err(RpcError::method_not_found(other)),
            }
        }

        async fn notification(&self, _method: &str, _params: Value) {
            self.notifications.fetch_add(1, Ordering::Relaxed);
        }

        fn is_concurrent(&self, method: &str) -> bool {
            // "reenter" calls back to the peer, so it must not hold the
            // loop -- the same reason `session/prompt` doesn't.
            method == "reenter"
        }
    }

    /// Drives one side of a duplex pipe as a scripted peer.
    async fn run_peer(lines: Vec<Value>) -> Vec<Value> {
        let (mine, theirs) = duplex(64 * 1024);
        let (their_read, their_write) = tokio::io::split(theirs);

        let served = tokio::spawn(async move {
            serve(their_read, their_write, |connection| {
                Arc::new(TestHandler {
                    connection,
                    notifications: AtomicUsize::new(0),
                })
            })
            .await
        });

        let (read_half, mut write_half) = tokio::io::split(mine);
        for line in lines {
            write_half
                .write_all(format!("{line}\n").as_bytes())
                .await
                .unwrap();
        }

        let mut received = Vec::new();
        let mut peer_lines = BufReader::new(read_half).lines();
        // Read until the far side goes quiet, then hang up.
        while let Ok(Some(line)) = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            peer_lines.next_line(),
        )
        .await
        .unwrap_or(Ok(None))
        {
            let value: Value = serde_json::from_str(&line).unwrap();
            let is_ping = value.get("method").and_then(Value::as_str) == Some("ping");
            received.push(value.clone());
            if is_ping {
                // Answer the agent's own request so `reenter` can finish.
                let id = value.get("id").cloned().unwrap();
                write_half
                    .write_all(
                        format!(
                            "{}\n",
                            json!({"jsonrpc": "2.0", "id": id, "result": "pong"})
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        }

        // Both halves have to go: a `DuplexStream` only reports EOF to the
        // far side once every handle to this side is dropped, so holding
        // the reader would leave `serve` running forever.
        drop(peer_lines);
        drop(write_half);
        tokio::time::timeout(std::time::Duration::from_secs(5), served)
            .await
            .expect("serve should return once the peer hangs up")
            .unwrap()
            .unwrap();
        received
    }

    #[tokio::test]
    async fn answers_a_request_with_a_matching_id() {
        let received = run_peer(vec![
            json!({"jsonrpc": "2.0", "id": 7, "method": "echo", "params": {"a": 1}}),
        ])
        .await;
        assert_eq!(received.len(), 1);
        assert_eq!(received[0]["id"], json!(7));
        assert_eq!(received[0]["result"], json!({"a": 1}));
    }

    #[tokio::test]
    async fn unknown_methods_get_a_method_not_found_error() {
        let received = run_peer(vec![json!({"jsonrpc": "2.0", "id": 1, "method": "nope"})]).await;
        assert_eq!(received[0]["error"]["code"], json!(-32601));
    }

    #[tokio::test]
    async fn notifications_get_no_response_at_all() {
        let received = run_peer(vec![
            json!({"jsonrpc": "2.0", "method": "some_notification", "params": {}}),
            json!({"jsonrpc": "2.0", "id": 1, "method": "echo", "params": "x"}),
        ])
        .await;
        // Only the request produced a line; the notification produced none.
        assert_eq!(received.len(), 1);
        assert_eq!(received[0]["id"], json!(1));
    }

    #[tokio::test]
    async fn handles_an_outbound_request_issued_while_serving_an_inbound_one() {
        let received = run_peer(vec![
            json!({"jsonrpc": "2.0", "id": 3, "method": "reenter"}),
        ])
        .await;
        // The agent's `ping` went out first, then its answer to `reenter`.
        assert_eq!(received[0]["method"], json!("ping"));
        assert_eq!(received[1]["id"], json!(3));
        assert_eq!(received[1]["result"], json!("pong"));
    }

    #[tokio::test]
    async fn pipelined_requests_are_answered_in_the_order_they_arrived() {
        // A client is entitled to send several messages without waiting.
        // Answering them out of order breaks protocols with ordering
        // requirements -- ACP's `initialize` before anything else.
        let received = run_peer(vec![
            json!({"jsonrpc": "2.0", "id": 1, "method": "echo", "params": "first"}),
            json!({"jsonrpc": "2.0", "id": 2, "method": "echo", "params": "second"}),
            json!({"jsonrpc": "2.0", "id": 3, "method": "echo", "params": "third"}),
        ])
        .await;

        let ids: Vec<_> = received
            .iter()
            .map(|message| message["id"].clone())
            .collect();
        assert_eq!(ids, vec![json!(1), json!(2), json!(3)]);
    }

    #[tokio::test]
    async fn queued_answers_still_go_out_when_the_peer_closes_the_pipe_at_once() {
        // Two independent pipes, the way a subprocess really has them:
        // closing the agent's stdin must not also close its stdout. (A
        // single `duplex` can't model this -- it only reports EOF once
        // both halves are dropped, which would take the answers with it.)
        let (mut write_half, agent_stdin) = duplex(64 * 1024);
        let (agent_stdout, read_half) = duplex(64 * 1024);
        let served = tokio::spawn(async move {
            serve(agent_stdin, agent_stdout, |connection| {
                Arc::new(TestHandler {
                    connection,
                    notifications: AtomicUsize::new(0),
                })
            })
            .await
        });

        for id in 1..=2 {
            write_half
                .write_all(
                    format!(
                        "{}\n",
                        json!({"jsonrpc": "2.0", "id": id, "method": "echo", "params": id})
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        }
        // Hang up without waiting for either answer. Both are still owed.
        drop(write_half);

        let mut answers = Vec::new();
        let mut peer_lines = BufReader::new(read_half).lines();
        while let Some(line) = peer_lines.next_line().await.unwrap() {
            answers.push(serde_json::from_str::<Value>(&line).unwrap());
        }

        assert_eq!(answers.len(), 2, "both requests should still be answered");
        assert_eq!(answers[0]["result"], json!(1));
        assert_eq!(answers[1]["result"], json!(2));

        drop(peer_lines);
        tokio::time::timeout(std::time::Duration::from_secs(5), served)
            .await
            .expect("serve should return once the peer hangs up")
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn a_malformed_line_gets_a_parse_error_without_killing_the_connection() {
        let (mine, theirs) = duplex(64 * 1024);
        let (their_read, their_write) = tokio::io::split(theirs);
        let served = tokio::spawn(async move {
            serve(their_read, their_write, |connection| {
                Arc::new(TestHandler {
                    connection,
                    notifications: AtomicUsize::new(0),
                })
            })
            .await
        });

        let (read_half, mut write_half) = tokio::io::split(mine);
        write_half.write_all(b"{not json\n").await.unwrap();
        write_half
            .write_all(
                format!(
                    "{}\n",
                    json!({"jsonrpc": "2.0", "id": 2, "method": "echo", "params": "still here"})
                )
                .as_bytes(),
            )
            .await
            .unwrap();

        let mut peer_lines = BufReader::new(read_half).lines();
        let parse_error: Value =
            serde_json::from_str(&peer_lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(parse_error["error"]["code"], json!(-32700));
        assert_eq!(parse_error["id"], Value::Null);

        let echoed: Value =
            serde_json::from_str(&peer_lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(echoed["result"], json!("still here"));

        drop(peer_lines);
        drop(write_half);
        tokio::time::timeout(std::time::Duration::from_secs(5), served)
            .await
            .expect("serve should return once the peer hangs up")
            .unwrap()
            .unwrap();
    }
}
