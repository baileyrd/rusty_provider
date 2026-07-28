//! The tools the model can call, and their execution against the ACP
//! client.
//!
//! Every tool here is a thin shim over a client method: the agent never
//! touches the filesystem or spawns a process itself. That's ACP's design,
//! not an accident of this implementation -- routing edits through the
//! editor is what lets it show unsaved buffers, render diffs inline, and
//! stream a command's output while it runs.
//!
//! Which tools exist at all depends on what the connected client
//! advertised in `initialize`. A client without `fs.writeTextFile` never
//! sees `write_file` in the tool list, so the model can't propose an edit
//! that could only ever fail.

use serde_json::{json, Value};

use rp_core::{FunctionDef, Tool};

use crate::jsonrpc::{Connection, TransportError};
use crate::schema::{
    method, ClientCapabilities, ContentBlock, CreateTerminalRequest, CreateTerminalResponse,
    PermissionOption, PermissionOptionKind, PermissionToolCall, ReadTextFileRequest,
    ReadTextFileResponse, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SessionNotification, SessionUpdate, TerminalOutputResponse,
    TerminalRequest, ToolCallContent, ToolCallLocation, ToolCallStatus, ToolKind,
    WaitForTerminalExitResponse, WriteTextFileRequest,
};
use crate::session::{Grant, Session};

pub const READ_FILE: &str = "read_file";
pub const WRITE_FILE: &str = "write_file";
pub const EDIT_FILE: &str = "edit_file";
pub const EXECUTE_COMMAND: &str = "execute_command";
pub const UPDATE_PLAN: &str = "update_plan";

/// How much of a command's output to hand back to the model. The client
/// keeps the full text (and renders it live); this only bounds what gets
/// spent on context.
const MAX_OUTPUT_CHARS_FOR_MODEL: usize = 16_000;
/// What the client is asked to buffer per terminal.
const TERMINAL_OUTPUT_BYTE_LIMIT: u64 = 1024 * 1024;

/// A tool call ended the whole turn rather than producing a result the
/// model can react to.
#[derive(Debug)]
pub enum ToolError {
    /// The user cancelled -- either by rejecting at the permission prompt
    /// with the turn cancelled, or via `session/cancel`.
    Cancelled,
    /// The client is unreachable, so there is nothing to continue for.
    Transport(TransportError),
}

impl From<TransportError> for ToolError {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

/// The result of a tool call that the loop can keep going from.
pub struct ToolOutcome {
    /// What gets appended to the conversation as the tool's result.
    pub result: String,
    /// What the client renders under the tool call.
    pub content: Vec<ToolCallContent>,
    pub raw_output: Option<Value>,
    /// A failure the *model* should see and recover from (a missing file,
    /// a command exiting non-zero), as opposed to a [`ToolError`].
    pub failed: bool,
}

impl ToolOutcome {
    fn ok(result: impl Into<String>) -> Self {
        Self {
            result: result.into(),
            content: Vec::new(),
            raw_output: None,
            failed: false,
        }
    }

    fn failure(result: impl Into<String>) -> Self {
        Self {
            result: result.into(),
            content: Vec::new(),
            raw_output: None,
            failed: true,
        }
    }

    fn with_content(mut self, content: Vec<ToolCallContent>) -> Self {
        self.content = content;
        self
    }

    fn with_raw_output(mut self, raw: Value) -> Self {
        self.raw_output = Some(raw);
        self
    }
}

/// How a tool call is labelled and categorised in the client's UI, decided
/// before the call runs so the `tool_call` update can be sent up front.
pub struct Presentation {
    pub title: String,
    pub kind: ToolKind,
    pub locations: Vec<ToolCallLocation>,
}

/// The tool list offered to the model, filtered by client capability.
pub fn definitions(capabilities: &ClientCapabilities) -> Vec<Tool> {
    let mut tools = Vec::new();

    if capabilities.fs.read_text_file {
        tools.push(function(
            READ_FILE,
            "Read a text file from the user's workspace. Prefer this over guessing a \
             file's contents. Returns the file's text, which reflects the editor's \
             unsaved buffer if one is open.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file, absolute or relative to the workspace root."},
                    "line": {"type": "integer", "description": "1-based line to start reading from."},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read."},
                },
                "required": ["path"],
            }),
        ));
    }

    if capabilities.fs.write_text_file {
        tools.push(function(
            WRITE_FILE,
            "Create a file, or replace an existing file's entire contents. For a \
             targeted change to part of an existing file, use edit_file instead.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file, absolute or relative to the workspace root."},
                    "content": {"type": "string", "description": "The file's complete new contents."},
                },
                "required": ["path", "content"],
            }),
        ));
    }

    // An edit is read-modify-write, so it needs both halves of the fs
    // capability, not just the write half.
    if capabilities.fs.read_text_file && capabilities.fs.write_text_file {
        tools.push(function(
            EDIT_FILE,
            "Replace an exact span of text in an existing file. old_text must match \
             the file exactly, including indentation, and must be unique unless \
             replace_all is set.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file, absolute or relative to the workspace root."},
                    "old_text": {"type": "string", "description": "The exact text to replace."},
                    "new_text": {"type": "string", "description": "The text to replace it with."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."},
                },
                "required": ["path", "old_text", "new_text"],
            }),
        ));
    }

    if capabilities.terminal {
        tools.push(function(
            EXECUTE_COMMAND,
            "Run a command in the user's workspace and return its exit status and \
             output. Use this for builds, tests, linters, and version control.",
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The program to run."},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Arguments to pass to it."},
                    "cwd": {"type": "string", "description": "Working directory; defaults to the workspace root."},
                },
                "required": ["command"],
            }),
        ));
    }

    // Always available: it needs nothing from the client beyond the
    // session/update notification every ACP client handles.
    tools.push(function(
        UPDATE_PLAN,
        "Record or update the plan for a multi-step task, so the user can follow \
         along. Send the whole plan each time, not just what changed.",
        json!({
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "What this step does."},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["content", "priority", "status"],
                    },
                },
            },
            "required": ["entries"],
        }),
    ));

    tools
}

fn function(name: &str, description: &str, parameters: Value) -> Tool {
    Tool {
        kind: "function".to_string(),
        function: FunctionDef {
            name: name.to_string(),
            description: Some(description.to_string()),
            parameters: Some(parameters),
        },
    }
}

/// Whether a tool mutates the user's machine and therefore needs consent
/// before it runs. Reads and plan updates don't; writes and commands do.
pub fn requires_permission(name: &str) -> bool {
    matches!(name, WRITE_FILE | EDIT_FILE | EXECUTE_COMMAND)
}

/// Labels a call for the client's UI. Falls back to something honest for
/// a tool name the model invented, since that still gets reported before
/// it's rejected.
pub fn presentation(name: &str, arguments: &Value, session: &Session) -> Presentation {
    let path = arguments.get("path").and_then(Value::as_str);
    let location = |path: &str| {
        vec![ToolCallLocation {
            path: session.resolve_path(path),
            line: arguments
                .get("line")
                .and_then(Value::as_u64)
                .map(|line| line as u32),
        }]
    };

    match (name, path) {
        (READ_FILE, Some(path)) => Presentation {
            title: format!("Read {path}"),
            kind: ToolKind::Read,
            locations: location(path),
        },
        (WRITE_FILE, Some(path)) => Presentation {
            title: format!("Write {path}"),
            kind: ToolKind::Edit,
            locations: location(path),
        },
        (EDIT_FILE, Some(path)) => Presentation {
            title: format!("Edit {path}"),
            kind: ToolKind::Edit,
            locations: location(path),
        },
        (EXECUTE_COMMAND, _) => {
            let command = arguments
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("command");
            let args = arguments
                .get("args")
                .and_then(Value::as_array)
                .map(|args| {
                    args.iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            Presentation {
                title: if args.is_empty() {
                    command.to_string()
                } else {
                    format!("{command} {args}")
                },
                kind: ToolKind::Execute,
                locations: Vec::new(),
            }
        }
        (UPDATE_PLAN, _) => Presentation {
            title: "Update plan".to_string(),
            kind: ToolKind::Think,
            locations: Vec::new(),
        },
        _ => Presentation {
            title: name.to_string(),
            kind: ToolKind::Other,
            locations: Vec::new(),
        },
    }
}

/// Everything one tool call needs to talk back to the client.
pub struct ToolContext<'a> {
    pub connection: &'a Connection,
    pub session: &'a Session,
    pub tool_call_id: String,
}

impl ToolContext<'_> {
    fn notify(&self, update: SessionUpdate) {
        let notification = SessionNotification {
            session_id: self.session.id.clone(),
            update,
        };
        if let Err(e) = self
            .connection
            .notify(method::SESSION_UPDATE, &notification)
        {
            tracing::debug!("dropping session update: {e}");
        }
    }

    /// Asks the user to approve a mutating call, honouring (and recording)
    /// the "always" answers.
    async fn request_permission(
        &self,
        tool: &str,
        presentation: &Presentation,
        content: Vec<ToolCallContent>,
        raw_input: Option<Value>,
    ) -> Result<bool, ToolError> {
        match self.session.state().await.grants.get(tool).copied() {
            Some(Grant::Always) => return Ok(true),
            Some(Grant::Never) => return Ok(false),
            None => {}
        }

        let request = RequestPermissionRequest {
            session_id: self.session.id.clone(),
            tool_call: PermissionToolCall {
                tool_call_id: self.tool_call_id.clone(),
                title: presentation.title.clone(),
                kind: presentation.kind,
                status: ToolCallStatus::Pending,
                content,
                locations: presentation.locations.clone(),
                raw_input,
            },
            options: vec![
                PermissionOption {
                    option_id: "allow_once".into(),
                    name: "Allow".into(),
                    kind: PermissionOptionKind::AllowOnce,
                },
                PermissionOption {
                    option_id: "allow_always".into(),
                    name: "Always allow".into(),
                    kind: PermissionOptionKind::AllowAlways,
                },
                PermissionOption {
                    option_id: "reject_once".into(),
                    name: "Reject".into(),
                    kind: PermissionOptionKind::RejectOnce,
                },
                PermissionOption {
                    option_id: "reject_always".into(),
                    name: "Always reject".into(),
                    kind: PermissionOptionKind::RejectAlways,
                },
            ],
        };

        let response: RequestPermissionResponse = self
            .connection
            .request(method::SESSION_REQUEST_PERMISSION, &request)
            .await?;

        match response.outcome {
            RequestPermissionOutcome::Cancelled => Err(ToolError::Cancelled),
            RequestPermissionOutcome::Selected { option_id } => match option_id.as_str() {
                "allow_once" => Ok(true),
                "allow_always" => {
                    self.session
                        .state()
                        .await
                        .grants
                        .insert(tool.to_string(), Grant::Always);
                    Ok(true)
                }
                "reject_always" => {
                    self.session
                        .state()
                        .await
                        .grants
                        .insert(tool.to_string(), Grant::Never);
                    Ok(false)
                }
                // "reject_once", plus any option id a client echoes back
                // that we didn't offer: refuse rather than assume consent.
                _ => Ok(false),
            },
        }
    }
}

/// Runs one tool call.
pub async fn execute(
    context: &ToolContext<'_>,
    name: &str,
    arguments: &str,
    presentation: &Presentation,
) -> Result<ToolOutcome, ToolError> {
    // The model hands arguments over as a JSON-encoded string, which it
    // does occasionally get wrong. That's a result the model can fix on
    // the next iteration, not a turn-ending error.
    let arguments: Value = if arguments.trim().is_empty() {
        json!({})
    } else {
        match serde_json::from_str(arguments) {
            Ok(value) => value,
            Err(e) => {
                return Ok(ToolOutcome::failure(format!(
                    "Arguments were not valid JSON: {e}"
                )))
            }
        }
    };

    if requires_permission(name) {
        let preview = permission_preview(context, name, &arguments).await?;
        let allowed = context
            .request_permission(name, presentation, preview, Some(arguments.clone()))
            .await?;
        if !allowed {
            return Ok(ToolOutcome::failure(
                "The user rejected this action. Do not retry it; ask them how to proceed.",
            ));
        }
    }

    match name {
        READ_FILE => read_file(context, &arguments).await,
        WRITE_FILE => write_file(context, &arguments).await,
        EDIT_FILE => edit_file(context, &arguments).await,
        EXECUTE_COMMAND => execute_command(context, &arguments).await,
        UPDATE_PLAN => update_plan(context, &arguments),
        other => Ok(ToolOutcome::failure(format!(
            "Unknown tool {other:?}. Use only the tools you were given."
        ))),
    }
}

/// The diff shown in the permission prompt, so the user approves a
/// concrete change rather than an abstract "may write files".
async fn permission_preview(
    context: &ToolContext<'_>,
    name: &str,
    arguments: &Value,
) -> Result<Vec<ToolCallContent>, ToolError> {
    let Some(path) = arguments.get("path").and_then(Value::as_str) else {
        return Ok(Vec::new());
    };
    let path = context.session.resolve_path(path);

    match name {
        WRITE_FILE => {
            let new_text = arguments
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            // A read failure here just means the file is new, which is a
            // perfectly good diff to show ("no old text").
            let old_text = read_text(context, &path, None, None).await.ok();
            Ok(vec![ToolCallContent::Diff {
                path,
                old_text,
                new_text,
            }])
        }
        EDIT_FILE => {
            let (Some(old), Some(new)) = (
                arguments.get("old_text").and_then(Value::as_str),
                arguments.get("new_text").and_then(Value::as_str),
            ) else {
                return Ok(Vec::new());
            };
            let Ok(current) = read_text(context, &path, None, None).await else {
                return Ok(Vec::new());
            };
            let replace_all = arguments
                .get("replace_all")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            match apply_replacement(&current, old, new, replace_all) {
                Ok(updated) => Ok(vec![ToolCallContent::Diff {
                    path,
                    old_text: Some(current),
                    new_text: updated,
                }]),
                // The edit won't apply. Prompting with no diff is fine --
                // execution below produces the real error for the model.
                Err(_) => Ok(Vec::new()),
            }
        }
        _ => Ok(Vec::new()),
    }
}

async fn read_text(
    context: &ToolContext<'_>,
    path: &str,
    line: Option<u32>,
    limit: Option<u32>,
) -> Result<String, TransportError> {
    let request = ReadTextFileRequest {
        session_id: context.session.id.clone(),
        path: path.to_string(),
        line,
        limit,
    };
    let response: ReadTextFileResponse = context
        .connection
        .request(method::FS_READ_TEXT_FILE, &request)
        .await?;
    Ok(response.content)
}

async fn read_file(context: &ToolContext<'_>, arguments: &Value) -> Result<ToolOutcome, ToolError> {
    let Some(path) = arguments.get("path").and_then(Value::as_str) else {
        return Ok(ToolOutcome::failure("Missing required argument \"path\"."));
    };
    let path = context.session.resolve_path(path);
    let line = arguments
        .get("line")
        .and_then(Value::as_u64)
        .map(|n| n as u32);
    let limit = arguments
        .get("limit")
        .and_then(Value::as_u64)
        .map(|n| n as u32);

    match read_text(context, &path, line, limit).await {
        Ok(content) => {
            Ok(
                ToolOutcome::ok(content.clone()).with_content(vec![ToolCallContent::Content {
                    content: ContentBlock::text(content),
                }]),
            )
        }
        // A client-side error here (no such file, unreadable) is
        // information the model needs, not a broken connection.
        Err(TransportError::Rpc(e)) => {
            Ok(ToolOutcome::failure(format!("Could not read {path}: {e}")))
        }
        Err(e) => Err(ToolError::Transport(e)),
    }
}

async fn write_text(
    context: &ToolContext<'_>,
    path: &str,
    content: &str,
) -> Result<(), TransportError> {
    let request = WriteTextFileRequest {
        session_id: context.session.id.clone(),
        path: path.to_string(),
        content: content.to_string(),
    };
    // The response is an empty object; nothing to read out of it.
    let _: Value = context
        .connection
        .request(method::FS_WRITE_TEXT_FILE, &request)
        .await?;
    Ok(())
}

async fn write_file(
    context: &ToolContext<'_>,
    arguments: &Value,
) -> Result<ToolOutcome, ToolError> {
    let (Some(path), Some(content)) = (
        arguments.get("path").and_then(Value::as_str),
        arguments.get("content").and_then(Value::as_str),
    ) else {
        return Ok(ToolOutcome::failure(
            "write_file requires both \"path\" and \"content\".",
        ));
    };
    let path = context.session.resolve_path(path);
    let old_text = read_text(context, &path, None, None).await.ok();

    match write_text(context, &path, content).await {
        Ok(()) => Ok(ToolOutcome::ok(format!("Wrote {path}.")).with_content(vec![
            ToolCallContent::Diff {
                path,
                old_text,
                new_text: content.to_string(),
            },
        ])),
        Err(TransportError::Rpc(e)) => {
            Ok(ToolOutcome::failure(format!("Could not write {path}: {e}")))
        }
        Err(e) => Err(ToolError::Transport(e)),
    }
}

async fn edit_file(context: &ToolContext<'_>, arguments: &Value) -> Result<ToolOutcome, ToolError> {
    let (Some(path), Some(old), Some(new)) = (
        arguments.get("path").and_then(Value::as_str),
        arguments.get("old_text").and_then(Value::as_str),
        arguments.get("new_text").and_then(Value::as_str),
    ) else {
        return Ok(ToolOutcome::failure(
            "edit_file requires \"path\", \"old_text\" and \"new_text\".",
        ));
    };
    let path = context.session.resolve_path(path);
    let replace_all = arguments
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let current = match read_text(context, &path, None, None).await {
        Ok(content) => content,
        Err(TransportError::Rpc(e)) => {
            return Ok(ToolOutcome::failure(format!("Could not read {path}: {e}")))
        }
        Err(e) => return Err(ToolError::Transport(e)),
    };

    let updated = match apply_replacement(&current, old, new, replace_all) {
        Ok(updated) => updated,
        Err(e) => return Ok(ToolOutcome::failure(e)),
    };

    match write_text(context, &path, &updated).await {
        Ok(()) => Ok(
            ToolOutcome::ok(format!("Edited {path}.")).with_content(vec![ToolCallContent::Diff {
                path,
                old_text: Some(current),
                new_text: updated,
            }]),
        ),
        Err(TransportError::Rpc(e)) => {
            Ok(ToolOutcome::failure(format!("Could not write {path}: {e}")))
        }
        Err(e) => Err(ToolError::Transport(e)),
    }
}

/// Exact-match replacement with the usual uniqueness rule: an ambiguous
/// match is refused rather than guessed at, since picking the wrong
/// occurrence silently corrupts the file.
fn apply_replacement(
    current: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<String, String> {
    if old.is_empty() {
        return Err("\"old_text\" must not be empty.".to_string());
    }
    let occurrences = current.matches(old).count();
    match occurrences {
        0 => Err(
            "\"old_text\" does not appear in the file. Read the file and match its \
             exact text, including indentation."
                .to_string(),
        ),
        _ if replace_all => Ok(current.replace(old, new)),
        1 => Ok(current.replacen(old, new, 1)),
        n => Err(format!(
            "\"old_text\" appears {n} times. Include more surrounding context to make \
             it unique, or set replace_all."
        )),
    }
}

async fn execute_command(
    context: &ToolContext<'_>,
    arguments: &Value,
) -> Result<ToolOutcome, ToolError> {
    let Some(command) = arguments.get("command").and_then(Value::as_str) else {
        return Ok(ToolOutcome::failure(
            "execute_command requires \"command\".",
        ));
    };
    let args: Vec<String> = arguments
        .get("args")
        .and_then(Value::as_array)
        .map(|args| {
            args.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let cwd = arguments
        .get("cwd")
        .and_then(Value::as_str)
        .map(|cwd| context.session.resolve_path(cwd))
        .or_else(|| Some(context.session.cwd.to_string_lossy().into_owned()));

    let create = CreateTerminalRequest {
        session_id: context.session.id.clone(),
        command: command.to_string(),
        args,
        cwd,
        output_byte_limit: Some(TERMINAL_OUTPUT_BYTE_LIMIT),
    };
    let terminal: CreateTerminalResponse = match context
        .connection
        .request(method::TERMINAL_CREATE, &create)
        .await
    {
        Ok(response) => response,
        Err(TransportError::Rpc(e)) => {
            return Ok(ToolOutcome::failure(format!(
                "Could not start {command}: {e}"
            )))
        }
        Err(e) => return Err(ToolError::Transport(e)),
    };
    let terminal_id = terminal.terminal_id;

    // Embedding the terminal now, before waiting, is what makes the
    // client show output as it's produced rather than in one dump at the
    // end.
    context.notify(SessionUpdate::ToolCallUpdate {
        tool_call_id: context.tool_call_id.clone(),
        status: Some(ToolCallStatus::InProgress),
        title: None,
        content: Some(vec![ToolCallContent::Terminal {
            terminal_id: terminal_id.clone(),
        }]),
        raw_output: None,
    });

    let terminal_request = TerminalRequest {
        session_id: context.session.id.clone(),
        terminal_id: terminal_id.clone(),
    };

    let exit = wait_for_exit(context, &terminal_request).await?;

    let output: Result<TerminalOutputResponse, _> = context
        .connection
        .request(method::TERMINAL_OUTPUT, &terminal_request)
        .await;

    // Release regardless of how the wait went, so a cancelled or failed
    // command doesn't leak a terminal on the client side.
    let _: Result<Value, _> = context
        .connection
        .request(method::TERMINAL_RELEASE, &terminal_request)
        .await;

    let exit = match exit {
        Some(exit) => exit,
        None => return Err(ToolError::Cancelled),
    };

    let output = match output {
        Ok(output) => output,
        Err(TransportError::Rpc(e)) => {
            return Ok(ToolOutcome::failure(format!(
                "Command ran but its output could not be read: {e}"
            )))
        }
        Err(e) => return Err(ToolError::Transport(e)),
    };

    let status = match (exit.exit_code, &exit.signal) {
        (_, Some(signal)) => format!("killed by signal {signal}"),
        (Some(code), None) => format!("exit code {code}"),
        (None, None) => "exited".to_string(),
    };
    let mut text = output.output;
    if output.truncated {
        text.push_str("\n[output truncated by the client's byte limit]");
    }
    let text = truncate_for_model(&text);

    let failed = exit.exit_code.unwrap_or(0) != 0 || exit.signal.is_some();
    let result = format!("{status}\n\n{text}");
    let outcome = if failed {
        ToolOutcome::failure(result)
    } else {
        ToolOutcome::ok(result)
    };

    Ok(outcome
        .with_content(vec![ToolCallContent::Terminal { terminal_id }])
        .with_raw_output(json!({
            "exitCode": exit.exit_code,
            "signal": exit.signal,
            "truncated": output.truncated,
        })))
}

/// Waits for the command to finish, giving up if the turn is cancelled --
/// otherwise `session/cancel` during a hung build would never be honoured.
/// Returns `None` if cancelled.
async fn wait_for_exit(
    context: &ToolContext<'_>,
    request: &TerminalRequest,
) -> Result<Option<WaitForTerminalExitResponse>, ToolError> {
    let wait = context
        .connection
        .request::<_, WaitForTerminalExitResponse>(method::TERMINAL_WAIT_FOR_EXIT, request);
    tokio::pin!(wait);

    loop {
        tokio::select! {
            result = &mut wait => {
                return match result {
                    Ok(exit) => Ok(Some(exit)),
                    Err(TransportError::Rpc(e)) => Ok(Some(WaitForTerminalExitResponse {
                        exit_code: None,
                        signal: Some(format!("client error: {e}")),
                    })),
                    Err(e) => Err(ToolError::Transport(e)),
                };
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if context.session.is_cancelled() {
                    let _: Result<Value, _> = context
                        .connection
                        .request(method::TERMINAL_KILL, request)
                        .await;
                    return Ok(None);
                }
            }
        }
    }
}

fn truncate_for_model(text: &str) -> String {
    if text.chars().count() <= MAX_OUTPUT_CHARS_FOR_MODEL {
        return text.to_string();
    }
    // Keep the tail: compiler and test output puts the summary last.
    let kept: String = text
        .chars()
        .skip(text.chars().count() - MAX_OUTPUT_CHARS_FOR_MODEL)
        .collect();
    format!("[earlier output omitted]\n{kept}")
}

fn update_plan(context: &ToolContext<'_>, arguments: &Value) -> Result<ToolOutcome, ToolError> {
    let entries = match arguments.get("entries") {
        Some(entries) => match serde_json::from_value(entries.clone()) {
            Ok(entries) => entries,
            Err(e) => {
                return Ok(ToolOutcome::failure(format!(
                    "Plan entries were not in the expected shape: {e}"
                )))
            }
        },
        None => return Ok(ToolOutcome::failure("update_plan requires \"entries\".")),
    };

    context.notify(SessionUpdate::Plan { entries });
    Ok(ToolOutcome::ok("Plan updated."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FileSystemCapabilities;
    use std::path::PathBuf;

    fn capabilities(read: bool, write: bool, terminal: bool) -> ClientCapabilities {
        ClientCapabilities {
            fs: FileSystemCapabilities {
                read_text_file: read,
                write_text_file: write,
            },
            terminal,
        }
    }

    fn names(capabilities: &ClientCapabilities) -> Vec<String> {
        definitions(capabilities)
            .into_iter()
            .map(|tool| tool.function.name)
            .collect()
    }

    #[test]
    fn a_client_with_no_capabilities_only_gets_the_plan_tool() {
        assert_eq!(names(&capabilities(false, false, false)), vec![UPDATE_PLAN]);
    }

    #[test]
    fn editing_requires_both_halves_of_the_fs_capability() {
        // Write-only: no edit_file, because an edit has to read first.
        let write_only = names(&capabilities(false, true, false));
        assert!(write_only.contains(&WRITE_FILE.to_string()));
        assert!(!write_only.contains(&EDIT_FILE.to_string()));

        let both = names(&capabilities(true, true, false));
        assert!(both.contains(&EDIT_FILE.to_string()));
    }

    #[test]
    fn the_terminal_tool_tracks_the_terminal_capability() {
        assert!(!names(&capabilities(true, true, false)).contains(&EXECUTE_COMMAND.to_string()));
        assert!(names(&capabilities(true, true, true)).contains(&EXECUTE_COMMAND.to_string()));
    }

    #[test]
    fn only_mutating_tools_ask_for_permission() {
        assert!(!requires_permission(READ_FILE));
        assert!(!requires_permission(UPDATE_PLAN));
        assert!(requires_permission(WRITE_FILE));
        assert!(requires_permission(EDIT_FILE));
        assert!(requires_permission(EXECUTE_COMMAND));
    }

    #[test]
    fn a_unique_match_is_replaced() {
        assert_eq!(
            apply_replacement("a b c", "b", "B", false).unwrap(),
            "a B c"
        );
    }

    #[test]
    fn an_ambiguous_match_is_refused_rather_than_guessed() {
        let err = apply_replacement("x x", "x", "y", false).unwrap_err();
        assert!(err.contains("appears 2 times"), "{err}");
    }

    #[test]
    fn replace_all_takes_every_occurrence() {
        assert_eq!(apply_replacement("x x", "x", "y", true).unwrap(), "y y");
    }

    #[test]
    fn a_missing_match_explains_what_to_do() {
        let err = apply_replacement("abc", "zzz", "y", false).unwrap_err();
        assert!(err.contains("does not appear"), "{err}");
    }

    #[test]
    fn an_empty_old_text_is_rejected() {
        assert!(apply_replacement("abc", "", "y", false).is_err());
    }

    #[test]
    fn presentation_titles_name_the_target() {
        let session = crate::session::Session::new(
            "sess_1".into(),
            PathBuf::from("/repo"),
            ClientCapabilities::default(),
        );

        let read = presentation(READ_FILE, &json!({"path": "src/main.rs"}), &session);
        assert_eq!(read.title, "Read src/main.rs");
        assert!(matches!(read.kind, ToolKind::Read));
        assert_eq!(read.locations[0].path, "/repo/src/main.rs");

        let run = presentation(
            EXECUTE_COMMAND,
            &json!({"command": "cargo", "args": ["test", "--workspace"]}),
            &session,
        );
        assert_eq!(run.title, "cargo test --workspace");
        assert!(matches!(run.kind, ToolKind::Execute));
    }

    #[test]
    fn long_output_keeps_the_tail_where_the_summary_is() {
        let text = format!("{}\nSUMMARY LINE", "x".repeat(MAX_OUTPUT_CHARS_FOR_MODEL));
        let truncated = truncate_for_model(&text);
        assert!(truncated.starts_with("[earlier output omitted]"));
        assert!(truncated.ends_with("SUMMARY LINE"));
    }

    #[test]
    fn short_output_is_passed_through_untouched() {
        assert_eq!(truncate_for_model("hello"), "hello");
    }
}
