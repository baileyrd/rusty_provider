//! Agent Client Protocol wire types, protocol version 1.
//!
//! Hand-written against the published JSON Schema
//! (`schema/v1/schema.json` in the `agentclientprotocol/agent-client-protocol`
//! repository), for the same reason the provider adapters hand-write the
//! Anthropic and Gemini wire formats instead of pulling in vendor SDKs:
//! the surface is small and stable, and owning the types keeps the
//! dependency graph flat.
//!
//! Only the subset this agent actually speaks is modelled. ACP is
//! explicitly extensible -- unknown fields on incoming messages are
//! ignored rather than rejected, and optional capabilities we don't
//! implement (session modes, elicitation, MCP-over-ACP, session
//! list/resume/delete) are simply never advertised, which per the spec
//! means a client must not use them.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The ACP protocol version this agent implements. Wire compatibility is
/// negotiated by this number during `initialize`, independently of the
/// schema-artifact version the types were generated from.
pub const PROTOCOL_VERSION: u32 = 1;

/// Method names, spelled exactly as `schema/v1/meta.json` defines them.
pub mod method {
    // Client -> agent.
    pub const INITIALIZE: &str = "initialize";
    pub const AUTHENTICATE: &str = "authenticate";
    pub const SESSION_NEW: &str = "session/new";
    pub const SESSION_LOAD: &str = "session/load";
    pub const SESSION_PROMPT: &str = "session/prompt";
    pub const SESSION_CANCEL: &str = "session/cancel";

    // Agent -> client.
    pub const SESSION_UPDATE: &str = "session/update";
    pub const SESSION_REQUEST_PERMISSION: &str = "session/request_permission";
    pub const FS_READ_TEXT_FILE: &str = "fs/read_text_file";
    pub const FS_WRITE_TEXT_FILE: &str = "fs/write_text_file";
    pub const TERMINAL_CREATE: &str = "terminal/create";
    pub const TERMINAL_OUTPUT: &str = "terminal/output";
    pub const TERMINAL_WAIT_FOR_EXIT: &str = "terminal/wait_for_exit";
    pub const TERMINAL_KILL: &str = "terminal/kill";
    pub const TERMINAL_RELEASE: &str = "terminal/release";
}

/// JSON-RPC error codes, including the ACP-specific ones carved out of
/// the reserved `-32000..=-32099` range.
pub mod error_code {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    pub const AUTH_REQUIRED: i32 = -32000;
}

pub type SessionId = String;
pub type ToolCallId = String;
pub type TerminalId = String;

// --- initialize ------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeRequest {
    pub protocol_version: u32,
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
}

/// What the *client* can do for us. This is the whole basis for which
/// tools we expose to the model: an editor that can't write files gets no
/// `write_file` tool, rather than a tool that always fails.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default)]
    pub fs: FileSystemCapabilities,
    #[serde(default)]
    pub terminal: bool,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileSystemCapabilities {
    #[serde(default)]
    pub read_text_file: bool,
    #[serde(default)]
    pub write_text_file: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResponse {
    pub protocol_version: u32,
    pub agent_capabilities: AgentCapabilities,
    pub auth_methods: Vec<AuthMethod>,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    pub load_session: bool,
    pub prompt_capabilities: PromptCapabilities,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCapabilities {
    pub image: bool,
    pub audio: bool,
    pub embedded_context: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthMethod {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

// --- session lifecycle -----------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionRequest {
    pub cwd: String,
    /// MCP servers the client wants attached to the session. We don't
    /// advertise `mcpCapabilities`, so a spec-conformant client sends an
    /// empty list here; the field is still parsed (and ignored) so a
    /// client that sends one anyway doesn't get a hard protocol error.
    #[serde(default)]
    pub mcp_servers: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionResponse {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptRequest {
    pub session_id: SessionId,
    pub prompt: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResponse {
    pub stop_reason: StopReason,
}

/// Why a prompt turn ended. `EndTurn` is the model finishing normally;
/// everything else is a bounded stop the client renders differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelNotification {
    pub session_id: SessionId,
}

// --- content ---------------------------------------------------------------

/// A piece of prompt or message content. Mirrors MCP's content shape,
/// which is where ACP borrowed it from.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
    },
    Audio {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// A pointer to context the client has *not* inlined. We can't fetch
    /// arbitrary URIs, so these reach the model as a textual mention.
    ResourceLink {
        uri: String,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    /// Context the client inlined for us (an `@`-mentioned file, usually).
    Resource {
        resource: EmbeddedResource,
    },
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
}

/// Untagged because the schema distinguishes the two purely by which of
/// `text`/`blob` is present.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EmbeddedResource {
    Text {
        uri: String,
        text: String,
        #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    Blob {
        uri: String,
        blob: String,
        #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
}

// --- session/update --------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNotification {
    pub session_id: SessionId,
    pub update: SessionUpdate,
}

/// Everything the agent streams back mid-turn. The client renders these
/// live; nothing here is a request, so none of it can fail or block.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
pub enum SessionUpdate {
    AgentMessageChunk {
        content: ContentBlock,
    },
    /// The model's reasoning trace, when the configured model produces one
    /// and the client asked for it. Rendered separately from the answer.
    AgentThoughtChunk {
        content: ContentBlock,
    },
    #[serde(rename_all = "camelCase")]
    ToolCall {
        tool_call_id: ToolCallId,
        title: String,
        kind: ToolKind,
        status: ToolCallStatus,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        content: Vec<ToolCallContent>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        locations: Vec<ToolCallLocation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        raw_input: Option<Value>,
    },
    #[serde(rename_all = "camelCase")]
    ToolCallUpdate {
        tool_call_id: ToolCallId,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<ToolCallStatus>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Vec<ToolCallContent>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        raw_output: Option<Value>,
    },
    Plan {
        entries: Vec<PlanEntry>,
    },
    /// Context-window and cumulative-cost accounting for the session.
    /// Only sent when the router knows the model's context length (i.e.
    /// it has a `[[pricing]]` entry with `context_length`), since `size`
    /// is required and guessing it would mis-render the client's gauge.
    #[serde(rename_all = "camelCase")]
    UsageUpdate {
        used: u64,
        size: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        cost: Option<Cost>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct Cost {
    pub amount: f64,
    pub currency: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// The category icon/affordance the client shows for a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    SwitchMode,
    Other,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallContent {
    Content {
        content: ContentBlock,
    },
    /// A proposed or applied file change, which clients render as a real
    /// diff rather than as text.
    #[serde(rename_all = "camelCase")]
    Diff {
        path: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        old_text: Option<String>,
        new_text: String,
    },
    /// Embeds a live terminal by id, so the client can stream command
    /// output as it happens instead of waiting for the tool to finish.
    #[serde(rename_all = "camelCase")]
    Terminal {
        terminal_id: TerminalId,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallLocation {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanEntry {
    pub content: String,
    pub priority: PlanEntryPriority,
    pub status: PlanEntryStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanEntryPriority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanEntryStatus {
    Pending,
    InProgress,
    Completed,
}

// --- session/request_permission --------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionRequest {
    pub session_id: SessionId,
    pub tool_call: PermissionToolCall,
    pub options: Vec<PermissionOption>,
}

/// The `ToolCallUpdate` shape, restricted to the fields worth showing in
/// a permission prompt.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionToolCall {
    pub tool_call_id: ToolCallId,
    pub title: String,
    pub kind: ToolKind,
    pub status: ToolCallStatus,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ToolCallContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<ToolCallLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionOption {
    pub option_id: String,
    pub name: String,
    pub kind: PermissionOptionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionOptionKind {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionResponse {
    pub outcome: RequestPermissionOutcome,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum RequestPermissionOutcome {
    /// The turn was cancelled while the prompt was on screen.
    Cancelled,
    #[serde(rename_all = "camelCase")]
    Selected { option_id: String },
}

// --- fs/* ------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadTextFileRequest {
    pub session_id: SessionId,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReadTextFileResponse {
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteTextFileRequest {
    pub session_id: SessionId,
    pub path: String,
    pub content: String,
}

// --- terminal/* ------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateTerminalRequest {
    pub session_id: SessionId,
    pub command: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Caps how much output the client buffers for us. Without it a
    /// runaway command's output is unbounded on the client side.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_byte_limit: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateTerminalResponse {
    pub terminal_id: TerminalId,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalRequest {
    pub session_id: SessionId,
    pub terminal_id: TerminalId,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalOutputResponse {
    pub output: String,
    #[serde(default)]
    pub truncated: bool,
    #[serde(default)]
    pub exit_status: Option<TerminalExitStatus>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WaitForTerminalExitResponse {
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub signal: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalExitStatus {
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub signal: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_update_serializes_with_the_schemas_internal_tag() {
        let update = SessionUpdate::AgentMessageChunk {
            content: ContentBlock::text("hi"),
        };
        assert_eq!(
            serde_json::to_value(&update).unwrap(),
            serde_json::json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "hi"},
            })
        );
    }

    #[test]
    fn tool_call_update_omits_absent_fields_rather_than_sending_nulls() {
        let update = SessionUpdate::ToolCallUpdate {
            tool_call_id: "call_1".into(),
            status: Some(ToolCallStatus::Completed),
            title: None,
            content: None,
            raw_output: None,
        };
        assert_eq!(
            serde_json::to_value(&update).unwrap(),
            serde_json::json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_1",
                "status": "completed",
            })
        );
    }

    #[test]
    fn diff_content_uses_camel_case_keys() {
        let content = ToolCallContent::Diff {
            path: "/repo/a.rs".into(),
            old_text: Some("old".into()),
            new_text: "new".into(),
        };
        assert_eq!(
            serde_json::to_value(&content).unwrap(),
            serde_json::json!({
                "type": "diff",
                "path": "/repo/a.rs",
                "oldText": "old",
                "newText": "new",
            })
        );
    }

    #[test]
    fn permission_outcome_deserializes_both_variants() {
        let selected: RequestPermissionResponse = serde_json::from_value(serde_json::json!({
            "outcome": {"outcome": "selected", "optionId": "allow_once"}
        }))
        .unwrap();
        assert!(matches!(
            selected.outcome,
            RequestPermissionOutcome::Selected { ref option_id } if option_id == "allow_once"
        ));

        let cancelled: RequestPermissionResponse = serde_json::from_value(serde_json::json!({
            "outcome": {"outcome": "cancelled"}
        }))
        .unwrap();
        assert!(matches!(
            cancelled.outcome,
            RequestPermissionOutcome::Cancelled
        ));
    }

    #[test]
    fn initialize_request_tolerates_capabilities_it_doesnt_model() {
        // ACP is extensible: a newer client sends fields (elicitation,
        // session capabilities, _meta) this agent has never heard of, and
        // that must not be a parse error.
        let req: InitializeRequest = serde_json::from_value(serde_json::json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": true, "writeTextFile": true},
                "terminal": true,
                "elicitation": {"forms": {}},
                "_meta": {"vendor": "whatever"},
            },
            "clientInfo": {"name": "zed", "version": "1.0"},
        }))
        .unwrap();
        assert_eq!(req.protocol_version, 1);
        assert!(req.client_capabilities.fs.read_text_file);
        assert!(req.client_capabilities.terminal);
    }

    #[test]
    fn prompt_content_blocks_round_trip_through_their_tag() {
        let blocks: Vec<ContentBlock> = serde_json::from_value(serde_json::json!([
            {"type": "text", "text": "explain this"},
            {"type": "resource", "resource": {"uri": "file:///a.rs", "text": "fn main() {}"}},
            {"type": "resource_link", "uri": "file:///b.rs", "name": "b.rs"},
        ]))
        .unwrap();
        assert_eq!(blocks.len(), 3);
        assert!(matches!(blocks[0], ContentBlock::Text { .. }));
        assert!(matches!(blocks[1], ContentBlock::Resource { .. }));
        assert!(matches!(blocks[2], ContentBlock::ResourceLink { .. }));
    }
}
