//! An Agent Client Protocol (ACP) coding agent backed by the router.
//!
//! ACP is the protocol code editors use to drive coding agents: the editor
//! spawns the agent as a subprocess and speaks JSON-RPC 2.0 over its
//! stdin/stdout. This crate implements the *agent* half, turning each
//! `session/prompt` into a tool-calling loop whose model requests go
//! through [`rp_router::Router`] -- so an ACP session gets the same
//! fallback chains, presets, guardrails, moderation, budgets, caching and
//! usage accounting as an HTTP request to `/v1/chat/completions`.
//!
//! The editor owns the workspace, not the agent: file reads and writes go
//! back over the protocol as `fs/read_text_file` and `fs/write_text_file`,
//! and commands run through the client's `terminal/*` methods. That is
//! what lets the editor show unsaved buffer state, render diffs, and stream
//! command output live -- and it means the tools this agent offers the
//! model are gated on what the connected client actually advertises.

pub mod agent;
pub mod jsonrpc;
pub mod schema;
pub mod session;
pub mod tools;
pub mod turn;

pub use agent::Agent;
pub use jsonrpc::{serve, Connection};
