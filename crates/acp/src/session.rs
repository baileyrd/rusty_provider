//! Per-session state: the conversation the model sees, the permissions
//! the user has granted for the rest of the session, and the cancellation
//! flag a `session/cancel` notification flips.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rp_core::ChatMessage;

use crate::schema::{ClientCapabilities, SessionId};

/// A standing decision the user made about one tool, from picking
/// "always" on a permission prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grant {
    Always,
    Never,
}

/// Everything a turn mutates, behind an async mutex because a turn holds
/// it across `await` points (model dispatch, tool execution).
#[derive(Debug, Default)]
pub struct SessionState {
    /// The full conversation, in the router's own message type -- user
    /// prompts, assistant replies, and the tool-result messages the loop
    /// feeds back in.
    pub history: Vec<ChatMessage>,
    pub grants: HashMap<String, Grant>,
}

pub struct Session {
    pub id: SessionId,
    /// The workspace root the client opened the session for. Relative
    /// paths from the model resolve against it, since `fs/*` requires
    /// absolute paths on the wire.
    pub cwd: PathBuf,
    pub client_capabilities: ClientCapabilities,
    /// Deliberately outside `state`: `session/cancel` arrives on a
    /// different task while the turn holds the state lock, so cancellation
    /// has to be observable without acquiring it.
    cancelled: AtomicBool,
    state: tokio::sync::Mutex<SessionState>,
}

impl Session {
    pub fn new(id: SessionId, cwd: PathBuf, client_capabilities: ClientCapabilities) -> Self {
        Self {
            id,
            cwd,
            client_capabilities,
            cancelled: AtomicBool::new(false),
            state: tokio::sync::Mutex::new(SessionState::default()),
        }
    }

    pub async fn state(&self) -> tokio::sync::MutexGuard<'_, SessionState> {
        self.state.lock().await
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Clears any cancellation left over from a previous turn. A cancel
    /// that arrives between turns would otherwise poison the next prompt.
    pub fn begin_turn(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }

    /// Makes a model-supplied path absolute, since `fs/read_text_file` and
    /// `fs/write_text_file` both require an absolute path and models
    /// routinely answer with workspace-relative ones.
    pub fn resolve_path(&self, path: &str) -> String {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            candidate.to_string_lossy().into_owned()
        } else {
            self.cwd.join(candidate).to_string_lossy().into_owned()
        }
    }
}

/// The set of live sessions, keyed by the id handed to the client.
#[derive(Default)]
pub struct SessionStore {
    sessions: Mutex<HashMap<SessionId, Arc<Session>>>,
    next_id: Mutex<u64>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Session ids only need to be opaque and unique for this process --
    /// the client treats them as a handle and never parses them.
    pub fn create(&self, cwd: PathBuf, client_capabilities: ClientCapabilities) -> Arc<Session> {
        let id = {
            let mut next = self.next_id.lock().expect("session id counter poisoned");
            *next += 1;
            format!("sess_{next}")
        };
        let session = Arc::new(Session::new(id.clone(), cwd, client_capabilities));
        self.sessions
            .lock()
            .expect("session store poisoned")
            .insert(id, session.clone());
        session
    }

    /// Registers a session under a client-supplied id, for `session/load`.
    pub fn insert(&self, session: Arc<Session>) {
        self.sessions
            .lock()
            .expect("session store poisoned")
            .insert(session.id.clone(), session);
    }

    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        self.sessions
            .lock()
            .expect("session store poisoned")
            .get(id)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        Session::new(
            "sess_1".into(),
            PathBuf::from("/repo"),
            ClientCapabilities::default(),
        )
    }

    #[test]
    fn relative_paths_resolve_against_the_session_cwd() {
        assert_eq!(session().resolve_path("src/main.rs"), "/repo/src/main.rs");
    }

    #[test]
    fn absolute_paths_are_left_alone() {
        assert_eq!(session().resolve_path("/etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn beginning_a_turn_clears_a_stale_cancellation() {
        let session = session();
        session.cancel();
        assert!(session.is_cancelled());
        session.begin_turn();
        assert!(!session.is_cancelled());
    }

    #[test]
    fn created_sessions_get_distinct_ids_and_are_retrievable() {
        let store = SessionStore::new();
        let first = store.create(PathBuf::from("/a"), ClientCapabilities::default());
        let second = store.create(PathBuf::from("/b"), ClientCapabilities::default());
        assert_ne!(first.id, second.id);
        assert_eq!(store.get(&first.id).unwrap().cwd, PathBuf::from("/a"));
        assert!(store.get("sess_nope").is_none());
    }
}
