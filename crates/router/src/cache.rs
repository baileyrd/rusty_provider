//! `[cache]` -- an opt-in, in-memory, exact-match cache of non-streaming
//! `Router::dispatch` responses. A cache key is a hash of the entire
//! incoming request, so any difference at all (a different message, a
//! different sampling parameter, a different `provider` preference)
//! misses -- there's no semantic/fuzzy matching, only "did this exact
//! request already run within `ttl_secs`."
//!
//! Scoped to `dispatch` only, not `dispatch_stream`: faithfully replaying
//! a stored response as a fresh SSE chunk sequence is meaningfully more
//! work than returning a stored `ChatResponse` as-is, and out of scope
//! for this first version.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use rp_core::{ChatRequest, ChatResponse};

use crate::config::CacheConfig;

/// Fixed-capacity, insertion-order-evicting, TTL-bounded cache of
/// `ChatResponse`s keyed by request hash. Not a general-purpose LRU --
/// same "insertion order only, no read-refresh" tradeoff `GenerationCache`
/// already makes, plus a TTL check on read.
pub(crate) struct ResponseCache {
    ttl: Duration,
    max_entries: usize,
    order: VecDeque<u64>,
    entries: HashMap<u64, (Instant, ChatResponse)>,
}

impl ResponseCache {
    pub(crate) fn new(config: &CacheConfig) -> Self {
        Self {
            ttl: Duration::from_secs(config.ttl_secs),
            max_entries: config.max_entries.max(1),
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    /// `req`'s cache key -- a hash of its full JSON serialization, so
    /// this is exact-match on every field without needing `ChatRequest`
    /// to implement `Hash` itself (several fields are `f32`/`f64`, which
    /// don't). Two requests that serialize identically always hash
    /// identically; a 64-bit hash carries a theoretical (astronomically
    /// unlikely) collision risk between two *different* requests, the
    /// same tradeoff the issue's own suggested design ("request-hash
    /// keyed") accepts.
    pub(crate) fn key_for(req: &ChatRequest) -> u64 {
        let json = serde_json::to_string(req).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        json.hash(&mut hasher);
        hasher.finish()
    }

    /// `None` for a miss -- either nothing was ever cached under `key`,
    /// or it was but has since aged out of `ttl`. An expired entry is
    /// removed on lookup rather than waiting for eviction, so a
    /// long-idle cache doesn't hold stale entries indefinitely just
    /// because nothing pushed them out.
    pub(crate) fn get(&mut self, key: u64) -> Option<ChatResponse> {
        let (inserted_at, resp) = self.entries.get(&key)?;
        if inserted_at.elapsed() > self.ttl {
            self.entries.remove(&key);
            return None;
        }
        Some(resp.clone())
    }

    pub(crate) fn insert(&mut self, key: u64, resp: ChatResponse) {
        if !self.entries.contains_key(&key) {
            self.order.push_back(key);
            if self.order.len() > self.max_entries {
                if let Some(oldest) = self.order.pop_front() {
                    self.entries.remove(&oldest);
                }
            }
        }
        self.entries.insert(key, (Instant::now(), resp));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rp_core::{ChatMessage, Choice};

    fn config(ttl_secs: u64, max_entries: usize) -> CacheConfig {
        CacheConfig {
            ttl_secs,
            max_entries,
        }
    }

    fn request(model: &str, text: &str) -> ChatRequest {
        serde_json::from_value(serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": text}]
        }))
        .unwrap()
    }

    fn response(id: &str) -> ChatResponse {
        ChatResponse {
            id: id.to_string(),
            object: "chat.completion",
            created: 0,
            model: "anthropic/m1".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage::assistant("ok"),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: None,
            cost_usd: None,
        }
    }

    // --- key_for ---------------------------------------------------------------

    #[test]
    fn key_for_is_identical_for_identical_requests() {
        let a = request("anthropic/m1", "hi");
        let b = request("anthropic/m1", "hi");
        assert_eq!(ResponseCache::key_for(&a), ResponseCache::key_for(&b));
    }

    #[test]
    fn key_for_differs_when_the_message_text_differs() {
        let a = request("anthropic/m1", "hi");
        let b = request("anthropic/m1", "bye");
        assert_ne!(ResponseCache::key_for(&a), ResponseCache::key_for(&b));
    }

    #[test]
    fn key_for_differs_when_the_model_differs() {
        let a = request("anthropic/m1", "hi");
        let b = request("anthropic/m2", "hi");
        assert_ne!(ResponseCache::key_for(&a), ResponseCache::key_for(&b));
    }

    #[test]
    fn key_for_differs_when_a_sampling_param_differs() {
        let mut a = request("anthropic/m1", "hi");
        let mut b = request("anthropic/m1", "hi");
        a.temperature = Some(0.2);
        b.temperature = Some(0.9);
        assert_ne!(ResponseCache::key_for(&a), ResponseCache::key_for(&b));
    }

    // --- get/insert --------------------------------------------------------------

    #[test]
    fn get_is_none_before_any_insert() {
        let mut cache = ResponseCache::new(&config(60, 10));
        assert!(cache
            .get(ResponseCache::key_for(&request("a/m1", "hi")))
            .is_none());
    }

    #[test]
    fn get_returns_what_was_inserted_under_the_same_key() {
        let mut cache = ResponseCache::new(&config(60, 10));
        let key = ResponseCache::key_for(&request("a/m1", "hi"));
        cache.insert(key, response("resp-1"));

        let hit = cache.get(key).expect("should be cached");
        assert_eq!(hit.id, "resp-1");
    }

    #[test]
    fn get_expires_an_entry_past_its_ttl() {
        // A 0-second TTL means "expired the instant it's inserted" --
        // any nonzero elapsed time (which `Instant::elapsed` always
        // reports, even immediately after insert) exceeds it.
        let mut cache = ResponseCache::new(&config(0, 10));
        let key = ResponseCache::key_for(&request("a/m1", "hi"));
        cache.insert(key, response("resp-1"));

        assert!(cache.get(key).is_none());
    }

    #[test]
    fn insert_evicts_the_oldest_entry_once_over_capacity() {
        let mut cache = ResponseCache::new(&config(60, 2));
        let key_a = ResponseCache::key_for(&request("a/m1", "a"));
        let key_b = ResponseCache::key_for(&request("a/m1", "b"));
        let key_c = ResponseCache::key_for(&request("a/m1", "c"));

        cache.insert(key_a, response("resp-a"));
        cache.insert(key_b, response("resp-b"));
        cache.insert(key_c, response("resp-c"));

        assert!(cache.get(key_a).is_none(), "oldest entry should be evicted");
        assert!(cache.get(key_b).is_some());
        assert!(cache.get(key_c).is_some());
    }

    #[test]
    fn insert_reinserting_an_existing_key_does_not_evict() {
        let mut cache = ResponseCache::new(&config(60, 2));
        let key_a = ResponseCache::key_for(&request("a/m1", "a"));
        let key_b = ResponseCache::key_for(&request("a/m1", "b"));

        cache.insert(key_a, response("resp-a"));
        cache.insert(key_b, response("resp-b"));
        cache.insert(key_a, response("resp-a-2"));

        assert_eq!(cache.get(key_a).unwrap().id, "resp-a-2");
        assert!(cache.get(key_b).is_some());
    }
}
