//! Durable, cross-process backing store for cumulative usage/cost stats,
//! backed by a single SQLite file. Entirely optional: a `Router` built
//! without `[persistence]` in its config never touches this module and
//! behaves exactly as it always has (in-memory only, reset on restart).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use rusqlite::Connection;

use crate::UsageStats;
use rp_core::Usage;

/// A cumulative usage/cost delta for one "provider/model" key, enqueued to
/// the background writer thread after every completed request/chunk.
struct UsageEvent {
    key: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    cost_usd: Option<f64>,
}

/// Durable, cross-process backing store for cumulative usage/cost stats.
/// Writes are handed off to a dedicated background thread -- SQLite only
/// supports one writer at a time, and this keeps the request-handling
/// path from ever blocking on file I/O. Reads (`snapshot`/`load_all`)
/// open a short-lived connection of their own; WAL mode lets them proceed
/// concurrently with the writer, including from another process sharing
/// the same file.
pub struct Persistence {
    path: PathBuf,
    tx: mpsc::Sender<UsageEvent>,
}

fn init_connection(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS usage_stats (
            key TEXT PRIMARY KEY,
            requests INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0.0
        )",
    )?;
    Ok(conn)
}

fn read_all(conn: &Connection) -> rusqlite::Result<HashMap<String, UsageStats>> {
    let mut stmt = conn.prepare(
        "SELECT key, requests, prompt_tokens, completion_tokens, cost_usd FROM usage_stats",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            UsageStats {
                requests: row.get::<_, i64>(1)? as u64,
                prompt_tokens: row.get::<_, i64>(2)? as u64,
                completion_tokens: row.get::<_, i64>(3)? as u64,
                cost_usd: row.get(4)?,
            },
        ))
    })?;
    rows.collect()
}

impl Persistence {
    /// Open (creating if needed) the SQLite database at `path`, ensure its
    /// schema, and start the background writer thread. Fails if the file
    /// can't be opened/created or the schema can't be ensured -- the
    /// caller is expected to treat this as a soft failure (log a warning
    /// and run without persistence) rather than refusing to start.
    pub fn open(path: impl Into<PathBuf>) -> rusqlite::Result<Self> {
        let path = path.into();
        // Ensure the file/schema exist and are writable up front, so a bad
        // path fails loudly here rather than only on the first write from
        // the (fire-and-forget) background thread.
        init_connection(&path)?;

        let (tx, rx) = mpsc::channel::<UsageEvent>();
        let writer_path = path.clone();
        std::thread::Builder::new()
            .name("rp-router-persistence-writer".to_string())
            .spawn(move || {
                let conn = match init_connection(&writer_path) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!(
                            "persistence writer thread failed to open {}: {e}",
                            writer_path.display()
                        );
                        return;
                    }
                };
                while let Ok(event) = rx.recv() {
                    let result = conn.execute(
                        "INSERT INTO usage_stats (key, requests, prompt_tokens, completion_tokens, cost_usd)
                         VALUES (?1, 1, ?2, ?3, ?4)
                         ON CONFLICT(key) DO UPDATE SET
                             requests = requests + 1,
                             prompt_tokens = prompt_tokens + ?2,
                             completion_tokens = completion_tokens + ?3,
                             cost_usd = cost_usd + ?4",
                        rusqlite::params![
                            event.key,
                            event.prompt_tokens,
                            event.completion_tokens,
                            event.cost_usd.unwrap_or(0.0),
                        ],
                    );
                    if let Err(e) = result {
                        tracing::warn!("failed to persist usage event: {e}");
                    }
                }
            })
            .expect("failed to spawn persistence writer thread");

        Ok(Self { path, tx })
    }

    /// Enqueue a usage delta for the background writer to persist. Never
    /// blocks on file I/O; silently drops the event (with a log line) if
    /// the writer thread has died, since usage persistence is best-effort
    /// and must never fail the request it's instrumenting.
    pub fn record(&self, key: &str, usage: &Usage, cost_usd: Option<f64>) {
        let event = UsageEvent {
            key: key.to_string(),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cost_usd,
        };
        if self.tx.send(event).is_err() {
            tracing::warn!("persistence writer thread is gone; dropping usage event");
        }
    }

    /// Synchronous full read of every persisted "provider/model" row, used
    /// once at startup to seed the in-memory cache.
    pub fn load_all(&self) -> rusqlite::Result<HashMap<String, UsageStats>> {
        let conn = Connection::open(&self.path)?;
        read_all(&conn)
    }

    /// Async full read for `GET /v1/usage`, run on a blocking thread pool
    /// since it's synchronous file I/O. Reflects every process's writes
    /// that have reached the shared file, not just this one's -- this is
    /// what makes usage reporting correct across multiple router
    /// processes sharing one database.
    pub async fn snapshot(&self) -> rusqlite::Result<HashMap<String, UsageStats>> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&path)?;
            read_all(&conn)
        })
        .await
        .expect("persistence snapshot task panicked")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    /// A fresh SQLite file path, guaranteed empty even if a prior test
    /// run left one behind at the same name -- the per-process counter
    /// restarts at 0 on every `cargo test` invocation, so without this a
    /// leftover file from an earlier run would silently carry stale rows
    /// into a test that assumes a brand-new database.
    fn unique_temp_path(label: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "rp_router_persistence_test_{label}_{}.db",
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
        path
    }

    /// The writer thread is asynchronous relative to `record()`; give it a
    /// moment to drain before asserting on what landed in the DB.
    fn wait_for_writer() {
        std::thread::sleep(Duration::from_millis(50));
    }

    #[test]
    fn open_creates_the_file_and_an_empty_table() {
        let path = unique_temp_path("open");
        let persistence = Persistence::open(&path).unwrap();
        assert!(path.exists());
        assert!(persistence.load_all().unwrap().is_empty());
    }

    #[test]
    fn record_persists_a_new_key() {
        let path = unique_temp_path("record_new");
        let persistence = Persistence::open(&path).unwrap();

        persistence.record(
            "anthropic/m1",
            &Usage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            },
            Some(0.5),
        );
        wait_for_writer();

        let stats = persistence.load_all().unwrap();
        let entry = &stats["anthropic/m1"];
        assert_eq!(entry.requests, 1);
        assert_eq!(entry.prompt_tokens, 100);
        assert_eq!(entry.completion_tokens, 50);
        assert_eq!(entry.cost_usd, 0.5);
    }

    #[test]
    fn record_accumulates_across_multiple_calls() {
        let path = unique_temp_path("record_accumulate");
        let persistence = Persistence::open(&path).unwrap();

        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        };
        persistence.record("anthropic/m1", &usage, Some(0.1));
        persistence.record("anthropic/m1", &usage, Some(0.1));
        wait_for_writer();

        let stats = persistence.load_all().unwrap();
        let entry = &stats["anthropic/m1"];
        assert_eq!(entry.requests, 2);
        assert_eq!(entry.prompt_tokens, 20);
        assert_eq!(entry.completion_tokens, 10);
        assert!((entry.cost_usd - 0.2).abs() < 1e-9);
    }

    #[test]
    fn record_with_no_cost_leaves_cost_usd_at_zero() {
        let path = unique_temp_path("record_no_cost");
        let persistence = Persistence::open(&path).unwrap();

        persistence.record(
            "anthropic/m1",
            &Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            None,
        );
        wait_for_writer();

        let stats = persistence.load_all().unwrap();
        assert_eq!(stats["anthropic/m1"].cost_usd, 0.0);
    }

    #[test]
    fn record_keys_are_independent() {
        let path = unique_temp_path("record_independent");
        let persistence = Persistence::open(&path).unwrap();
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        };

        persistence.record("anthropic/m1", &usage, Some(1.0));
        persistence.record("openai/m2", &usage, Some(2.0));
        wait_for_writer();

        let stats = persistence.load_all().unwrap();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats["anthropic/m1"].cost_usd, 1.0);
        assert_eq!(stats["openai/m2"].cost_usd, 2.0);
    }

    #[test]
    fn a_fresh_process_reopening_the_same_file_sees_prior_writes() {
        // Simulates a restart (or a second process): a brand new
        // Persistence handle pointed at the same file must see everything
        // the first handle wrote, without that handle needing to still be
        // alive.
        let path = unique_temp_path("reopen");
        {
            let persistence = Persistence::open(&path).unwrap();
            persistence.record(
                "anthropic/m1",
                &Usage {
                    prompt_tokens: 42,
                    completion_tokens: 7,
                    total_tokens: 49,
                },
                Some(0.9),
            );
            wait_for_writer();
        }

        let reopened = Persistence::open(&path).unwrap();
        let stats = reopened.load_all().unwrap();
        assert_eq!(stats["anthropic/m1"].requests, 1);
        assert_eq!(stats["anthropic/m1"].prompt_tokens, 42);
    }

    #[tokio::test]
    async fn snapshot_reflects_the_same_data_as_load_all() {
        let path = unique_temp_path("snapshot");
        let persistence = Persistence::open(&path).unwrap();
        persistence.record(
            "anthropic/m1",
            &Usage {
                prompt_tokens: 5,
                completion_tokens: 5,
                total_tokens: 10,
            },
            Some(0.3),
        );
        wait_for_writer();

        let snapshot = persistence.snapshot().await.unwrap();
        assert_eq!(snapshot["anthropic/m1"].requests, 1);
        assert_eq!(snapshot["anthropic/m1"].cost_usd, 0.3);
    }

    #[test]
    fn open_fails_when_the_parent_directory_does_not_exist() {
        // SQLite can create the database file itself, but not the
        // directories leading up to it.
        let path = std::env::temp_dir()
            .join("rp_router_persistence_test_nonexistent_parent_dir")
            .join("usage.db");
        assert!(Persistence::open(&path).is_err());
    }
}
