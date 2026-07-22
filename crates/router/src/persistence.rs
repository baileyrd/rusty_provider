//! Durable, cross-process backing store for cumulative usage/cost stats
//! and client spend budgets, backed by either a single SQLite file (one
//! host, or a shared local volume) or a shared Postgres database (usable
//! across multiple hosts). Entirely optional: a `Router` built without
//! `[persistence]` in its config never touches this module and behaves
//! exactly as it always has (in-memory only, reset on restart, never
//! shared across processes).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use rusqlite::{Connection, OptionalExtension};
use thiserror::Error;

use crate::config::PostgresTlsMode;
use crate::UsageStats;
use rp_core::Usage;

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    Postgres(#[from] tokio_postgres::Error),
    #[error("failed to set up TLS for the postgres connection: {0}")]
    Tls(String),
}

/// Where `Persistence::open` should connect to. Decoupled from
/// `PersistenceConfig` (the TOML-facing type) so this module doesn't need
/// to know about config parsing, env var resolution, etc. -- the caller
/// resolves all of that first.
pub enum PersistenceTarget {
    Sqlite(PathBuf),
    Postgres { url: String, tls: PostgresTlsMode },
}

/// A cumulative usage/cost delta for one "provider/model" key, enqueued to
/// the SQLite backend's background writer thread after every completed
/// request/chunk.
struct UsageEvent {
    key: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    cost_usd: Option<f64>,
}

// BIGINT (not INTEGER) for the integer columns: SQLite's type affinity
// doesn't care either way, but Postgres enforces the declared width, and
// every value bound to these columns is passed as an `i64` from Rust.
const SCHEMA_SQL: &str = "
    CREATE TABLE IF NOT EXISTS usage_stats (
        key TEXT PRIMARY KEY,
        requests BIGINT NOT NULL DEFAULT 0,
        prompt_tokens BIGINT NOT NULL DEFAULT 0,
        completion_tokens BIGINT NOT NULL DEFAULT 0,
        cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0
    );
    CREATE TABLE IF NOT EXISTS client_spend (
        client_name TEXT PRIMARY KEY,
        period_key BIGINT NOT NULL,
        spent_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0
    );
";

/// Durable, cross-process backing store for cumulative usage/cost stats
/// and client spend, dispatching to whichever backend `open` was pointed
/// at. Every method is backend-agnostic from the caller's side (`Router`
/// never needs to know which one is active).
pub struct Persistence {
    backend: Backend,
}

enum Backend {
    Sqlite(SqliteBackend),
    Postgres(PostgresBackend),
}

impl Persistence {
    /// Open (creating tables if needed) the configured backend. Fails if
    /// the database can't be reached/created -- the caller is expected to
    /// treat this as a soft failure (log a warning and run without
    /// persistence) rather than refusing to start.
    pub async fn open(target: PersistenceTarget) -> Result<Self, PersistenceError> {
        let backend = match target {
            PersistenceTarget::Sqlite(path) => Backend::Sqlite(SqliteBackend::open(path)?),
            PersistenceTarget::Postgres { url, tls } => {
                Backend::Postgres(PostgresBackend::connect(&url, tls).await?)
            }
        };
        Ok(Self { backend })
    }

    /// Enqueue a usage delta for the backend to persist. Never blocks the
    /// caller on I/O; best-effort (a failure is logged, not surfaced),
    /// since usage persistence must never fail the request it's
    /// instrumenting.
    pub fn record(&self, key: &str, usage: &Usage, cost_usd: Option<f64>) {
        match &self.backend {
            Backend::Sqlite(b) => b.record(key, usage, cost_usd),
            Backend::Postgres(b) => b.record(key, usage, cost_usd),
        }
    }

    /// Full read of every persisted "provider/model" row, used once at
    /// startup to seed the in-memory cache.
    pub async fn load_all(&self) -> Result<HashMap<String, UsageStats>, PersistenceError> {
        match &self.backend {
            Backend::Sqlite(b) => b.load_all().await,
            Backend::Postgres(b) => b.load_all().await,
        }
    }

    /// Full read for `GET /v1/usage`, reflecting every process's writes
    /// that have reached the shared backend, not just this one's -- this
    /// is what makes usage reporting correct across multiple router
    /// processes (or, with the Postgres backend, multiple hosts) sharing
    /// one database.
    pub async fn snapshot(&self) -> Result<HashMap<String, UsageStats>, PersistenceError> {
        match &self.backend {
            Backend::Sqlite(b) => b.snapshot().await,
            Backend::Postgres(b) => b.snapshot().await,
        }
    }

    /// Adds `cost_usd` to `client_name`'s spend, resetting to just this
    /// amount if `period_key` doesn't match whatever's already stored
    /// (i.e. a budget period rollover). Fire-and-forget, like `record`.
    pub fn record_client_spend(&self, client_name: &str, period_key: i64, cost_usd: f64) {
        match &self.backend {
            Backend::Sqlite(b) => b.record_client_spend(client_name, period_key, cost_usd),
            Backend::Postgres(b) => b.record_client_spend(client_name, period_key, cost_usd),
        }
    }

    /// Reads back `client_name`'s persisted `(period_key, spent_usd)`, or
    /// `None` if nothing has ever been recorded for it. The caller is
    /// responsible for comparing `period_key` against the current one --
    /// a stale row (from a past period) still reads back its old value
    /// here; rollover is only ever applied by `record_client_spend`.
    pub async fn client_spend(
        &self,
        client_name: &str,
    ) -> Result<Option<(i64, f64)>, PersistenceError> {
        match &self.backend {
            Backend::Sqlite(b) => b.client_spend(client_name).await,
            Backend::Postgres(b) => b.client_spend(client_name).await,
        }
    }

    /// Unconditionally zeroes `client_name`'s spend for `period_key`,
    /// regardless of what's currently stored -- unlike
    /// `record_client_spend`, this doesn't add to an existing value for a
    /// matching period, it always resets. Fire-and-forget, like `record`
    /// and `record_client_spend`; used by the admin API's manual budget
    /// reset.
    pub fn reset_client_spend(&self, client_name: &str, period_key: i64) {
        match &self.backend {
            Backend::Sqlite(b) => b.reset_client_spend(client_name, period_key),
            Backend::Postgres(b) => b.reset_client_spend(client_name, period_key),
        }
    }

    /// A trivial round trip (`SELECT 1`) confirming the backend is
    /// actually reachable right now, for `GET /ready`. Deliberately
    /// cheaper than `snapshot`/`load_all` -- readiness only needs to know
    /// the connection works, not read back any real data.
    pub async fn ping(&self) -> Result<(), PersistenceError> {
        match &self.backend {
            Backend::Sqlite(b) => b.ping().await,
            Backend::Postgres(b) => b.ping().await,
        }
    }
}

fn read_usage_rows<E>(
    rows: impl Iterator<Item = Result<(String, i64, i64, i64, f64), E>>,
) -> Result<HashMap<String, UsageStats>, E> {
    rows.map(|row| {
        row.map(
            |(key, requests, prompt_tokens, completion_tokens, cost_usd)| {
                (
                    key,
                    UsageStats {
                        requests: requests as u64,
                        prompt_tokens: prompt_tokens as u64,
                        completion_tokens: completion_tokens as u64,
                        cost_usd,
                    },
                )
            },
        )
    })
    .collect()
}

// --- SQLite backend --------------------------------------------------------------

/// Writes are handed off to a dedicated background thread -- SQLite only
/// supports one writer at a time, and this keeps the request-handling
/// path from ever blocking on file I/O. Reads open a short-lived
/// connection of their own; WAL mode lets them proceed concurrently with
/// the writer, including from another process sharing the same file.
struct SqliteBackend {
    path: PathBuf,
    tx: mpsc::Sender<SqliteWrite>,
}

enum SqliteWrite {
    Usage(UsageEvent),
    ClientSpend {
        client_name: String,
        period_key: i64,
        cost_usd: f64,
    },
    ResetClientSpend {
        client_name: String,
        period_key: i64,
    },
}

fn init_connection(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.execute_batch(SCHEMA_SQL)?;
    Ok(conn)
}

fn sqlite_read_all(conn: &Connection) -> rusqlite::Result<HashMap<String, UsageStats>> {
    let mut stmt = conn.prepare(
        "SELECT key, requests, prompt_tokens, completion_tokens, cost_usd FROM usage_stats",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, f64>(4)?,
        ))
    })?;
    read_usage_rows(rows)
}

impl SqliteBackend {
    fn open(path: PathBuf) -> rusqlite::Result<Self> {
        // Ensure the file/schema exist and are writable up front, so a bad
        // path fails loudly here rather than only on the first write from
        // the (fire-and-forget) background thread.
        init_connection(&path)?;

        let (tx, rx) = mpsc::channel::<SqliteWrite>();
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
                    let result = match &event {
                        SqliteWrite::Usage(event) => conn.execute(
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
                        ),
                        SqliteWrite::ClientSpend { client_name, period_key, cost_usd } => conn.execute(
                            "INSERT INTO client_spend (client_name, period_key, spent_usd)
                             VALUES (?1, ?2, ?3)
                             ON CONFLICT(client_name) DO UPDATE SET
                                 period_key = ?2,
                                 spent_usd = CASE WHEN client_spend.period_key = ?2 THEN client_spend.spent_usd + ?3 ELSE ?3 END",
                            rusqlite::params![client_name, period_key, cost_usd],
                        ),
                        SqliteWrite::ResetClientSpend { client_name, period_key } => conn.execute(
                            "INSERT INTO client_spend (client_name, period_key, spent_usd)
                             VALUES (?1, ?2, 0.0)
                             ON CONFLICT(client_name) DO UPDATE SET
                                 period_key = ?2,
                                 spent_usd = 0.0",
                            rusqlite::params![client_name, period_key],
                        ),
                    };
                    if let Err(e) = result {
                        tracing::warn!("failed to persist event to sqlite: {e}");
                    }
                }
            })
            .expect("failed to spawn persistence writer thread");

        Ok(Self { path, tx })
    }

    fn record(&self, key: &str, usage: &Usage, cost_usd: Option<f64>) {
        let event = SqliteWrite::Usage(UsageEvent {
            key: key.to_string(),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cost_usd,
        });
        if self.tx.send(event).is_err() {
            tracing::warn!("persistence writer thread is gone; dropping usage event");
        }
    }

    async fn load_all(&self) -> Result<HashMap<String, UsageStats>, PersistenceError> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&path)?;
            sqlite_read_all(&conn)
        })
        .await
        .expect("persistence load_all task panicked")
        .map_err(PersistenceError::from)
    }

    async fn snapshot(&self) -> Result<HashMap<String, UsageStats>, PersistenceError> {
        self.load_all().await
    }

    fn record_client_spend(&self, client_name: &str, period_key: i64, cost_usd: f64) {
        let event = SqliteWrite::ClientSpend {
            client_name: client_name.to_string(),
            period_key,
            cost_usd,
        };
        if self.tx.send(event).is_err() {
            tracing::warn!("persistence writer thread is gone; dropping client spend event");
        }
    }

    fn reset_client_spend(&self, client_name: &str, period_key: i64) {
        let event = SqliteWrite::ResetClientSpend {
            client_name: client_name.to_string(),
            period_key,
        };
        if self.tx.send(event).is_err() {
            tracing::warn!("persistence writer thread is gone; dropping client spend reset");
        }
    }

    async fn client_spend(
        &self,
        client_name: &str,
    ) -> Result<Option<(i64, f64)>, PersistenceError> {
        let path = self.path.clone();
        let client_name = client_name.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&path)?;
            conn.query_row(
                "SELECT period_key, spent_usd FROM client_spend WHERE client_name = ?1",
                rusqlite::params![client_name],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)),
            )
            .optional()
        })
        .await
        .expect("persistence client_spend task panicked")
        .map_err(PersistenceError::from)
    }

    async fn ping(&self) -> Result<(), PersistenceError> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = Connection::open(&path)?;
            conn.query_row("SELECT 1", [], |_| Ok(()))
        })
        .await
        .expect("persistence ping task panicked")
        .map_err(PersistenceError::from)
    }
}

// --- Postgres backend -------------------------------------------------------------

/// `tokio_postgres::Client` is safe to use concurrently (it pipelines
/// queries over one connection internally), so unlike the SQLite backend
/// this needs no hand-rolled writer thread/queue -- writes just spawn a
/// short-lived task that awaits the query without blocking the caller.
struct PostgresBackend {
    client: std::sync::Arc<tokio_postgres::Client>,
}

/// Builds a `rustls::ClientConfig` trusting the host's native root
/// certificate store -- the same set `reqwest` trusts for outbound
/// provider calls, so operators don't need a separate CA bundle just for
/// the Postgres connection. Explicitly bound to the `ring` crypto
/// provider (matching the one already pulled in by `reqwest`'s
/// `rustls-tls-native-roots` feature) rather than relying on a
/// process-wide default, so this doesn't race with -- or require -- any
/// other code installing one first.
fn build_rustls_client_config() -> Result<rustls::ClientConfig, String> {
    let mut roots = rustls::RootCertStore::empty();
    let loaded = rustls_native_certs::load_native_certs();
    for err in &loaded.errors {
        tracing::warn!("error loading a native root certificate: {err}");
    }
    for cert in loaded.certs {
        if let Err(e) = roots.add(cert) {
            tracing::warn!("failed to add a native root certificate: {e}");
        }
    }
    if roots.is_empty() {
        return Err("no usable root certificates found in the native trust store".to_string());
    }

    let builder = rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| format!("failed to configure TLS protocol versions: {e}"))?;
    Ok(builder.with_root_certificates(roots).with_no_client_auth())
}

impl PostgresBackend {
    async fn connect(url: &str, tls: PostgresTlsMode) -> Result<Self, PersistenceError> {
        let mut config: tokio_postgres::Config = url.parse()?;
        match tls {
            PostgresTlsMode::Disable => Self::connect_with(config, tokio_postgres::NoTls).await,
            PostgresTlsMode::Require => {
                // Set explicitly (rather than trusting the connection
                // string to say `sslmode=require`) so this mode always
                // means what it says: refuse to fall back to plaintext,
                // even against a server that doesn't support TLS.
                config.ssl_mode(tokio_postgres::config::SslMode::Require);
                let tls_config = build_rustls_client_config().map_err(PersistenceError::Tls)?;
                let tls = tokio_postgres_rustls::MakeRustlsConnect::new(tls_config);
                Self::connect_with(config, tls).await
            }
        }
    }

    async fn connect_with<T>(
        config: tokio_postgres::Config,
        tls: T,
    ) -> Result<Self, PersistenceError>
    where
        T: tokio_postgres::tls::MakeTlsConnect<tokio_postgres::Socket> + Send + 'static,
        T::TlsConnect: Send,
        T::Stream: Send,
        <T::TlsConnect as tokio_postgres::tls::TlsConnect<tokio_postgres::Socket>>::Future: Send,
    {
        let (client, connection) = config.connect(tls).await?;
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::error!("postgres persistence connection closed with an error: {e}");
            }
        });
        // `CREATE TABLE IF NOT EXISTS` isn't safe against concurrent first
        // creation -- two sessions can both pass the existence check and
        // then race inserting into pg_catalog, one losing with a spurious
        // duplicate-key error. An advisory lock (an arbitrary constant key,
        // scoped to this session and auto-released on disconnect) makes
        // concurrent `Persistence::open` calls against a fresh database
        // create the schema one at a time instead.
        client
            .batch_execute("SELECT pg_advisory_lock(72_727_100)")
            .await?;
        let schema_result = client.batch_execute(SCHEMA_SQL).await;
        client
            .batch_execute("SELECT pg_advisory_unlock(72_727_100)")
            .await?;
        schema_result?;
        Ok(Self {
            client: std::sync::Arc::new(client),
        })
    }

    fn record(&self, key: &str, usage: &Usage, cost_usd: Option<f64>) {
        let client = self.client.clone();
        let key = key.to_string();
        let (prompt_tokens, completion_tokens, cost_usd) = (
            usage.prompt_tokens as i64,
            usage.completion_tokens as i64,
            cost_usd.unwrap_or(0.0),
        );
        tokio::spawn(async move {
            let result = client
                .execute(
                    "INSERT INTO usage_stats (key, requests, prompt_tokens, completion_tokens, cost_usd)
                     VALUES ($1, 1, $2, $3, $4)
                     ON CONFLICT (key) DO UPDATE SET
                         requests = usage_stats.requests + 1,
                         prompt_tokens = usage_stats.prompt_tokens + $2,
                         completion_tokens = usage_stats.completion_tokens + $3,
                         cost_usd = usage_stats.cost_usd + $4",
                    &[&key, &prompt_tokens, &completion_tokens, &cost_usd],
                )
                .await;
            if let Err(e) = result {
                tracing::warn!("failed to persist usage event to postgres: {e}");
            }
        });
    }

    async fn load_all(&self) -> Result<HashMap<String, UsageStats>, PersistenceError> {
        let rows = self
            .client
            .query(
                "SELECT key, requests, prompt_tokens, completion_tokens, cost_usd FROM usage_stats",
                &[],
            )
            .await?;
        Ok(read_usage_rows(rows.iter().map(|row| {
            Ok::<_, tokio_postgres::Error>((
                row.get::<_, String>(0),
                row.get::<_, i64>(1),
                row.get::<_, i64>(2),
                row.get::<_, i64>(3),
                row.get::<_, f64>(4),
            ))
        }))?)
    }

    async fn snapshot(&self) -> Result<HashMap<String, UsageStats>, PersistenceError> {
        self.load_all().await
    }

    fn record_client_spend(&self, client_name: &str, period_key: i64, cost_usd: f64) {
        let client = self.client.clone();
        let client_name = client_name.to_string();
        tokio::spawn(async move {
            let result = client
                .execute(
                    "INSERT INTO client_spend (client_name, period_key, spent_usd)
                     VALUES ($1, $2, $3)
                     ON CONFLICT (client_name) DO UPDATE SET
                         period_key = $2,
                         spent_usd = CASE WHEN client_spend.period_key = $2 THEN client_spend.spent_usd + $3 ELSE $3 END",
                    &[&client_name, &period_key, &cost_usd],
                )
                .await;
            if let Err(e) = result {
                tracing::warn!("failed to persist client spend event to postgres: {e}");
            }
        });
    }

    fn reset_client_spend(&self, client_name: &str, period_key: i64) {
        let client = self.client.clone();
        let client_name = client_name.to_string();
        tokio::spawn(async move {
            let result = client
                .execute(
                    "INSERT INTO client_spend (client_name, period_key, spent_usd)
                     VALUES ($1, $2, 0.0)
                     ON CONFLICT (client_name) DO UPDATE SET
                         period_key = $2,
                         spent_usd = 0.0",
                    &[&client_name, &period_key],
                )
                .await;
            if let Err(e) = result {
                tracing::warn!("failed to persist client spend reset to postgres: {e}");
            }
        });
    }

    async fn client_spend(
        &self,
        client_name: &str,
    ) -> Result<Option<(i64, f64)>, PersistenceError> {
        let row = self
            .client
            .query_opt(
                "SELECT period_key, spent_usd FROM client_spend WHERE client_name = $1",
                &[&client_name],
            )
            .await?;
        Ok(row.map(|row| (row.get::<_, i64>(0), row.get::<_, f64>(1))))
    }

    async fn ping(&self) -> Result<(), PersistenceError> {
        self.client.execute("SELECT 1", &[]).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- rustls TLS config -------------------------------------------------------

    #[test]
    fn build_rustls_client_config_succeeds_with_the_hosts_native_roots() {
        // Doesn't touch the network -- just confirms the native trust
        // store loads and produces a usable `rustls::ClientConfig`,
        // independent of whether a TLS-enabled Postgres is reachable in
        // this environment.
        build_rustls_client_config().expect("native root certificates should load");
    }
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

    async fn open_sqlite(path: &Path) -> Persistence {
        Persistence::open(PersistenceTarget::Sqlite(path.to_path_buf()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn open_creates_the_file_and_an_empty_table() {
        let path = unique_temp_path("open");
        let persistence = open_sqlite(&path).await;
        assert!(path.exists());
        assert!(persistence.load_all().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn ping_succeeds_against_a_reachable_sqlite_file() {
        let path = unique_temp_path("ping");
        let persistence = open_sqlite(&path).await;
        persistence.ping().await.unwrap();
    }

    #[tokio::test]
    async fn record_persists_a_new_key() {
        let path = unique_temp_path("record_new");
        let persistence = open_sqlite(&path).await;

        persistence.record(
            "anthropic/m1",
            &Usage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                cached_tokens: None,
                cache_creation_tokens: None,
            },
            Some(0.5),
        );
        wait_for_writer();

        let stats = persistence.load_all().await.unwrap();
        let entry = &stats["anthropic/m1"];
        assert_eq!(entry.requests, 1);
        assert_eq!(entry.prompt_tokens, 100);
        assert_eq!(entry.completion_tokens, 50);
        assert_eq!(entry.cost_usd, 0.5);
    }

    #[tokio::test]
    async fn record_accumulates_across_multiple_calls() {
        let path = unique_temp_path("record_accumulate");
        let persistence = open_sqlite(&path).await;

        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cached_tokens: None,
            cache_creation_tokens: None,
        };
        persistence.record("anthropic/m1", &usage, Some(0.1));
        persistence.record("anthropic/m1", &usage, Some(0.1));
        wait_for_writer();

        let stats = persistence.load_all().await.unwrap();
        let entry = &stats["anthropic/m1"];
        assert_eq!(entry.requests, 2);
        assert_eq!(entry.prompt_tokens, 20);
        assert_eq!(entry.completion_tokens, 10);
        assert!((entry.cost_usd - 0.2).abs() < 1e-9);
    }

    #[tokio::test]
    async fn record_with_no_cost_leaves_cost_usd_at_zero() {
        let path = unique_temp_path("record_no_cost");
        let persistence = open_sqlite(&path).await;

        persistence.record(
            "anthropic/m1",
            &Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cached_tokens: None,
                cache_creation_tokens: None,
            },
            None,
        );
        wait_for_writer();

        let stats = persistence.load_all().await.unwrap();
        assert_eq!(stats["anthropic/m1"].cost_usd, 0.0);
    }

    #[tokio::test]
    async fn record_keys_are_independent() {
        let path = unique_temp_path("record_independent");
        let persistence = open_sqlite(&path).await;
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cached_tokens: None,
            cache_creation_tokens: None,
        };

        persistence.record("anthropic/m1", &usage, Some(1.0));
        persistence.record("openai/m2", &usage, Some(2.0));
        wait_for_writer();

        let stats = persistence.load_all().await.unwrap();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats["anthropic/m1"].cost_usd, 1.0);
        assert_eq!(stats["openai/m2"].cost_usd, 2.0);
    }

    #[tokio::test]
    async fn a_fresh_process_reopening_the_same_file_sees_prior_writes() {
        // Simulates a restart (or a second process): a brand new
        // Persistence handle pointed at the same file must see everything
        // the first handle wrote, without that handle needing to still be
        // alive.
        let path = unique_temp_path("reopen");
        {
            let persistence = open_sqlite(&path).await;
            persistence.record(
                "anthropic/m1",
                &Usage {
                    prompt_tokens: 42,
                    completion_tokens: 7,
                    total_tokens: 49,
                    cached_tokens: None,
                    cache_creation_tokens: None,
                },
                Some(0.9),
            );
            wait_for_writer();
        }

        let reopened = open_sqlite(&path).await;
        let stats = reopened.load_all().await.unwrap();
        assert_eq!(stats["anthropic/m1"].requests, 1);
        assert_eq!(stats["anthropic/m1"].prompt_tokens, 42);
    }

    #[tokio::test]
    async fn snapshot_reflects_the_same_data_as_load_all() {
        let path = unique_temp_path("snapshot");
        let persistence = open_sqlite(&path).await;
        persistence.record(
            "anthropic/m1",
            &Usage {
                prompt_tokens: 5,
                completion_tokens: 5,
                total_tokens: 10,
                cached_tokens: None,
                cache_creation_tokens: None,
            },
            Some(0.3),
        );
        wait_for_writer();

        let snapshot = persistence.snapshot().await.unwrap();
        assert_eq!(snapshot["anthropic/m1"].requests, 1);
        assert_eq!(snapshot["anthropic/m1"].cost_usd, 0.3);
    }

    #[tokio::test]
    async fn open_fails_when_the_parent_directory_does_not_exist() {
        // SQLite can create the database file itself, but not the
        // directories leading up to it.
        let path = std::env::temp_dir()
            .join("rp_router_persistence_test_nonexistent_parent_dir")
            .join("usage.db");
        assert!(Persistence::open(PersistenceTarget::Sqlite(path))
            .await
            .is_err());
    }

    // --- client_spend (sqlite) ----------------------------------------------------

    #[tokio::test]
    async fn client_spend_is_none_for_an_unrecorded_client() {
        let path = unique_temp_path("client_spend_none");
        let persistence = open_sqlite(&path).await;
        assert_eq!(persistence.client_spend("acme").await.unwrap(), None);
    }

    #[tokio::test]
    async fn record_client_spend_persists_a_new_client() {
        let path = unique_temp_path("client_spend_new");
        let persistence = open_sqlite(&path).await;
        persistence.record_client_spend("acme", 100, 5.0);
        wait_for_writer();
        assert_eq!(
            persistence.client_spend("acme").await.unwrap(),
            Some((100, 5.0))
        );
    }

    #[tokio::test]
    async fn record_client_spend_accumulates_within_the_same_period() {
        let path = unique_temp_path("client_spend_accumulate");
        let persistence = open_sqlite(&path).await;
        persistence.record_client_spend("acme", 100, 5.0);
        persistence.record_client_spend("acme", 100, 3.0);
        wait_for_writer();
        assert_eq!(
            persistence.client_spend("acme").await.unwrap(),
            Some((100, 8.0))
        );
    }

    #[tokio::test]
    async fn record_client_spend_resets_on_a_new_period_key() {
        let path = unique_temp_path("client_spend_rollover");
        let persistence = open_sqlite(&path).await;
        persistence.record_client_spend("acme", 100, 5.0);
        persistence.record_client_spend("acme", 101, 2.0);
        wait_for_writer();
        assert_eq!(
            persistence.client_spend("acme").await.unwrap(),
            Some((101, 2.0))
        );
    }

    #[tokio::test]
    async fn client_spend_is_independent_per_client() {
        let path = unique_temp_path("client_spend_independent");
        let persistence = open_sqlite(&path).await;
        persistence.record_client_spend("acme", 100, 5.0);
        persistence.record_client_spend("globex", 100, 9.0);
        wait_for_writer();
        assert_eq!(
            persistence.client_spend("acme").await.unwrap(),
            Some((100, 5.0))
        );
        assert_eq!(
            persistence.client_spend("globex").await.unwrap(),
            Some((100, 9.0))
        );
    }

    #[tokio::test]
    async fn reset_client_spend_zeroes_an_existing_balance() {
        let path = unique_temp_path("client_spend_reset");
        let persistence = open_sqlite(&path).await;
        persistence.record_client_spend("acme", 100, 5.0);
        wait_for_writer();
        persistence.reset_client_spend("acme", 100);
        wait_for_writer();
        assert_eq!(
            persistence.client_spend("acme").await.unwrap(),
            Some((100, 0.0))
        );
    }

    #[tokio::test]
    async fn reset_client_spend_creates_a_zeroed_row_for_an_unrecorded_client() {
        let path = unique_temp_path("client_spend_reset_new");
        let persistence = open_sqlite(&path).await;
        persistence.reset_client_spend("acme", 100);
        wait_for_writer();
        assert_eq!(
            persistence.client_spend("acme").await.unwrap(),
            Some((100, 0.0))
        );
    }

    #[tokio::test]
    async fn reset_client_spend_does_not_affect_other_clients() {
        let path = unique_temp_path("client_spend_reset_independent");
        let persistence = open_sqlite(&path).await;
        persistence.record_client_spend("acme", 100, 5.0);
        persistence.record_client_spend("globex", 100, 9.0);
        wait_for_writer();
        persistence.reset_client_spend("acme", 100);
        wait_for_writer();
        assert_eq!(
            persistence.client_spend("globex").await.unwrap(),
            Some((100, 9.0))
        );
    }

    // --- postgres backend ----------------------------------------------------------
    //
    // Gated on TEST_POSTGRES_URL so contributors without a local Postgres
    // aren't blocked; see the CI workflow for how this is provided there.

    fn test_postgres_url() -> Option<String> {
        std::env::var("TEST_POSTGRES_URL").ok()
    }

    async fn open_postgres_for_test(label: &str) -> Option<Persistence> {
        let base_url = test_postgres_url()?;
        let persistence = Persistence::open(PersistenceTarget::Postgres {
            url: base_url,
            tls: PostgresTlsMode::Disable,
        })
        .await
        .expect("TEST_POSTGRES_URL should be reachable");
        // Each test gets its own key prefix rather than its own database
        // (creating/dropping databases per test is slow and needs
        // elevated privileges) -- scoped via a unique key/client_name
        // prefix instead, mirroring the SQLite tests' unique-file-per-test
        // isolation.
        let _ = label;
        Some(persistence)
    }

    /// Unlike the SQLite tests' per-test temp file, the Postgres test
    /// database is never cleaned up -- rows from a previous `cargo test`
    /// invocation stick around. A per-process counter alone would collide
    /// with a prior run's (it also restarts at 0 each invocation), so the
    /// process id is folded in too.
    fn unique_key(label: &str) -> String {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        format!(
            "pg_test_{label}_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )
    }

    #[tokio::test]
    async fn postgres_ping_succeeds_against_a_reachable_database() {
        let Some(persistence) = open_postgres_for_test("ping").await else {
            eprintln!("skipping: TEST_POSTGRES_URL not set");
            return;
        };
        persistence.ping().await.unwrap();
    }

    #[tokio::test]
    async fn postgres_record_and_load_all_round_trip() {
        let Some(persistence) = open_postgres_for_test("round_trip").await else {
            eprintln!("skipping: TEST_POSTGRES_URL not set");
            return;
        };
        let key = unique_key("usage");
        persistence.record(
            &key,
            &Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cached_tokens: None,
                cache_creation_tokens: None,
            },
            Some(0.25),
        );
        // Postgres writes are fire-and-forget via tokio::spawn, not a
        // background thread -- give the spawned task a moment to land.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = persistence.load_all().await.unwrap();
        let entry = &stats[&key];
        assert_eq!(entry.requests, 1);
        assert_eq!(entry.prompt_tokens, 10);
        assert_eq!(entry.completion_tokens, 5);
        assert_eq!(entry.cost_usd, 0.25);
    }

    #[tokio::test]
    async fn postgres_record_accumulates_across_multiple_calls() {
        let Some(persistence) = open_postgres_for_test("accumulate").await else {
            eprintln!("skipping: TEST_POSTGRES_URL not set");
            return;
        };
        let key = unique_key("usage");
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cached_tokens: None,
            cache_creation_tokens: None,
        };
        persistence.record(&key, &usage, Some(0.1));
        persistence.record(&key, &usage, Some(0.1));
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = persistence.load_all().await.unwrap();
        let entry = &stats[&key];
        assert_eq!(entry.requests, 2);
        assert_eq!(entry.prompt_tokens, 20);
        assert!((entry.cost_usd - 0.2).abs() < 1e-9);
    }

    #[tokio::test]
    async fn postgres_two_handles_to_the_same_database_see_each_others_writes() {
        // Simulates two hosts/processes sharing one Postgres database.
        let Some(url) = test_postgres_url() else {
            eprintln!("skipping: TEST_POSTGRES_URL not set");
            return;
        };
        let handle_a = Persistence::open(PersistenceTarget::Postgres {
            url: url.clone(),
            tls: PostgresTlsMode::Disable,
        })
        .await
        .unwrap();
        let handle_b = Persistence::open(PersistenceTarget::Postgres {
            url,
            tls: PostgresTlsMode::Disable,
        })
        .await
        .unwrap();

        let key = unique_key("shared");
        handle_a.record(
            &key,
            &Usage {
                prompt_tokens: 7,
                completion_tokens: 3,
                total_tokens: 10,
                cached_tokens: None,
                cache_creation_tokens: None,
            },
            Some(0.4),
        );
        tokio::time::sleep(Duration::from_millis(200)).await;

        let snapshot = handle_b.snapshot().await.unwrap();
        assert_eq!(snapshot[&key].requests, 1);
        assert_eq!(snapshot[&key].cost_usd, 0.4);
    }

    #[tokio::test]
    async fn postgres_client_spend_round_trips_and_rolls_over() {
        let Some(persistence) = open_postgres_for_test("client_spend").await else {
            eprintln!("skipping: TEST_POSTGRES_URL not set");
            return;
        };
        let client_name = unique_key("client");

        assert_eq!(persistence.client_spend(&client_name).await.unwrap(), None);

        persistence.record_client_spend(&client_name, 100, 5.0);
        persistence.record_client_spend(&client_name, 100, 3.0);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            persistence.client_spend(&client_name).await.unwrap(),
            Some((100, 8.0))
        );

        persistence.record_client_spend(&client_name, 101, 1.5);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            persistence.client_spend(&client_name).await.unwrap(),
            Some((101, 1.5))
        );
    }

    #[tokio::test]
    async fn postgres_reset_client_spend_zeroes_an_existing_balance() {
        let Some(persistence) = open_postgres_for_test("client_spend_reset").await else {
            eprintln!("skipping: TEST_POSTGRES_URL not set");
            return;
        };
        let client_name = unique_key("client_reset");

        persistence.record_client_spend(&client_name, 100, 5.0);
        tokio::time::sleep(Duration::from_millis(200)).await;
        persistence.reset_client_spend(&client_name, 100);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            persistence.client_spend(&client_name).await.unwrap(),
            Some((100, 0.0))
        );
    }

    // --- postgres backend, TLS ------------------------------------------------------
    //
    // Separately gated on TEST_POSTGRES_TLS_URL (not TEST_POSTGRES_URL)
    // since this needs a Postgres with TLS actually enabled and a
    // certificate the test host's native trust store accepts -- more to
    // ask of a CI service container than the plaintext tests above, so
    // this stays opt-in even where TEST_POSTGRES_URL is provided.

    #[tokio::test]
    async fn postgres_require_tls_round_trip() {
        let Ok(url) = std::env::var("TEST_POSTGRES_TLS_URL") else {
            eprintln!("skipping: TEST_POSTGRES_TLS_URL not set");
            return;
        };
        let persistence = Persistence::open(PersistenceTarget::Postgres {
            url,
            tls: PostgresTlsMode::Require,
        })
        .await
        .expect("TEST_POSTGRES_TLS_URL should be reachable over TLS");

        let key = unique_key("tls");
        persistence.record(
            &key,
            &Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
                cached_tokens: None,
                cache_creation_tokens: None,
            },
            Some(0.05),
        );
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = persistence.load_all().await.unwrap();
        let entry = &stats[&key];
        assert_eq!(entry.requests, 1);
        assert_eq!(entry.prompt_tokens, 3);
        assert_eq!(entry.completion_tokens, 2);
    }
}
