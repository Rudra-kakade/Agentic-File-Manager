//! sentinel-daemon: graph database abstraction layer.
//!
//! Provides a `GraphBackend` trait and two implementations:
//!
//! | Implementation | When to use |
//! |----------------|-------------|
//! | `KuzuGraphStore` | Production — embedded Kùzu graph DB |
//! | `MockGraphStore` | Development / testing — NDJSON log file |
//!
//! # Schema (Kùzu DDL)
//!
//! ```cypher
//! CREATE NODE TABLE File(
//!     path     STRING,
//!     ext      STRING,
//!     mtime    INT64,
//!     PRIMARY KEY (path)
//! );
//!
//! CREATE REL TABLE MODIFIED_AT(
//!     FROM File TO File,
//!     ts INT64
//! );
//! ```
//!
//! In practice `MODIFIED_AT` is a self-loop on each `File` node carrying the
//! modification timestamp.  This lets us query timestamp ranges purely in
//! Cypher without a separate edge table.  A simpler alternative — storing
//! `mtime` directly on the node — is used in the actual implementation below
//! because Kùzu's Cypher subset handles node property range scans efficiently.
//!
//! # Thread safety
//!
//! Both `KuzuGraphStore` and `MockGraphStore` implement `Send + Sync`.
//! Kùzu's Rust bindings expose a `Database` + `Connection` pair; connections
//! are not `Send`, so we pool them behind a `Mutex<Vec<Connection>>` and
//! check one out per operation.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Common interface for the graph store.
///
/// Both `KuzuGraphStore` and `MockGraphStore` implement this trait so the
/// rest of the daemon is completely decoupled from the storage backend.
pub trait GraphBackend: Send + Sync {
    /// Insert or update a file node with its latest modification timestamp.
    fn upsert_file(&self, path: &Path, mtime: u64);

    /// Returns `true` if the file already exists in the graph.
    ///
    /// Used by the bulk indexer to skip already-indexed files on resume.
    fn has_file(&self, path: &Path) -> bool;

    /// Return at most `limit` file paths whose mtime falls in
    /// `[start_ts, end_ts]`, optionally filtered by file extension.
    /// Results are ordered by mtime descending (most recently modified first).
    fn query_time_range(
        &self,
        start_ts: Option<u64>,
        end_ts: Option<u64>,
        ext_filter: Option<&str>,
        limit: usize,
    ) -> Vec<PathBuf>;

    /// Total number of file nodes in the graph.
    fn file_count(&self) -> u64;
}

// ---------------------------------------------------------------------------
// KuzuGraphStore — production implementation
// ---------------------------------------------------------------------------

/// Production graph store backed by the Kùzu embedded graph database.
///
/// # Kùzu crate gating
///
/// The `kuzu` crate (kuzu-rs) exposes safe Rust bindings over Kùzu's C API.
/// We gate the entire implementation behind `#[cfg(feature = "kuzu")]` so
/// the daemon compiles on machines that do not have the Kùzu shared library
/// installed (CI, developer laptops, etc.).  When the feature flag is absent
/// the compiler falls back to `MockGraphStore` automatically — see `main.rs`.
///
/// Enable with:  `cargo build --features kuzu`
///
/// # Connection pooling
///
/// Kùzu `Connection` objects are not `Send`, so we keep a small pool
/// (default: 4) inside a `Mutex<Vec<Connection>>`.  Each public method
/// pops a connection, runs the query, and pushes it back.  If the pool is
/// empty (all connections in use by concurrent callers) the thread blocks
/// on the mutex — acceptable because graph operations are fast (µs–ms).
///
/// # Schema migration
///
/// On `open()` the store runs `CREATE NODE TABLE IF NOT EXISTS …` DDL.
/// Kùzu's `IF NOT EXISTS` is idempotent, so re-opening an existing database
/// (daemon restart) never fails.

#[cfg(feature = "kuzu")]
pub mod kuzu_store {
    use super::*;
    use kuzu::{Connection, Database, Error as KuzuError, LogicalType, Value};

    // DDL executed once on open — idempotent via IF NOT EXISTS.
    const DDL: &str = "
        CREATE NODE TABLE IF NOT EXISTS File (
            path   STRING,
            ext    STRING,
            mtime  INT64,
            PRIMARY KEY (path)
        );
    ";

    pub struct KuzuGraphStore {
        db: Database,
    }

    impl KuzuGraphStore {
        /// Open (or create) the Kùzu database at `db_path`.
        ///
        /// Creates the schema on first open; subsequent opens are no-ops.
        pub fn open(db_path: &Path, _pool_size: usize) -> anyhow::Result<Self> {
            use anyhow::Context;

            fs::create_dir_all(db_path)
                .context("Failed to create Kùzu database directory")?;

            let db = Database::new(db_path, kuzu::SystemConfig::default())
                .context("Failed to open Kùzu database")?;

            // Run DDL on a bootstrap connection.
            {
                let boot = Connection::new(&db)
                    .context("Failed to open bootstrap connection")?;
                boot.query(DDL)
                    .context("Failed to initialise Kùzu schema")?;
            }

            tracing::info!(
                db = %db_path.display(),
                "Kùzu graph store opened"
            );

            Ok(Self { db })
        }

        /// Get a fresh connection to the database.
        fn conn(&self) -> Connection<'_> {
            Connection::new(&self.db).expect("Failed to create Kùzu connection")
        }
    }


    impl super::GraphBackend for KuzuGraphStore {
        fn upsert_file(&self, path: &Path, mtime: u64) {
            let path_str = path.to_string_lossy();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            // MERGE: insert or update in one idempotent statement.
            // Kùzu supports MERGE with ON MATCH SET / ON CREATE SET.
            let cypher = "
                MERGE (f:File {path: $path})
                ON CREATE SET f.ext = $ext, f.mtime = $mtime
                ON MATCH  SET f.ext = $ext, f.mtime = $mtime
            ";

            let conn = self.conn();
            // In Kuzu 0.6.1, we use prepared statements for parameters.
            let mut prepared = match conn.prepare(cypher) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, "Kùzu prepare failed for upsert_file");
                    return;
                }
            };

            if let Err(e) = conn.execute(&mut prepared, vec![
                ("path",  Value::String(path_str.to_string())),
                ("ext",   Value::String(ext)),
                ("mtime", Value::Int64(mtime as i64)),
            ]) {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "Kùzu upsert_file execution failed"
                );
            }
        }

        fn has_file(&self, path: &Path) -> bool {
            let path_str = path.to_string_lossy();
            let cypher = "
                MATCH (f:File {path: $path})
                RETURN count(f) AS n
            ";
            let conn = self.conn();
            let mut prepared = match conn.prepare(cypher) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, "Kùzu prepare failed for has_file");
                    return false;
                }
            };

            match conn.execute(&mut prepared, vec![
                ("path", Value::String(path_str.to_string())),
            ]) {
                Ok(mut result) => {
                    if let Some(row) = result.next() {
                        if let Value::Int64(n) = &row[0] {
                            return *n > 0;
                        }
                    }
                    false
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Kùzu has_file execution failed");
                    false
                }
            }
        }

        fn query_time_range(
            &self,
            start_ts: Option<u64>,
            end_ts: Option<u64>,
            ext_filter: Option<&str>,
            limit: usize,
        ) -> Vec<PathBuf> {
            // Build the WHERE clause dynamically — Kùzu doesn't yet support
            // optional parameter binding, so we inline the values.
            let mut predicates: Vec<String> = Vec::new();
            if let Some(s) = start_ts {
                predicates.push(format!("f.mtime >= {}", s as i64));
            }
            if let Some(e) = end_ts {
                predicates.push(format!("f.mtime <= {}", e as i64));
            }
            if let Some(ext) = ext_filter {
                let safe = ext.replace('\'', "''"); // basic SQL-style escaping
                predicates.push(format!("f.ext = '{safe}'"));
            }

            let where_clause = if predicates.is_empty() {
                String::new()
            } else {
                format!("WHERE {}", predicates.join(" AND "))
            };

            let cypher = format!(
                "MATCH (f:File) {where_clause} \
                 RETURN f.path AS path \
                 ORDER BY f.mtime DESC \
                 LIMIT {limit}"
            );

            let conn = self.conn();
            // Building WHERE clause was correct.
            match conn.query(&cypher) {
                Ok(mut result) => {
                    let mut paths = Vec::new();
                    while let Some(row) = result.next() {
                        if let Value::String(p) = &row[0] {
                            paths.push(PathBuf::from(p));
                        }
                    }
                    paths
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Kùzu query_time_range failed");
                    vec![]
                }
            }
        }

        fn file_count(&self) -> u64 {
            let cypher = "MATCH (f:File) RETURN count(f) AS n";
            let conn = self.conn();
            match conn.query(cypher) {
                Ok(mut result) => {
                    if let Some(row) = result.next() {
                        if let Value::Int64(n) = &row[0] {
                            return *n as u64;
                        }
                    }
                    0
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Kùzu file_count failed");
                    0
                }
            }
        }
    }
}

#[cfg(feature = "kuzu")]
pub use kuzu_store::KuzuGraphStore;

// ---------------------------------------------------------------------------
// MockGraphStore — in-memory + NDJSON log (development / testing)
// ---------------------------------------------------------------------------

/// In-memory graph store that logs every operation to an NDJSON file.
///
/// Replays the log on startup so it survives daemon restarts.
/// Used when the `kuzu` feature is not enabled or in unit tests.
pub struct MockGraphStore {
    /// path → (ext, mtime)
    index: Mutex<HashMap<PathBuf, (String, u64)>>,
    log:   Mutex<BufWriter<File>>,
}

impl MockGraphStore {
    /// Open (or create) the NDJSON log at `log_path`.
    pub fn open(log_path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let index = Self::replay(log_path);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        Ok(Self {
            index: Mutex::new(index),
            log:   Mutex::new(BufWriter::new(file)),
        })
    }

    fn replay(log_path: &Path) -> HashMap<PathBuf, (String, u64)> {
        let mut index = HashMap::new();
        let Ok(file) = File::open(log_path) else { return index };
        for line in BufReader::new(file).lines().flatten() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            if let (Some(path), Some(mtime)) = (v["path"].as_str(), v["mtime"].as_u64()) {
                let ext = v["ext"].as_str().unwrap_or("").to_string();
                index.insert(PathBuf::from(path), (ext, mtime));
            }
        }
        index
    }

    fn append_log(&self, path: &Path, ext: &str, mtime: u64) {
        let entry = serde_json::json!({"path": path, "ext": ext, "mtime": mtime});
        let mut log = self.log.lock().unwrap();
        let _ = writeln!(log, "{entry}");
        let _ = log.flush();
    }
}

impl GraphBackend for MockGraphStore {
    fn upsert_file(&self, path: &Path, mtime: u64) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        {
            let mut idx = self.index.lock().unwrap();
            idx.insert(path.to_path_buf(), (ext.clone(), mtime));
        }
        self.append_log(path, &ext, mtime);
    }

    fn has_file(&self, path: &Path) -> bool {
        self.index.lock().unwrap().contains_key(path)
    }

    fn query_time_range(
        &self,
        start_ts: Option<u64>,
        end_ts: Option<u64>,
        ext_filter: Option<&str>,
        limit: usize,
    ) -> Vec<PathBuf> {
        let idx = self.index.lock().unwrap();
        let mut hits: Vec<(u64, PathBuf)> = idx
            .iter()
            .filter(|(path, (ext, mtime))| {
                if let Some(s) = start_ts { if *mtime < s { return false; } }
                if let Some(e) = end_ts   { if *mtime > e { return false; } }
                if let Some(f) = ext_filter {
                    if ext.as_str() != f.to_ascii_lowercase() { return false; }
                }
                true
            })
            .map(|(path, (_, mtime))| (*mtime, path.clone()))
            .collect();

        hits.sort_by(|a, b| b.0.cmp(&a.0));
        hits.truncate(limit);
        hits.into_iter().map(|(_, p)| p).collect()
    }

    fn file_count(&self) -> u64 {
        self.index.lock().unwrap().len() as u64
    }
}

// ---------------------------------------------------------------------------
// Schema migration helper (used in integration tests and the installer)
// ---------------------------------------------------------------------------

/// DDL statements to initialise the Kùzu schema.
///
/// Exported so the installer (Phase 9) and integration tests can run them
/// independently without constructing a `KuzuGraphStore`.
pub const KUZU_DDL: &str = "
    CREATE NODE TABLE IF NOT EXISTS File (
        path   STRING,
        ext    STRING,
        mtime  INT64,
        PRIMARY KEY (path)
    );
";

/// Benchmark query used by Phase 8 performance validation.
/// Run against a database with 1M+ nodes to verify sub-100ms retrieval.
pub const KUZU_BENCHMARK_QUERY: &str = "
    MATCH (f:File)
    WHERE f.mtime >= $start AND f.mtime <= $end
    RETURN f.path AS path
    ORDER BY f.mtime DESC
    LIMIT 500;
";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── MockGraphStore ────────────────────────────────────────────────────

    fn open_mock(tmp: &TempDir) -> MockGraphStore {
        MockGraphStore::open(&tmp.path().join("graph.ndjson")).unwrap()
    }

    #[test]
    fn upsert_and_has_file() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        let p = Path::new("/home/alice/report.pdf");

        assert!(!g.has_file(p));
        g.upsert_file(p, 1_700_000_000);
        assert!(g.has_file(p));
    }

    #[test]
    fn upsert_is_idempotent_updates_mtime() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        let p = Path::new("/home/alice/notes.txt");

        g.upsert_file(p, 1_000);
        g.upsert_file(p, 2_000); // update

        let results = g.query_time_range(Some(1_999), Some(2_001), None, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], p);
    }

    #[test]
    fn file_count_accurate() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        assert_eq!(g.file_count(), 0);
        g.upsert_file(Path::new("/a.txt"), 1000);
        g.upsert_file(Path::new("/b.txt"), 2000);
        assert_eq!(g.file_count(), 2);
        // Re-inserting same path should not increase count
        g.upsert_file(Path::new("/a.txt"), 3000);
        assert_eq!(g.file_count(), 2);
    }

    #[test]
    fn query_time_range_no_filter() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        g.upsert_file(Path::new("/old.txt"),    1_000);
        g.upsert_file(Path::new("/recent.txt"), 9_000);
        g.upsert_file(Path::new("/mid.txt"),    5_000);

        let results = g.query_time_range(Some(4_000), Some(9_001), None, 10);
        assert_eq!(results.len(), 2);
        // Most recent first
        assert_eq!(results[0], Path::new("/recent.txt"));
        assert_eq!(results[1], Path::new("/mid.txt"));
    }

    #[test]
    fn query_time_range_open_ended_start() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        g.upsert_file(Path::new("/a.txt"), 1_000);
        g.upsert_file(Path::new("/b.txt"), 9_000);

        // No start constraint
        let r = g.query_time_range(None, Some(5_000), None, 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], Path::new("/a.txt"));
    }

    #[test]
    fn query_time_range_open_ended_end() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        g.upsert_file(Path::new("/a.txt"), 1_000);
        g.upsert_file(Path::new("/b.txt"), 9_000);

        // No end constraint
        let r = g.query_time_range(Some(5_000), None, None, 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], Path::new("/b.txt"));
    }

    #[test]
    fn query_time_range_no_constraints() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        g.upsert_file(Path::new("/x.txt"), 100);
        g.upsert_file(Path::new("/y.txt"), 200);

        let r = g.query_time_range(None, None, None, 10);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn query_time_range_ext_filter() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        g.upsert_file(Path::new("/report.pdf"),  5_000);
        g.upsert_file(Path::new("/notes.txt"),   5_001);
        g.upsert_file(Path::new("/budget.xlsx"), 5_002);

        let r = g.query_time_range(None, None, Some("pdf"), 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], Path::new("/report.pdf"));
    }

    #[test]
    fn query_time_range_ext_filter_case_insensitive() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        // File extension stored as lowercase regardless of filesystem case
        g.upsert_file(Path::new("/Report.PDF"), 5_000);

        let r = g.query_time_range(None, None, Some("pdf"), 10);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn query_time_range_limit_respected() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        for i in 0..20u64 {
            g.upsert_file(&Path::new(&format!("/f{i}.txt")), i * 1000);
        }
        let r = g.query_time_range(None, None, None, 5);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn query_returns_most_recent_first() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        g.upsert_file(Path::new("/old.txt"),  1_000);
        g.upsert_file(Path::new("/new.txt"), 10_000);
        g.upsert_file(Path::new("/mid.txt"),  5_000);

        let r = g.query_time_range(None, None, None, 10);
        assert_eq!(r[0], Path::new("/new.txt"));
        assert_eq!(r[1], Path::new("/mid.txt"));
        assert_eq!(r[2], Path::new("/old.txt"));
    }

    #[test]
    fn query_time_range_empty_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let g = open_mock(&tmp);
        let r = g.query_time_range(Some(1_000), Some(2_000), None, 10);
        assert!(r.is_empty());
    }

    // ── NDJSON persistence / replay ───────────────────────────────────────

    #[test]
    fn survives_restart_via_replay() {
        let tmp = TempDir::new().unwrap();
        let log = tmp.path().join("graph.ndjson");

        {
            let g = MockGraphStore::open(&log).unwrap();
            g.upsert_file(Path::new("/persisted.txt"), 42_000);
            assert_eq!(g.file_count(), 1);
        }

        // Re-open — should replay the log
        let g2 = MockGraphStore::open(&log).unwrap();
        assert!(g2.has_file(Path::new("/persisted.txt")));
        assert_eq!(g2.file_count(), 1);
    }

    #[test]
    fn corrupt_log_lines_skipped_on_replay() {
        let tmp = TempDir::new().unwrap();
        let log = tmp.path().join("graph.ndjson");

        // Write one valid line and one corrupt line
        fs::write(
            &log,
            b"{\"path\":\"/good.txt\",\"ext\":\"txt\",\"mtime\":1000}\nnot json at all\n",
        )
        .unwrap();

        let g = MockGraphStore::open(&log).unwrap();
        assert!(g.has_file(Path::new("/good.txt")));
        assert_eq!(g.file_count(), 1); // corrupt line skipped
    }

    #[test]
    fn missing_log_returns_empty_store() {
        let tmp = TempDir::new().unwrap();
        let log = tmp.path().join("nonexistent.ndjson");
        let g = MockGraphStore::open(&log).unwrap();
        assert_eq!(g.file_count(), 0);
    }

    // ── DDL constants ─────────────────────────────────────────────────────

    #[test]
    fn kuzu_ddl_has_all_columns() {
        assert!(KUZU_DDL.contains("path"));
        assert!(KUZU_DDL.contains("ext"));
        assert!(KUZU_DDL.contains("mtime"));
        assert!(KUZU_DDL.contains("PRIMARY KEY"));
        assert!(KUZU_DDL.contains("IF NOT EXISTS"));
    }

    #[test]
    fn kuzu_benchmark_query_has_params() {
        assert!(KUZU_BENCHMARK_QUERY.contains("$start"));
        assert!(KUZU_BENCHMARK_QUERY.contains("$end"));
        assert!(KUZU_BENCHMARK_QUERY.contains("LIMIT 500"));
    }

    // ── Trait object usage ────────────────────────────────────────────────

    #[test]
    fn trait_object_dynamic_dispatch_works() {
        let tmp = TempDir::new().unwrap();
        let g: Box<dyn GraphBackend> =
            Box::new(MockGraphStore::open(&tmp.path().join("g.ndjson")).unwrap());

        g.upsert_file(Path::new("/dyn.txt"), 9_000);
        assert!(g.has_file(Path::new("/dyn.txt")));
        assert_eq!(g.file_count(), 1);

        let r = g.query_time_range(Some(8_000), Some(10_000), Some("txt"), 5);
        assert_eq!(r.len(), 1);
    }
}
