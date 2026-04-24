// src/graph/mod.rs
//
// Knowledge Graph abstraction.
//
// Wraps Kùzu (or any graph backend) behind a simple Rust trait so the
// storage engine is swappable without touching the rest of the daemon.
//
// Schema (Kùzu Cypher DDL — run once on first open):
//
//   CREATE NODE TABLE IF NOT EXISTS File (
//       id     STRING,   -- SHA-256 of canonical path (stable across renames if content same)
//       path   STRING,
//       name   STRING,
//       ext    STRING,
//       size   INT64,
//       PRIMARY KEY (id)
//   );
//
//   CREATE NODE TABLE IF NOT EXISTS Process (
//       pid  INT64,
//       name STRING,
//       PRIMARY KEY (pid)
//   );
//
//   CREATE REL TABLE IF NOT EXISTS MODIFIED_BY (
//       FROM File TO Process,
//       timestamp INT64
//   );
//
//   CREATE REL TABLE IF NOT EXISTS CREATED_BY (
//       FROM File TO Process,
//       timestamp INT64
//   );
//
// NOTE: Until kuzu-rs reaches 1.0, we provide a `MockGraphStore` that logs
//       all writes.  Swap in `KuzuGraphStore` when the crate is available.

use std::path::{Path, PathBuf};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventKind {
    Created,
    Modified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEvent {
    pub path:      PathBuf,
    pub kind:      EventKind,
    pub pid:       u32,
    pub timestamp: u64,   // Unix epoch seconds (UTC)
}

/// Result row from a graph query.
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub path:      String,
    pub timestamp: u64,
    pub score:     f32,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

pub trait GraphBackend: Send {
    fn record_event(&mut self, ev: &FileEvent) -> Result<()>;
    fn query_modified_range(&self, start: u64, end: u64) -> Result<Vec<QueryResult>>;
}

// ── Public facade ─────────────────────────────────────────────────────────────

/// `GraphStore` is the type used everywhere in the daemon.
/// Currently backed by `MockGraphStore`; replace with `KuzuGraphStore` when
/// the kuzu-rs FFI binding is stable.
pub struct GraphStore(Box<dyn GraphBackend>);

impl GraphStore {
    pub fn open(path: &Path) -> Result<Self> {
        // TODO: switch to KuzuGraphStore::open(path)?
        Ok(Self(Box::new(MockGraphStore::new(path))))
    }

    pub fn record_event(&mut self, ev: &FileEvent) -> Result<()> {
        self.0.record_event(ev)
    }

    pub fn query_modified_range(&self, start: u64, end: u64) -> Result<Vec<QueryResult>> {
        self.0.query_modified_range(start, end)
    }
}

// ── MockGraphStore ────────────────────────────────────────────────────────────
// Logs all events to a newline-delimited JSON file.
// Provides correct range query semantics so the rest of the system
// can be built and tested without a live Kùzu instance.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};

struct MockGraphStore {
    log: File,
    log_path: PathBuf,
}

impl MockGraphStore {
    fn new(dir: &Path) -> Self {
        std::fs::create_dir_all(dir).ok();
        let log_path = dir.join("events.ndjson");
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .expect("Failed to open event log");
        info!(path = ?log_path, "MockGraphStore: logging to file");
        Self { log, log_path }
    }
}

impl GraphBackend for MockGraphStore {
    fn record_event(&mut self, ev: &FileEvent) -> Result<()> {
        let line = serde_json::to_string(ev)?;
        writeln!(self.log, "{line}")?;
        Ok(())
    }

    fn query_modified_range(&self, start: u64, end: u64) -> Result<Vec<QueryResult>> {
        let f   = BufReader::new(File::open(&self.log_path)?);
        let mut results = Vec::new();
        for line in f.lines().flatten() {
            if let Ok(ev) = serde_json::from_str::<FileEvent>(&line) {
                if ev.timestamp >= start && ev.timestamp <= end {
                    results.push(QueryResult {
                        path:      ev.path.to_string_lossy().to_string(),
                        timestamp: ev.timestamp,
                        score:     1.0,
                    });
                }
            }
        }
        Ok(results)
    }
}

// ── KuzuGraphStore stub ───────────────────────────────────────────────────────
// Uncomment and flesh out when kuzu-rs FFI is available.
//
// pub struct KuzuGraphStore { db: kuzu::Database, conn: kuzu::Connection }
// impl KuzuGraphStore {
//     pub fn open(path: &Path) -> Result<Self> {
//         let db   = kuzu::Database::new(path, kuzu::SystemConfig::default())?;
//         let conn = kuzu::Connection::new(&db)?;
//         // Run DDL migrations
//         conn.query(include_str!("schema.cypher"))?;
//         Ok(Self { db, conn })
//     }
// }
// impl GraphBackend for KuzuGraphStore {
//     fn record_event(&mut self, ev: &FileEvent) -> Result<()> {
//         let kind_rel = match ev.kind {
//             EventKind::Created  => "CREATED_BY",
//             EventKind::Modified => "MODIFIED_BY",
//         };
//         let path_id = sha256_hex(&ev.path.to_string_lossy());
//         self.conn.query(&format!(
//             "MERGE (f:File {{id: '{path_id}', path: '{}', name: '{}', ext: '{}'}})
//              MERGE (p:Process {{pid: {pid}}})
//              CREATE (f)-[:{kind_rel} {{timestamp: {ts}}}]->(p)",
//             ev.path.display(),
//             ev.path.file_name().unwrap_or_default().to_string_lossy(),
//             ev.path.extension().unwrap_or_default().to_string_lossy(),
//             pid = ev.pid,
//             ts  = ev.timestamp,
//         ))?;
//         Ok(())
//     }
//
//     fn query_modified_range(&self, start: u64, end: u64) -> Result<Vec<QueryResult>> {
//         let q = format!(
//             "MATCH (f:File)-[r:MODIFIED_BY]->(p:Process)
//              WHERE r.timestamp >= {start} AND r.timestamp <= {end}
//              RETURN f.path, r.timestamp ORDER BY r.timestamp DESC"
//         );
//         // ... parse result set ...
//         todo!()
//     }
// }
