//! sentinel-daemon: graph database abstraction.
//!
//! `GraphBackend` is a trait so we can swap `MockGraphStore` (NDJSON, used
//! during development) for `KuzuGraphStore` (Phase 8) without touching any
//! calling code.

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

pub trait GraphBackend {
    /// Upsert a `MODIFIED_BY` edge: File(path) → User, with timestamp.
    fn upsert_file(&self, path: &Path, ts: u64);

    /// Returns true if the file is already present in the graph.
    ///
    /// Used by the bulk indexer to skip files that are already indexed so a
    /// re-run after an interrupted cold-start doesn't re-index everything.
    fn has_file(&self, path: &Path) -> bool;

    /// Query files modified in a time range, optionally filtered by extension.
    ///
    /// Returns absolute paths sorted by modification time, most recent first.
    fn query_time_range(
        &self,
        start_ts: Option<u64>,
        end_ts: Option<u64>,
        ext_filter: Option<&str>,
        limit: usize,
    ) -> Vec<PathBuf>;

    /// Total number of files in the graph.
    fn file_count(&self) -> u64;
}

// ---------------------------------------------------------------------------
// MockGraphStore — NDJSON log for development / testing
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;

/// Development-time graph store that writes operations to an NDJSON log.
///
/// Replace with `KuzuGraphStore` (Phase 8) when kuzu-rs FFI is stable.
pub struct MockGraphStore {
    /// In-memory index: path → modification timestamp
    index: Mutex<HashMap<PathBuf, u64>>,
    /// Append-only NDJSON log for inspection / debugging
    log: Mutex<BufWriter<File>>,
}

impl MockGraphStore {
    pub fn open(log_path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        // Replay existing log into the in-memory index so we survive restarts.
        let index = Self::replay(log_path);

        Ok(Self {
            index: Mutex::new(index),
            log: Mutex::new(BufWriter::new(file)),
        })
    }

    fn replay(log_path: &Path) -> HashMap<PathBuf, u64> {
        use std::io::{BufRead, BufReader};
        let mut index = HashMap::new();
        if let Ok(f) = File::open(log_path) {
            for line in BufReader::new(f).lines().flatten() {
                if let Ok(entry) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let (Some(path), Some(ts)) = (
                        entry["path"].as_str(),
                        entry["ts"].as_u64(),
                    ) {
                        index.insert(PathBuf::from(path), ts);
                    }
                }
            }
        }
        index
    }

    fn write_log(&self, path: &Path, ts: u64) {
        let mut log = self.log.lock().unwrap();
        let entry = serde_json::json!({"path": path, "ts": ts});
        let _ = writeln!(log, "{entry}");
        let _ = log.flush();
    }
}

impl GraphBackend for MockGraphStore {
    fn upsert_file(&self, path: &Path, ts: u64) {
        let mut index = self.index.lock().unwrap();
        index.insert(path.to_path_buf(), ts);
        drop(index);
        self.write_log(path, ts);
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
        let index = self.index.lock().unwrap();
        let mut results: Vec<(u64, PathBuf)> = index
            .iter()
            .filter(|(path, &ts)| {
                // Time range filter
                if let Some(s) = start_ts {
                    if ts < s {
                        return false;
                    }
                }
                if let Some(e) = end_ts {
                    if ts > e {
                        return false;
                    }
                }
                // Extension filter
                if let Some(ext) = ext_filter {
                    let matches = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case(ext))
                        .unwrap_or(false);
                    if !matches {
                        return false;
                    }
                }
                true
            })
            .map(|(path, &ts)| (ts, path.clone()))
            .collect();

        // Sort most recent first
        results.sort_by(|a, b| b.0.cmp(&a.0));
        results.truncate(limit);
        results.into_iter().map(|(_, p)| p).collect()
    }

    fn file_count(&self) -> u64 {
        self.index.lock().unwrap().len() as u64
    }
}

// ---------------------------------------------------------------------------
// KuzuGraphStore stub (Phase 8)
// ---------------------------------------------------------------------------

// pub struct KuzuGraphStore { … }
//
// impl GraphBackend for KuzuGraphStore {
//     fn upsert_file(&self, path: &Path, ts: u64) { … }
//     fn has_file(&self, path: &Path) -> bool { … }
//     fn query_time_range(&self, …) -> Vec<PathBuf> { … }
//     fn file_count(&self) -> u64 { … }
// }
