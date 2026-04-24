//! sentinel-daemon: daemon subsystem module.

pub mod bulk_indexer;
pub mod embedder;
pub mod fanotify;

use std::path::PathBuf;

/// A file that needs to be embedded and indexed.
///
/// Produced by the fanotify listener (real-time) and the bulk indexer
/// (cold-start).  Consumed by the `Embedder` worker.
#[derive(Debug, Clone)]
pub struct IndexTask {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// File modification time as a UTC Unix timestamp.
    pub modified_ts: u64,
}
