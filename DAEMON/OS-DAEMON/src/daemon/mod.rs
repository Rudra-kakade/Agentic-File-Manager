// src/daemon/mod.rs
//
// Top-level module for the daemon subsystems (fanotify listener + embedder worker).

pub mod fanotify;
pub mod embedder;

use std::path::PathBuf;

/// Task sent from the fanotify handler to the embedding worker.
#[derive(Debug, Clone)]
pub struct IndexTask {
    pub path:      PathBuf,
    pub timestamp: u64,
}
