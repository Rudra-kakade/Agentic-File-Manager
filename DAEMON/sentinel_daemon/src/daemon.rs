//! sentinel-daemon: index engine module.
//!
//! Includes the IndexTask type and the Cold-Start Bulk Indexer.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::sleep;
use walkdir::WalkDir;

use crate::graph::GraphBackend;

// ---------------------------------------------------------------------------
// Shared Types
// ---------------------------------------------------------------------------

/// A file that needs to be embedded and indexed.
#[derive(Debug, Clone)]
pub struct IndexTask {
    pub path: PathBuf,
    pub modified_ts: u64,
}

// ---------------------------------------------------------------------------
// Progress event (forwarded to the UI via IPC)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    pub event_type: &'static str, // always "bulk_progress"
    pub indexed: u64,
    pub total: u64,
    pub skipped: u64,
    pub state: IndexingState,
    pub fraction: f32,
    pub eta_seconds: Option<u64>,
    pub throughput_fps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexingState {
    Running,
    Paused,
    Complete,
    Cancelled,
    Error,
}

// ---------------------------------------------------------------------------
// Persistent checkpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BulkIndexState {
    pub indexed_paths: HashSet<PathBuf>,
    pub total_files: u64,
    pub started_at: u64,
    pub last_checkpoint_at: u64,
}

impl BulkIndexState {
    fn load(path: &Path) -> Self {
        match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension("tmp");
        let bytes = serde_json::to_vec_pretty(self)
            .expect("BulkIndexState serialisation is infallible");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&tmp, &bytes)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BulkIndexConfig {
    pub watch_paths: Vec<PathBuf>,
    pub state_path: PathBuf,
    pub batch_size: usize,
    pub governor_poll_interval: Duration,
    pub skip_extensions: HashSet<String>,
    pub max_file_bytes: u64,
}

impl Default for BulkIndexConfig {
    fn default() -> Self {
        Self {
            watch_paths: vec![
                PathBuf::from("/home"),
                PathBuf::from("/root"),
            ],
            state_path: PathBuf::from("/var/sentinel/state/bulk_index.json"),
            batch_size: 64,
            governor_poll_interval: Duration::from_millis(500),
            skip_extensions: [
                "mp4", "mkv", "avi", "mov", "mp3", "flac", "wav", "ogg",
                "jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff",
                "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
                "iso", "img", "dmg",
                "o", "a", "so", "dylib", "dll", "exe",
                "pyc", "class", "jar",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            max_file_bytes: 50 * 1024 * 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Bulk indexer
// ---------------------------------------------------------------------------

pub struct BulkIndexer<G: GraphBackend> {
    config: BulkIndexConfig,
    graph: Arc<G>,
    index_tx: mpsc::Sender<IndexTask>,
    governor_pause: Arc<AtomicBool>,
    progress_tx: mpsc::Sender<ProgressEvent>,
    state: BulkIndexState,
}

impl<G: GraphBackend + Send + Sync + 'static> BulkIndexer<G> {
    pub fn new(
        config: BulkIndexConfig,
        graph: Arc<G>,
        index_tx: mpsc::Sender<IndexTask>,
        governor_pause: Arc<AtomicBool>,
        progress_tx: mpsc::Sender<ProgressEvent>,
    ) -> Self {
        Self {
            config,
            graph,
            index_tx,
            governor_pause,
            progress_tx,
            state: BulkIndexState::default(),
        }
    }

    pub fn load_state(&mut self) {
        self.state = BulkIndexState::load(&self.config.state_path);
    }

    pub fn count_files(&mut self) -> u64 {
        if self.state.total_files > 0 {
            return self.state.total_files;
        }
        let mut count: u64 = 0;
        for root in &self.config.watch_paths {
            for entry in WalkDir::new(root)
                .follow_links(false)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file() && self.is_indexable(entry.path()) {
                    count += 1;
                }
            }
        }
        self.state.total_files = count;
        count
    }

    pub async fn run(&mut self, cancel: tokio::sync::CancellationToken) {
        let total = self.state.total_files;
        let mut indexed: u64 = 0;
        let mut skipped: u64 = 0;
        let mut batch: Vec<PathBuf> = Vec::with_capacity(self.config.batch_size);
        let mut throughput_window: std::collections::VecDeque<(Instant, u64)> =
            std::collections::VecDeque::with_capacity(6);

        if self.state.started_at == 0 {
            self.state.started_at = now_unix();
        }

        self.emit_progress(indexed, skipped, total, IndexingState::Running, &throughput_window).await;

        'walk: for root in self.config.watch_paths.clone() {
            for entry in WalkDir::new(&root)
                .follow_links(false)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if cancel.is_cancelled() {
                    self.flush_batch(&mut batch, &mut indexed).await;
                    self.save_checkpoint();
                    self.emit_progress(indexed, skipped, total, IndexingState::Cancelled, &throughput_window).await;
                    return;
                }
                let path = entry.into_path();
                if !path.is_file() { continue; }
                if !self.is_indexable(&path) { skipped += 1; continue; }
                if self.state.indexed_paths.contains(&path) { skipped += 1; continue; }
                if self.graph.has_file(&path) {
                    self.state.indexed_paths.insert(path);
                    skipped += 1;
                    continue;
                }
                batch.push(path);
                if batch.len() >= self.config.batch_size {
                    while self.governor_pause.load(Ordering::Relaxed) {
                        if cancel.is_cancelled() { break 'walk; }
                        self.emit_progress(indexed, skipped, total, IndexingState::Paused, &throughput_window).await;
                        sleep(self.config.governor_poll_interval).await;
                    }
                    let batch_count = batch.len() as u64;
                    self.flush_batch(&mut batch, &mut indexed).await;
                    throughput_window.push_back((Instant::now(), batch_count));
                    if throughput_window.len() > 5 { throughput_window.pop_front(); }
                    self.save_checkpoint();
                    self.emit_progress(indexed, skipped, total, IndexingState::Running, &throughput_window).await;
                }
            }
        }
        self.flush_batch(&mut batch, &mut indexed).await;
        self.save_checkpoint();
        self.emit_progress(indexed, skipped, total, IndexingState::Complete, &throughput_window).await;
    }

    async fn flush_batch(&mut self, batch: &mut Vec<PathBuf>, indexed: &mut u64) {
        for path in batch.drain(..) {
            let mtime = file_mtime(&path).unwrap_or_else(now_unix);
            let task = IndexTask { path: path.clone(), modified_ts: mtime };
            match self.index_tx.try_send(task) {
                Ok(_) => {
                    self.graph.upsert_file(&path, mtime);
                    self.state.indexed_paths.insert(path);
                    *indexed += 1;
                }
                Err(_) => {} // silent drop for now
            }
        }
    }

    fn is_indexable(&self, path: &Path) -> bool {
        for component in path.components() {
            if let std::path::Component::Normal(name) = component {
                if name.to_str().map(|s| s.starts_with('.')).unwrap_or(false) { return false; }
            }
        }
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if self.config.skip_extensions.contains(&ext.to_lowercase()) { return false; }
        }
        if let Ok(meta) = path.metadata() {
            if meta.len() > self.config.max_file_bytes { return false; }
        }
        path.metadata().map(|m| m.is_file()).unwrap_or(false)
    }

    fn save_checkpoint(&mut self) {
        self.state.last_checkpoint_at = now_unix();
        let _ = self.state.save(&self.config.state_path);
    }

    async fn emit_progress(&self, indexed: u64, skipped: u64, total: u64, state: IndexingState, window: &std::collections::VecDeque<(Instant, u64)>) {
        let (throughput_fps, eta_seconds) = compute_throughput_eta(indexed, total, window);
        let event = ProgressEvent {
            event_type: "bulk_progress",
            indexed, total, skipped, state,
            fraction: if total > 0 { indexed as f32 / total as f32 } else { 0.0 },
            eta_seconds, throughput_fps,
        };
        let _ = self.progress_tx.try_send(event);
    }
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn file_mtime(path: &Path) -> Option<u64> {
    path.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs())
}

fn compute_throughput_eta(indexed: u64, total: u64, window: &std::collections::VecDeque<(Instant, u64)>) -> (f32, Option<u64>) {
    if window.len() < 2 { return (0.0, None); }
    let oldest = window.front().unwrap().0;
    let elapsed_secs = oldest.elapsed().as_secs_f32();
    let window_files: u64 = window.iter().map(|(_, n)| n).sum();
    if elapsed_secs < 0.001 { return (0.0, None); }
    let fps = window_files as f32 / elapsed_secs;
    let remaining = total.saturating_sub(indexed);
    let eta = if fps > 0.1 { Some((remaining as f32 / fps) as u64) } else { None };
    (fps, eta)
}
