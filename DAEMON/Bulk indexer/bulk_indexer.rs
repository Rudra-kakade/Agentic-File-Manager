//! sentinel-daemon: Cold-Start Bulk Indexer
//!
//! Walks configured filesystem paths on first install and submits every
//! discovered file into the same `IndexTask` pipeline used by the real-time
//! fanotify daemon.  Without this, the system would be blind to files that
//! already exist and only learn about them the first time they are modified.
//!
//! # Key design properties
//!
//! | Property | Implementation |
//! |----------|----------------|
//! | Governor-aware | Polls the shared `AtomicBool` every batch; pauses with 500 ms sleeps while CPU > 15% |
//! | Resumable | Persists a `BulkIndexState` checkpoint to `/var/sentinel/state/bulk_index.json` after every batch; re-reads on startup and skips already-indexed paths |
//! | No duplicate work | Checks `GraphBackend::has_file()` before enqueueing — files already in the graph are skipped even on a fresh run |
//! | Progress reporting | Emits `ProgressEvent` structs after every batch; the IPC layer forwards them to any connected UI client |
//! | Single-threaded walk | `walkdir` does the heavy lifting; we never spawn extra threads — CPU budget is tight |
//! | Graceful shutdown | Watches a `CancellationToken`; drops cleanly mid-walk and saves checkpoint |
//!
//! # Startup flow (called from `main.rs`)
//!
//! ```text
//! 1. BulkIndexer::new(paths, graph, index_tx, governor, state_path)
//! 2. indexer.load_state()          // read checkpoint if it exists
//! 3. indexer.count_files()         // fast pre-scan to know the total
//! 4. indexer.run(cancel_token)     // walk + enqueue; saves checkpoint on exit
//! ```

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

use crate::daemon::IndexTask;
use crate::graph::GraphBackend;

// ---------------------------------------------------------------------------
// Progress event (forwarded to the UI via IPC)
// ---------------------------------------------------------------------------

/// Emitted after every processed batch so the UI can show a live progress bar.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    pub event_type: &'static str, // always "bulk_progress"
    pub indexed: u64,
    pub total: u64,
    pub skipped: u64,
    pub state: IndexingState,
    /// Fractional progress in [0.0, 1.0].
    pub fraction: f32,
    /// Estimated seconds remaining.  None until we have throughput data.
    pub eta_seconds: Option<u64>,
    /// Files indexed per second over the last measurement window.
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

/// Written to disk after every batch so a restart can resume where it left off.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BulkIndexState {
    /// Absolute paths that have already been submitted to the index pipeline.
    /// Using a `HashSet` means resume is O(1) per path lookup.
    pub indexed_paths: HashSet<PathBuf>,
    /// Total file count from the pre-scan (cached to avoid re-counting on resume).
    pub total_files: u64,
    /// Unix timestamp when the indexing run started.
    pub started_at: u64,
    /// Unix timestamp of the last checkpoint write.
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
        // Write to a temp file then rename for atomic update — we never want
        // a partially written checkpoint.
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

/// Tuning knobs — all have sensible defaults.
#[derive(Debug, Clone)]
pub struct BulkIndexConfig {
    /// Filesystem paths to walk.  Typically `["/home", "/root"]` or configured
    /// by the user via `sentinel-daemon.toml`.
    pub watch_paths: Vec<PathBuf>,

    /// Where to persist the checkpoint between runs.
    pub state_path: PathBuf,

    /// How many files to process before checking the governor, saving a
    /// checkpoint, and emitting a progress event.
    /// Smaller → more responsive to CPU pressure; larger → less I/O overhead.
    pub batch_size: usize,

    /// How long to sleep between governor polls when paused.
    pub governor_poll_interval: Duration,

    /// File extensions to skip entirely (binary blobs that yield no text).
    /// The extractor already handles unknown types gracefully, so this is
    /// purely a performance optimisation.
    pub skip_extensions: HashSet<String>,

    /// Maximum file size to index.  Files larger than this are skipped to
    /// avoid saturating the embedding queue with huge binaries.
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
                // Large media/archive formats — no useful text content
                "mp4", "mkv", "avi", "mov", "mp3", "flac", "wav", "ogg",
                "jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff",
                "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
                "iso", "img", "dmg",
                // Compiled objects
                "o", "a", "so", "dylib", "dll", "exe",
                // Package manager caches
                "pyc", "class", "jar",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            max_file_bytes: 50 * 1024 * 1024, // 50 MB
        }
    }
}

// ---------------------------------------------------------------------------
// Bulk indexer
// ---------------------------------------------------------------------------

/// Drives the cold-start indexing run.
///
/// Constructed once in `main.rs` and driven with `.run(cancel_token)`.
pub struct BulkIndexer<G: GraphBackend> {
    config: BulkIndexConfig,
    graph: Arc<G>,
    /// Channel to the `Embedder` worker — same channel the fanotify daemon uses.
    index_tx: mpsc::Sender<IndexTask>,
    /// Resource governor: `true` = system CPU is over threshold, pause work.
    governor_pause: Arc<AtomicBool>,
    /// Progress events are sent here; the IPC layer forwards them to UI clients.
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

    // ------------------------------------------------------------------
    // Phase 1: load checkpoint
    // ------------------------------------------------------------------

    /// Load an existing checkpoint from disk (if any).
    /// Called once before `count_files` so we can reuse the cached total.
    pub fn load_state(&mut self) {
        self.state = BulkIndexState::load(&self.config.state_path);
        if !self.state.indexed_paths.is_empty() {
            tracing::info!(
                already_indexed = self.state.indexed_paths.len(),
                "Resuming bulk index from checkpoint"
            );
        }
    }

    // ------------------------------------------------------------------
    // Phase 2: count files (pre-scan for accurate ETA)
    // ------------------------------------------------------------------

    /// Walk all configured paths and count indexable files.
    ///
    /// Uses the cached total from the checkpoint if available so a resumed run
    /// doesn't need to re-scan the entire filesystem.
    pub fn count_files(&mut self) -> u64 {
        if self.state.total_files > 0 {
            tracing::debug!(
                total = self.state.total_files,
                "Using cached file count from checkpoint"
            );
            return self.state.total_files;
        }

        tracing::info!("Pre-scanning filesystem to count indexable files …");
        let t0 = Instant::now();
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
        tracing::info!(
            total = count,
            elapsed_ms = t0.elapsed().as_millis(),
            "Pre-scan complete"
        );
        count
    }

    // ------------------------------------------------------------------
    // Phase 3: walk and index
    // ------------------------------------------------------------------

    /// Main indexing loop.  Walks all configured paths and submits each
    /// indexable file to the `IndexTask` channel.
    ///
    /// `cancel` is polled after every batch — signal it to stop cleanly.
    pub async fn run(&mut self, cancel: tokio::sync::CancellationToken) {
        let total = self.state.total_files;
        let mut indexed: u64 = 0;
        let mut skipped: u64 = 0;
        let mut batch: Vec<PathBuf> = Vec::with_capacity(self.config.batch_size);

        // Throughput tracking — sliding window over the last 5 batches.
        let mut throughput_window: std::collections::VecDeque<(Instant, u64)> =
            std::collections::VecDeque::with_capacity(6);
        let run_start = Instant::now();

        if self.state.started_at == 0 {
            self.state.started_at = now_unix();
        }

        tracing::info!(total, "Starting bulk index run");
        self.emit_progress(indexed, skipped, total, IndexingState::Running, &throughput_window).await;

        'walk: for root in self.config.watch_paths.clone() {
            for entry in WalkDir::new(&root)
                .follow_links(false)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                // --- Cancellation check ---
                if cancel.is_cancelled() {
                    tracing::info!("Bulk indexer cancelled mid-walk");
                    self.flush_batch(&mut batch, &mut indexed).await;
                    self.save_checkpoint();
                    self.emit_progress(indexed, skipped, total, IndexingState::Cancelled, &throughput_window).await;
                    return;
                }

                let path = entry.into_path();

                if !path.is_file() {
                    continue;
                }

                // --- Skip non-indexable files ---
                if !self.is_indexable(&path) {
                    skipped += 1;
                    continue;
                }

                // --- Skip already-indexed (resume support) ---
                if self.state.indexed_paths.contains(&path) {
                    skipped += 1;
                    continue;
                }

                // --- Skip files already in the graph ---
                if self.graph.has_file(&path) {
                    self.state.indexed_paths.insert(path);
                    skipped += 1;
                    continue;
                }

                batch.push(path);

                if batch.len() >= self.config.batch_size {
                    // --- Governor: pause if CPU is over budget ---
                    while self.governor_pause.load(Ordering::Relaxed) {
                        if cancel.is_cancelled() {
                            break 'walk;
                        }
                        self.emit_progress(indexed, skipped, total, IndexingState::Paused, &throughput_window).await;
                        sleep(self.config.governor_poll_interval).await;
                    }

                    // --- Process batch ---
                    let batch_count = batch.len() as u64;
                    self.flush_batch(&mut batch, &mut indexed).await;

                    // Update throughput window
                    throughput_window.push_back((Instant::now(), batch_count));
                    if throughput_window.len() > 5 {
                        throughput_window.pop_front();
                    }

                    // --- Checkpoint ---
                    self.save_checkpoint();

                    // --- Progress ---
                    self.emit_progress(indexed, skipped, total, IndexingState::Running, &throughput_window).await;
                }
            }
        }

        // Flush the final partial batch
        self.flush_batch(&mut batch, &mut indexed).await;
        self.save_checkpoint();

        let elapsed = run_start.elapsed();
        tracing::info!(
            indexed,
            skipped,
            total,
            elapsed_secs = elapsed.as_secs(),
            "Bulk index complete"
        );

        self.emit_progress(indexed, skipped, total, IndexingState::Complete, &throughput_window).await;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Submit a batch of paths to the IndexTask channel and record them in
    /// the checkpoint set.
    async fn flush_batch(&mut self, batch: &mut Vec<PathBuf>, indexed: &mut u64) {
        for path in batch.drain(..) {
            let mtime = file_mtime(&path).unwrap_or_else(now_unix);
            let task = IndexTask {
                path: path.clone(),
                modified_ts: mtime,
            };
            // Non-blocking send — if the channel is full (embedder can't keep up)
            // we drop the task rather than blocking the walk.  The file will be
            // picked up by fanotify the next time it is modified, or the user
            // can re-run the bulk indexer.
            match self.index_tx.try_send(task) {
                Ok(_) => {
                    self.state.indexed_paths.insert(path);
                    *indexed += 1;
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    tracing::warn!("Index channel full — dropping bulk task for {:?}", path);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    tracing::error!("Index channel closed — aborting bulk indexer");
                    break;
                }
            }
        }
    }

    /// Returns true if the file should be indexed.
    fn is_indexable(&self, path: &Path) -> bool {
        // Skip hidden files and directories (dot-prefixed names)
        for component in path.components() {
            if let std::path::Component::Normal(name) = component {
                if name.to_str().map(|s| s.starts_with('.')).unwrap_or(false) {
                    return false;
                }
            }
        }

        // Skip known-skippable extensions
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if self.config.skip_extensions.contains(&ext.to_lowercase()) {
                return false;
            }
        }

        // Skip oversized files
        if let Ok(meta) = path.metadata() {
            if meta.len() > self.config.max_file_bytes {
                return false;
            }
        }

        // Skip special files (symlinks already filtered by follow_links=false,
        // but be defensive about sockets, pipes, etc.)
        match path.metadata() {
            Ok(m) if m.is_file() => true,
            _ => false,
        }
    }

    fn save_checkpoint(&mut self) {
        self.state.last_checkpoint_at = now_unix();
        if let Err(e) = self.state.save(&self.config.state_path) {
            tracing::warn!(error = %e, "Failed to save bulk index checkpoint");
        }
    }

    async fn emit_progress(
        &self,
        indexed: u64,
        skipped: u64,
        total: u64,
        state: IndexingState,
        window: &std::collections::VecDeque<(Instant, u64)>,
    ) {
        let (throughput_fps, eta_seconds) = compute_throughput_eta(indexed, total, window);

        let event = ProgressEvent {
            event_type: "bulk_progress",
            indexed,
            total,
            skipped,
            state,
            fraction: if total > 0 { indexed as f32 / total as f32 } else { 0.0 },
            eta_seconds,
            throughput_fps,
        };

        // Best-effort — if the UI isn't connected the channel may be full.
        let _ = self.progress_tx.try_send(event);
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn file_mtime(path: &Path) -> Option<u64> {
    path.metadata()
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

/// Compute throughput (files/sec) and ETA from the sliding window.
///
/// The window contains (timestamp, files_indexed_in_that_batch) pairs.
/// We take the elapsed time across the entire window and sum the files.
fn compute_throughput_eta(
    indexed: u64,
    total: u64,
    window: &std::collections::VecDeque<(Instant, u64)>,
) -> (f32, Option<u64>) {
    if window.len() < 2 {
        return (0.0, None);
    }

    let oldest = window.front().unwrap().0;
    let elapsed_secs = oldest.elapsed().as_secs_f32();
    let window_files: u64 = window.iter().map(|(_, n)| n).sum();

    if elapsed_secs < 0.001 {
        return (0.0, None);
    }

    let fps = window_files as f32 / elapsed_secs;
    let remaining = total.saturating_sub(indexed);
    let eta = if fps > 0.1 {
        Some((remaining as f32 / fps) as u64)
    } else {
        None
    };

    (fps, eta)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use tempfile::TempDir;
    use tokio::sync::mpsc;

    // Minimal stub implementing GraphBackend for tests.
    struct TestGraph {
        known: HashSet<PathBuf>,
    }

    impl TestGraph {
        fn new(known: impl IntoIterator<Item = PathBuf>) -> Self {
            Self { known: known.into_iter().collect() }
        }
    }

    impl crate::graph::GraphBackend for TestGraph {
        fn upsert_file(&self, _path: &Path, _ts: u64) {}
        fn has_file(&self, path: &Path) -> bool {
            self.known.contains(path)
        }
        fn query_time_range(
            &self,
            _start: Option<u64>,
            _end: Option<u64>,
            _ext: Option<&str>,
            _limit: usize,
        ) -> Vec<PathBuf> {
            vec![]
        }
        fn file_count(&self) -> u64 {
            self.known.len() as u64
        }
    }

    fn make_cancel() -> tokio::sync::CancellationToken {
        tokio::sync::CancellationToken::new()
    }

    // ── is_indexable ──────────────────────────────────────────────────────

    #[test]
    fn skips_hidden_files() {
        let cfg = BulkIndexConfig::default();
        let indexer = make_dummy_indexer(cfg);
        assert!(!indexer.is_indexable(Path::new("/home/alice/.bashrc")));
        assert!(!indexer.is_indexable(Path::new("/home/alice/.config/app/settings.json")));
    }

    #[test]
    fn skips_known_binary_extensions() {
        let cfg = BulkIndexConfig::default();
        let indexer = make_dummy_indexer(cfg);
        assert!(!indexer.is_indexable(Path::new("/home/alice/video.mp4")));
        assert!(!indexer.is_indexable(Path::new("/home/alice/archive.zip")));
    }

    #[test]
    fn accepts_text_files() {
        let tmp = TempDir::new().unwrap();
        let f = tmp.path().join("notes.txt");
        std::fs::write(&f, "hello world").unwrap();

        let cfg = BulkIndexConfig::default();
        let indexer = make_dummy_indexer(cfg);
        assert!(indexer.is_indexable(&f));
    }

    #[test]
    fn skips_oversized_files() {
        let tmp = TempDir::new().unwrap();
        let f = tmp.path().join("huge.bin");
        // Write 51 MB — over the 50 MB limit
        std::fs::write(&f, vec![0u8; 51 * 1024 * 1024]).unwrap();

        let cfg = BulkIndexConfig::default();
        let indexer = make_dummy_indexer(cfg);
        assert!(!indexer.is_indexable(&f));
    }

    // ── BulkIndexState checkpoint ─────────────────────────────────────────

    #[test]
    fn checkpoint_round_trips() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bulk_index.json");

        let mut state = BulkIndexState::default();
        state.total_files = 1234;
        state.indexed_paths.insert(PathBuf::from("/home/alice/a.txt"));
        state.indexed_paths.insert(PathBuf::from("/home/alice/b.pdf"));
        state.save(&path).unwrap();

        let loaded = BulkIndexState::load(&path);
        assert_eq!(loaded.total_files, 1234);
        assert!(loaded.indexed_paths.contains(Path::new("/home/alice/a.txt")));
        assert!(loaded.indexed_paths.contains(Path::new("/home/alice/b.pdf")));
    }

    #[test]
    fn checkpoint_load_returns_default_on_missing_file() {
        let state = BulkIndexState::load(Path::new("/nonexistent/state.json"));
        assert_eq!(state.total_files, 0);
        assert!(state.indexed_paths.is_empty());
    }

    #[test]
    fn checkpoint_save_is_atomic() {
        // The save writes to a .tmp file then renames — the target file
        // should only ever be fully written or absent.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bulk_index.json");

        let mut state = BulkIndexState::default();
        state.total_files = 99;
        state.save(&path).unwrap();

        // The temp file must not exist after save.
        assert!(!path.with_extension("tmp").exists());
        // The target file must exist.
        assert!(path.exists());
    }

    // ── count_files ───────────────────────────────────────────────────────

    #[test]
    fn count_files_correct() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "a").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "b").unwrap();
        std::fs::write(tmp.path().join("c.mp4"), vec![0u8; 100]).unwrap(); // skipped

        let mut cfg = BulkIndexConfig::default();
        cfg.watch_paths = vec![tmp.path().to_path_buf()];

        let mut indexer = make_dummy_indexer(cfg);
        let count = indexer.count_files();
        assert_eq!(count, 2); // .mp4 is in skip_extensions
    }

    #[test]
    fn count_files_uses_cache() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "a").unwrap();

        let mut cfg = BulkIndexConfig::default();
        cfg.watch_paths = vec![tmp.path().to_path_buf()];

        let mut indexer = make_dummy_indexer(cfg);
        indexer.state.total_files = 999; // pre-cached
        let count = indexer.count_files();
        assert_eq!(count, 999); // returned from cache, not re-counted
    }

    // ── full run ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn indexes_all_files_in_directory() {
        let tmp = TempDir::new().unwrap();
        for i in 0..10 {
            std::fs::write(tmp.path().join(format!("file{i}.txt")), "content").unwrap();
        }

        let (index_tx, mut index_rx) = mpsc::channel(128);
        let (progress_tx, _progress_rx) = mpsc::channel(32);
        let governor = Arc::new(AtomicBool::new(false));
        let graph = Arc::new(TestGraph::new([]));

        let mut cfg = BulkIndexConfig::default();
        cfg.watch_paths = vec![tmp.path().to_path_buf()];
        cfg.batch_size = 3;
        cfg.state_path = tmp.path().join("state.json");

        let mut indexer = BulkIndexer::new(cfg, graph, index_tx, governor, progress_tx);
        indexer.count_files();
        indexer.run(make_cancel()).await;

        drop(indexer); // close the sender so we can drain the channel
        let mut received = Vec::new();
        while let Ok(task) = index_rx.try_recv() {
            received.push(task.path);
        }
        assert_eq!(received.len(), 10);
    }

    #[tokio::test]
    async fn skips_files_already_in_graph() {
        let tmp = TempDir::new().unwrap();
        let a = tmp.path().join("a.txt");
        let b = tmp.path().join("b.txt");
        std::fs::write(&a, "a").unwrap();
        std::fs::write(&b, "b").unwrap();

        let (index_tx, mut index_rx) = mpsc::channel(32);
        let (progress_tx, _) = mpsc::channel(32);
        let governor = Arc::new(AtomicBool::new(false));
        // 'a' is already in the graph — should be skipped
        let graph = Arc::new(TestGraph::new([a.clone()]));

        let mut cfg = BulkIndexConfig::default();
        cfg.watch_paths = vec![tmp.path().to_path_buf()];
        cfg.state_path = tmp.path().join("state.json");

        let mut indexer = BulkIndexer::new(cfg, graph, index_tx, governor, progress_tx);
        indexer.count_files();
        indexer.run(make_cancel()).await;

        drop(indexer);
        let mut received = Vec::new();
        while let Ok(task) = index_rx.try_recv() {
            received.push(task.path);
        }
        assert_eq!(received.len(), 1);
        assert_eq!(received[0], b);
    }

    #[tokio::test]
    async fn resumes_from_checkpoint() {
        let tmp = TempDir::new().unwrap();
        let a = tmp.path().join("a.txt");
        let b = tmp.path().join("b.txt");
        std::fs::write(&a, "a").unwrap();
        std::fs::write(&b, "b").unwrap();

        let state_path = tmp.path().join("state.json");
        // Pre-populate checkpoint: 'a' already indexed
        let mut state = BulkIndexState::default();
        state.total_files = 2;
        state.indexed_paths.insert(a.clone());
        state.save(&state_path).unwrap();

        let (index_tx, mut index_rx) = mpsc::channel(32);
        let (progress_tx, _) = mpsc::channel(32);
        let governor = Arc::new(AtomicBool::new(false));
        let graph = Arc::new(TestGraph::new([]));

        let mut cfg = BulkIndexConfig::default();
        cfg.watch_paths = vec![tmp.path().to_path_buf()];
        cfg.state_path = state_path;

        let mut indexer = BulkIndexer::new(cfg, graph, index_tx, governor, progress_tx);
        indexer.load_state();
        indexer.count_files();
        indexer.run(make_cancel()).await;

        drop(indexer);
        let mut received = Vec::new();
        while let Ok(task) = index_rx.try_recv() {
            received.push(task.path);
        }
        // Only 'b' should be indexed — 'a' was in the checkpoint
        assert_eq!(received.len(), 1);
        assert_eq!(received[0], b);
    }

    #[tokio::test]
    async fn cancellation_stops_walk_cleanly() {
        let tmp = TempDir::new().unwrap();
        for i in 0..100 {
            std::fs::write(tmp.path().join(format!("f{i}.txt")), "x").unwrap();
        }

        let (index_tx, _index_rx) = mpsc::channel(4); // tiny buffer — will block
        let (progress_tx, _) = mpsc::channel(32);
        let governor = Arc::new(AtomicBool::new(false));
        let graph = Arc::new(TestGraph::new([]));

        let cancel = make_cancel();

        let mut cfg = BulkIndexConfig::default();
        cfg.watch_paths = vec![tmp.path().to_path_buf()];
        cfg.batch_size = 2;
        cfg.state_path = tmp.path().join("state.json");

        let mut indexer = BulkIndexer::new(cfg, graph, index_tx, governor, progress_tx);
        indexer.count_files();

        // Cancel after 10 ms — the run should return promptly without panic
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            cancel_clone.cancel();
        });

        // Should complete without hanging
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            indexer.run(cancel),
        ).await;
        assert!(result.is_ok(), "run() did not complete after cancellation");
    }

    #[tokio::test]
    async fn governor_pause_resumes_on_clear() {
        let tmp = TempDir::new().unwrap();
        for i in 0..6 {
            std::fs::write(tmp.path().join(format!("f{i}.txt")), "x").unwrap();
        }

        let (index_tx, _index_rx) = mpsc::channel(64);
        let (progress_tx, _) = mpsc::channel(32);
        let governor = Arc::new(AtomicBool::new(true)); // start paused
        let graph = Arc::new(TestGraph::new([]));
        let cancel = make_cancel();

        let mut cfg = BulkIndexConfig::default();
        cfg.watch_paths = vec![tmp.path().to_path_buf()];
        cfg.batch_size = 2;
        cfg.governor_poll_interval = Duration::from_millis(20);
        cfg.state_path = tmp.path().join("state.json");

        let mut indexer = BulkIndexer::new(
            cfg,
            graph,
            index_tx,
            Arc::clone(&governor),
            progress_tx,
        );
        indexer.count_files();

        // Clear the governor after 50 ms
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            governor.store(false, Ordering::Relaxed);
        });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            indexer.run(cancel),
        ).await;
        assert!(result.is_ok(), "run() did not resume after governor cleared");
    }

    // ── progress events ───────────────────────────────────────────────────

    #[tokio::test]
    async fn emits_progress_events() {
        let tmp = TempDir::new().unwrap();
        for i in 0..5 {
            std::fs::write(tmp.path().join(format!("f{i}.txt")), "x").unwrap();
        }

        let (index_tx, _) = mpsc::channel(64);
        let (progress_tx, mut progress_rx) = mpsc::channel(64);
        let governor = Arc::new(AtomicBool::new(false));
        let graph = Arc::new(TestGraph::new([]));

        let mut cfg = BulkIndexConfig::default();
        cfg.watch_paths = vec![tmp.path().to_path_buf()];
        cfg.batch_size = 2;
        cfg.state_path = tmp.path().join("state.json");

        let mut indexer = BulkIndexer::new(cfg, graph, index_tx, governor, progress_tx);
        indexer.count_files();
        indexer.run(make_cancel()).await;

        drop(indexer);

        let mut events = Vec::new();
        while let Ok(e) = progress_rx.try_recv() {
            events.push(e);
        }

        // At minimum we expect a "running" event and a "complete" event
        assert!(!events.is_empty());
        let last = events.last().unwrap();
        assert_eq!(last.state, IndexingState::Complete);
        assert_eq!(last.indexed, 5);
    }

    // ── throughput / ETA ──────────────────────────────────────────────────

    #[test]
    fn throughput_eta_with_empty_window() {
        let window = std::collections::VecDeque::new();
        let (fps, eta) = compute_throughput_eta(0, 100, &window);
        assert_eq!(fps, 0.0);
        assert!(eta.is_none());
    }

    #[test]
    fn throughput_eta_with_data() {
        let mut window = std::collections::VecDeque::new();
        // Simulate 2 batches of 64 files, 1 second apart
        window.push_back((Instant::now() - Duration::from_secs(1), 64));
        window.push_back((Instant::now(), 64));
        let (fps, eta) = compute_throughput_eta(128, 1128, &window);
        // fps should be ~64/s (64 files over ~1s window)
        assert!(fps > 10.0, "fps={fps} — expected > 10");
        // eta should be ~1000/64 ≈ 15 s
        assert!(eta.is_some());
    }

    // ── helper to build a BulkIndexer without requiring a real filesystem ──

    fn make_dummy_indexer(cfg: BulkIndexConfig) -> BulkIndexer<TestGraph> {
        let (tx, _) = mpsc::channel(1);
        let (ptx, _) = mpsc::channel(1);
        BulkIndexer::new(
            cfg,
            Arc::new(TestGraph::new([])),
            tx,
            Arc::new(AtomicBool::new(false)),
            ptx,
        )
    }
}
