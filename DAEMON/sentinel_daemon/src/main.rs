#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// ===========================================================================
// sentinel-daemon: Monolithic Distribution (Isolated Logic)
// ===========================================================================

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, Mutex};
use tokio::time::{interval, sleep};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

#[cfg(feature = "kuzu")]
use kuzu::{Database, SystemConfig, Connection, Value};

// ---------------------------------------------------------------------------
// 1. Shared Types & Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct IndexTask {
    pub path: PathBuf,
    pub modified_ts: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    pub event_type: &'static str,
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

struct Config {
    watch_paths:            Vec<PathBuf>,
    cpu_threshold_pct:      f32,
    kuzu_db_path:           PathBuf,
    graph_log_path:         PathBuf,
    bulk_index_state_path:  PathBuf,
    daemon_socket_path:     PathBuf,
}

impl Config {
    fn load() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let base = PathBuf::from(home).join(".sentinel");
        
        // Ensure directories exist
        let _ = fs::create_dir_all(&base);
        let _ = fs::create_dir_all(base.join("state"));

        Self {
            watch_paths: std::env::var("SENTINEL_WATCH_PATHS")
                .map(|s| s.split(':').map(PathBuf::from).collect())
                .unwrap_or_else(|_| vec![
                    PathBuf::from("/home/rudra"),
                ]),
            cpu_threshold_pct: std::env::var("SENTINEL_CPU_THRESHOLD")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(15.0),
            kuzu_db_path: PathBuf::from(
                std::env::var("SENTINEL_KUZU_PATH")
                    .unwrap_or_else(|_| base.join("graph").to_string_lossy().into()),
            ),
            graph_log_path: base.join("state/graph.ndjson"),
            bulk_index_state_path: base.join("state/bulk_index.json"),
            daemon_socket_path: base.join("daemon.sock"),
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Resource Governor
// ---------------------------------------------------------------------------

pub struct Governor {
    threshold: f32,
    should_yield: Arc<AtomicBool>,
}

impl Governor {
    pub fn new(threshold_pct: f32) -> Self {
        Self {
            threshold: threshold_pct,
            should_yield: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn start(&self) {
        let flag = self.should_yield.clone();
        let threshold = self.threshold;

        tokio::spawn(async move {
            use sysinfo::{System, RefreshKind, CpuRefreshKind};
            let mut sys = System::new_with_specifics(
                RefreshKind::new().with_cpu(CpuRefreshKind::everything()),
            );
            let mut tick = interval(Duration::from_secs(2));
            info!(threshold, "Resource Governor started");

            loop {
                tick.tick().await;
                sys.refresh_cpu_usage();
                let usage = sys.global_cpu_info().cpu_usage();
                let yield_now = usage >= threshold;
                flag.store(yield_now, Ordering::Relaxed);
            }
        });
    }

    pub fn pause_flag(&self) -> Arc<AtomicBool> {
        self.should_yield.clone()
    }
}

// ---------------------------------------------------------------------------
// 3. Text Extraction
// ---------------------------------------------------------------------------

const MAX_FILE_SIZE: u64 = 16 * 1024 * 1024;

pub async fn extract_text(path: &Path) -> Result<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let meta = std::fs::metadata(&path)?;
        if meta.len() > MAX_FILE_SIZE {
            anyhow::bail!("File too large");
        }
        std::fs::read_to_string(&path).context("Failed to read file")
    }).await?
}

// ---------------------------------------------------------------------------
// 4. Graph Backend
// ---------------------------------------------------------------------------

pub trait GraphBackend: Send + Sync {
    fn upsert_file(&self, path: &Path, ts: u64);
    fn has_file(&self, path: &Path) -> bool;
    fn query_time_range(&self, start: Option<u64>, end: Option<u64>, ext: Option<&str>, limit: usize) -> Vec<PathBuf>;
    fn file_count(&self) -> u64;
}

#[cfg(feature = "kuzu")]
pub struct KuzuGraphStore {
    db: Arc<Database>,
}

#[cfg(feature = "kuzu")]
impl KuzuGraphStore {
    pub fn open(path: &Path, pool_size: usize) -> Result<Self> {
        let db = Database::new(path, SystemConfig::default())?;
        let store = Self { db: Arc::new(db) };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = Connection::new(&self.db)?;
        let _ = conn.query("CREATE NODE TABLE IF NOT EXISTS File(path STRING, modified_ts INT64, PRIMARY KEY (path))");
        Ok(())
    }
}

#[cfg(feature = "kuzu")]
impl GraphBackend for KuzuGraphStore {
    fn upsert_file(&self, path: &Path, ts: u64) {
        if let Ok(conn) = Connection::new(&self.db) {
            let q = "MERGE (f:File {path: $p}) ON CREATE SET f.modified_ts = $ts ON MATCH SET f.modified_ts = $ts";
            if let Ok(mut prepared) = conn.prepare(q) {
                let params = vec![
                    ("p", Value::String(path.to_string_lossy().to_string())),
                    ("ts", Value::Int64(ts as i64)),
                ];
                let _ = conn.execute(&mut prepared, params);
            }
        }
    }

    fn has_file(&self, path: &Path) -> bool {
        if let Ok(conn) = Connection::new(&self.db) {
            let q = "MATCH (f:File {path: $p}) RETURN count(f)";
            if let Ok(mut prepared) = conn.prepare(q) {
                let params = vec![
                    ("p", Value::String(path.to_string_lossy().to_string())),
                ];
                if let Ok(result) = conn.execute(&mut prepared, params) {
                    if let Some(row) = result.into_iter().next() {
                        if let Some(val) = row.get(0) {
                            return val.to_string().parse::<u64>().unwrap_or(0) > 0;
                        }
                    }
                }
            }
        }
        false
    }

    fn query_time_range(&self, start: Option<u64>, end: Option<u64>, _ext: Option<&str>, limit: usize) -> Vec<PathBuf> {
        let mut results = Vec::new();
        if let Ok(conn) = Connection::new(&self.db) {
            let q = "MATCH (f:File) WHERE f.modified_ts >= $s AND f.modified_ts <= $e RETURN f.path LIMIT $l";
            if let Ok(mut prepared) = conn.prepare(q) {
                let params = vec![
                    ("s", Value::Int64(start.unwrap_or(0) as i64)),
                    ("e", Value::Int64(end.unwrap_or(u64::MAX) as i64)),
                    ("l", Value::Int64(limit as i64)),
                ];
                if let Ok(res) = conn.execute(&mut prepared, params) {
                    for row in res {
                        if let Some(v) = row.get(0) {
                            results.push(PathBuf::from(v.to_string()));
                        }
                    }
                }
            }
        }
        results
    }

    fn file_count(&self) -> u64 {
        if let Ok(conn) = Connection::new(&self.db) {
            if let Ok(result) = conn.query("MATCH (f:File) RETURN count(*)") {
                if let Some(row) = result.into_iter().next() {
                    if let Some(val) = row.get(0) {
                        return val.to_string().parse().unwrap_or(0);
                    }
                }
            }
        }
        0
    }
}

pub struct MockGraphStore;
impl MockGraphStore {
    pub fn open(_: &Path) -> Result<Self> { Ok(Self) }
}
impl GraphBackend for MockGraphStore {
    fn upsert_file(&self, _: &Path, _: u64) {}
    fn has_file(&self, _: &Path) -> bool { false }
    fn query_time_range(&self, _: Option<u64>, _: Option<u64>, _: Option<&str>, _: usize) -> Vec<PathBuf> { vec![] }
    fn file_count(&self) -> u64 { 0 }
}

// ---------------------------------------------------------------------------
// 5. Bulk Indexer
// ---------------------------------------------------------------------------

pub struct BulkIndexer {
    watch_paths: Vec<PathBuf>,
    state_path: PathBuf,
    graph: Arc<dyn GraphBackend>,
    index_tx: mpsc::Sender<IndexTask>,
    gov: Arc<AtomicBool>,
    progress_tx: mpsc::Sender<ProgressEvent>,
}

impl BulkIndexer {
    pub fn new(watch_paths: Vec<PathBuf>, state_path: PathBuf, graph: Arc<dyn GraphBackend>, index_tx: mpsc::Sender<IndexTask>, gov: Arc<AtomicBool>, progress_tx: mpsc::Sender<ProgressEvent>) -> Self {
        Self { watch_paths, state_path, graph, index_tx, gov, progress_tx }
    }

    pub async fn run(&mut self, cancel: CancellationToken) {
        info!("BulkIndexer running...");
        for root in &self.watch_paths {
            for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
                if cancel.is_cancelled() { return; }
                if entry.file_type().is_file() {
                    let path = entry.into_path();
                    if !self.graph.has_file(&path) {
                        let mtime = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
                        let _ = self.index_tx.try_send(IndexTask { path: path.clone(), modified_ts: mtime });
                        self.graph.upsert_file(&path, mtime);
                    }
                }
            }
        }
        info!("BulkIndexer complete.");
    }
}

// ---------------------------------------------------------------------------
// 6. IPC Server
// ---------------------------------------------------------------------------

pub struct Server {
    socket_path: PathBuf,
    graph: Arc<dyn GraphBackend>,
    index_tx: mpsc::Sender<IndexTask>,
    progress_rx: Arc<Mutex<mpsc::Receiver<ProgressEvent>>>,
}

impl Server {
    pub fn new(socket_path: PathBuf, graph: Arc<dyn GraphBackend>, index_tx: mpsc::Sender<IndexTask>, progress_rx: mpsc::Receiver<ProgressEvent>) -> Self {
        Self { socket_path, graph, index_tx, progress_rx: Arc::new(Mutex::new(progress_rx)) }
    }

    pub async fn run(&self, cancel: CancellationToken) -> Result<()> {
        let _ = std::fs::remove_file(&self.socket_path);
        if let Some(parent) = self.socket_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let listener = UnixListener::bind(&self.socket_path)?;
        info!("IPC listening at {:?}", self.socket_path);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                accept_res = listener.accept() => {
                    if let Ok((mut stream, _)) = accept_res {
                        tokio::spawn(async move {
                            let mut buf = [0u8; 1024];
                            if let Ok(n) = stream.read(&mut buf).await {
                               let _ = stream.write_all(b"{\"ok\":true,\"status\":\"sentinel-daemon monolith\"}").await;
                            }
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 7. Main Application Entry Point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("SentinelAI Daemon monolith started");
    let config = Config::load();
    
    #[cfg(feature = "kuzu")]
    let graph: Arc<dyn GraphBackend> = Arc::new(KuzuGraphStore::open(&config.kuzu_db_path, 4)?);
    #[cfg(not(feature = "kuzu"))]
    let graph: Arc<dyn GraphBackend> = Arc::new(MockGraphStore::open(&config.graph_log_path)?);

    info!("Graph count: {}", graph.file_count());
    
    let (index_tx, _index_rx) = mpsc::channel::<IndexTask>(1024);
    let (progress_tx, progress_rx) = mpsc::channel::<ProgressEvent>(64);
    
    let gov_global = Arc::new(Governor::new(config.cpu_threshold_pct));
    gov_global.start();
    let cancel_global = CancellationToken::new();

    if graph.file_count() == 0 {
        let watch_paths = config.watch_paths.clone();
        let state_path = config.bulk_index_state_path.clone();
        let g_indexer = graph.clone();
        let tx_indexer = index_tx.clone();
        let gov_yield = gov_global.should_yield.clone();
        let c_indexer = cancel_global.clone();
        
        tokio::spawn(async move {
            let mut indexer = BulkIndexer::new(watch_paths, state_path, g_indexer, tx_indexer, gov_yield, progress_tx);
            indexer.run(c_indexer).await;
        });
    }

    let server = Server::new(config.daemon_socket_path, graph, index_tx, progress_rx);
    server.run(cancel_global).await
}
