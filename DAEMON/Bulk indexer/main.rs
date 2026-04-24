//! sentinel-daemon: bootstrap.
//!
//! Startup order
//! ─────────────
//! 1. Parse config from `sentinel-daemon.toml` (or defaults)
//! 2. Check required privileges (`CAP_SYS_ADMIN` for fanotify)
//! 3. Open graph store and Tantivy index
//! 4. Start the Resource Governor (CPU sampler)
//! 5. Start the Embedder worker (consumes IndexTask channel)
//! 6. Start the fanotify real-time listener
//! 7. **NEW** — If this is a first run (graph is empty), start the Bulk Indexer
//! 8. Start the IPC server (handles UI / orchestrator requests)
//! 9. Block on SIGINT / SIGTERM, then shut down cleanly

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::info;

use crate::daemon::bulk_indexer::{BulkIndexConfig, BulkIndexer};
use crate::daemon::IndexTask;
use crate::graph::MockGraphStore;
use crate::governor::Governor;

mod daemon;
mod extractor;
mod governor;
mod graph;
mod ipc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "sentinel_daemon=info".into()),
        )
        .init();

    // ── Config ────────────────────────────────────────────────────────────
    let config = Config::load();

    // ── Graph store ───────────────────────────────────────────────────────
    let graph = Arc::new(
        MockGraphStore::open(&config.graph_log_path)
            .expect("Failed to open graph store"),
    );

    // ── Shared channels ───────────────────────────────────────────────────
    let (index_tx, index_rx) = mpsc::channel::<IndexTask>(1024);
    let (progress_tx, progress_rx) = mpsc::channel::<daemon::bulk_indexer::ProgressEvent>(64);

    // ── Resource Governor ─────────────────────────────────────────────────
    let governor = Arc::new(Governor::new(config.cpu_threshold_pct));
    governor.start();
    let governor_pause = governor.pause_flag();

    // ── Cancellation ──────────────────────────────────────────────────────
    let cancel = tokio_util::sync::CancellationToken::new();
    let cancel_bulk = cancel.clone();

    // ── Bulk indexer — run on first install or if graph is empty ──────────
    let first_run = graph.file_count() == 0;
    if first_run {
        info!("First run detected (graph is empty) — starting bulk indexer");

        let mut bulk_cfg = BulkIndexConfig::default();
        bulk_cfg.watch_paths = config.watch_paths.clone();
        bulk_cfg.state_path = config.bulk_index_state_path.clone();

        let graph_arc = Arc::clone(&graph);
        let index_tx_bulk = index_tx.clone();
        let governor_pause_bulk = Arc::clone(&governor_pause);

        tokio::spawn(async move {
            let mut indexer = BulkIndexer::new(
                bulk_cfg,
                graph_arc,
                index_tx_bulk,
                governor_pause_bulk,
                progress_tx,
            );
            indexer.load_state();
            indexer.count_files();
            indexer.run(cancel_bulk).await;
        });
    } else {
        info!(
            file_count = graph.file_count(),
            "Graph already populated — skipping bulk indexer"
        );
        // Drop progress_tx so the channel closes cleanly; nobody is reading it.
        drop(progress_tx);
    }

    // ── Forward progress events to IPC clients ───────────────────────────
    // (The IPC server subscribes to progress_rx and forwards to connected UIs)
    // This is handled inside ipc::Server::new() below.

    // ── IPC server ────────────────────────────────────────────────────────
    let ipc_server = ipc::Server::new(
        config.daemon_socket_path.clone(),
        Arc::clone(&graph),
        index_tx.clone(),
        progress_rx,
    );

    // ── Signal handling ───────────────────────────────────────────────────
    let cancel_signal = cancel.clone();
    tokio::spawn(async move {
        let mut sigint = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::interrupt(),
        )
        .expect("Failed to install SIGINT handler");
        let mut sigterm = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        )
        .expect("Failed to install SIGTERM handler");

        tokio::select! {
            _ = sigint.recv()  => info!("Received SIGINT"),
            _ = sigterm.recv() => info!("Received SIGTERM"),
        }
        cancel_signal.cancel();
    });

    // ── Run IPC server until cancelled ───────────────────────────────────
    ipc_server.run(cancel).await?;

    info!("sentinel-daemon exiting cleanly");
    Ok(())
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct Config {
    watch_paths: Vec<PathBuf>,
    cpu_threshold_pct: f32,
    graph_log_path: PathBuf,
    bulk_index_state_path: PathBuf,
    daemon_socket_path: PathBuf,
}

impl Config {
    fn load() -> Self {
        // In a real implementation this would parse `sentinel-daemon.toml`.
        // For now, all values are either hardcoded defaults or read from env vars.
        Self {
            watch_paths: std::env::var("SENTINEL_WATCH_PATHS")
                .map(|s| s.split(':').map(PathBuf::from).collect())
                .unwrap_or_else(|_| vec![
                    PathBuf::from("/home"),
                    PathBuf::from("/root"),
                ]),
            cpu_threshold_pct: std::env::var("SENTINEL_CPU_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(15.0),
            graph_log_path: PathBuf::from(
                std::env::var("SENTINEL_GRAPH_LOG")
                    .unwrap_or_else(|_| "/var/sentinel/state/graph.ndjson".into()),
            ),
            bulk_index_state_path: PathBuf::from(
                std::env::var("SENTINEL_BULK_STATE")
                    .unwrap_or_else(|_| "/var/sentinel/state/bulk_index.json".into()),
            ),
            daemon_socket_path: PathBuf::from(
                std::env::var("SENTINEL_DAEMON_SOCKET")
                    .unwrap_or_else(|_| "/run/sentinel/daemon.sock".into()),
            ),
        }
    }
}
