// src/main.rs
// SentinelAI — OS Event Daemon
// Entry point: validates privileges, loads config, starts subsystems.

mod daemon;
mod graph;
mod governor;
mod ipc;
mod extractor;

use anyhow::{Context, Result};
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> Result<()> {
    // ── Logging ─────────────────────────────────────────────────────────────
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .json()
        .init();

    info!(version = env!("CARGO_PKG_VERSION"), "SentinelAI daemon starting");

    // ── Privilege check ──────────────────────────────────────────────────────
    // fanotify with FAN_CLASS_CONTENT requires CAP_SYS_ADMIN.
    privilege::assert_capable().context("Privilege check failed")?;

    // ── Config ───────────────────────────────────────────────────────────────
    let cfg = config::DaemonConfig::load().context("Failed to load config")?;
    info!(config = ?cfg, "Configuration loaded");

    // ── Graph DB ─────────────────────────────────────────────────────────────
    let graph = graph::GraphStore::open(&cfg.graph_db_path)
        .context("Failed to open graph database")?;
    let graph = std::sync::Arc::new(tokio::sync::Mutex::new(graph));

    // ── Tantivy lexical index ────────────────────────────────────────────────
    let lexical = tantivy_index::LexicalIndex::open_or_create(&cfg.tantivy_index_path)
        .context("Failed to open tantivy index")?;
    let lexical = std::sync::Arc::new(tokio::sync::Mutex::new(lexical));

    // ── Resource Governor ────────────────────────────────────────────────────
    let governor = governor::ResourceGovernor::new(cfg.cpu_threshold_pct);

    // ── IPC server (Unix domain socket) ──────────────────────────────────────
    let ipc_server = ipc::IpcServer::bind(&cfg.ipc_socket_path)
        .context("Failed to bind IPC socket")?;

    // ── Indexing queue ───────────────────────────────────────────────────────
    // Bounded channel between fanotify handler → embedding worker
    let (index_tx, index_rx) =
        tokio::sync::mpsc::channel::<daemon::IndexTask>(cfg.index_queue_depth);

    // ── Spawn subsystems ──────────────────────────────────────────────────────
    let graph_c  = graph.clone();
    let lexical_c = lexical.clone();

    // ── Run bulk index (for WSL without fanotify) ─────────────────────────────
    // Run this in a background task so it doesn't block IPC or fanotify
    let bulk_watch = cfg.watch_paths.clone();
    let bulk_tx = index_tx.clone();
    tokio::spawn(async move {
        run_bulk_index(bulk_watch, bulk_tx).await;
    });


    // 1. fanotify listener — writes to graph + queues for embedding
    let fan_handle = tokio::spawn(daemon::fanotify::run(
        cfg.watch_paths.clone(),
        graph_c,
        index_tx,
    ));

    // 2. Embedding dispatch worker — reads from queue, respects governor
    let embed_handle = tokio::spawn(daemon::embedder::run(
        index_rx,
        governor,
        lexical_c,
        cfg.embedding_socket_path.clone(),
    ));

    // 3. IPC command handler — serves query engine requests
    let graph_c2 = graph.clone();
    let ipc_handle = tokio::spawn(ipc::serve(ipc_server, graph_c2));

    // ── Signal handling ───────────────────────────────────────────────────────
    tokio::select! {
        res = fan_handle    => { error!("fanotify task exited: {:?}", res); }
        res = embed_handle  => { error!("Embedder task exited: {:?}", res); }
        res = ipc_handle    => { error!("IPC task exited: {:?}", res); }
        _ = shutdown_signal() => {
            info!("Shutdown signal received — exiting cleanly");
        }
    }

    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let mut sigint  = signal(SignalKind::interrupt()).expect("SIGINT handler");
    tokio::select! {
        _ = sigterm.recv() => {}
        _ = sigint.recv()  => {}
    }
}

// ── Inline modules for config, privilege, tantivy wrapper ────────────────────

mod config {
    use anyhow::Result;
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Debug, Deserialize, Clone)]
    pub struct DaemonConfig {
        /// Paths to monitor (e.g., ["/home", "/var/projects"])
        pub watch_paths: Vec<PathBuf>,
        /// Path to the Kùzu graph database directory
        pub graph_db_path: PathBuf,
        /// Path to the Tantivy index directory
        pub tantivy_index_path: PathBuf,
        /// Unix socket path for IPC
        pub ipc_socket_path: PathBuf,
        /// Unix socket path for embedding service
        pub embedding_socket_path: PathBuf,
        /// CPU usage ceiling (0–100) before governor pauses embedding
        pub cpu_threshold_pct: f32,
        /// Depth of the in-memory indexing queue
        pub index_queue_depth: usize,
    }

    impl DaemonConfig {
        pub fn load() -> Result<Self> {
            let cfg = ::config::Config::builder()
                .add_source(::config::File::with_name("/etc/sentinel/daemon").required(false))
                .add_source(::config::File::with_name("sentinel-daemon").required(false))
                .add_source(::config::Environment::with_prefix("SENTINEL"))
                .set_default("cpu_threshold_pct", 15.0)?
                .set_default("index_queue_depth", 1024)?
                .set_default("ipc_socket_path", "/run/sentinel/daemon.sock")?
                .set_default("embedding_socket_path", "/run/sentinel/embedding.sock")?
                .set_default("graph_db_path", "/var/sentinel/graph")?
                .set_default("tantivy_index_path", "/var/sentinel/tantivy")?
                .set_default("watch_paths", vec!["/home"])?
                .build()?;
            Ok(cfg.try_deserialize()?)
        }
    }
}

mod privilege {
    use anyhow::{bail, Result};

    pub fn assert_capable() -> Result<()> {
        // Simplest portable check: effective UID 0 implies all capabilities.
        // For production: use the `caps` crate to check CAP_SYS_ADMIN directly.
        let euid = unsafe { libc::geteuid() };
        if euid != 0 {
            bail!(
                "sentinel-daemon requires CAP_SYS_ADMIN (fanotify). \
                 Run as root or grant the capability via: \
                 setcap cap_sys_admin+ep /usr/bin/sentinel-daemon"
            );
        }
        Ok(())
    }
}

mod tantivy_index {
    use anyhow::Result;
    use std::path::Path;
    use tantivy::{
        doc,
        schema::{Schema, TEXT, STORED, STRING, Value},
        Index, IndexWriter, TantivyDocument,
        query::QueryParser,
        collector::TopDocs,
    };

    pub struct LexicalIndex {
        index:  Index,
        writer: IndexWriter,
        // Schema fields
        pub f_path:    tantivy::schema::Field,
        pub f_name:    tantivy::schema::Field,
        pub f_content: tantivy::schema::Field,
    }

    impl LexicalIndex {
        pub fn open_or_create(path: &Path) -> Result<Self> {
            std::fs::create_dir_all(path)?;

            let mut schema_builder = Schema::builder();
            let f_path    = schema_builder.add_text_field("path",    STRING | STORED);
            let f_name    = schema_builder.add_text_field("name",    TEXT   | STORED);
            let f_content = schema_builder.add_text_field("content", TEXT);
            let schema    = schema_builder.build();

            let index = if path.join("meta.json").exists() {
                Index::open_in_dir(path)?
            } else {
                Index::create_in_dir(path, schema)?
            };

            let writer = index.writer(50_000_000)?; // 50 MB heap
            Ok(Self { index, writer, f_path, f_name, f_content })
        }

        /// Upsert a document. Tantivy doesn't support true upsert — we
        /// delete-by-path then re-add. Safe because we hold the writer lock.
        pub fn upsert(&mut self, path: &str, name: &str, content: &str) -> Result<()> {
            use tantivy::Term;
            let term = Term::from_field_text(self.f_path, path);
            self.writer.delete_term(term);
            self.writer.add_document(doc!(
                self.f_path    => path,
                self.f_name    => name,
                self.f_content => content,
            ))?;
            Ok(())
        }

        pub fn commit(&mut self) -> Result<()> {
            self.writer.commit()?;
            Ok(())
        }

        /// BM25 search — Tier 3 fallback retrieval.
        pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<String>> {
            let reader  = self.index.reader()?;
            let searcher = reader.searcher();
            let qp = QueryParser::for_index(
                &self.index,
                vec![self.f_name, self.f_content],
            );
            let query   = qp.parse_query(query_str)?;
            let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
            let mut results = Vec::new();
            for (_score, addr) in top_docs {
                let doc: TantivyDocument = searcher.doc(addr)?;
                if let Some(v) = doc.get_first(self.f_path) {
                    if let Some(s) = v.as_str() {
                        results.push(s.to_string());
                    }
                }
            }
            Ok(results)
        }
    }
}

async fn run_bulk_index(watch_paths: Vec<std::path::PathBuf>, tx: tokio::sync::mpsc::Sender<crate::daemon::IndexTask>) {
    use std::time::{SystemTime, UNIX_EPOCH};
    use walkdir::WalkDir;
    tracing::info!("Starting bulk index run on paths: {:?}", watch_paths);
    let mut count = 0;
    tokio::task::spawn_blocking(move || {
        for root in watch_paths {
            for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
                if entry.file_type().is_file() {
                    let path = entry.into_path();
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        let ext = ext.to_lowercase();
                        if ["mp4", "mkv", "mp3", "jpg", "png", "zip", "tar", "gz", "exe", "dll", "so", "o", "a", "pyc", "class"].contains(&ext.as_str()) {
                            continue;
                        }
                    }
                    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    let task = crate::daemon::IndexTask { path, timestamp };
                    if let Err(e) = tx.blocking_send(task) {
                        tracing::error!("Failed to send bulk index task: {}", e);
                        break;
                    }
                    count += 1;
                }
            }
        }
        tracing::info!("Bulk index complete. Queued {} files.", count);
    }).await.unwrap();
}
