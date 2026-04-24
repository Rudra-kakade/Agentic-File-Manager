// src/ipc.rs
//
// Unix domain socket IPC server for the daemon.
// Accepts commands from the orchestrator / query engine.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::graph::GraphBackend;
use crate::daemon::{IndexTask, ProgressEvent};

pub struct Server {
    socket_path: PathBuf,
    graph: Arc<dyn GraphBackend>,
    _index_tx: mpsc::Sender<IndexTask>,
    _progress_rx: Arc<Mutex<mpsc::Receiver<ProgressEvent>>>,
}

impl Server {
    pub fn new(
        socket_path: PathBuf,
        graph: Arc<dyn GraphBackend>,
        index_tx: mpsc::Sender<IndexTask>,
        progress_rx: mpsc::Receiver<ProgressEvent>,
    ) -> Self {
        Self {
            socket_path,
            graph,
            _index_tx: index_tx,
            _progress_rx: Arc::new(Mutex::new(progress_rx)),
        }
    }

    /// Run the server until the cancellation token is triggered.
    pub async fn run(&self, cancel: CancellationToken) -> Result<()> {
        // Remove stale socket if it exists
        let _ = std::fs::remove_file(&self.socket_path);
        if let Some(parent) = self.socket_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let listener = UnixListener::bind(&self.socket_path)?;
        info!(path = ?self.socket_path, "IPC server listening");

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    info!("IPC server shutting down");
                    break;
                }
                accept_res = listener.accept() => {
                    match accept_res {
                        Ok((stream, _addr)) => {
                            let graph = self.graph.clone();
                            tokio::spawn(async move {
                                if let Err(e) = handle_client(stream, graph).await {
                                    warn!("IPC client error: {e}");
                                }
                            });
                        }
                        Err(e) => {
                            error!("IPC accept error: {e}");
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

async fn handle_client(mut stream: UnixStream, graph: Arc<dyn GraphBackend>) -> Result<()> {
    loop {
        // Length-prefixed framing: read 4-byte LE u32, then payload
        let mut len_buf = [0u8; 4];
        if stream.read_exact(&mut len_buf).await.is_err() {
            return Ok(()); // Connection closed
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        
        // Safety: don't allocate more than 10MB
        if len > 10 * 1024 * 1024 {
            warn!(len, "IPC payload too large");
            return Ok(());
        }

        let mut buf = vec![0u8; len];
        if stream.read_exact(&mut buf).await.is_err() {
            return Ok(());
        }

        // Parse and handle command
        match serde_json::from_slice::<serde_json::Value>(&buf) {
            Ok(cmd) => {
                let response = handle_command(cmd, &graph).await;
                let resp_bytes = serde_json::to_vec(&response)?;
                let resp_len = (resp_bytes.len() as u32).to_le_bytes();
                stream.write_all(&resp_len).await?;
                stream.write_all(&resp_bytes).await?;
            }
            Err(e) => {
                error!("Invalid IPC payload: {e}");
            }
        }
    }
}

async fn handle_command(
    cmd: serde_json::Value,
    graph: &Arc<dyn GraphBackend>,
) -> serde_json::Value {
    // Simple query handler: { "type": "query_range", "start": u64, "end": u64 }
    if let Some(typ) = cmd.get("type").and_then(|v| v.as_str()) {
        match typ {
            "query_range" => {
                let start = cmd.get("start").and_then(|v| v.as_u64());
                let end = cmd.get("end").and_then(|v| v.as_u64());
                let results = graph.query_time_range(start, end, None, 100);
                serde_json::json!({ "ok": true, "results": results })
            }
            "file_count" => {
                let count = graph.file_count();
                serde_json::json!({ "ok": true, "count": count })
            }
            _ => serde_json::json!({ "ok": false, "error": format!("Unknown command: {typ}") }),
        }
    } else {
        serde_json::json!({ "ok": false, "error": "Missing 'type' field" })
    }
}

