// src/ipc.rs
//
// Unix domain socket IPC server for the daemon.
// Accepts commands from the orchestrator / query engine.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::Mutex;
use tracing::{error, info};

use crate::graph::GraphStore;

pub struct IpcServer {
    listener: UnixListener,
}

impl IpcServer {
    pub fn bind(path: &Path) -> Result<Self> {
        // Remove stale socket if it exists
        let _ = std::fs::remove_file(path);
        let listener = UnixListener::bind(path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o666));
        }
        info!(path = ?path, "IPC socket bound");
        Ok(Self { listener })
    }
}

/// Accept connections and handle commands.
pub async fn serve(server: IpcServer, graph: Arc<Mutex<GraphStore>>) -> Result<()> {
    loop {
        match server.listener.accept().await {
            Ok((mut stream, _addr)) => {
                let graph = graph.clone();
                tokio::spawn(async move {
                    // Length-prefixed framing: read 4-byte LE u32, then payload
                    let mut len_buf = [0u8; 4];
                    if stream.read_exact(&mut len_buf).await.is_err() {
                        return;
                    }
                    let len = u32::from_le_bytes(len_buf) as usize;
                    let mut buf = vec![0u8; len];
                    if stream.read_exact(&mut buf).await.is_err() {
                        return;
                    }

                    // Parse and handle command
                    match serde_json::from_slice::<serde_json::Value>(&buf) {
                        Ok(cmd) => {
                            let response = handle_command(cmd, &graph).await;
                            let resp_bytes = serde_json::to_vec(&response)
                                .unwrap_or_default();
                            let resp_len = (resp_bytes.len() as u32).to_le_bytes();
                            let _ = stream.write_all(&resp_len).await;
                            let _ = stream.write_all(&resp_bytes).await;
                        }
                        Err(e) => {
                            error!("Invalid IPC payload: {e}");
                        }
                    }
                });
            }
            Err(e) => {
                error!("IPC accept error: {e}");
            }
        }
    }
}

async fn handle_command(
    cmd: serde_json::Value,
    graph: &Arc<Mutex<GraphStore>>,
) -> serde_json::Value {
    // Simple query handler: { "type": "query_range", "start": u64, "end": u64 }
    if let Some(typ) = cmd.get("type").and_then(|v| v.as_str()) {
        match typ {
            "query_range" => {
                let start = cmd.get("start").and_then(|v| v.as_u64()).unwrap_or(0);
                let end = cmd.get("end").and_then(|v| v.as_u64()).unwrap_or(u64::MAX);
                let g = graph.lock().await;
                match g.query_modified_range(start, end) {
                    Ok(results) => serde_json::json!({ "ok": true, "results": results }),
                    Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
                }
            }
            _ => serde_json::json!({ "ok": false, "error": format!("Unknown command: {typ}") }),
        }
    } else {
        serde_json::json!({ "ok": false, "error": "Missing 'type' field" })
    }
}
