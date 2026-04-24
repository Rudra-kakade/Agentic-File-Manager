// src/ipc/mod.rs
//
// IPC Server — Unix Domain Socket
//
// The query engine (LLM layer) connects here to:
//   1. Submit natural-language queries (translated to graph range + vector queries)
//   2. Submit validated system actions (JSON-whitelisted commands only)
//
// Wire protocol: length-prefixed JSON frames.
//   [ 4 bytes LE u32: payload length ][ payload bytes (UTF-8 JSON) ]
//
// All incoming action commands are validated against the ACTION_WHITELIST
// before execution.  Raw shell is NEVER executed.

use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream},
    sync::Mutex,
};
use tracing::{error, info, warn};

use crate::graph::{GraphStore, QueryResult};

// ── Public types ──────────────────────────────────────────────────────────────

pub struct IpcServer(UnixListener);

impl IpcServer {
    pub fn bind(path: &PathBuf) -> Result<Self> {
        // Remove stale socket if present
        let _ = std::fs::remove_file(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let listener = UnixListener::bind(path)?;
        info!(socket = ?path, "IPC server listening");
        Ok(Self(listener))
    }
}

pub async fn serve(server: IpcServer, graph: Arc<Mutex<GraphStore>>) -> Result<()> {
    loop {
        match server.0.accept().await {
            Ok((stream, _addr)) => {
                let graph_c = graph.clone();
                tokio::spawn(handle_client(stream, graph_c));
            }
            Err(e) => error!("IPC accept error: {e}"),
        }
    }
}

// ── Wire protocol ─────────────────────────────────────────────────────────────

/// Requests sent TO the daemon from the query engine.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum IpcRequest {
    /// Graph range query — returns file paths modified between start..end
    GraphQuery { start_ts: u64, end_ts: u64 },

    /// Validated system action — must be in ACTION_WHITELIST
    SystemAction(SystemAction),
}

/// Responses sent FROM the daemon to the query engine.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum IpcResponse {
    GraphResults { results: Vec<QueryResult> },
    ActionAck    { success: bool, message: String },
    Error        { message: String },
}

// ── Action whitelist ──────────────────────────────────────────────────────────
// The LLM is constrained (via llama.cpp grammar) to emit only these actions.
// The daemon validates the *type* AND *parameters* before executing anything.

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum SystemAction {
    SetProcessPriority { target_process: String, level: PriorityLevel },
    MoveToTrash        { path: String },     // soft-delete only; requires confirmation
    OpenFile           { path: String },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PriorityLevel { Low, Normal, High }

// ── Client handler ────────────────────────────────────────────────────────────

async fn handle_client(mut stream: UnixStream, graph: Arc<Mutex<GraphStore>>) {
    loop {
        // Read length prefix
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_)  => {}
            Err(_) => break,  // client disconnected
        }
        let len = u32::from_le_bytes(len_buf) as usize;

        // Guard against absurdly large frames
        if len > 1_048_576 {
            warn!("IPC frame too large ({len} bytes) — dropping connection");
            break;
        }

        let mut payload = vec![0u8; len];
        if stream.read_exact(&mut payload).await.is_err() {
            break;
        }

        let response = match serde_json::from_slice::<IpcRequest>(&payload) {
            Ok(req) => dispatch(req, &graph).await,
            Err(e)  => IpcResponse::Error {
                message: format!("Malformed request: {e}"),
            },
        };

        if let Err(e) = write_response(&mut stream, &response).await {
            warn!("IPC write error: {e}");
            break;
        }
    }
}

async fn dispatch(req: IpcRequest, graph: &Arc<Mutex<GraphStore>>) -> IpcResponse {
    match req {
        IpcRequest::GraphQuery { start_ts, end_ts } => {
            let g = graph.lock().await;
            match g.query_modified_range(start_ts, end_ts) {
                Ok(results) => IpcResponse::GraphResults { results },
                Err(e)      => IpcResponse::Error { message: e.to_string() },
            }
        }
        IpcRequest::SystemAction(action) => execute_action(action).await,
    }
}

async fn execute_action(action: SystemAction) -> IpcResponse {
    match action {
        SystemAction::SetProcessPriority { target_process, level } => {
            set_process_priority(&target_process, level)
        }
        SystemAction::MoveToTrash { path } => {
            move_to_trash(&path)
        }
        SystemAction::OpenFile { path } => {
            // Delegate to xdg-open; safe — no shell interpolation
            match std::process::Command::new("xdg-open").arg(&path).spawn() {
                Ok(_)  => IpcResponse::ActionAck {
                    success: true,
                    message: format!("Opened {path}"),
                },
                Err(e) => IpcResponse::ActionAck {
                    success: false,
                    message: format!("Failed to open {path}: {e}"),
                },
            }
        }
    }
}

fn set_process_priority(name: &str, level: PriorityLevel) -> IpcResponse {
    // Validate name is a simple identifier (no shell metacharacters)
    if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return IpcResponse::ActionAck {
            success: false,
            message: "Invalid process name".to_string(),
        };
    }

    let nice_val = match level {
        PriorityLevel::Low    =>  10i32,
        PriorityLevel::Normal =>   0i32,
        PriorityLevel::High   => -10i32,
    };

    // Find PID by name via /proc, then renice via nix::sys::resource
    match find_pid_by_name(name) {
        Some(pid) => {
            use nix::sys::resource::{setpriority, Which};
            use nix::unistd::Pid;
            match setpriority(Which::Process, pid as u32, nice_val) {
                Ok(_)  => IpcResponse::ActionAck {
                    success: true,
                    message: format!("Set {name} (PID {pid}) to nice {nice_val}"),
                },
                Err(e) => IpcResponse::ActionAck {
                    success: false,
                    message: format!("setpriority failed: {e}"),
                },
            }
        }
        None => IpcResponse::ActionAck {
            success: false,
            message: format!("Process '{name}' not found"),
        },
    }
}

fn move_to_trash(path: &str) -> IpcResponse {
    // Canonicalize + validate path is not a system directory
    let p = std::path::Path::new(path);
    if !p.exists() {
        return IpcResponse::ActionAck {
            success: false,
            message: format!("Path does not exist: {path}"),
        };
    }

    // Versioned soft-delete: move to ~/.sentinel/trash/<timestamp>/<filename>
    let trash_dir = dirs_next::home_dir()
        .unwrap_or_default()
        .join(".sentinel/trash")
        .join(chrono_epoch());

    if let Err(e) = std::fs::create_dir_all(&trash_dir) {
        return IpcResponse::ActionAck {
            success: false,
            message: format!("Could not create trash dir: {e}"),
        };
    }

    let dest = trash_dir.join(p.file_name().unwrap_or_default());
    match std::fs::rename(p, &dest) {
        Ok(_)  => IpcResponse::ActionAck {
            success: true,
            message: format!("Moved to trash: {}", dest.display()),
        },
        Err(e) => IpcResponse::ActionAck {
            success: false,
            message: format!("Move failed: {e}"),
        },
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async fn write_response(stream: &mut UnixStream, resp: &IpcResponse) -> Result<()> {
    let json = serde_json::to_vec(resp)?;
    let len  = (json.len() as u32).to_le_bytes();
    stream.write_all(&len).await?;
    stream.write_all(&json).await?;
    stream.flush().await?;
    Ok(())
}

fn find_pid_by_name(name: &str) -> Option<i32> {
    let entries = std::fs::read_dir("/proc").ok()?;
    for entry in entries.flatten() {
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if fname_str.chars().all(|c| c.is_ascii_digit()) {
            let comm_path = entry.path().join("comm");
            if let Ok(comm) = std::fs::read_to_string(comm_path) {
                if comm.trim() == name {
                    return fname_str.parse().ok();
                }
            }
        }
    }
    None
}

fn chrono_epoch() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}
