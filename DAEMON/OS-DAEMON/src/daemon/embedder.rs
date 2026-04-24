// src/daemon/embedder.rs
//
// Embedding dispatch worker.
//
// Reads IndexTasks from the bounded channel, checks the ResourceGovernor
// before each task, extracts file text, and forwards to the MiniLM engine
// via Unix-domain socket IPC.  On IPC failure it falls back to just
// updating the Tantivy BM25 index so Tier-3 retrieval still works.

use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::Result;
use tokio::{
    io::AsyncWriteExt,
    net::UnixStream,
    sync::{mpsc, Mutex},
    time::sleep,
};
use tracing::{debug, info, warn};

use crate::{
    daemon::IndexTask,
    extractor,
    governor::ResourceGovernor,
    tantivy_index::LexicalIndex,
};

/// Entry point for the embedding worker task.
pub async fn run(
    mut index_rx: mpsc::Receiver<IndexTask>,
    governor: ResourceGovernor,
    lexical: Arc<Mutex<LexicalIndex>>,
    ipc_sock: PathBuf,
) -> Result<()> {
    info!("Embedder worker started");

    // Tantivy commit batching
    let mut uncommitted: u32 = 0;
    const COMMIT_EVERY: u32 = 50;

    while let Some(task) = index_rx.recv().await {
        // ── Governor check ────────────────────────────────────────────────
        // If CPU is above threshold, park until it drops.
        while governor.should_yield() {
            debug!("Governor: CPU threshold exceeded — yielding");
            sleep(Duration::from_secs(2)).await;
        }

        // ── Text extraction ───────────────────────────────────────────────
        let content = match extractor::extract_text(&task.path).await {
            Ok(c)  => c,
            Err(e) => {
                warn!(path = ?task.path, "Extraction failed: {e:#}");
                String::new()
            }
        };

        let path_str = task.path.to_string_lossy().to_string();
        let file_name = task.path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // ── Tantivy upsert (always, regardless of IPC) ────────────────────
        {
            let mut lex = lexical.lock().await;
            if let Err(e) = lex.upsert(&path_str, &file_name, &content) {
                warn!("Tantivy upsert error: {e:#}");
            }
            uncommitted += 1;
            if uncommitted >= COMMIT_EVERY {
                let _ = lex.commit();
                uncommitted = 0;
            }
        }

        // ── Forward to MiniLM engine via IPC ──────────────────────────────
        if !content.is_empty() {
            let msg = EmbedRequest {
                path:      path_str,
                timestamp: task.timestamp,
                content:   content.chars().take(8192).collect(), // cap at ~8K chars
            };

            if let Err(e) = send_embed_request(&ipc_sock, &msg).await {
                warn!("IPC embed send failed (MiniLM engine may be offline): {e:#}");
                // Non-fatal: Tantivy already indexed it for Tier-3 fallback
            }
        }
    }

    // Flush any remaining tantivy documents on shutdown
    let mut lex = lexical.lock().await;
    let _ = lex.commit();

    info!("Embedder worker exiting");
    Ok(())
}

#[derive(serde::Serialize)]
struct EmbedRequest {
    path:      String,
    timestamp: u64,
    content:   String,
}

async fn send_embed_request(sock_path: &PathBuf, req: &EmbedRequest) -> Result<()> {
    let json    = serde_json::to_vec(req)?;
    let mut stream = UnixStream::connect(sock_path).await?;
    // Length-prefixed framing: 4-byte little-endian u32 length + payload
    let len = (json.len() as u32).to_le_bytes();
    stream.write_all(&len).await?;
    stream.write_all(&json).await?;
    stream.flush().await?;
    Ok(())
}
