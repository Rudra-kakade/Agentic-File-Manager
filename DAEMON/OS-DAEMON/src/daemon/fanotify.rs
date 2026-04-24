// src/daemon/fanotify.rs
//
// Wraps the Linux fanotify API via `nix` to listen for file create/modify
// events across one or more mount points.  For each event it:
//   1. Writes an edge to the Kùzu Knowledge Graph  (CREATED_BY / MODIFIED_BY)
//   2. Sends an IndexTask to the embedding queue
//
// Design notes:
//   • We use FAN_CLASS_NOTIF (read-only, no permission events) to keep the
//     daemon non-blocking on the watched processes.
//   • fanotify gives us an open fd per event; we immediately resolve the path
//     via /proc/self/fd/<fd> and close the fd to avoid fd exhaustion.
//   • All graph writes are batched in a small Vec and flushed every 200 events
//     or 500 ms to reduce lock contention.

use std::{
    os::unix::io::{AsRawFd, AsFd},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use nix::sys::fanotify::{
    Fanotify, EventFFlags, InitFlags, MarkFlags, MaskFlags,
};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, warn};

use crate::{
    daemon::IndexTask,
    graph::{EventKind, FileEvent, GraphStore},
};

/// Spawn the fanotify listener.  Runs forever; returns only on fatal error.
pub async fn run(
    watch_paths: Vec<PathBuf>,
    graph: Arc<Mutex<GraphStore>>,
    index_tx: mpsc::Sender<IndexTask>,
) -> Result<()> {
    // fanotify is a synchronous (blocking) API.  We run it on a dedicated
    // blocking thread so it never stalls the Tokio executor.
    let (ev_tx, mut ev_rx) =
        mpsc::channel::<FileEvent>(4096);

    // ── Blocking thread: raw fanotify loop ──────────────────────────────────
    std::thread::spawn(move || {
        if let Err(e) = fanotify_loop(&watch_paths, ev_tx) {
            error!("fanotify loop exited with error: {e:#}");
        }
    });

    // ── Async side: consume events, write graph, enqueue embedding ──────────
    let mut batch: Vec<FileEvent> = Vec::with_capacity(200);
    let flush_interval = Duration::from_millis(500);
    let mut interval = tokio::time::interval(flush_interval);

    loop {
        tokio::select! {
            Some(ev) = ev_rx.recv() => {
                batch.push(ev);
                if batch.len() >= 200 {
                    flush(&mut batch, &graph, &index_tx).await;
                }
            }
            _ = interval.tick() => {
                if !batch.is_empty() {
                    flush(&mut batch, &graph, &index_tx).await;
                }
            }
        }
    }
}

/// Flush a batch of events to the graph and the embedding queue.
async fn flush(
    batch: &mut Vec<FileEvent>,
    graph: &Arc<Mutex<GraphStore>>,
    index_tx: &mpsc::Sender<IndexTask>,
) {
    {
        let mut g = graph.lock().await;
        for ev in batch.iter() {
            if let Err(e) = g.record_event(ev) {
                warn!(path = ?ev.path, "Graph write error: {e:#}");
            }
        }
    }

    for ev in batch.drain(..) {
        let task = IndexTask {
            path:      ev.path.clone(),
            timestamp: ev.timestamp,
        };
        if index_tx.try_send(task).is_err() {
            warn!(path = ?ev.path, "Index queue full — dropping embedding task");
        }
    }
}

// ── Blocking fanotify loop (runs on std::thread) ─────────────────────────────

fn fanotify_loop(
    watch_paths: &[PathBuf],
    ev_tx: mpsc::Sender<FileEvent>,
) -> Result<()> {
    // FAN_CLASS_NOTIF: notification-only (no blocking/permission events)
    // FAN_CLOEXEC + FAN_NONBLOCK: fd lifecycle hygiene
    let fan = Fanotify::init(
        InitFlags::FAN_CLASS_NOTIF | InitFlags::FAN_CLOEXEC | InitFlags::FAN_NONBLOCK,
        EventFFlags::O_RDONLY | EventFFlags::O_LARGEFILE | EventFFlags::O_CLOEXEC,
    )?;

    // Mark each configured path (entire mount point beneath it)
    for path in watch_paths {
        fan.mark(
            MarkFlags::FAN_MARK_ADD | MarkFlags::FAN_MARK_MOUNT,
            MaskFlags::FAN_CLOSE_WRITE | MaskFlags::FAN_CREATE | MaskFlags::FAN_MOVED_TO,
            None,   // dirfd = AT_FDCWD
            Some(path.as_path()),
        )?;
        tracing::info!(path = ?path, "fanotify mark registered");
    }

    let mut poll_fds = [nix::poll::PollFd::new(
        fan.as_fd(),
        nix::poll::PollFlags::POLLIN,
    )];

    loop {
        // Block until events arrive (1 second timeout for graceful shutdown)
        match nix::poll::poll(&mut poll_fds, nix::poll::PollTimeout::from(1000u16)) {
            Ok(0)  => continue,  // timeout — loop again
            Ok(_)  => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(e) => return Err(e.into()),
        }

        let events = fan.read_events()?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for event in events {
            // Resolve path via /proc/self/fd/<fd>
            if let Some(fd) = event.fd() {
                let proc_path = format!("/proc/self/fd/{}", fd.as_raw_fd());
                match std::fs::read_link(&proc_path) {
                    Ok(real_path) => {
                        // Skip our own daemon writes to avoid feedback loops
                        if event.pid() == std::process::id() as i32 {
                            continue;
                        }

                        let kind = mask_to_kind(event.mask());
                        let file_ev = FileEvent {
                            path:      real_path,
                            kind,
                            pid:       event.pid() as u32,
                            timestamp: now,
                        };

                        debug!(path = ?file_ev.path, kind = ?file_ev.kind, "fanotify event");

                        // Non-blocking send; if buffer is full we'll warn upstream
                        let _ = ev_tx.blocking_send(file_ev);
                    }
                    Err(e) => {
                        warn!("Could not resolve fd path {proc_path}: {e}");
                    }
                }
                // CRITICAL: Close the fd immediately to avoid fd exhaustion.
                // The nix wrapper does this on drop.
            }
        }
    }
}

fn mask_to_kind(mask: MaskFlags) -> EventKind {
    if mask.contains(MaskFlags::FAN_CREATE) || mask.contains(MaskFlags::FAN_MOVED_TO) {
        EventKind::Created
    } else {
        EventKind::Modified
    }
}
