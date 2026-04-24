# sentinel-daemon

The core OS event daemon for SentinelAI.  Written in Rust.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     sentinel-daemon                          │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │  fanotify   │───▶│ Graph Store  │    │  IPC Server     │ │
│  │  Listener   │    │ (Kùzu/Mock)  │◀───│  (Unix socket)  │ │
│  └──────┬──────┘    └──────────────┘    └─────────────────┘ │
│         │                                                    │
│         ▼  IndexTask                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │  Embedder   │───▶│  Resource    │    │ Tantivy Lexical │ │
│  │  Worker     │    │  Governor    │    │ Index (BM25)    │ │
│  └──────┬──────┘    └──────────────┘    └─────────────────┘ │
│         │                                                    │
│         ▼ IPC (Unix socket)                                  │
│  ┌─────────────┐                                             │
│  │  MiniLM     │   (separate process / Python)               │
│  │  Engine     │                                             │
│  └─────────────┘                                             │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Linux kernel ≥ 5.1 (fanotify `FAN_REPORT_FID`)
- Rust stable ≥ 1.75
- `poppler-utils` (for PDF extraction): `apt install poppler-utils`
- Root or `CAP_SYS_ADMIN` capability

## Build

```bash
cargo build --release
```

## Install

```bash
# Copy binary
sudo cp target/release/sentinel-daemon /usr/bin/

# Copy config
sudo mkdir -p /etc/sentinel
sudo cp sentinel-daemon.toml /etc/sentinel/daemon.toml

# Create data directories
sudo mkdir -p /var/sentinel/{graph,tantivy}
sudo mkdir -p /run/sentinel

# Install systemd service
sudo cp sentinel-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentinel-daemon

# Verify
sudo systemctl status sentinel-daemon
journalctl -u sentinel-daemon -f
```

## Configuration

| Key                  | Default                       | Description                              |
|----------------------|-------------------------------|------------------------------------------|
| `watch_paths`        | `["/home", "/var/projects"]`  | Paths to monitor (fanotify marks mounts) |
| `graph_db_path`      | `/var/sentinel/graph`         | Kùzu graph database directory            |
| `tantivy_index_path` | `/var/sentinel/tantivy`       | Tantivy BM25 index directory             |
| `ipc_socket_path`    | `/run/sentinel/daemon.sock`   | Unix socket for query engine IPC         |
| `cpu_threshold_pct`  | `15.0`                        | CPU % ceiling before embedding pauses    |
| `index_queue_depth`  | `1024`                        | Bounded queue depth for indexing tasks   |

Override via:
1. `/etc/sentinel/daemon.toml`
2. `sentinel-daemon.toml` (current directory)
3. Environment variables prefixed `SENTINEL_` (e.g. `SENTINEL_CPU_THRESHOLD_PCT=10`)

## IPC Protocol

The query engine connects to the Unix socket and sends length-prefixed JSON frames:

```
[ 4 bytes LE u32: payload_len ][ payload_len bytes: UTF-8 JSON ]
```

### Request: Graph range query
```json
{ "type": "graph_query", "start_ts": 1709472000, "end_ts": 1710076800 }
```

### Request: System action
```json
{ "type": "system_action", "action": "set_process_priority",
  "target_process": "vscode", "level": "high" }
```

### Response
```json
{ "type": "graph_results", "results": [ { "path": "/home/...", "timestamp": ..., "score": 1.0 } ] }
```

## Module Map

| Module              | Responsibility                                          |
|---------------------|---------------------------------------------------------|
| `main.rs`           | Bootstrap, config, signal handling                      |
| `daemon/fanotify`   | Raw fanotify listener; writes graph + queues embedding  |
| `daemon/embedder`   | Queue consumer; extracts text; calls MiniLM via IPC     |
| `graph/`            | Kùzu abstraction + MockGraphStore                       |
| `governor/`         | CPU sampling; `should_yield()` predicate                |
| `ipc/`              | Unix socket server; action whitelist enforcement        |
| `extractor/`        | File content extraction (PDF, DOCX, XLSX, text, code)   |

## Enabling Kùzu (when kuzu-rs is stable)

1. Add to `Cargo.toml`: `kuzu = "0.3"`
2. In `src/graph/mod.rs`, uncomment `KuzuGraphStore` and the DDL
3. Change `GraphStore::open` to use `KuzuGraphStore::open(path)?`

## Next Steps

- [ ] MiniLM embedding engine (Python / ONNX sidecar)
- [ ] hnswlib vector store integration
- [ ] LLM query translation layer (Llama-3 8B GGUF)
- [ ] Installer + sysctl configuration helper
