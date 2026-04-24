# sentinel-embedding

MiniLM embedding engine + hnswlib vector store for SentinelAI.

## Architecture

```
sentinel-daemon (Rust)
    │
    │  IndexTask (path, timestamp, content)
    │  Unix socket: /run/sentinel/embedding.sock
    ▼
sentinel-embedding                         sentinel-vectorstore
┌─────────────────────────────┐           ┌────────────────────────────┐
│  EmbeddingEngine            │           │  VectorStoreServer         │
│                             │           │                            │
│  asyncio socket server      │  upsert   │  hnswlib.Index             │
│  → dynamic batching (32)    │ ────────▶ │  dim=384, cosine space     │
│  → ONNX Runtime inference   │           │  max_elements=2M           │
│  → MiniLM L6 v2             │  search   │                            │
│                             │ ◀──────── │  path↔id mapping           │
│  ResourceGovernor monitor   │           │  soft-delete + compaction  │
│  (pauses if CPU > 15%)      │           │  auto-persist every 60s    │
└─────────────────────────────┘           └────────────────────────────┘
                                                    ▲
                              search(vec, k,        │
                              filter_paths)         │
                                                    │
                                          Query Engine (future)
```

## Quick Start

### 1. Install Python dependencies

```bash
pip install -e ".[dev]"
```

### 2. Download and export the model (one-time)

```bash
pip install -e ".[download]"
python scripts/download_model.py --output /var/sentinel/models/minilm
```

This downloads `all-MiniLM-L6-v2` from HuggingFace (~90 MB) and exports it
to ONNX format (~22 MB).  PyTorch is only needed for this step.

### 3. Start the vector store

```bash
SENTINEL_VECTORSTORE_PATH=/var/sentinel/vectorstore sentinel-vectorstore
```

### 4. Start the embedding engine

```bash
SENTINEL_MODEL_PATH=/var/sentinel/models/minilm sentinel-embedding
```

### 5. Install as systemd services

```bash
sudo cp sentinel-vectorstore.service /etc/systemd/system/
sudo cp sentinel-embedding.service   /etc/systemd/system/

sudo useradd --system --no-create-home --shell /usr/sbin/nologin sentinel

sudo systemctl daemon-reload
sudo systemctl enable --now sentinel-vectorstore sentinel-embedding

journalctl -u sentinel-vectorstore -u sentinel-embedding -f
```

## Module Map

| File                    | Role                                                   |
|-------------------------|--------------------------------------------------------|
| `src/engine.py`         | Main embedding engine — socket server + batch worker   |
| `src/model.py`          | ONNX Runtime wrapper — tokenise + mean-pool + L2 norm  |
| `src/vector_store.py`   | hnswlib server — upsert, search, persist, compact      |
| `src/vector_store_client.py` | Async client for both embedding engine + query engine |
| `src/config.py`         | Config dataclasses — override via env vars             |
| `scripts/download_model.py` | One-time model download + ONNX export              |
| `tests/test_pipeline.py`| Unit + integration tests                               |

## IPC Protocol

Same length-prefixed JSON as the daemon:
`[ 4 bytes LE uint32: len ][ len bytes UTF-8 JSON ]`

### Embedding engine (listens on `SENTINEL_EMBEDDING_SOCKET`)

Accepts IndexTask frames from the Rust daemon — no response needed (fire-and-forget).

```json
{"path": "/home/user/report.pdf", "timestamp": 1709472000, "content": "HDFS replication..."}
```

### Vector store (listens on `SENTINEL_VECTORSTORE_SOCKET`)

**Upsert** (from embedding engine):
```json
{"op": "upsert", "path": "/home/user/report.pdf", "timestamp": 1709472000, "vector": [...384 floats...]}
```

**Hybrid search** (from query engine — graph pre-filtered):
```json
{"op": "search", "vector": [...384 floats...], "k": 10, "filter_paths": ["/home/...", ...]}
```

**Pure semantic search** (Tier-2 fallback):
```json
{"op": "search", "vector": [...384 floats...], "k": 10}
```

**Response**:
```json
{"ok": true, "results": [{"path": "/home/user/report.pdf", "score": 0.94}, ...]}
```

## Memory Sizing

| Component       | ~Memory for 1M vectors |
|-----------------|------------------------|
| hnswlib index   | 1.5 GB (dim=384, M=16) |
| id↔path mapping | ~150 MB (avg 150-char path) |
| ONNX model      | ~200 MB (model + runtime) |
| **Total**       | **~2 GB**              |

## Running Tests

```bash
# Without model (index + server tests only):
pytest tests/ -v

# With model (full semantic tests):
python scripts/download_model.py --output /tmp/test-model
SENTINEL_MODEL_PATH=/tmp/test-model pytest tests/ -v
```

## Environment Variables

| Variable                      | Default                          | Description                     |
|-------------------------------|----------------------------------|---------------------------------|
| `SENTINEL_MODEL_PATH`         | `/var/sentinel/models/minilm`    | ONNX model directory            |
| `SENTINEL_EMBEDDING_SOCKET`   | `/run/sentinel/embedding.sock`   | Engine listen socket            |
| `SENTINEL_VECTORSTORE_SOCKET` | `/run/sentinel/vectorstore.sock` | Vector store socket             |
| `SENTINEL_BATCH_SIZE`         | `32`                             | Max docs per inference batch    |
| `SENTINEL_BATCH_TIMEOUT_S`    | `1.0`                            | Max seconds before flush        |
| `SENTINEL_INFERENCE_THREADS`  | `4`                              | ONNX Runtime thread count       |
| `SENTINEL_VECTORSTORE_PATH`   | `/var/sentinel/vectorstore`      | Index persistence directory     |
| `SENTINEL_VECTORSTORE_MAX_EL` | `2000000`                        | Max vectors (pre-allocated)     |
| `SENTINEL_VECTORSTORE_PERSIST_S` | `60.0`                        | Auto-persist interval (seconds) |
| `SENTINEL_LOG_LEVEL`          | `INFO`                           | Logging level                   |

## Next Steps

- [ ] LLM query translation layer (Llama-3 8B GGUF) — translates NL to graph query + embedding
- [ ] Hybrid retrieval orchestrator — ties graph results + vector search together
- [ ] Cold-start indexer — bulk-processes existing files at install time
