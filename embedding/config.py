"""
sentinel-embedding/src/config.py

Configuration for the embedding engine and vector store server.
All values can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(f"SENTINEL_{key}", default)


@dataclass
class EngineConfig:
    """Configuration for the MiniLM embedding engine process."""

    # Path to directory containing model.onnx + tokenizer.json
    model_path: Path = field(default_factory=lambda: Path("/var/sentinel/models/minilm"))

    # Unix socket this engine listens on (daemon connects here)
    listen_socket: Path = field(default_factory=lambda: Path("/run/sentinel/embedding.sock"))

    # Unix socket of the vector store server (engine connects here)
    vector_store_socket: Path = field(default_factory=lambda: Path("/run/sentinel/vectorstore.sock"))

    # Unix socket for governor yield signals from daemon (optional)
    governor_socket: Path | None = None

    # Dynamic batching: flush when either limit is reached
    batch_size: int      = 32
    batch_timeout_s: float = 1.0

    # Depth of in-memory task queue
    queue_depth: int = 512

    # Number of threads for ONNX inference
    inference_threads: int = 4

    @classmethod
    def from_env(cls) -> "EngineConfig":
        gov_sock_str = _env("GOVERNOR_SOCKET", "")
        return cls(
            model_path          = Path(_env("MODEL_PATH",           "/var/sentinel/models/minilm")),
            listen_socket       = Path(_env("EMBEDDING_SOCKET",     "/run/sentinel/embedding.sock")),
            vector_store_socket = Path(_env("VECTORSTORE_SOCKET",   "/run/sentinel/vectorstore.sock")),
            governor_socket     = Path(gov_sock_str) if gov_sock_str else None,
            batch_size          = int(_env("BATCH_SIZE",            "32")),
            batch_timeout_s     = float(_env("BATCH_TIMEOUT_S",     "1.0")),
            queue_depth         = int(_env("QUEUE_DEPTH",           "512")),
            inference_threads   = int(_env("INFERENCE_THREADS",     "4")),
        )


@dataclass
class VectorStoreConfig:
    """Configuration for the hnswlib vector store server process."""

    # Directory where hnsw.bin and meta.json are stored
    index_path: Path = field(default_factory=lambda: Path("/var/sentinel/vectorstore"))

    # Unix socket this server listens on
    listen_socket: Path = field(default_factory=lambda: Path("/run/sentinel/vectorstore.sock"))

    # Maximum number of vectors the index can hold
    # Pre-allocate for 1M+ files; hnswlib requires this at init time.
    max_elements: int = 2_000_000

    # Auto-persist interval in seconds
    persist_interval_s: float = 60.0

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        return cls(
            index_path         = Path(_env("VECTORSTORE_PATH",      "/var/sentinel/vectorstore")),
            listen_socket      = Path(_env("VECTORSTORE_SOCKET",    "/run/sentinel/vectorstore.sock")),
            max_elements       = int(_env("VECTORSTORE_MAX_EL",     "2000000")),
            persist_interval_s = float(_env("VECTORSTORE_PERSIST_S","60.0")),
        )
