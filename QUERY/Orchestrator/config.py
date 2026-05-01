"""
sentinel-query: configuration dataclasses.

All values can be overridden via SENTINEL_* environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Path to the Llama-3 8B GGUF file.
    # The installer places it here after download_model.py runs.
    model_path: str = field(
        default_factory=lambda: os.environ.get(
            "SENTINEL_LLM_MODEL_PATH",
            "/var/sentinel/models/llama-3-8b-instruct.Q4_K_M.gguf",
        )
    )

    # Number of context tokens.  512 is more than enough for a single
    # NL query + structured JSON output.
    n_ctx: int = int(os.environ.get("SENTINEL_LLM_N_CTX", "512"))

    # CPU-only: never offload layers to GPU.
    n_gpu_layers: int = int(os.environ.get("SENTINEL_LLM_N_GPU_LAYERS", "0"))

    # Number of threads for inference.  Default: half the logical CPUs so we
    # stay within the 15 % system-CPU budget at query time.
    n_threads: int = int(
        os.environ.get(
            "SENTINEL_LLM_N_THREADS",
            str(max(1, (os.cpu_count() or 2) // 2)),
        )
    )

    # mmap is always True — the 8B model is ~4.5 GB; we must NOT load it fully
    # into RAM on startup.  Pages are faulted in on demand.
    use_mmap: bool = True

    # mlock: keep loaded pages in RAM so inference doesn't stall on page faults.
    # Disabled by default because it requires elevated ulimits and is optional.
    use_mlock: bool = bool(int(os.environ.get("SENTINEL_LLM_USE_MLOCK", "0")))

    # Maximum tokens the model may emit for a single query translation.
    # The JSON output is small — 128 tokens is a generous ceiling.
    max_tokens: int = int(os.environ.get("SENTINEL_LLM_MAX_TOKENS", "128"))

    # Temperature = 0 gives deterministic, greedy output.  We want
    # consistency, not creativity, for structured JSON generation.
    temperature: float = float(os.environ.get("SENTINEL_LLM_TEMPERATURE", "0.0"))


@dataclass
class ServerConfig:
    # Unix socket path where this server listens for NL query requests.
    socket_path: str = os.environ.get(
        "SENTINEL_ORCHESTRATOR_SOCKET",
        os.environ.get("SENTINEL_QUERY_SOCKET", "/run/sentinel/query.sock")
    )

    # Downstream sockets (defined in the shared IPC socket map).
    daemon_socket: str = os.environ.get(
        "SENTINEL_DAEMON_SOCKET", "/run/sentinel/daemon.sock"
    )
    vectorstore_socket: str = os.environ.get(
        "SENTINEL_VECTORSTORE_SOCKET", "/run/sentinel/vectorstore.sock"
    )
    embedding_socket: str = os.environ.get(
        "SENTINEL_EMBEDDING_SOCKET", "/run/sentinel/embedding.sock"
    )

    # How many concurrent client connections to accept.  Inference is
    # serialised internally (one Llama instance), so this is just the
    # accept-queue depth.
    max_connections: int = int(os.environ.get("SENTINEL_QUERY_MAX_CONN", "8"))


@dataclass
class QueryConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
