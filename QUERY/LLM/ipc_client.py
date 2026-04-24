"""
sentinel-query: IPC client for the shared Unix socket wire protocol.

Wire protocol (identical across all sentinel services):
  [ 4-byte little-endian uint32  length of UTF-8 JSON payload ]
  [ UTF-8 JSON payload                                         ]

This module provides:
  • IpcClient       — single-connection async client with auto-reconnect
  • DaemonClient    — typed client for sentinel-daemon (GraphQuery, SystemAction)
  • VectorStoreClient — typed client for sentinel-vectorstore (search)

Used by the orchestrator (Phase 5) and the query server (Phase 4) to talk
to downstream services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Header format: 4-byte little-endian unsigned int
_HEADER_FMT = "<I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


# ---------------------------------------------------------------------------
# Low-level framed client
# ---------------------------------------------------------------------------

class IpcClient:
    """
    Persistent async client for the sentinel length-prefixed JSON protocol.

    Features
    ────────
    • Automatic reconnection with exponential back-off (max 30 s).
    • Thread-safe: uses a single asyncio.Lock so multiple coroutines can share
      one connection without interleaving frames.
    • Framing: every message is preceded by a 4-byte LE uint32 length header.

    Example
    ───────
        client = IpcClient("/run/sentinel/daemon.sock")
        await client.connect()
        resp = await client.send_recv({"type": "graph_query", ...})
        await client.close()
    """

    _MAX_BACKOFF = 30.0
    _INITIAL_BACKOFF = 0.5

    def __init__(self, socket_path: str) -> None:
        self._path = socket_path
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()
        self._backoff = self._INITIAL_BACKOFF

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Unix socket connection.  Retries with back-off on failure."""
        while True:
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(
                    self._path
                )
                self._backoff = self._INITIAL_BACKOFF
                logger.debug("Connected to %s", self._path)
                return
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                logger.warning(
                    "Cannot connect to %s (%s); retrying in %.1f s …",
                    self._path, exc, self._backoff,
                )
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self._MAX_BACKOFF)

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except OSError:
                pass
            self._writer = None
            self._reader = None

    @property
    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    # ------------------------------------------------------------------
    # Send / receive
    # ------------------------------------------------------------------

    async def send_recv(self, payload: dict) -> dict:
        """
        Send *payload* and return the parsed response dict.

        Automatically reconnects if the connection was lost.
        """
        async with self._lock:
            for attempt in range(2):
                if not self.is_connected:
                    await self.connect()
                try:
                    await self._send(payload)
                    return await self._recv()
                except (ConnectionResetError, BrokenPipeError, OSError) as exc:
                    logger.warning(
                        "Connection to %s lost (%s); reconnecting …",
                        self._path, exc,
                    )
                    await self.close()
                    if attempt == 1:
                        raise
        raise RuntimeError("unreachable")

    async def _send(self, payload: dict) -> None:
        assert self._writer is not None
        data = json.dumps(payload).encode("utf-8")
        header = struct.pack(_HEADER_FMT, len(data))
        self._writer.write(header + data)
        await self._writer.drain()

    async def _recv(self) -> dict:
        assert self._reader is not None
        header = await self._reader.readexactly(_HEADER_SIZE)
        (length,) = struct.unpack(_HEADER_FMT, header)
        if length == 0:
            return {}
        data = await self._reader.readexactly(length)
        return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------------
# Typed daemon client
# ---------------------------------------------------------------------------

class DaemonClient(IpcClient):
    """
    Client for sentinel-daemon (/run/sentinel/daemon.sock).

    The daemon handles two request types:
      • GraphQuery   — returns a list of file paths matching a timestamp range
      • BM25Query    — returns a list of (path, score) pairs from Tantivy
      • SystemAction — perform a whitelisted file system action
    """

    async def graph_query(
        self,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        file_type_filter: Optional[str] = None,
        limit: int = 500,
    ) -> list[str]:
        """
        Query the graph for file paths modified in the given time range.

        Returns a list of absolute paths, sorted by modification time
        (most recent first).
        """
        body: dict[str, Any] = {"type": "graph_query", "limit": limit}
        if start_ts is not None:
            body["start_ts"] = start_ts
        if end_ts is not None:
            body["end_ts"] = end_ts
        if file_type_filter is not None:
            body["file_type_filter"] = file_type_filter

        resp = await self.send_recv(body)
        paths: list[str] = resp.get("paths", [])
        return paths

    async def bm25_query(
        self,
        query_text: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        BM25 (Tantivy) keyword search.

        Returns a list of {"path": str, "score": float} dicts.
        """
        resp = await self.send_recv({
            "type": "bm25_query",
            "query": query_text,
            "limit": limit,
        })
        return resp.get("results", [])

    async def system_action(
        self,
        action: str,
        path: str,
        **kwargs: Any,
    ) -> dict:
        """
        Execute a whitelisted system action.

        action must be one of: set_priority, move_to_trash, open_file
        """
        _ALLOWED = {"set_priority", "move_to_trash", "open_file"}
        if action not in _ALLOWED:
            raise ValueError(f"Action {action!r} not in whitelist {_ALLOWED}")
        return await self.send_recv({
            "type": "system_action",
            "action": action,
            "path": path,
            **kwargs,
        })


# ---------------------------------------------------------------------------
# Typed vector store client
# ---------------------------------------------------------------------------

class VectorStoreClient(IpcClient):
    """
    Client for sentinel-vectorstore (/run/sentinel/vectorstore.sock).
    """

    async def search(
        self,
        vector: list[float],
        k: int = 20,
        filter_paths: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Nearest-neighbour search.

        Parameters
        ──────────
        vector       : 384-dim unit vector from the embedding engine.
        k            : Number of results to return.
        filter_paths : If provided, only search within these paths (Tier-1
                       hybrid retrieval).  None = search entire index (Tier-2).

        Returns list of {"path": str, "score": float} dicts, sorted by score
        descending (1.0 = perfect match in cosine space).
        """
        body: dict[str, Any] = {"type": "search", "vector": vector, "k": k}
        if filter_paths is not None:
            body["filter_paths"] = filter_paths
        resp = await self.send_recv(body)
        return resp.get("results", [])

    async def stats(self) -> dict:
        """Return vector store statistics."""
        return await self.send_recv({"type": "stats"})


# ---------------------------------------------------------------------------
# Typed embedding client (used by the query server to embed the semantic_query)
# ---------------------------------------------------------------------------

class EmbeddingClient(IpcClient):
    """
    Client for sentinel-embedding (/run/sentinel/embedding.sock).

    The query server needs to embed the `semantic_query` string into a
    384-dim vector before passing it to the vector store.
    """

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Returns a 384-dim L2-normalised unit vector.
        """
        resp = await self.send_recv({"type": "embed", "text": text})
        vec: list[float] = resp.get("vector", [])
        if len(vec) != 384:
            raise RuntimeError(
                f"Unexpected embedding dimension: {len(vec)} (expected 384)"
            )
        return vec
