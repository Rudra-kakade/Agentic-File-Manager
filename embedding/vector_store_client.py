"""
sentinel-embedding/src/vector_store_client.py

Async client for talking to the Vector Store server over a Unix socket.
Used by the embedding engine to forward (path, vector) pairs after inference.
Used by the query engine to run similarity searches.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("sentinel.vs_client")

_RECONNECT_DELAY = 2.0   # seconds between reconnection attempts


class VectorStoreClient:
    """
    Persistent async client with automatic reconnection.
    Safe to use from a single asyncio task.
    """

    def __init__(self, socket_path: Path) -> None:
        self._sock  = socket_path
        self._reader: Optional[asyncio.StreamReader]  = None
        self._writer: Optional[asyncio.StreamWriter]  = None
        self._lock   = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def upsert(self, path: str, timestamp: int, vector: np.ndarray) -> bool:
        req = {
            "op":        "upsert",
            "path":      path,
            "timestamp": timestamp,
            "vector":    vector.tolist(),
        }
        resp = await self._rpc(req)
        return bool(resp.get("ok", False))

    async def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        filter_paths: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Semantic search.

        filter_paths — if provided, the vector store only considers these
        paths (used for Hybrid Retrieval step after graph pre-filtering).
        """
        req: dict = {
            "op":     "search",
            "vector": vector.tolist(),
            "k":      k,
        }
        if filter_paths is not None:
            req["filter_paths"] = filter_paths

        resp = await self._rpc(req)
        if not resp.get("ok"):
            log.warning("search failed: %s", resp.get("error"))
            return []
        return [(r["path"], r["score"]) for r in resp.get("results", [])]

    async def delete(self, path: str) -> bool:
        resp = await self._rpc({"op": "delete", "path": path})
        return bool(resp.get("ok", False))

    async def stats(self) -> dict:
        return await self._rpc({"op": "stats"})

    async def persist(self) -> bool:
        resp = await self._rpc({"op": "persist"})
        return bool(resp.get("ok", False))

    # ── RPC layer ─────────────────────────────────────────────────────────────

    async def _rpc(self, req: dict) -> dict:
        async with self._lock:
            for attempt in range(3):
                try:
                    await self._ensure_connected()
                    await _write_frame(self._writer, req)
                    resp = await _read_frame(self._reader)
                    if resp is None:
                        raise ConnectionError("EOF from vector store")
                    return json.loads(resp)
                except (ConnectionError, OSError, asyncio.IncompleteReadError) as e:
                    log.warning("VS RPC attempt %d failed: %s", attempt + 1, e)
                    await self._close()
                    await asyncio.sleep(_RECONNECT_DELAY)
            return {"ok": False, "error": "Could not reach vector store after 3 attempts"}

    async def _ensure_connected(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        self._reader, self._writer = await asyncio.open_unix_connection(str(self._sock))
        log.debug("Connected to vector store at %s", self._sock)

    async def _close(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None


# ── Wire protocol ─────────────────────────────────────────────────────────────

async def _write_frame(writer: asyncio.StreamWriter, obj: dict) -> None:
    data   = json.dumps(obj).encode()
    header = struct.pack("<I", len(data))
    writer.write(header + data)
    await writer.drain()


async def _read_frame(reader: asyncio.StreamReader) -> Optional[bytes]:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    length = struct.unpack("<I", header)[0]
    if length == 0 or length > 50 * 1024 * 1024:
        return None
    return await reader.readexactly(length)
