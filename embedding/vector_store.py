"""
sentinel-embedding/src/vector_store.py

hnswlib Vector Store — Tier-1 semantic retrieval.

This module runs as a small standalone asyncio server, separate from the
embedding engine process.  Keeping it separate means:
  - The Rust daemon can query it directly without going through the Python
    embedding engine (which may be idle/paused by the governor).
  - The index persists independently and survives embedding engine restarts.

Architecture:
  ┌──────────────┐  upsert(path, vec)   ┌─────────────────────┐
  │  Embedding   │ ──────────────────▶  │  Vector Store       │
  │  Engine      │                      │  (this file)        │
  └──────────────┘                      │                      │
                                        │  - hnswlib index     │
  ┌──────────────┐  search(vec, k)      │  - id↔path mapping  │
  │  Query       │ ──────────────────▶  │  - disk persistence  │
  │  Engine      │ ◀──────────────────  └─────────────────────┘
  └──────────────┘  [(path, score), …]

Wire protocol: same length-prefixed JSON as everywhere else.

Requests:
  {"op": "upsert", "path": "...", "timestamp": 12345, "vector": [...384 floats...]}
  {"op": "search", "vector": [...384 floats...], "k": 20, "filter_paths": ["..."]}
  {"op": "delete", "path": "..."}
  {"op": "persist"}
  {"op": "stats"}

Responses:
  {"ok": true}
  {"ok": true, "results": [{"path": "...", "score": 0.92}, ...]}
  {"ok": false, "error": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import struct
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import hnswlib

from config import VectorStoreConfig

log = logging.getLogger("sentinel.vectorstore")

EMBEDDING_DIM = 384


# ── Index wrapper ─────────────────────────────────────────────────────────────

class HnswIndex:
    """
    Thread-safe wrapper around hnswlib.Index.

    hnswlib uses internal C++ locks for concurrent reads, but add_items()
    is not safe to call concurrently.  We use a Python RLock on the write
    path only; searches are lock-free.

    ID mapping:
      hnswlib works with integer IDs.  We maintain two dicts:
        _path_to_id : str  → int
        _id_to_path : int  → (str, int)   # path, timestamp
    """

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg        = cfg
        self._lock       = threading.RLock()
        self._next_id    = 0
        self._path_to_id: Dict[str, int]         = {}
        self._id_to_path: Dict[int, Tuple[str, int]] = {}   # id → (path, ts)
        self._deleted:    set[int]               = set()

        # hnswlib index
        self._index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)

        index_file = cfg.index_path / "hnsw.bin"
        meta_file  = cfg.index_path / "meta.json"

        if index_file.exists() and meta_file.exists():
            self._load(index_file, meta_file, cfg.max_elements)
        else:
            self._create(cfg.max_elements)

    # ── HNSW parameters ───────────────────────────────────────────────────────
    # M        : number of bi-directional links per node.  Higher = better recall,
    #            more memory.  16 is a good default.
    # ef_construction: size of the dynamic candidate list during build.
    #            Higher = slower build, better index quality.  200 is solid.
    # ef       : query-time candidate list.  100 gives excellent recall@10.

    def _create(self, max_elements: int) -> None:
        self._index.init_index(
            max_elements    = max_elements,
            ef_construction = 200,
            M               = 16,
        )
        self._index.set_ef(100)
        log.info("Created new hnswlib index (max_elements=%d)", max_elements)

    def _load(self, index_file: Path, meta_file: Path, max_elements: int) -> None:
        self._index.load_index(str(index_file), max_elements=max_elements)
        self._index.set_ef(100)
        meta = json.loads(meta_file.read_text())
        self._next_id    = meta["next_id"]
        self._path_to_id = meta["path_to_id"]
        self._id_to_path = {int(k): tuple(v) for k, v in meta["id_to_path"].items()}
        self._deleted    = set(meta.get("deleted", []))
        log.info(
            "Loaded hnswlib index: %d vectors (%d deleted)",
            len(self._path_to_id), len(self._deleted),
        )

    # ── Public API (called from async tasks via run_in_executor) ──────────────

    def upsert(self, path: str, timestamp: int, vector: np.ndarray) -> None:
        with self._lock:
            # If path already exists, mark old id deleted (hnswlib has no remove)
            if path in self._path_to_id:
                old_id = self._path_to_id[path]
                self._deleted.add(old_id)

            new_id = self._next_id
            self._next_id += 1

            self._index.add_items(
                vector.reshape(1, -1),
                ids = np.array([new_id]),
            )
            self._path_to_id[path]    = new_id
            self._id_to_path[new_id]  = (path, timestamp)

        log.debug("Upserted: %s (id=%d)", path, new_id)

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter_paths: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k results as (path, cosine_similarity) pairs.

        filter_paths: if provided, only return results whose path is in this set
        (used for Hybrid Retrieval — graph pre-filters, then vector re-ranks).
        """
        # Request more candidates than k to account for deleted ids and filtering
        over_fetch  = k * 5 + len(self._deleted)
        fetch_count = min(over_fetch, self._index.element_count)

        if fetch_count == 0:
            return []

        labels, distances = self._index.knn_query(
            vector.reshape(1, -1),
            k = fetch_count,
        )
        # cosine distance → cosine similarity
        ids    = labels[0]
        scores = 1.0 - distances[0]   # hnswlib cosine distance is 1 - similarity

        filter_set = set(filter_paths) if filter_paths else None
        results: List[Tuple[str, float]] = []

        for id_, score in zip(ids, scores):
            if id_ in self._deleted:
                continue
            entry = self._id_to_path.get(int(id_))
            if entry is None:
                continue
            path_, _ = entry
            if filter_set and path_ not in filter_set:
                continue
            results.append((path_, float(score)))
            if len(results) >= k:
                break

        return results

    def delete(self, path: str) -> bool:
        with self._lock:
            id_ = self._path_to_id.pop(path, None)
            if id_ is None:
                return False
            self._deleted.add(id_)
        log.debug("Soft-deleted: %s", path)
        return True

    def persist(self) -> None:
        """Save index and metadata to disk."""
        with self._lock:
            self._cfg.index_path.mkdir(parents=True, exist_ok=True)
            self._index.save_index(str(self._cfg.index_path / "hnsw.bin"))
            meta = {
                "next_id":    self._next_id,
                "path_to_id": self._path_to_id,
                "id_to_path": {str(k): list(v) for k, v in self._id_to_path.items()},
                "deleted":    list(self._deleted),
            }
            (self._cfg.index_path / "meta.json").write_text(json.dumps(meta, indent=2))
        log.info("Index persisted (%d vectors)", len(self._path_to_id))

    def stats(self) -> dict:
        return {
            "total_vectors": len(self._path_to_id),
            "deleted":       len(self._deleted),
            "next_id":       self._next_id,
        }

    def maybe_compact(self) -> None:
        """
        Rebuild the index from scratch, removing deleted vectors.
        Triggered automatically when deleted count exceeds 20% of total.
        Expensive — only called during idle periods.
        """
        if not self._path_to_id:
            return
        frac = len(self._deleted) / max(len(self._path_to_id), 1)
        if frac < 0.2:
            return

        log.info("Compacting index (%.0f%% deleted)", frac * 100)
        with self._lock:
            new_index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
            new_index.init_index(
                max_elements    = self._cfg.max_elements,
                ef_construction = 200,
                M               = 16,
            )
            new_index.set_ef(100)

            for path, old_id in self._path_to_id.items():
                if old_id in self._deleted:
                    continue
                try:
                    vec = self._index.get_items([old_id])[0]
                    new_id = len(self._path_to_id)
                    new_index.add_items(vec.reshape(1, -1), ids=np.array([new_id]))
                except Exception:
                    pass

            self._index   = new_index
            self._deleted = set()
            # Remap IDs
            new_path_to_id = {}
            new_id_to_path = {}
            for i, (path, (_, ts)) in enumerate(
                [(p, self._id_to_path[old_id]) for p, old_id in self._path_to_id.items()]
            ):
                new_path_to_id[path]  = i
                new_id_to_path[i]     = (path, ts)
            self._path_to_id = new_path_to_id
            self._id_to_path = new_id_to_path
            self._next_id    = len(new_path_to_id)

        self.persist()
        log.info("Compaction complete")


# ── Server ────────────────────────────────────────────────────────────────────

class VectorStoreServer:
    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg      = cfg
        self._index    = HnswIndex(cfg)
        self._executor = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="hnsw"
        )
        self._shutdown = asyncio.Event()

        # Auto-persist every N seconds
        self._persist_interval = cfg.persist_interval_s

    async def run(self) -> None:
        sock_path = self._cfg.listen_socket
        _remove_stale(sock_path)
        sock_path.parent.mkdir(parents=True, exist_ok=True)

        server = await asyncio.start_unix_server(
            self._handle, path=str(sock_path)
        )
        log.info("Vector store listening on %s", sock_path)

        async with server:
            await asyncio.gather(
                self._shutdown.wait(),
                self._auto_persist(),
            )

        # Final persist on shutdown
        await asyncio.get_event_loop().run_in_executor(
            self._executor, self._index.persist
        )

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                frame = await _read_frame(reader)
                if frame is None:
                    break
                try:
                    req  = json.loads(frame)
                    resp = await self._dispatch(req)
                except Exception as e:
                    resp = {"ok": False, "error": str(e)}
                await _write_frame(writer, resp)
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()

    async def _dispatch(self, req: dict) -> dict:
        op   = req.get("op")
        loop = asyncio.get_event_loop()

        if op == "upsert":
            vec = np.array(req["vector"], dtype=np.float32)
            await loop.run_in_executor(
                self._executor,
                self._index.upsert,
                req["path"], int(req["timestamp"]), vec,
            )
            return {"ok": True}

        elif op == "search":
            vec          = np.array(req["vector"], dtype=np.float32)
            k            = int(req.get("k", 10))
            filter_paths = req.get("filter_paths")   # None = no filter (pure semantic)
            results      = await loop.run_in_executor(
                self._executor,
                self._index.search,
                vec, k, filter_paths,
            )
            return {
                "ok":      True,
                "results": [{"path": p, "score": s} for p, s in results],
            }

        elif op == "delete":
            ok = await loop.run_in_executor(
                self._executor, self._index.delete, req["path"]
            )
            return {"ok": ok}

        elif op == "persist":
            await loop.run_in_executor(self._executor, self._index.persist)
            return {"ok": True}

        elif op == "stats":
            return {"ok": True, **self._index.stats()}

        elif op == "compact":
            await loop.run_in_executor(self._executor, self._index.maybe_compact)
            return {"ok": True}

        else:
            return {"ok": False, "error": f"Unknown op: {op}"}

    async def _auto_persist(self) -> None:
        loop = asyncio.get_event_loop()
        while not self._shutdown.is_set():
            await asyncio.sleep(self._persist_interval)
            await loop.run_in_executor(self._executor, self._index.persist)


# ── Wire protocol ─────────────────────────────────────────────────────────────

async def _read_frame(reader: asyncio.StreamReader) -> Optional[bytes]:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    length = struct.unpack("<I", header)[0]
    if length == 0 or length > 50 * 1024 * 1024:
        return None
    return await reader.readexactly(length)


async def _write_frame(writer: asyncio.StreamWriter, obj: dict) -> None:
    data   = json.dumps(obj).encode()
    header = struct.pack("<I", len(data))
    writer.write(header + data)
    await writer.drain()


def _remove_stale(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main() -> None:
    logging.basicConfig(
        level  = os.environ.get("SENTINEL_LOG_LEVEL", "INFO").upper(),
        format = "%(asctime)s %(name)s %(levelname)s %(message)s",
        stream = sys.stdout,
    )
    cfg    = VectorStoreConfig.from_env()
    server = VectorStoreServer(cfg)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(
        signal.SIGTERM, lambda: asyncio.create_task(server._shutdown.set())
    )
    loop.add_signal_handler(
        signal.SIGINT,  lambda: asyncio.create_task(server._shutdown.set())
    )

    await server.run()


def main() -> None:
    asyncio.run(_main())
