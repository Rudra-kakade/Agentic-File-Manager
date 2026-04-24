"""
sentinel-embedding/src/engine.py

SentinelAI Embedding Engine — MiniLM sidecar.

Responsibilities:
  1. Listen on a Unix domain socket for IndexTask messages from the Rust daemon.
  2. Batch incoming tasks and run them through all-MiniLM-L6-v2 (ONNX).
  3. Forward (path, embedding_vector) pairs to the vector store (hnswlib) via
     a second Unix socket.
  4. Respect the governor signal: if the daemon signals high CPU, drain the
     current batch then pause until load drops.

Wire protocol (both sockets): length-prefixed JSON frames.
  [ 4 bytes LE uint32: payload_len ][ payload_len bytes: UTF-8 JSON ]

Design decisions:
  - ONNX Runtime over PyTorch: ~10× smaller, no CUDA dependency, CPU-optimised.
  - Dynamic batching: accumulate up to BATCH_SIZE tasks or flush after
    BATCH_TIMEOUT_S seconds, whichever comes first. Maximises throughput
    without introducing runaway latency on quiet systems.
  - Asyncio + threading: socket I/O is async; ONNX inference runs in a
    dedicated ThreadPoolExecutor so it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np

from .model import EmbeddingModel
from .vector_store_client import VectorStoreClient
from .config import EngineConfig

log = logging.getLogger("sentinel.engine")


# ── Data classes ──────────────────────────────────────────────────────────────

class IndexTask:
    __slots__ = ("path", "timestamp", "content")

    def __init__(self, path: str, timestamp: int, content: str) -> None:
        self.path      = path
        self.timestamp = timestamp
        self.content   = content


# ── Engine ────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg   = cfg
        self.model = EmbeddingModel(cfg.model_path)
        self.vs    = VectorStoreClient(cfg.vector_store_socket)
        self._executor = ThreadPoolExecutor(
            max_workers=cfg.inference_threads,
            thread_name_prefix="onnx-infer",
        )
        self._queue: asyncio.Queue[IndexTask] = asyncio.Queue(
            maxsize=cfg.queue_depth
        )
        self._shutdown = asyncio.Event()
        self._yielding = False  # set True when governor signals high CPU

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        log.info("Embedding engine starting (model=%s)", self.cfg.model_path)

        # Pre-warm the ONNX session so the first real batch isn't slow
        await asyncio.get_event_loop().run_in_executor(
            self._executor, self.model.warmup
        )
        log.info("Model warmed up — ready")

        await asyncio.gather(
            self._accept_tasks(),
            self._batch_worker(),
            self._governor_monitor(),
        )

    async def shutdown(self) -> None:
        log.info("Shutdown requested")
        self._shutdown.set()

    # ── Task ingestion ────────────────────────────────────────────────────────

    async def _accept_tasks(self) -> None:
        """Accept connections from the Rust daemon and deserialise IndexTasks."""
        sock_path = self.cfg.listen_socket
        _remove_stale(sock_path)

        server = await asyncio.start_unix_server(
            self._handle_connection, path=str(sock_path)
        )
        log.info("Listening on %s", sock_path)

        async with server:
            await self._shutdown.wait()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<daemon>")
        log.debug("Connection from %s", peer)
        try:
            while True:
                frame = await _read_frame(reader)
                if frame is None:
                    break
                try:
                    data = json.loads(frame)
                    task = IndexTask(
                        path      = data["path"],
                        timestamp = int(data["timestamp"]),
                        content   = data.get("content", ""),
                    )
                except (KeyError, ValueError) as e:
                    log.warning("Malformed task frame: %s", e)
                    continue

                try:
                    self._queue.put_nowait(task)
                    log.debug("Queued task path=%s", task.path)
                except asyncio.QueueFull:
                    log.warning("Queue full — dropping task for %s", task.path)
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()

    # ── Batch worker ──────────────────────────────────────────────────────────

    async def _batch_worker(self) -> None:
        """
        Accumulate tasks into batches, run ONNX inference, forward embeddings
        to the vector store.
        """
        loop  = asyncio.get_event_loop()
        batch: List[IndexTask] = []
        deadline = time.monotonic() + self.cfg.batch_timeout_s

        while not self._shutdown.is_set():
            # ── Governor yield ────────────────────────────────────────────
            if self._yielding:
                log.debug("Governor yield — pausing batch worker")
                await asyncio.sleep(2.0)
                continue

            # ── Collect tasks until batch full or timeout ──────────────────
            now = time.monotonic()
            timeout = max(0.0, deadline - now)
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                batch.append(task)
            except asyncio.TimeoutError:
                pass

            flush = (
                len(batch) >= self.cfg.batch_size
                or time.monotonic() >= deadline
            )

            if flush and batch:
                await self._process_batch(batch, loop)
                batch = []
                deadline = time.monotonic() + self.cfg.batch_timeout_s

        # Drain remaining on shutdown
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._process_batch(batch, loop)

    async def _process_batch(
        self,
        batch: List[IndexTask],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        texts = [t.content if t.content else t.path for t in batch]
        log.info("Processing batch of %d tasks", len(batch))

        t0 = time.perf_counter()
        embeddings: np.ndarray = await loop.run_in_executor(
            self._executor,
            self.model.encode,
            texts,
        )
        elapsed = time.perf_counter() - t0
        log.info(
            "Inference: %d docs in %.3fs (%.1f docs/s)",
            len(batch), elapsed, len(batch) / max(elapsed, 1e-6),
        )

        # Forward to vector store
        for task, vec in zip(batch, embeddings):
            await self.vs.upsert(task.path, task.timestamp, vec)

    # ── Governor monitor ──────────────────────────────────────────────────────

    async def _governor_monitor(self) -> None:
        """
        Read governor signals from the Rust daemon over a dedicated socket.
        Protocol: single-line JSON  {"yield": true|false}
        """
        gov_sock = self.cfg.governor_socket
        if not gov_sock:
            return  # governor integration disabled

        while not self._shutdown.is_set():
            try:
                reader, writer = await asyncio.open_unix_connection(str(gov_sock))
                async for line in reader:
                    try:
                        msg = json.loads(line.decode())
                        self._yielding = bool(msg.get("yield", False))
                        log.debug("Governor: yielding=%s", self._yielding)
                    except json.JSONDecodeError:
                        pass
                writer.close()
            except (ConnectionRefusedError, FileNotFoundError):
                await asyncio.sleep(5.0)  # daemon not ready yet


# ── Wire protocol helpers ─────────────────────────────────────────────────────

async def _read_frame(reader: asyncio.StreamReader) -> Optional[bytes]:
    """Read one length-prefixed frame.  Returns None on EOF."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    length = struct.unpack("<I", header)[0]
    if length == 0 or length > 10 * 1024 * 1024:  # 10 MB max frame
        log.warning("Invalid frame length %d — closing connection", length)
        return None
    return await reader.readexactly(length)


def _remove_stale(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main() -> None:
    logging.basicConfig(
        level    = os.environ.get("SENTINEL_LOG_LEVEL", "INFO").upper(),
        format   = "%(asctime)s %(name)s %(levelname)s %(message)s",
        stream   = sys.stdout,
    )

    cfg    = EngineConfig.from_env()
    engine = EmbeddingEngine(cfg)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(engine.shutdown()))

    await engine.run()


def main() -> None:
    asyncio.run(_main())
