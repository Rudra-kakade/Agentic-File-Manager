"""
sentinel-query: Orchestrator server.

This is the single front door the UI (Phase 10) talks to.  It wraps the full
pipeline — translation → embedding → three-tier retrieval — behind one Unix
socket with a clean request/response protocol.

Wire protocol (same length-prefixed JSON as all sentinel services)
──────────────────────────────────────────────────────────────────
  Inbound request:
    {
      "query": "<NL string>",
      "now_ts": <optional int>,   // defaults to server wall clock
      "k":      <optional int>    // max results, default 20
    }

  Outbound response (success):
    {
      "results": [
        {
          "path":        "/home/alice/report.pdf",
          "filename":    "report.pdf",
          "extension":   "pdf",
          "score":       0.9124,
          "tier":        "graph+vector",
          "size_bytes":  204800,
          "modified_ts": 1710028800
        },
        ...
      ],
      "tier_used":               "graph+vector",
      "total_latency_ms":        312.4,
      "translation_latency_ms":  287.1,
      "query_text":              "find the PDF I edited last week about HDFS",
      "semantic_query":          "HDFS distributed file system replication",
      "graph_query":             {"start_ts": 1709424000, "end_ts": 1710028800},
      "file_type_filter":        "pdf",
      "result_count":            7,
      "error":                   null
    }

  Outbound response (error):
    {"error": "<message>", "results": [], ...}

All frames: [ 4-byte LE uint32 length ][ UTF-8 JSON payload ]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import struct
import time
from typing import Optional

from .config import QueryConfig
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_HEADER_FMT = "<I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_DEFAULT_K = 20
_MAX_K = 100


# ---------------------------------------------------------------------------
# Framing helpers
# ---------------------------------------------------------------------------

async def _read_frame(reader: asyncio.StreamReader) -> Optional[bytes]:
    try:
        header = await reader.readexactly(_HEADER_SIZE)
    except asyncio.IncompleteReadError:
        return None
    (length,) = struct.unpack(_HEADER_FMT, header)
    if length == 0:
        return b""
    return await reader.readexactly(length)


async def _write_frame(writer: asyncio.StreamWriter, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    header = struct.pack(_HEADER_FMT, len(data))
    writer.write(header + data)
    await writer.drain()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class OrchestratorServer:
    """
    Asyncio Unix socket server wrapping the full retrieval pipeline.

    Lifecycle
    ─────────
        server = OrchestratorServer(config)
        server.start()   # loads model + event loop — blocks forever
    """

    def __init__(self, config: Optional[QueryConfig] = None) -> None:
        self._config = config or QueryConfig()
        self._orchestrator = Orchestrator(self._config)
        self._server: Optional[asyncio.AbstractServer] = None

    # ------------------------------------------------------------------
    # Public start — called from main()
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load model, connect to downstream services, run the event loop."""
        logger.info("Loading LLM translation model …")
        self._orchestrator.load_model()
        logger.info("Model ready.  Starting orchestrator server on %s",
                    self._config.server.socket_path)
        asyncio.run(self._run())

    # ------------------------------------------------------------------
    # Asyncio event loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        # Connect to all downstream services.
        await self._orchestrator.connect()

        socket_path = self._config.server.socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        os.makedirs(os.path.dirname(socket_path), exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=socket_path,
        )
        os.chmod(socket_path, 0o660)

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        logger.info("sentinel-orchestrator listening on %s", socket_path)

        async with self._server:
            await stop_event.wait()

        logger.info("Shutting down orchestrator …")
        await self._orchestrator.close()

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<unix>")
        logger.debug("New connection from %s", peer)
        try:
            while True:
                frame = await _read_frame(reader)
                if frame is None:
                    break
                response = await self._handle_frame(frame)
                await _write_frame(writer, response)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    async def _handle_frame(self, frame: bytes) -> dict:
        # ── Parse request ─────────────────────────────────────────────────
        try:
            req = json.loads(frame.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"error": f"Malformed request: {exc}", "results": []}

        query = req.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return {"error": "Missing or empty 'query' field", "results": []}
        if query.strip() == "sentinel status probe":
            return {"status": "ok", "tier_used": "none", "results": []}

        now_ts: Optional[int] = req.get("now_ts")
        k = int(req.get("k", _DEFAULT_K))
        k = max(1, min(k, _MAX_K))

        # ── Run the full pipeline ─────────────────────────────────────────
        result_set = await self._orchestrator.query(
            nl_query=query.strip(),
            now_ts=now_ts,
            k=k,
        )
        return result_set.to_dict()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SentinelAI orchestrator — full NL → file retrieval pipeline"
    )
    parser.add_argument("--model-path", help="Path to Llama-3 8B GGUF file")
    parser.add_argument("--socket-path", help="Unix socket to listen on")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    config = QueryConfig()
    if args.model_path:
        config.model.model_path = args.model_path
    if args.socket_path:
        config.server.socket_path = args.socket_path

    OrchestratorServer(config).start()


if __name__ == "__main__":
    main()
