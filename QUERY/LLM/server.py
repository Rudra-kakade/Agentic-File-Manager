"""
sentinel-query: asyncio Unix socket server.

This server is the entry point for the query pipeline.  It:
  1. Receives a natural language query over a Unix socket.
  2. Calls the QueryTranslator (Llama-3 GGUF + GBNF grammar).
  3. Embeds the resulting semantic_query via the embedding engine.
  4. Returns the full StructuredQuery + embedded vector to the caller.

The *orchestrator* (Phase 5) is the consumer of this server.  The UI layer
(Phase 10) talks to the orchestrator, not directly to this server.

Wire protocol (shared across all sentinel services)
────────────────────────────────────────────────────
  Inbound:   {"query": "<NL string>", "now_ts": <optional int>}
  Outbound:  {
               "graph_query":      null | {"start_ts": int, "end_ts": int},
               "semantic_query":   "<string>",
               "file_type_filter": null | "<extension>",
               "vector":           [<384 floats>],
               "latency_ms":       <float>,
               "used_llm":         <bool>
             }
  On error:  {"error": "<message>"}

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
from .ipc_client import EmbeddingClient
from .translator import QueryTranslator

logger = logging.getLogger(__name__)

_HEADER_FMT = "<I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


# ---------------------------------------------------------------------------
# Frame-level helpers (same as in all other sentinel services)
# ---------------------------------------------------------------------------

async def _read_frame(reader: asyncio.StreamReader) -> Optional[bytes]:
    """Read one length-prefixed frame.  Returns None on EOF."""
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
# Request handler
# ---------------------------------------------------------------------------

class QueryServer:
    """
    Asyncio Unix socket server for the query translation pipeline.

    Lifecycle
    ─────────
        server = QueryServer(config)
        server.start()      # loads model (blocking), then starts event loop
    """

    def __init__(self, config: Optional[QueryConfig] = None) -> None:
        self._config = config or QueryConfig()
        self._translator = QueryTranslator(self._config.model)
        self._embedding_client = EmbeddingClient(
            self._config.server.embedding_socket
        )
        self._server: Optional[asyncio.AbstractServer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Load the model then run the asyncio event loop.  Blocks forever.

        Signal handlers for SIGINT / SIGTERM trigger a clean shutdown.
        """
        logger.info("Loading LLM model (this may take a few seconds) …")
        self._translator.load()
        logger.info("Model ready.  Starting query server on %s",
                    self._config.server.socket_path)

        asyncio.run(self._run())

    # ------------------------------------------------------------------
    # Asyncio entry point
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        socket_path = self._config.server.socket_path

        # Remove stale socket file from a previous run.
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        # Ensure the parent directory exists.
        os.makedirs(os.path.dirname(socket_path), exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=socket_path,
        )

        # Restrict socket permissions: only the sentinel group may connect.
        os.chmod(socket_path, 0o660)

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        logger.info("sentinel-query listening on %s", socket_path)

        async with self._server:
            await stop_event.wait()

        logger.info("Shutting down …")
        self._translator.unload()
        await self._embedding_client.close()

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<unknown>")
        logger.debug("New connection from %s", peer)

        try:
            while True:
                frame = await _read_frame(reader)
                if frame is None:
                    break   # client disconnected

                response = await self._handle_frame(frame)
                await _write_frame(writer, response)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            logger.debug("Connection from %s closed", peer)

    async def _handle_frame(self, frame: bytes) -> dict:
        t0 = time.monotonic()

        # ── Parse request ─────────────────────────────────────────────────
        try:
            req = json.loads(frame.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"error": f"Malformed request: {exc}"}

        query = req.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return {"error": "Missing or empty 'query' field"}

        now_ts: Optional[int] = req.get("now_ts")
        if now_ts is not None and not isinstance(now_ts, int):
            now_ts = None

        # ── Translate NL → structured query ───────────────────────────────
        try:
            structured = await self._translator.translate(query, now_ts)
        except Exception as exc:
            logger.error("Unexpected translation error: %s", exc, exc_info=True)
            return {"error": f"Translation failed: {exc}"}

        # ── Embed the semantic query ───────────────────────────────────────
        try:
            vector = await self._embedding_client.embed(structured.semantic_query)
        except Exception as exc:
            logger.warning("Embedding failed: %s — returning result without vector", exc)
            vector = []

        total_ms = (time.monotonic() - t0) * 1000

        # ── Build response ────────────────────────────────────────────────
        resp = structured.to_ipc_dict()
        resp["vector"] = vector
        resp["latency_ms"] = round(total_ms, 1)
        resp["used_llm"] = structured.used_llm

        logger.info(
            "Translated %r → semantic=%r  graph=%r  ft=%r  (%.0f ms, llm=%s)",
            query,
            structured.semantic_query,
            resp["graph_query"],
            structured.file_type_filter,
            total_ms,
            structured.used_llm,
        )

        return resp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="sentinel-query translation server")
    parser.add_argument(
        "--model-path",
        help="Path to the Llama-3 8B GGUF file "
             "(overrides SENTINEL_LLM_MODEL_PATH env var)",
    )
    parser.add_argument(
        "--socket-path",
        help="Unix socket to listen on "
             "(overrides SENTINEL_QUERY_SOCKET env var)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    config = QueryConfig()
    if args.model_path:
        config.model.model_path = args.model_path
    if args.socket_path:
        config.server.socket_path = args.socket_path

    QueryServer(config).start()


if __name__ == "__main__":
    main()
