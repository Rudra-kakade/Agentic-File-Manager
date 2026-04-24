"""
sentinel-query: test suite.

Run with:
    cd sentinel-query
    python -m pytest tests/ -v

The tests are divided into three groups:

  1. Time parser   — pure-Python, no external deps, always runs.
  2. Grammar       — validates the GBNF string is well-formed (structural only;
                     llama-cpp-python is not required for these tests).
  3. Translator    — integration tests that stub out the Llama model so we can
                     test the full translation + validation pipeline without
                     needing a GGUF file.
  4. IPC framing   — tests the wire protocol framing against a loopback socket.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> int:
    return int(time.time())


# ---------------------------------------------------------------------------
# 1 · Time parser tests
# ---------------------------------------------------------------------------

class TestTimeParser:
    from src.time_parser import parse_time_expression, validate_range

    def test_today(self):
        from src.time_parser import parse_time_expression
        now = _now()
        r = parse_time_expression("find files I edited today", now)
        assert r.start_ts is not None
        assert r.end_ts is not None
        assert r.start_ts <= now
        assert r.end_ts <= now + 1
        # start should be today at midnight
        assert (now - r.start_ts) < 86400

    def test_yesterday(self):
        from src.time_parser import parse_time_expression
        now = _now()
        r = parse_time_expression("what did I work on yesterday", now)
        assert r.start_ts is not None and r.end_ts is not None
        # yesterday span should be exactly 86400 seconds
        assert abs((r.end_ts - r.start_ts) - 86400) < 2

    def test_last_week(self):
        from src.time_parser import parse_time_expression
        now = _now()
        r = parse_time_expression("the PDF I edited last week about HDFS", now)
        assert r.start_ts is not None and r.end_ts is not None
        # last week is exactly 7 days
        assert 5 * 86400 <= (r.end_ts - r.start_ts) <= 8 * 86400

    def test_last_n_days(self):
        from src.time_parser import parse_time_expression
        now = _now()
        r = parse_time_expression("files from the last 30 days", now)
        assert r.start_ts is not None
        assert abs((now - r.start_ts) - 30 * 86400) < 5

    def test_last_month(self):
        from src.time_parser import parse_time_expression
        now = _now()
        r = parse_time_expression("last month's budget report", now)
        assert r.start_ts is not None and r.end_ts is not None
        assert r.start_ts < r.end_ts

    def test_no_time_expression(self):
        from src.time_parser import parse_time_expression
        r = parse_time_expression("find my resume", _now())
        assert r.start_ts is None
        assert r.end_ts is None
        assert r.as_dict() is None

    def test_named_month_with_year(self):
        from src.time_parser import parse_time_expression
        r = parse_time_expression("the contract I signed in March 2023", _now())
        assert r.start_ts is not None
        # March 2023 starts at 2023-03-01 00:00:00 UTC
        assert r.start_ts == 1677628800
        # March 2023 ends at 2023-04-01 00:00:00 UTC
        assert r.end_ts == 1680307200

    def test_validate_range_valid(self):
        from src.time_parser import validate_range
        assert validate_range(1000000000, 1100000000) is True

    def test_validate_range_inverted(self):
        from src.time_parser import validate_range
        assert validate_range(1100000000, 1000000000) is False

    def test_validate_range_future(self):
        from src.time_parser import validate_range
        future = _now() + 86400 * 365
        assert validate_range(None, future) is False

    def test_validate_range_before_2000(self):
        from src.time_parser import validate_range
        assert validate_range(946684799, None) is False

    def test_as_dict_with_both(self):
        from src.time_parser import TimeRange
        r = TimeRange(1000, 2000)
        d = r.as_dict()
        assert d == {"start_ts": 1000, "end_ts": 2000}

    def test_as_dict_start_only(self):
        from src.time_parser import TimeRange
        r = TimeRange(1000, None)
        d = r.as_dict()
        assert d == {"start_ts": 1000}
        assert "end_ts" not in d

    def test_as_dict_none(self):
        from src.time_parser import TimeRange
        r = TimeRange(None, None)
        assert r.as_dict() is None


# ---------------------------------------------------------------------------
# 2 · Grammar tests
# ---------------------------------------------------------------------------

class TestGrammar:
    def test_grammar_is_string(self):
        from src.grammar import get_grammar
        g = get_grammar()
        assert isinstance(g, str)
        assert len(g) > 100

    def test_grammar_has_required_rules(self):
        from src.grammar import get_grammar
        g = get_grammar()
        assert "root" in g
        assert "semantic_query" in g
        assert "graph_query" in g
        assert "file_type_filter" in g

    def test_grammar_has_common_extensions(self):
        from src.grammar import get_grammar
        g = get_grammar()
        for ext in ("pdf", "docx", "xlsx", "py", "rs", "md"):
            assert f'"{ext}"' in g, f"Extension '{ext}' missing from grammar"

    def test_grammar_has_null_alternatives(self):
        from src.grammar import get_grammar
        g = get_grammar()
        # Both gq-val and ft-val must accept "null"
        assert g.count('"null"') >= 2

    def test_example_output_is_valid_json(self):
        from src.grammar import EXAMPLE_OUTPUT, EXAMPLE_OUTPUT_NO_CONSTRAINTS
        d1 = json.loads(EXAMPLE_OUTPUT)
        assert "semantic_query" in d1
        assert "graph_query" in d1
        assert "file_type_filter" in d1

        d2 = json.loads(EXAMPLE_OUTPUT_NO_CONSTRAINTS)
        assert d2["graph_query"] is None
        assert d2["file_type_filter"] is None


# ---------------------------------------------------------------------------
# 3 · Translator tests (stubbed Llama)
# ---------------------------------------------------------------------------

class TestTranslator:
    """
    These tests stub out llama_cpp.Llama so we can exercise the full
    translation + validation pipeline without a GGUF file on disk.
    """

    def _make_llm_response(self, content: str) -> dict:
        """Mimic llama-cpp-python's create_chat_completion response."""
        return {"choices": [{"message": {"content": content}}]}

    def _make_translator_with_mock(self, llm_output: str):
        """
        Return a QueryTranslator whose internal Llama instance is replaced
        with a MagicMock that returns *llm_output*.
        """
        from src.config import ModelConfig
        from src.translator import QueryTranslator

        cfg = ModelConfig(model_path="/tmp/nonexistent.gguf")
        t = QueryTranslator(cfg)

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = (
            self._make_llm_response(llm_output)
        )
        mock_grammar = MagicMock()

        t._llm = mock_llm
        t._grammar = mock_grammar
        return t

    def test_full_query_with_time_and_filetype(self):
        output = json.dumps({
            "graph_query": {"start_ts": 1709424000, "end_ts": 1710028800},
            "semantic_query": "HDFS distributed file system replication",
            "file_type_filter": "pdf",
        })
        t = self._make_translator_with_mock(output)
        result = asyncio.run(
            t.translate("find the PDF I edited last week about HDFS")
        )
        assert result.semantic_query == "HDFS distributed file system replication"
        assert result.graph_query is not None
        assert result.graph_query.start_ts == 1709424000
        assert result.graph_query.end_ts == 1710028800
        assert result.file_type_filter == "pdf"
        assert result.used_llm is True

    def test_no_time_no_filetype(self):
        output = json.dumps({
            "graph_query": None,
            "semantic_query": "quarterly budget spreadsheet finance",
            "file_type_filter": None,
        })
        t = self._make_translator_with_mock(output)
        result = asyncio.run(t.translate("find the budget spreadsheet"))
        assert result.graph_query is None
        assert result.file_type_filter is None
        assert "budget" in result.semantic_query

    def test_invalid_timestamp_range_discarded(self):
        """LLM emits start > end — graph_query should be set to None."""
        output = json.dumps({
            "graph_query": {"start_ts": 1710028800, "end_ts": 1709424000},  # inverted
            "semantic_query": "invoices",
            "file_type_filter": None,
        })
        t = self._make_translator_with_mock(output)
        result = asyncio.run(t.translate("invoices from last week"))
        assert result.graph_query is None   # discarded due to validation failure

    def test_future_timestamp_discarded(self):
        """LLM hallucinates a far-future timestamp — should be discarded."""
        future = _now() + 86400 * 365 * 10
        output = json.dumps({
            "graph_query": {"start_ts": future - 86400, "end_ts": future},
            "semantic_query": "meeting notes",
            "file_type_filter": None,
        })
        t = self._make_translator_with_mock(output)
        result = asyncio.run(t.translate("recent meeting notes"))
        assert result.graph_query is None

    def test_empty_semantic_query_falls_back_to_raw(self):
        """If LLM returns empty semantic_query, we use the raw query."""
        output = json.dumps({
            "graph_query": None,
            "semantic_query": "   ",
            "file_type_filter": None,
        })
        t = self._make_translator_with_mock(output)
        raw = "find my resume cv"
        result = asyncio.run(t.translate(raw))
        assert result.semantic_query == raw

    def test_fallback_when_not_loaded(self):
        """Translator with no model loaded returns a fallback query."""
        from src.config import ModelConfig
        from src.translator import QueryTranslator
        t = QueryTranslator(ModelConfig())
        # _llm is None — model not loaded
        result = asyncio.run(
            t.translate("find my HDFS notes from last week", now_ts=1710000000)
        )
        assert result.used_llm is False
        assert result.semantic_query == "find my HDFS notes from last week"
        # time_parser should have found "last week"
        assert result.graph_query is not None

    def test_inference_exception_triggers_fallback(self):
        """If _infer raises, the translator returns a fallback."""
        from src.config import ModelConfig
        from src.translator import QueryTranslator
        t = QueryTranslator(ModelConfig())
        t._llm = MagicMock()
        t._llm.create_chat_completion.side_effect = RuntimeError("CUDA OOM")
        t._grammar = MagicMock()
        result = asyncio.run(t.translate("my notes"))
        assert result.used_llm is False

    def test_to_ipc_dict_structure(self):
        """to_ipc_dict() must match the wire format contract."""
        output = json.dumps({
            "graph_query": {"start_ts": 1709424000, "end_ts": 1710028800},
            "semantic_query": "HDFS notes",
            "file_type_filter": "txt",
        })
        t = self._make_translator_with_mock(output)
        result = asyncio.run(t.translate("my HDFS notes from last week"))
        d = result.to_ipc_dict()
        assert set(d.keys()) == {"graph_query", "semantic_query", "file_type_filter"}
        assert isinstance(d["graph_query"], dict)
        assert isinstance(d["semantic_query"], str)


# ---------------------------------------------------------------------------
# 4 · IPC framing tests
# ---------------------------------------------------------------------------

class TestIpcFraming:
    """
    Test the wire protocol framing using a loopback asyncio Unix socket pair.
    No real sentinel services need to be running.
    """

    @pytest.fixture
    def socket_path(self, tmp_path):
        return str(tmp_path / "test.sock")

    def _pack(self, payload: dict) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        return struct.pack("<I", len(data)) + data

    def _unpack(self, raw: bytes) -> dict:
        length = struct.unpack("<I", raw[:4])[0]
        return json.loads(raw[4:4 + length].decode("utf-8"))

    def test_frame_round_trip(self, socket_path):
        """Server echoes the request back; verify framing is symmetric."""
        async def _echo_server(reader, writer):
            header = await reader.readexactly(4)
            (n,) = struct.unpack("<I", header)
            data = await reader.readexactly(n)
            writer.write(header + data)
            await writer.drain()
            writer.close()

        async def _run():
            import os
            server = await asyncio.start_unix_server(_echo_server, path=socket_path)
            async with server:
                reader, writer = await asyncio.open_unix_connection(socket_path)
                payload = {"query": "hello world", "now_ts": 1234567890}
                frame = self._pack(payload)
                writer.write(frame)
                await writer.drain()

                header = await reader.readexactly(4)
                (n,) = struct.unpack("<I", header)
                body = await reader.readexactly(n)
                received = json.loads(body.decode("utf-8"))
                writer.close()
                return received

        received = asyncio.run(_run())
        assert received["query"] == "hello world"
        assert received["now_ts"] == 1234567890

    def test_empty_query_returns_error_via_server(self, socket_path):
        """QueryServer returns {"error": ...} for an empty query field."""
        from src.server import QueryServer
        from src.config import QueryConfig

        # We need to test handle_frame directly without starting a real server.
        # Patch the translator and embedding client.
        cfg = QueryConfig()
        server = QueryServer(cfg)

        async def _run():
            frame = json.dumps({"query": ""}).encode("utf-8")
            return await server._handle_frame(frame)

        result = asyncio.run(_run())
        assert "error" in result

    def test_malformed_json_returns_error(self, socket_path):
        from src.config import QueryConfig
        from src.server import QueryServer

        server = QueryServer(QueryConfig())

        async def _run():
            return await server._handle_frame(b"not json at all")

        result = asyncio.run(_run())
        assert "error" in result
