"""
sentinel-query: orchestrator test suite.

All tests use AsyncMock stubs for the IPC clients and translator so they
run without any real sentinel services or GGUF model present.

Test groups
───────────
  1. Result types     — SearchResult, ResultSet, serialisation
  2. Score utils      — normalisation, deduplication, metadata enrichment
  3. Tier cascade     — Tier-1, Tier-2, Tier-3 routing and fallback logic
  4. File-type filter — post-filter applied consistently across all tiers
  5. Orchestrator E2E — full pipeline with all stubs wired together
  6. Server framing   — handle_frame error paths and response contract
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import tempfile
import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vector(dim: int = 384, val: float = 0.1) -> list[float]:
    return [val] * dim


def _raw_vs_result(path: str, score: float) -> dict:
    """Mimic the dict that VectorStoreClient.search() returns."""
    return {"path": path, "score": score}


def _raw_bm25_result(path: str, score: float) -> dict:
    return {"path": path, "score": score}


# ---------------------------------------------------------------------------
# 1 · Result types
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_filename_extracted(self):
        from src.result import SearchResult, Tier
        r = SearchResult("/home/alice/docs/report.pdf", 0.9, Tier.GRAPH_VECTOR)
        assert r.filename == "report.pdf"
        assert r.extension == "pdf"

    def test_extension_lowercased(self):
        from src.result import SearchResult, Tier
        r = SearchResult("/tmp/Archive.ZIP", 0.5, Tier.VECTOR)
        assert r.extension == "zip"

    def test_no_extension(self):
        from src.result import SearchResult, Tier
        r = SearchResult("/tmp/Makefile", 0.5, Tier.VECTOR)
        assert r.extension == ""

    def test_to_dict_keys(self):
        from src.result import SearchResult, Tier
        r = SearchResult("/tmp/x.py", 0.75, Tier.BM25, raw_score=3.5)
        d = r.to_dict()
        assert set(d.keys()) == {
            "path", "filename", "extension", "score", "tier",
            "size_bytes", "modified_ts",
        }
        assert d["tier"] == "bm25"
        assert d["score"] == 0.75

    def test_score_rounded_to_4dp(self):
        from src.result import SearchResult, Tier
        r = SearchResult("/tmp/x.txt", 0.12345678, Tier.VECTOR)
        assert r.to_dict()["score"] == 0.1235


class TestResultSet:
    def test_to_dict_structure(self):
        from src.result import ResultSet, SearchResult, Tier
        rs = ResultSet(
            results=[SearchResult("/a.pdf", 0.9, Tier.GRAPH_VECTOR)],
            tier_used=Tier.GRAPH_VECTOR,
            total_latency_ms=300.0,
            translation_latency_ms=250.0,
            query_text="find my pdf",
            semantic_query="pdf document",
            graph_query={"start_ts": 1000, "end_ts": 2000},
            file_type_filter="pdf",
        )
        d = rs.to_dict()
        assert d["result_count"] == 1
        assert d["tier_used"] == "graph+vector"
        assert d["graph_query"] == {"start_ts": 1000, "end_ts": 2000}
        assert d["error"] is None

    def test_empty_results_ok(self):
        from src.result import ResultSet, Tier
        rs = ResultSet(
            results=[], tier_used=Tier.BM25,
            total_latency_ms=100.0, translation_latency_ms=90.0,
            query_text="x", semantic_query="x",
        )
        d = rs.to_dict()
        assert d["result_count"] == 0
        assert d["results"] == []


# ---------------------------------------------------------------------------
# 2 · Score utilities
# ---------------------------------------------------------------------------

class TestScoreUtils:
    def test_normalise_vector_clamps(self):
        from src.result import SearchResult, Tier, normalise_vector_scores
        results = [
            SearchResult("/a", 0.0, Tier.VECTOR, raw_score=1.2),  # over 1
            SearchResult("/b", 0.0, Tier.VECTOR, raw_score=-0.1), # negative
            SearchResult("/c", 0.0, Tier.VECTOR, raw_score=0.7),  # normal
        ]
        normalise_vector_scores(results)
        assert results[0].score == 1.0
        assert results[1].score == 0.0
        assert results[2].score == 0.7

    def test_normalise_bm25_max_gets_one(self):
        from src.result import SearchResult, Tier, normalise_bm25_scores
        results = [
            SearchResult("/a", 0.0, Tier.BM25, raw_score=10.0),
            SearchResult("/b", 0.0, Tier.BM25, raw_score=5.0),
            SearchResult("/c", 0.0, Tier.BM25, raw_score=2.5),
        ]
        normalise_bm25_scores(results)
        assert results[0].score == 1.0
        assert results[1].score == 0.5
        assert abs(results[2].score - 0.25) < 1e-9

    def test_normalise_bm25_single_result(self):
        from src.result import SearchResult, Tier, normalise_bm25_scores
        results = [SearchResult("/a", 0.0, Tier.BM25, raw_score=7.3)]
        normalise_bm25_scores(results)
        assert results[0].score == 1.0

    def test_normalise_bm25_all_zero(self):
        from src.result import SearchResult, Tier, normalise_bm25_scores
        results = [SearchResult("/a", 0.0, Tier.BM25, raw_score=0.0)]
        normalise_bm25_scores(results)
        assert results[0].score == 0.0

    def test_deduplicate_keeps_highest_score(self):
        from src.result import SearchResult, Tier, deduplicate
        results = [
            SearchResult("/dup.txt", 0.8, Tier.VECTOR),
            SearchResult("/dup.txt", 0.6, Tier.GRAPH_VECTOR),
            SearchResult("/other.txt", 0.5, Tier.VECTOR),
        ]
        out = deduplicate(results)
        paths = [r.path for r in out]
        assert len(out) == 2
        assert "/dup.txt" in paths
        dup = next(r for r in out if r.path == "/dup.txt")
        assert dup.score == 0.8

    def test_enrich_metadata_existing_file(self, tmp_path):
        from src.result import SearchResult, Tier, enrich_metadata
        f = tmp_path / "test.txt"
        f.write_text("hello")
        r = SearchResult(str(f), 1.0, Tier.VECTOR)
        enrich_metadata([r])
        assert r.size_bytes == 5
        assert r.modified_ts is not None and r.modified_ts > 0

    def test_enrich_metadata_missing_file(self):
        from src.result import SearchResult, Tier, enrich_metadata
        r = SearchResult("/nonexistent/path/file.txt", 1.0, Tier.VECTOR)
        enrich_metadata([r])  # should not raise
        assert r.size_bytes is None
        assert r.modified_ts is None


# ---------------------------------------------------------------------------
# 3 · Tier cascade
# ---------------------------------------------------------------------------

class TestTierCascade:
    """Test the Tier-1 → Tier-2 → Tier-3 fallback logic."""

    def _make_orchestrator(self) -> "Orchestrator":
        """Build an Orchestrator with all IPC clients replaced by AsyncMocks."""
        from src.orchestrator import Orchestrator
        from src.config import QueryConfig
        o = Orchestrator(QueryConfig())
        o._daemon = AsyncMock()
        o._vectorstore = AsyncMock()
        o._embedding = AsyncMock()
        o._translator = AsyncMock()
        return o

    def _make_structured(
        self,
        semantic: str = "HDFS notes",
        has_graph: bool = True,
        file_type: Optional[str] = None,
    ):
        from src.translator import GraphQuery, StructuredQuery
        gq = GraphQuery(start_ts=1709424000, end_ts=1710028800) if has_graph else None
        return StructuredQuery(
            semantic_query=semantic,
            graph_query=gq,
            file_type_filter=file_type,
            latency_ms=42.0,
            used_llm=True,
        )

    def test_tier1_used_when_graph_and_vector_results(self):
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=True)
        o._daemon.graph_query = AsyncMock(return_value=["/a.txt", "/b.txt"])
        o._vectorstore.search = AsyncMock(return_value=[
            {"path": "/a.txt", "score": 0.9},
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("find HDFS notes from last week"))
        assert result.tier_used.value == "graph+vector"
        assert len(result.results) == 1
        assert result.results[0].path == "/a.txt"

    def test_tier2_fallback_when_graph_returns_empty(self):
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=True)
        o._daemon.graph_query = AsyncMock(return_value=[])   # Tier-1 gets nothing
        o._vectorstore.search = AsyncMock(return_value=[
            {"path": "/c.txt", "score": 0.75},
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("find something"))
        assert result.tier_used.value == "vector"
        assert result.results[0].path == "/c.txt"

    def test_tier2_fallback_when_vector_filtered_results_empty(self):
        """Tier-1 graph returns paths but vector re-rank gets 0 matches."""
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=True)
        o._daemon.graph_query = AsyncMock(return_value=["/a.txt"])
        # First call (Tier-1 vector search) → empty; second call (Tier-2) → results
        o._vectorstore.search = AsyncMock(side_effect=[
            [],                                          # Tier-1 returns nothing
            [{"path": "/d.txt", "score": 0.8}],         # Tier-2 returns something
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("query"))
        assert result.tier_used.value == "vector"

    def test_tier3_fallback_when_both_vector_tiers_empty(self):
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=True)
        o._daemon.graph_query = AsyncMock(return_value=["/a.txt"])
        o._vectorstore.search = AsyncMock(return_value=[])  # both tiers empty
        o._daemon.bm25_query = AsyncMock(return_value=[
            {"path": "/e.txt", "score": 5.0},
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("query"))
        assert result.tier_used.value == "bm25"
        assert result.results[0].path == "/e.txt"
        assert result.results[0].score == 1.0  # only result → normalised to 1.0

    def test_no_graph_query_skips_tier1(self):
        """When structured.graph_query is None, we go straight to Tier-2."""
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=False)
        o._vectorstore.search = AsyncMock(return_value=[
            {"path": "/f.txt", "score": 0.85},
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("find anything"))
        assert result.tier_used.value == "vector"
        # graph_query should NOT have been called
        o._daemon.graph_query.assert_not_called()

    def test_results_sorted_by_score_descending(self):
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=False)
        o._vectorstore.search = AsyncMock(return_value=[
            {"path": "/low.txt",  "score": 0.3},
            {"path": "/high.txt", "score": 0.9},
            {"path": "/mid.txt",  "score": 0.6},
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("x"))
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)
        assert result.results[0].path == "/high.txt"

    def test_all_tiers_empty_returns_empty_result_set(self):
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=True)
        o._daemon.graph_query = AsyncMock(return_value=[])
        o._vectorstore.search = AsyncMock(return_value=[])
        o._daemon.bm25_query = AsyncMock(return_value=[])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("unindexable query"))
        assert result.results == []
        assert result.tier_used.value == "bm25"

    def test_embedding_failure_forces_tier3(self):
        """If embedding returns empty vector, Tier-1 and Tier-2 are skipped."""
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=True)
        o._embedding.embed = AsyncMock(side_effect=RuntimeError("embedding down"))
        o._daemon.bm25_query = AsyncMock(return_value=[
            {"path": "/g.txt", "score": 3.0}
        ])
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("find my notes"))
        assert result.tier_used.value == "bm25"
        assert result.results[0].path == "/g.txt"

    def test_graph_query_failure_falls_back(self):
        """DaemonClient.graph_query raises → graceful fallback to Tier-2."""
        o = self._make_orchestrator()
        structured = self._make_structured(has_graph=True)
        o._daemon.graph_query = AsyncMock(side_effect=ConnectionError("daemon down"))
        o._vectorstore.search = AsyncMock(return_value=[
            {"path": "/h.txt", "score": 0.7}
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("find notes"))
        assert result.tier_used.value == "vector"


# ---------------------------------------------------------------------------
# 4 · File-type filter
# ---------------------------------------------------------------------------

class TestFileTypeFilter:
    def _make_orchestrator(self):
        from src.orchestrator import Orchestrator
        from src.config import QueryConfig
        o = Orchestrator(QueryConfig())
        o._daemon = AsyncMock()
        o._vectorstore = AsyncMock()
        o._embedding = AsyncMock()
        o._translator = AsyncMock()
        return o

    def test_type_filter_removes_wrong_extensions(self):
        """Post-filter drops files that don't match file_type_filter."""
        from src.translator import StructuredQuery
        from src.config import QueryConfig
        from src.orchestrator import Orchestrator

        o = self._make_orchestrator()
        from src.translator import GraphQuery, StructuredQuery
        structured = StructuredQuery(
            semantic_query="HDFS notes",
            graph_query=None,
            file_type_filter="pdf",
            latency_ms=10.0,
            used_llm=True,
        )
        o._vectorstore.search = AsyncMock(return_value=[
            {"path": "/a.pdf",  "score": 0.9},
            {"path": "/b.docx", "score": 0.8},  # wrong type — should be filtered
            {"path": "/c.pdf",  "score": 0.7},
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("HDFS notes"))
        paths = [r.path for r in result.results]
        assert "/b.docx" not in paths
        assert "/a.pdf" in paths and "/c.pdf" in paths

    def test_no_filter_returns_all_types(self):
        o = self._make_orchestrator()
        from src.translator import StructuredQuery
        structured = StructuredQuery(
            semantic_query="meeting notes",
            graph_query=None,
            file_type_filter=None,
            latency_ms=10.0,
            used_llm=True,
        )
        o._vectorstore.search = AsyncMock(return_value=[
            {"path": "/a.pdf",  "score": 0.9},
            {"path": "/b.docx", "score": 0.8},
            {"path": "/c.py",   "score": 0.7},
        ])
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._translator.translate = AsyncMock(return_value=structured)

        result = asyncio.run(o.query("meeting notes"))
        assert len(result.results) == 3


# ---------------------------------------------------------------------------
# 5 · Orchestrator E2E (all stubs wired)
# ---------------------------------------------------------------------------

class TestOrchestratorE2E:
    def test_result_set_has_all_fields(self):
        from src.orchestrator import Orchestrator
        from src.config import QueryConfig
        from src.translator import GraphQuery, StructuredQuery

        o = Orchestrator(QueryConfig())
        o._daemon = AsyncMock()
        o._vectorstore = AsyncMock()
        o._embedding = AsyncMock()
        o._translator = AsyncMock()

        structured = StructuredQuery(
            semantic_query="HDFS distributed file system",
            graph_query=GraphQuery(start_ts=1709424000, end_ts=1710028800),
            file_type_filter="pdf",
            latency_ms=280.0,
            used_llm=True,
        )
        o._translator.translate = AsyncMock(return_value=structured)
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._daemon.graph_query = AsyncMock(return_value=["/report.pdf"])
        o._vectorstore.search = AsyncMock(return_value=[
            {"path": "/report.pdf", "score": 0.93},
        ])

        rs = asyncio.run(o.query(
            "find the PDF I edited last week about HDFS",
            now_ts=1710000000,
        ))

        d = rs.to_dict()
        assert d["result_count"] == 1
        assert d["tier_used"] == "graph+vector"
        assert d["query_text"] == "find the PDF I edited last week about HDFS"
        assert d["semantic_query"] == "HDFS distributed file system"
        assert d["file_type_filter"] == "pdf"
        assert d["graph_query"] == {"start_ts": 1709424000, "end_ts": 1710028800}
        assert d["translation_latency_ms"] == 280.0
        assert d["total_latency_ms"] > 0
        assert d["error"] is None

    def test_latency_fields_present(self):
        from src.orchestrator import Orchestrator
        from src.config import QueryConfig
        from src.translator import StructuredQuery

        o = Orchestrator(QueryConfig())
        o._daemon = AsyncMock()
        o._vectorstore = AsyncMock()
        o._embedding = AsyncMock()
        o._translator = AsyncMock()

        structured = StructuredQuery(
            semantic_query="budget", graph_query=None,
            file_type_filter=None, latency_ms=50.0, used_llm=False,
        )
        o._translator.translate = AsyncMock(return_value=structured)
        o._embedding.embed = AsyncMock(return_value=_make_vector())
        o._vectorstore.search = AsyncMock(return_value=[])
        o._daemon.bm25_query = AsyncMock(return_value=[])

        rs = asyncio.run(o.query("budget"))
        assert rs.total_latency_ms > 0
        assert rs.translation_latency_ms == 50.0


# ---------------------------------------------------------------------------
# 6 · Server frame handling
# ---------------------------------------------------------------------------

class TestOrchestratorServerFraming:
    def _make_server(self):
        from src.orchestrator_server import OrchestratorServer
        from src.config import QueryConfig
        s = OrchestratorServer(QueryConfig())
        s._orchestrator = MagicMock()
        return s

    def test_empty_query_returns_error(self):
        s = self._make_server()
        frame = json.dumps({"query": ""}).encode()
        result = asyncio.run(s._handle_frame(frame))
        assert "error" in result
        assert result["results"] == []

    def test_missing_query_returns_error(self):
        s = self._make_server()
        frame = json.dumps({"k": 10}).encode()
        result = asyncio.run(s._handle_frame(frame))
        assert "error" in result

    def test_malformed_json_returns_error(self):
        s = self._make_server()
        result = asyncio.run(s._handle_frame(b"{{broken"))
        assert "error" in result

    def test_valid_query_calls_orchestrator(self):
        from src.result import ResultSet, Tier
        s = self._make_server()
        mock_rs = ResultSet(
            results=[], tier_used=Tier.BM25,
            total_latency_ms=100.0, translation_latency_ms=80.0,
            query_text="find notes", semantic_query="notes",
        )
        s._orchestrator.query = AsyncMock(return_value=mock_rs)
        frame = json.dumps({"query": "find notes", "k": 5}).encode()
        result = asyncio.run(s._handle_frame(frame))
        assert "error" not in result or result.get("error") is None
        assert "results" in result
        s._orchestrator.query.assert_called_once()

    def test_k_clamped_to_max(self):
        from src.result import ResultSet, Tier
        s = self._make_server()
        mock_rs = ResultSet(
            results=[], tier_used=Tier.BM25,
            total_latency_ms=1.0, translation_latency_ms=0.5,
            query_text="x", semantic_query="x",
        )
        s._orchestrator.query = AsyncMock(return_value=mock_rs)
        frame = json.dumps({"query": "find something", "k": 9999}).encode()
        asyncio.run(s._handle_frame(frame))
        call_kwargs = s._orchestrator.query.call_args
        # k should be clamped to _MAX_K (100)
        assert call_kwargs.kwargs.get("k") == 100 or call_kwargs.args[2] == 100

    def test_response_is_json_serialisable(self):
        from src.result import ResultSet, SearchResult, Tier
        s = self._make_server()
        mock_rs = ResultSet(
            results=[SearchResult("/a.txt", 0.9, Tier.GRAPH_VECTOR)],
            tier_used=Tier.GRAPH_VECTOR,
            total_latency_ms=200.0, translation_latency_ms=180.0,
            query_text="q", semantic_query="q",
        )
        s._orchestrator.query = AsyncMock(return_value=mock_rs)
        frame = json.dumps({"query": "q"}).encode()
        result = asyncio.run(s._handle_frame(frame))
        # Must be round-trippable through JSON
        serialised = json.dumps(result)
        parsed = json.loads(serialised)
        assert parsed["results"][0]["path"] == "/a.txt"
