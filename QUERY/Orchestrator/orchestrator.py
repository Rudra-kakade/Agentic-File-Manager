"""
sentinel-query: Hybrid Retrieval Orchestrator.

This is the brain of the retrieval pipeline.  It accepts a natural language
query, calls the translator (Phase 4) to get a StructuredQuery, then drives
three-tier retrieval using all three downstream services.

Retrieval cascade
─────────────────
  Tier-1  Graph + Vector (hybrid)
  │  graph_query → DaemonClient.graph_query() → list[path]
  │  embed(semantic_query) → VectorStoreClient.search(filter_paths=paths)
  │  Result: up to `k` paths, re-ranked by cosine similarity
  │
  │  If zero results ──────────────────────────────────────────────────────
  ▼
  Tier-2  Pure semantic search
  │  VectorStoreClient.search(vector, k, filter_paths=None)
  │  Full index scan — no graph filter
  │
  │  If zero results ──────────────────────────────────────────────────────
  ▼
  Tier-3  BM25 keyword search
     DaemonClient.bm25_query(semantic_query)
     Tantivy full-text index in the daemon

Design principles
─────────────────
• The orchestrator is stateless between queries.  All state (vector index,
  graph, BM25) lives in the downstream services.
• All downstream I/O is async.  Tier-1 fires graph query and embedding
  concurrently (they are independent) then fans in.
• File-type filtering is applied at two points:
    - In the graph query (MIME/extension filter via the daemon)
    - As a post-filter on vector results (path suffix check)
  This is belt-and-suspenders: if one service doesn't support it natively,
  the post-filter catches it.
• k_graph (how many graph candidates to fetch) is deliberately larger than
  k_final (how many results to return) — graph retrieval is cheap, so we
  cast a wide net and let the vector re-ranker trim it down.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .config import QueryConfig
from LLM.ipc_client import DaemonClient, EmbeddingClient, VectorStoreClient
from .result import (
    ResultSet,
    SearchResult,
    Tier,
    deduplicate,
    enrich_metadata,
    normalise_bm25_scores,
    normalise_vector_scores,
)
from LLM.translator import GraphQuery, QueryTranslator, StructuredQuery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator configuration knobs
# ---------------------------------------------------------------------------

# How many candidate paths to fetch from the graph before vector re-ranking.
# Larger → better recall at the cost of a bigger filter_paths list in the
# vector search (still O(k_graph) not O(N)).
_K_GRAPH = 500

# How many vector results to return from Tier-1 and Tier-2.
_K_VECTOR = 20

# How many BM25 results to return in Tier-3.
_K_BM25 = 20

# Minimum cosine similarity score to include in results.
# Scores below this are almost certainly noise.
_MIN_VECTOR_SCORE = 0.15

# Minimum BM25 score (raw) to include.
_MIN_BM25_SCORE = 0.0   # BM25 already filters by relevance; 0 = keep all


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Hybrid retrieval orchestrator.

    Usage (single query)
    ────────────────────
        orch = Orchestrator(config)
        await orch.connect()            # open IPC connections
        result = await orch.query("find my HDFS pdf from last week")
        await orch.close()

    Usage (server loop)
    ───────────────────
        orch = Orchestrator(config)
        await orch.connect()
        while True:
            nl_query = await receive_from_ui()
            result = await orch.query(nl_query)
            await send_to_ui(result.to_dict())
    """

    def __init__(self, config: Optional[QueryConfig] = None) -> None:
        self._config = config or QueryConfig()

        # ── Downstream IPC clients ─────────────────────────────────────────
        self._daemon = DaemonClient(self._config.server.daemon_socket)
        self._vectorstore = VectorStoreClient(self._config.server.vectorstore_socket)
        self._embedding = EmbeddingClient(self._config.server.embedding_socket)

        # ── LLM translator (runs in-process; no extra socket needed) ──────
        self._translator = QueryTranslator(self._config.model)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open all downstream IPC connections."""
        await asyncio.gather(
            self._daemon.connect(),
            self._vectorstore.connect(),
            self._embedding.connect(),
        )
        logger.info("Orchestrator: all IPC connections established.")

    async def close(self) -> None:
        await asyncio.gather(
            self._daemon.close(),
            self._vectorstore.close(),
            self._embedding.close(),
        )

    def load_model(self) -> None:
        """Load the LLM translator model.  Blocks; call once at startup."""
        self._translator.load()

    # ------------------------------------------------------------------
    # Public query entry point
    # ------------------------------------------------------------------

    async def query(
        self,
        nl_query: str,
        now_ts: Optional[int] = None,
        k: int = _K_VECTOR,
    ) -> ResultSet:
        """
        Execute a full three-tier retrieval for *nl_query*.

        Parameters
        ──────────
        nl_query : Natural language query from the user.
        now_ts   : Current UTC Unix timestamp.  Defaults to time.time().
        k        : Number of results to return (default 20).

        Returns a ResultSet — never raises.
        """
        t_start = time.monotonic()
        if now_ts is None:
            now_ts = int(time.time())

        # ── Step 1: Translate NL → StructuredQuery ────────────────────────
        try:
            structured = await self._translator.translate(nl_query, now_ts)
        except Exception as exc:
            logger.error("Translation failed: %s", exc)
            # Return an error result set — the UI should show a friendly message.
            return ResultSet(
                results=[],
                tier_used=Tier.BM25,
                total_latency_ms=(time.monotonic() - t_start) * 1000,
                translation_latency_ms=0.0,
                query_text=nl_query,
                semantic_query=nl_query,
                error=f"Query translation failed: {exc}",
            )

        logger.info(
            "Translated %r → semantic=%r  graph=%r  ft=%r  (%.0f ms, llm=%s)",
            nl_query,
            structured.semantic_query,
            structured.graph_query.as_dict() if structured.graph_query else None,
            structured.file_type_filter,
            structured.latency_ms,
            structured.used_llm,
        )

        # ── Step 2: Embed the semantic query ──────────────────────────────
        try:
            vector = await self._embedding.embed(structured.semantic_query)
        except Exception as exc:
            logger.error("Embedding failed: %s — forcing Tier-3 fallback", exc)
            vector = []

        # ── Step 3: Drive the retrieval cascade ───────────────────────────
        results, tier = await self._cascade(structured, vector, k)

        # ── Step 4: Enrich with filesystem metadata ────────────────────────
        results = enrich_metadata(results)

        total_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "Query complete: %d results via %s in %.0f ms total",
            len(results), tier.value, total_ms,
        )

        return ResultSet(
            results=results,
            tier_used=tier,
            total_latency_ms=total_ms,
            translation_latency_ms=structured.latency_ms,
            query_text=nl_query,
            semantic_query=structured.semantic_query,
            graph_query=structured.graph_query.as_dict() if structured.graph_query else None,
            file_type_filter=structured.file_type_filter,
        )

    # ------------------------------------------------------------------
    # Three-tier cascade
    # ------------------------------------------------------------------

    async def _cascade(
        self,
        structured: StructuredQuery,
        vector: list[float],
        k: int,
    ) -> tuple[list[SearchResult], Tier]:
        """
        Run Tier-1 → Tier-2 → Tier-3 until we get results.

        Returns (results, tier_that_produced_them).
        """
        # ── Tier-1: Graph-filtered vector search ──────────────────────────
        if vector and structured.graph_query is not None:
            results = await self._tier1(structured, vector, k)
            if results:
                return results, Tier.GRAPH_VECTOR
            logger.info("Tier-1 returned 0 results — falling back to Tier-2")

        # ── Tier-2: Pure semantic vector search ───────────────────────────
        if vector:
            results = await self._tier2(structured, vector, k)
            if results:
                return results, Tier.VECTOR
            logger.info("Tier-2 returned 0 results — falling back to Tier-3")

        # ── Tier-3: BM25 keyword search ───────────────────────────────────
        results = await self._tier3(structured, k)
        return results, Tier.BM25

    # ------------------------------------------------------------------
    # Tier-1: Graph + Vector (hybrid)
    # ------------------------------------------------------------------

    async def _tier1(
        self,
        structured: StructuredQuery,
        vector: list[float],
        k: int,
    ) -> list[SearchResult]:
        """
        Graph query → candidate paths → vector re-rank within candidates.

        The two sub-steps (graph query and embedding) are already done before
        this call.  The fan-out here is:
          1. DaemonClient.graph_query() → list[path]   (timestamp + type filter)
          2. VectorStoreClient.search(vector, filter_paths=paths)  (re-rank)
        """
        gq = structured.graph_query
        assert gq is not None  # caller checks this

        try:
            # Step 1: graph candidates (larger window to maximise recall)
            graph_paths = await self._daemon.graph_query(
                start_ts=gq.start_ts,
                end_ts=gq.end_ts,
                file_type_filter=structured.file_type_filter,
                limit=_K_GRAPH,
            )
        except Exception as exc:
            logger.warning("Tier-1 graph query failed: %s", exc)
            return []

        if not graph_paths:
            logger.debug("Tier-1: graph returned 0 paths")
            return []

        logger.debug("Tier-1: graph returned %d candidate paths", len(graph_paths))

        try:
            # Step 2: vector re-rank within the candidate set
            raw_results = await self._vectorstore.search(
                vector=vector,
                k=k,
                filter_paths=graph_paths,
            )
        except Exception as exc:
            logger.warning("Tier-1 vector search failed: %s", exc)
            return []

        return self._build_results(raw_results, Tier.GRAPH_VECTOR, structured.file_type_filter)

    # ------------------------------------------------------------------
    # Tier-2: Pure semantic search
    # ------------------------------------------------------------------

    async def _tier2(
        self,
        structured: StructuredQuery,
        vector: list[float],
        k: int,
    ) -> list[SearchResult]:
        """Full-index vector search with optional path-suffix post-filter."""
        try:
            raw_results = await self._vectorstore.search(
                vector=vector,
                k=k * 3 if structured.file_type_filter else k,
                # Fetch 3× when we have a type filter; we post-filter below.
                filter_paths=None,
            )
        except Exception as exc:
            logger.warning("Tier-2 vector search failed: %s", exc)
            return []

        results = self._build_results(raw_results, Tier.VECTOR, structured.file_type_filter)
        return results[:k]

    # ------------------------------------------------------------------
    # Tier-3: BM25 (Tantivy)
    # ------------------------------------------------------------------

    async def _tier3(
        self,
        structured: StructuredQuery,
        k: int,
    ) -> list[SearchResult]:
        """Keyword search via the Tantivy index in sentinel-daemon."""
        try:
            raw_results = await self._daemon.bm25_query(
                query_text=structured.semantic_query,
                limit=k * 3 if structured.file_type_filter else k,
            )
        except Exception as exc:
            logger.warning("Tier-3 BM25 search failed: %s", exc)
            return []

        results: list[SearchResult] = []
        for item in raw_results:
            path = item.get("path", "")
            raw_score = float(item.get("score", 0.0))
            if not path or raw_score < _MIN_BM25_SCORE:
                continue
            r = SearchResult(path=path, score=0.0, tier=Tier.BM25, raw_score=raw_score)
            results.append(r)

        results = normalise_bm25_scores(results)
        results = _apply_type_filter(results, structured.file_type_filter)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    # ------------------------------------------------------------------
    # Shared result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_results(
        raw: list[dict],
        tier: Tier,
        file_type_filter: Optional[str],
    ) -> list[SearchResult]:
        """
        Convert raw vector-store dicts to SearchResult objects.

        Applies:
          • Minimum score threshold
          • File-type post-filter (belt-and-suspenders alongside graph filter)
          • Score normalisation
          • Deduplication (paths should be unique from the vector store,
            but defensive)
          • Sort by score descending
        """
        results: list[SearchResult] = []
        for item in raw:
            path = item.get("path", "")
            raw_score = float(item.get("score", 0.0))
            if not path or raw_score < _MIN_VECTOR_SCORE:
                continue
            r = SearchResult(path=path, score=0.0, tier=tier, raw_score=raw_score)
            results.append(r)

        results = normalise_vector_scores(results)
        results = _apply_type_filter(results, file_type_filter)
        results = deduplicate(results)
        results.sort(key=lambda r: r.score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_type_filter(
    results: list[SearchResult],
    file_type_filter: Optional[str],
) -> list[SearchResult]:
    """
    Post-filter results by file extension.

    This is belt-and-suspenders — the graph query and the vector store's
    filter_paths already constrain to the right type in most cases.
    We apply it again here to catch anything that slipped through.
    """
    if not file_type_filter:
        return results
    target = file_type_filter.lower().lstrip(".")
    return [r for r in results if r.extension == target]
