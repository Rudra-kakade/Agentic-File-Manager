"""
sentinel-query: result types, score normalisation, and deduplication.

All three retrieval tiers produce scores on different scales:

  Tier-1 / Tier-2  — hnswlib cosine similarity: range [0, 1]
                      1.0 = perfect match, 0.0 = orthogonal
  Tier-3           — Tantivy BM25: unbounded positive float
                      typical values 0.5 – 20.0 for real queries

This module normalises all scores into a single [0, 1] range and attaches
enough metadata for the UI layer to render rich result cards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Retrieval tier labels
# ---------------------------------------------------------------------------

class Tier(str, Enum):
    GRAPH_VECTOR = "graph+vector"   # Tier-1: graph-filtered vector search
    VECTOR       = "vector"         # Tier-2: pure semantic search
    BM25         = "bm25"           # Tier-3: keyword search (Tantivy)


# ---------------------------------------------------------------------------
# Single search result
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """One file returned by any retrieval tier."""

    # Absolute path to the file on disk.
    path: str

    # Normalised relevance score in [0, 1].
    # Higher = more relevant.  Comparable across tiers after normalisation.
    score: float

    # Which tier produced this result (before deduplication).
    tier: Tier

    # Raw score as emitted by the retrieval engine (before normalisation).
    raw_score: float = 0.0

    # ── File metadata (populated by the orchestrator post-retrieval) ───────
    filename: str = field(init=False)
    extension: str = field(init=False)
    size_bytes: Optional[int] = None
    modified_ts: Optional[int] = None   # Unix timestamp

    def __post_init__(self) -> None:
        self.filename = os.path.basename(self.path)
        ext = os.path.splitext(self.filename)[1].lstrip(".")
        self.extension = ext.lower() if ext else ""

    def to_dict(self) -> dict:
        """Wire format for the UI layer."""
        return {
            "path":        self.path,
            "filename":    self.filename,
            "extension":   self.extension,
            "score":       round(self.score, 4),
            "tier":        self.tier.value,
            "size_bytes":  self.size_bytes,
            "modified_ts": self.modified_ts,
        }


# ---------------------------------------------------------------------------
# Result set (output of the orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class ResultSet:
    """
    The complete response from the orchestrator for one query.

    Attributes
    ──────────
    results        : Deduplicated, normalised, sorted results (best first).
    tier_used      : Which tier produced the results (after fallback cascade).
    total_latency_ms : End-to-end wall time including translation + retrieval.
    translation_latency_ms : Time spent in the LLM translator alone.
    query_text     : Original NL query (echoed back for the UI).
    semantic_query : Cleaned-up content query after translation.
    graph_query    : Timestamp range used (if any).
    file_type_filter : File type constraint applied (if any).
    """
    results:                 list[SearchResult]
    tier_used:               Tier
    total_latency_ms:        float
    translation_latency_ms:  float
    query_text:              str
    semantic_query:          str
    graph_query:             Optional[dict]        = None
    file_type_filter:        Optional[str]         = None
    error:                   Optional[str]         = None

    def to_dict(self) -> dict:
        """Wire format for the UI layer."""
        return {
            "results":                [r.to_dict() for r in self.results],
            "tier_used":              self.tier_used.value,
            "total_latency_ms":       round(self.total_latency_ms, 1),
            "translation_latency_ms": round(self.translation_latency_ms, 1),
            "query_text":             self.query_text,
            "semantic_query":         self.semantic_query,
            "graph_query":            self.graph_query,
            "file_type_filter":       self.file_type_filter,
            "error":                  self.error,
            "result_count":           len(self.results),
        }


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def normalise_vector_scores(results: list[SearchResult]) -> list[SearchResult]:
    """
    Vector (cosine) scores are already in [0, 1].

    hnswlib returns cosine *distance* in some configurations, but our
    vector_store.py converts to similarity before sending (1 - distance).
    We clamp to [0, 1] as a safety measure.
    """
    for r in results:
        r.score = max(0.0, min(1.0, r.raw_score))
    return results


def normalise_bm25_scores(results: list[SearchResult]) -> list[SearchResult]:
    """
    BM25 scores are unbounded positive floats.  We normalise using the
    max-score in the result set so the best document gets score=1.0.

    If there is only one result (or all scores are zero), we assign 1.0.
    """
    if not results:
        return results
    max_score = max(r.raw_score for r in results)
    if max_score <= 0:
        for r in results:
            r.score = 0.0
        return results
    for r in results:
        r.score = r.raw_score / max_score
    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """
    Remove duplicate paths, keeping the entry with the highest score.

    When Tier-1 and Tier-2 are both tried (which doesn't happen in the
    current cascade, but could happen if we ever union tiers), the same
    path might appear twice.  We keep the best-scored instance.
    """
    seen: dict[str, SearchResult] = {}
    for r in results:
        if r.path not in seen or r.score > seen[r.path].score:
            seen[r.path] = r
    return list(seen.values())


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def enrich_metadata(results: list[SearchResult]) -> list[SearchResult]:
    """
    Populate size_bytes and modified_ts from the filesystem for each result.

    We do a best-effort stat() call.  Missing files (deleted since indexing)
    are left with None metadata — the UI handles them gracefully.
    """
    for r in results:
        try:
            st = os.stat(r.path)
            r.size_bytes = st.st_size
            r.modified_ts = int(st.st_mtime)
        except OSError:
            pass   # file may have been moved or deleted
    return results
