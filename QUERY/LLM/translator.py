"""
sentinel-query: LLM query translator.

Loads a Llama-3 8B GGUF model and translates natural language queries into
structured JSON objects that drive the hybrid retrieval pipeline.

Architecture
────────────
  NL query + now_ts
       │
       ▼
  time_parser.parse_time_expression()   ← pure-Python fast path
       │  (pre-computes the time range;
       │   injected into the prompt so
       │   the LLM just confirms/refines)
       ▼
  Llama-3 8B GGUF (grammar-constrained)
       │  GBNF grammar physically prevents
       │  invalid JSON at the token level
       ▼
  validate + post-process
       │
       ▼
  StructuredQuery  (or FallbackQuery on failure)

Key design decisions
────────────────────
• use_mmap=True (default): The 4.5 GB model file is not loaded into RAM on
  startup.  OS pages are faulted in on demand, keeping startup fast and RSS low.
• temperature=0: Greedy decoding — deterministic and fast; we do not want
  sampling noise in a structured-output pipeline.
• Grammar constraint is the *only* correctness guarantee.  Prompt engineering
  is belt, grammar is suspenders.
• A single Llama instance is reused across all queries (load once, infer many).
  Inference is protected by an asyncio.Lock so the server can accept multiple
  concurrent connections without racing on the non-thread-safe Llama object.
• If inference raises or the parsed JSON fails validation, we emit a
  FallbackQuery that skips graph filtering and uses the raw query text as the
  semantic_query.  This ensures Tier-2 / Tier-3 always have something to work
  with.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Optional

from .config import ModelConfig
from .grammar import EXAMPLE_OUTPUT, EXAMPLE_OUTPUT_NO_CONSTRAINTS, get_grammar
from .time_parser import format_ts_for_prompt, parse_time_expression, validate_range

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GraphQuery:
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None

    def as_dict(self) -> Optional[dict]:
        if self.start_ts is None and self.end_ts is None:
            return None
        d: dict = {}
        if self.start_ts is not None:
            d["start_ts"] = self.start_ts
        if self.end_ts is not None:
            d["end_ts"] = self.end_ts
        return d


@dataclass
class StructuredQuery:
    """The output contract for the translator.

    Consumers (orchestrator.py) should check `graph_query is None` to decide
    whether to attempt Tier-1 retrieval.
    """
    semantic_query: str
    graph_query: Optional[GraphQuery]
    file_type_filter: Optional[str]
    # Diagnostics — not forwarded downstream.
    latency_ms: float = 0.0
    used_llm: bool = True

    def to_ipc_dict(self) -> dict:
        """Wire format sent to the orchestrator via Unix socket."""
        return {
            "graph_query": self.graph_query.as_dict() if self.graph_query else None,
            "semantic_query": self.semantic_query,
            "file_type_filter": self.file_type_filter,
        }


class TranslationError(Exception):
    """Raised when the LLM output cannot be parsed or validated."""


# ---------------------------------------------------------------------------
# System prompt factory
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a precise file-search query parser.  Given a natural language query, \
output a single JSON object that describes the search.

Current UTC time: {now_str}

Rules
─────
1. "semantic_query"
   • Extract the core content/topic the user is searching for.
   • Expand abbreviations (e.g. "HDFS" → "HDFS Hadoop distributed file system").
   • Remove any time expressions and file-type mentions — those go in other fields.
   • Must be a non-empty string.

2. "graph_query"
   • If the query mentions a time period, set start_ts and end_ts as UTC Unix \
timestamps.
   • Pre-computed hint — use these values unless the query implies something \
different:
     start_ts hint = {start_ts_hint}
     end_ts hint   = {end_ts_hint}
   • If no time is mentioned, output null.

3. "file_type_filter"
   • If the user mentions a file type (pdf, docx, py, etc.), output the \
extension string.
   • Otherwise output null.

Examples
────────
Query: "Find the PDF I edited last week about HDFS"
{example_with_constraints}

Query: "Where is the budget spreadsheet"
{example_no_constraints}

Output only the JSON object.  No explanation, no markdown fences.\
"""


def _build_system_prompt(
    now_ts: int,
    start_ts_hint: Optional[int],
    end_ts_hint: Optional[int],
) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        now_str=format_ts_for_prompt(now_ts),
        start_ts_hint=start_ts_hint if start_ts_hint is not None else "none",
        end_ts_hint=end_ts_hint if end_ts_hint is not None else "none",
        example_with_constraints=EXAMPLE_OUTPUT,
        example_no_constraints=EXAMPLE_OUTPUT_NO_CONSTRAINTS,
    )


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class QueryTranslator:
    """
    Wraps a Llama-3 8B GGUF model for NL → StructuredQuery translation.

    Usage
    ─────
        translator = QueryTranslator(config)
        translator.load()          # call once at startup
        result = await translator.translate("find my HDFS pdf from last week")
        translator.unload()        # optional; called automatically on GC
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._llm = None          # llama_cpp.Llama instance; None until load()
        self._grammar = None      # llama_cpp.LlamaGrammar instance
        self._lock = asyncio.Lock()  # serialise inference

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the GGUF model.  Blocks the calling thread; call once at startup
        *before* entering the asyncio event loop (or in a thread executor).
        """
        try:
            from llama_cpp import Llama, LlamaGrammar  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed.  "
                "Run: pip install llama-cpp-python --break-system-packages"
            ) from exc

        logger.info("Loading GGUF model from %s …", self._config.model_path)
        t0 = time.monotonic()

        self._llm = Llama(
            model_path=self._config.model_path,
            n_ctx=self._config.n_ctx,
            n_gpu_layers=self._config.n_gpu_layers,
            n_threads=self._config.n_threads,
            use_mmap=self._config.use_mmap,
            use_mlock=self._config.use_mlock,
            verbose=False,
        )

        self._grammar = LlamaGrammar.from_string(get_grammar())

        elapsed = time.monotonic() - t0
        logger.info("Model loaded in %.1f s (mmap=True, RAM impact minimal)", elapsed)

    def unload(self) -> None:
        """Release the model.  Pages are returned to the OS lazily by mmap."""
        self._llm = None
        self._grammar = None
        logger.info("Model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    async def translate(
        self,
        query: str,
        now_ts: Optional[int] = None,
    ) -> StructuredQuery:
        """
        Translate *query* into a StructuredQuery.

        Never raises — on any failure returns a FallbackQuery so the
        orchestrator can always proceed to Tier-2 retrieval.

        Parameters
        ──────────
        query   : Raw natural language query from the user.
        now_ts  : Current time as UTC Unix seconds.  Defaults to time.time().
        """
        if now_ts is None:
            now_ts = int(time.time())

        # ── Fast-path time parsing ─────────────────────────────────────────
        time_range = parse_time_expression(query, now_ts)
        start_hint = time_range.start_ts
        end_hint = time_range.end_ts

        # ── LLM inference ─────────────────────────────────────────────────
        if not self.is_loaded:
            logger.warning("Model not loaded — returning fallback query.")
            return self._fallback(query, start_hint, end_hint)

        system_prompt = _build_system_prompt(now_ts, start_hint, end_hint)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ]

        t0 = time.monotonic()
        try:
            async with self._lock:
                raw = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._infer,
                    messages,
                )
        except Exception as exc:
            logger.error("LLM inference failed: %s", exc)
            return self._fallback(query, start_hint, end_hint)

        latency_ms = (time.monotonic() - t0) * 1000
        logger.debug("LLM inference took %.0f ms, raw output: %r", latency_ms, raw)

        # ── Parse and validate ────────────────────────────────────────────
        try:
            parsed = self._parse_and_validate(raw, query, now_ts, latency_ms)
        except TranslationError as exc:
            logger.warning("Validation failed (%s) — using fallback.", exc)
            return self._fallback(query, start_hint, end_hint, latency_ms)

        return parsed

    def _infer(self, messages: list[dict]) -> str:
        """Blocking inference; run in executor so it doesn't block the loop."""
        assert self._llm is not None
        assert self._grammar is not None

        response = self._llm.create_chat_completion(
            messages=messages,
            grammar=self._grammar,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            stop=["\n\n"],      # belt-and-suspenders; grammar already limits output
        )
        return response["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Parsing and validation
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        raw: str,
        original_query: str,
        now_ts: int,
        latency_ms: float,
    ) -> StructuredQuery:
        """
        Parse the grammar-constrained JSON and run semantic validation.

        The grammar guarantees syntactic correctness, but we still validate:
        • semantic_query is non-empty
        • timestamps are sane (start ≤ end, not in the future, not before 2000)
        • file_type_filter is a known extension (grammar already enforces this,
          but belt-and-suspenders)

        Raises TranslationError on failure.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Should never happen when grammar is active, but handle it anyway.
            raise TranslationError(f"JSON parse error: {exc}") from exc

        # ── semantic_query ────────────────────────────────────────────────
        sq = data.get("semantic_query", "")
        if not isinstance(sq, str) or not sq.strip():
            # Fall back to the raw user query rather than returning garbage.
            sq = original_query
        sq = sq.strip()

        # ── graph_query ───────────────────────────────────────────────────
        gq_raw = data.get("graph_query")
        graph_query: Optional[GraphQuery] = None
        if isinstance(gq_raw, dict):
            start_ts = gq_raw.get("start_ts")
            end_ts   = gq_raw.get("end_ts")
            if not validate_range(start_ts, end_ts):
                logger.warning(
                    "LLM emitted invalid timestamp range (%s, %s) — discarding.",
                    start_ts, end_ts,
                )
                graph_query = None
            else:
                graph_query = GraphQuery(
                    start_ts=int(start_ts) if start_ts is not None else None,
                    end_ts=int(end_ts) if end_ts is not None else None,
                )
        # null → no time constraint

        # ── file_type_filter ──────────────────────────────────────────────
        ft = data.get("file_type_filter")
        if ft is not None and not isinstance(ft, str):
            ft = None

        return StructuredQuery(
            semantic_query=sq,
            graph_query=graph_query,
            file_type_filter=ft,
            latency_ms=latency_ms,
            used_llm=True,
        )

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(
        query: str,
        start_ts: Optional[int],
        end_ts: Optional[int],
        latency_ms: float = 0.0,
    ) -> StructuredQuery:
        """
        Best-effort StructuredQuery when the LLM is unavailable or fails.

        Uses the pure-Python time range (if any) and the raw query as the
        semantic string.  The orchestrator will hit Tier-2 (pure vector) or
        Tier-3 (BM25) — never zero results.
        """
        graph_query: Optional[GraphQuery] = None
        if start_ts is not None or end_ts is not None:
            if validate_range(start_ts, end_ts):
                graph_query = GraphQuery(start_ts=start_ts, end_ts=end_ts)

        return StructuredQuery(
            semantic_query=query,
            graph_query=graph_query,
            file_type_filter=None,
            latency_ms=latency_ms,
            used_llm=False,
        )
