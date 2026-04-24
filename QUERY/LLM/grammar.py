"""
sentinel-query: GBNF grammar for grammar-constrained JSON output.

llama.cpp's GBNF (GGML BNF) grammar engine enforces this at the token-sampling
level — the model literally cannot emit a token that would violate the grammar.
No prompt-only guardrails; structural correctness is physically guaranteed.

Output schema
─────────────
{
  "graph_query":      null  |  {"start_ts": <int>, "end_ts": <int>}
                                (either or both timestamp keys are optional)
  "semantic_query":   "<string>"          (always present, never null)
  "file_type_filter": null  |  "<one of the known extensions>"
}

Why this exact schema?
  • graph_query drives the Kùzu timestamp-range edge filter.
    Null means "no time constraint".
  • semantic_query is passed to the MiniLM embedding engine.
    It should be a clean, expanded description of the *content* the user wants
    — time expressions and file-type words stripped out.
  • file_type_filter is passed to both the graph query (MIME filter) and the
    vector store's path-suffix filter.  Null means "any file type".
"""

# ---------------------------------------------------------------------------
# Enumerated file-type strings the model may emit.
# Keeping this list finite prevents the grammar from accepting
# hallucinated extensions.
# ---------------------------------------------------------------------------
_KNOWN_EXTENSIONS = [
    "pdf", "docx", "xlsx", "csv",
    "txt", "md", "rst",
    "py", "rs", "js", "ts", "go", "c", "cpp", "h", "java", "rb", "sh",
    "json", "yaml", "toml", "xml",
    "jpg", "jpeg", "png", "gif", "svg", "webp",
    "mp4", "mkv", "avi", "mov",
    "mp3", "flac", "wav", "ogg",
    "zip", "tar", "gz",
]

# Build the alternation for file-type strings:  "pdf" | "docx" | ...
_FT_ALTERNATIVES = " | ".join(f'"{ext}"' for ext in _KNOWN_EXTENSIONS)


# ---------------------------------------------------------------------------
# The grammar itself.
#
# Formatting notes
#   • llama.cpp GBNF uses the same rule-reference syntax as PEG/EBNF.
#   • ws* between structural tokens handles whatever whitespace the model
#     naturally produces inside JSON objects.
#   • char* in string allows any non-quote, non-backslash character plus
#     the standard JSON escape sequences.
# ---------------------------------------------------------------------------
QUERY_GRAMMAR = r"""
root   ::= "{" ws
           "\"graph_query\""      ws ":" ws gq-val      ws "," ws
           "\"semantic_query\""   ws ":" ws str-val      ws "," ws
           "\"file_type_filter\"" ws ":" ws ft-val
           ws "}"

# ── graph_query ──────────────────────────────────────────────────────────────
gq-val  ::= "null" | "{" ws gq-body ws "}"

# Either key may appear alone or both may appear (in either order).
gq-body ::= gq-pair (ws "," ws gq-pair)*
gq-pair ::= "\"start_ts\"" ws ":" ws integer
           | "\"end_ts\""   ws ":" ws integer

# ── semantic_query ────────────────────────────────────────────────────────────
str-val ::= "\"" char* "\""
char    ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])

# ── file_type_filter ─────────────────────────────────────────────────────────
ft-val  ::= "null" | ft-str
ft-str  ::= """ + _FT_ALTERNATIVES + r"""

# ── primitives ───────────────────────────────────────────────────────────────
integer ::= [0-9]+
ws      ::= [ \t\n\r]*
"""


def get_grammar() -> str:
    """Return the GBNF grammar string ready for llama-cpp-python."""
    return QUERY_GRAMMAR.strip()


# ---------------------------------------------------------------------------
# Convenience: a human-readable example of valid output.
# Used in the system prompt to show the model what to produce.
# ---------------------------------------------------------------------------
EXAMPLE_OUTPUT = """\
{
  "graph_query": {"start_ts": 1709424000, "end_ts": 1710028800},
  "semantic_query": "HDFS distributed file system replication factor",
  "file_type_filter": "pdf"
}"""

# Example with no time constraint and no file type filter:
EXAMPLE_OUTPUT_NO_CONSTRAINTS = """\
{
  "graph_query": null,
  "semantic_query": "quarterly budget spreadsheet finance department",
  "file_type_filter": null
}"""
