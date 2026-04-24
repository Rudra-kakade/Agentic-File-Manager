#!/usr/bin/env python3
"""
sentinel-daemon: Kùzu schema migration + 1M-edge benchmark script.

Usage
─────
    # Apply DDL only (safe to run on an existing database)
    python scripts/migrate_kuzu.py --db /var/sentinel/graph

    # Apply DDL + run the 1M-edge performance benchmark
    python scripts/migrate_kuzu.py --db /var/sentinel/graph --benchmark

    # Seed a test database with synthetic data for development
    python scripts/migrate_kuzu.py --db /tmp/sentinel_test --seed 100000

The script uses kuzu-python (pip install kuzu) to talk to the same database
that the Rust daemon uses — the on-disk format is identical.

Phase 8 acceptance criteria
────────────────────────────
    • Schema migration is idempotent (safe to re-run after daemon restarts).
    • query_time_range on 1M+ nodes completes in < 100 ms (p95).
    • upsert_file throughput >= 10,000 nodes/second.

Exit codes
──────────
    0  — success
    1  — kuzu-python not installed
    2  — migration failed
    3  — benchmark failed (latency > 100 ms p95)
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# DDL — must match the Rust KUZU_DDL constant in graph/mod.rs
# ---------------------------------------------------------------------------

DDL = """
    CREATE NODE TABLE IF NOT EXISTS File (
        path   STRING,
        ext    STRING,
        mtime  INT64,
        PRIMARY KEY (path)
    );
"""

BENCHMARK_QUERY = """
    MATCH (f:File)
    WHERE f.mtime >= $start AND f.mtime <= $end
    RETURN f.path AS path
    ORDER BY f.mtime DESC
    LIMIT 500;
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_db(db_path: str):
    """Open a Kùzu database; exit with code 1 if kuzu-python is missing."""
    try:
        import kuzu  # type: ignore[import]
    except ImportError:
        print(
            "Error: kuzu-python is not installed.\n"
            "Run: pip install kuzu",
            file=sys.stderr,
        )
        sys.exit(1)
    Path(db_path).mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(db_path)
    return kuzu.Connection(db), db


def migrate(conn, db_path: str) -> None:
    """Apply DDL. Safe to call on an existing database."""
    print(f"Applying schema migration to {db_path} …")
    try:
        conn.execute(DDL)
        print("Schema migration complete (IF NOT EXISTS — idempotent).")
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        sys.exit(2)


def seed(conn, n: int) -> None:
    """Insert *n* synthetic File nodes for benchmark / dev purposes."""
    exts = ["pdf", "docx", "txt", "py", "rs", "md", "xlsx", "csv"]
    batch_size = 500
    now = int(time.time())
    inserted = 0

    print(f"Seeding {n:,} synthetic File nodes …")
    t0 = time.monotonic()

    for i in range(0, n, batch_size):
        chunk = min(batch_size, n - i)
        values = []
        for j in range(chunk):
            idx = i + j
            ext = exts[idx % len(exts)]
            mtime = now - random.randint(0, 86400 * 365 * 3)  # within 3 years
            path = f"/home/user{idx % 100}/docs/file_{idx}.{ext}"
            # Escape single quotes in path (defensive)
            safe_path = path.replace("'", "''")
            values.append(f"('{safe_path}', '{ext}', {mtime})")

        cypher = (
            f"MERGE (f:File {{path: v.path}}) "
            f"ON CREATE SET f.ext = v.ext, f.mtime = v.mtime "
            f"ON MATCH  SET f.ext = v.ext, f.mtime = v.mtime "
            f"WITH [{', '.join(values)}] AS rows "
            f"UNWIND rows AS v "
        )
        # Simpler: use COPY or batched CREATE for seeding
        # Kùzu supports CREATE with literal values
        for val in values:
            path_v, ext_v, mtime_v = val.strip("()").split(", ", 2)
            conn.execute(
                "MERGE (f:File {path: $p}) "
                "ON CREATE SET f.ext = $e, f.mtime = $m "
                "ON MATCH  SET f.ext = $e, f.mtime = $m",
                parameters={"p": path_v.strip("'"), "e": ext_v.strip("'"), "m": int(mtime_v)},
            )
            inserted += 1

        elapsed = time.monotonic() - t0
        fps = inserted / elapsed if elapsed > 0 else 0
        print(f"\r  {inserted:>8,} / {n:,}  ({fps:,.0f} nodes/s)", end="", flush=True)

    elapsed = time.monotonic() - t0
    fps = n / elapsed if elapsed > 0 else 0
    print(f"\nSeeded {n:,} nodes in {elapsed:.1f}s  ({fps:,.0f} nodes/s)")

    if fps < 10_000:
        print(
            f"WARNING: upsert throughput {fps:,.0f} nodes/s is below the "
            f"10,000 nodes/s target.",
            file=sys.stderr,
        )


def run_benchmark(conn, iterations: int = 50) -> None:
    """
    Run the timestamp-range query *iterations* times and report p50/p95/p99.

    Acceptance criterion: p95 < 100 ms on a 1M+ node graph.
    """
    # Count nodes first
    result = conn.execute("MATCH (f:File) RETURN count(f) AS n")
    row = result.get_next()
    count = row[0] if row else 0
    print(f"\nBenchmarking timestamp-range query on {count:,} File nodes …")

    if count < 100_000:
        print(
            "WARNING: graph has fewer than 100,000 nodes. "
            "Seed more data for a meaningful benchmark (--seed 1000000).",
            file=sys.stderr,
        )

    now = int(time.time())
    latencies: list[float] = []

    for i in range(iterations):
        # Random 7-day window somewhere in the past 3 years
        end = now - random.randint(0, 86400 * 365 * 3)
        start = end - 86400 * 7

        t0 = time.monotonic()
        r = conn.execute(
            "MATCH (f:File) "
            "WHERE f.mtime >= $start AND f.mtime <= $end "
            "RETURN f.path AS path "
            "ORDER BY f.mtime DESC "
            "LIMIT 500",
            parameters={"start": start, "end": end},
        )
        # Consume all results (materialise the query)
        rows = []
        while r.has_next():
            rows.append(r.get_next())
        elapsed_ms = (time.monotonic() - t0) * 1000
        latencies.append(elapsed_ms)

    p50  = statistics.median(latencies)
    p95  = sorted(latencies)[int(len(latencies) * 0.95)]
    p99  = sorted(latencies)[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)

    print(f"\n  Iterations : {iterations}")
    print(f"  Mean       : {mean:.1f} ms")
    print(f"  p50        : {p50:.1f} ms")
    print(f"  p95        : {p95:.1f} ms")
    print(f"  p99        : {p99:.1f} ms")

    if p95 > 100.0:
        print(
            f"\nFAIL: p95 latency {p95:.1f} ms exceeds the 100 ms target.",
            file=sys.stderr,
        )
        sys.exit(3)
    else:
        print(f"\nPASS: p95 latency {p95:.1f} ms is within the 100 ms target.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kùzu schema migration and benchmark tool for sentinel-daemon"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the Kùzu database directory",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the 1M-edge performance benchmark after migration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Seed the database with N synthetic File nodes (for dev/benchmarking)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark query iterations (default: 50)",
    )
    args = parser.parse_args()

    conn, db = open_db(args.db)
    migrate(conn, args.db)

    if args.seed > 0:
        seed(conn, args.seed)

    if args.benchmark:
        run_benchmark(conn, args.iterations)

    print("\nDone.")


if __name__ == "__main__":
    main()
