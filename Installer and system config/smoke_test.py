#!/usr/bin/env python3
"""
SentinelAI smoke test.

Verifies the full pipeline end-to-end:
  1. Creates a uniquely-named temporary file with known content
  2. Waits for the file to appear in the orchestrator's index
  3. Queries the orchestrator with a phrase from the file content
  4. Asserts the file appears in the top results

Usage
─────
    python3 scripts/smoke_test.py
    python3 scripts/smoke_test.py --socket /run/sentinel/orchestrator.sock
    python3 scripts/smoke_test.py --timeout 120   # wait up to 2 min for indexing
    python3 scripts/smoke_test.py --no-cleanup     # leave test file on disk

Exit codes
──────────
    0  — all checks passed
    1  — connection failed (services not running)
    2  — test file not found in results within timeout
    3  — unexpected error
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Wire protocol (identical to all sentinel services)
# ---------------------------------------------------------------------------

def send_recv(sock_path: str, payload: dict, timeout: float = 15.0) -> dict:
    """Send one framed JSON request and return the parsed response."""
    data = json.dumps(payload).encode("utf-8")
    frame = struct.pack("<I", len(data)) + data

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(sock_path)
        sock.sendall(frame)

        header = sock.recv(4)
        if len(header) < 4:
            raise RuntimeError("Connection closed before response header")
        (length,) = struct.unpack("<I", header)

        body = b""
        while len(body) < length:
            chunk = sock.recv(length - len(body))
            if not chunk:
                raise RuntimeError("Connection closed mid-response")
            body += chunk

        return json.loads(body.decode("utf-8"))
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_services_reachable(socket_path: str) -> None:
    """Fail fast if the orchestrator socket doesn't exist."""
    if not Path(socket_path).exists():
        print(f"ERROR: Orchestrator socket not found: {socket_path}", file=sys.stderr)
        print("       Is sentinel-orchestrator running?", file=sys.stderr)
        print("       Try: systemctl status sentinel-orchestrator", file=sys.stderr)
        sys.exit(1)


def check_service_stats(socket_path: str) -> dict:
    """Query the orchestrator with an empty probe to verify it responds."""
    try:
        resp = send_recv(socket_path, {"query": "sentinel status probe", "k": 1})
        return resp
    except (ConnectionRefusedError, FileNotFoundError, TimeoutError) as exc:
        print(f"ERROR: Cannot connect to orchestrator: {exc}", file=sys.stderr)
        sys.exit(1)


def wait_for_file_in_index(
    socket_path: str,
    file_path: str,
    search_phrase: str,
    timeout: int,
) -> bool:
    """
    Poll the orchestrator every 5 seconds until *file_path* appears in
    the results for *search_phrase*, or *timeout* seconds have elapsed.
    """
    deadline = time.monotonic() + timeout
    attempt = 0

    print(f"  Waiting up to {timeout}s for test file to be indexed …")

    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = send_recv(socket_path, {"query": search_phrase, "k": 20})
        except Exception as exc:
            print(f"  [attempt {attempt}] Query error: {exc}")
            time.sleep(5)
            continue

        result_paths = [r.get("path", "") for r in resp.get("results", [])]
        if file_path in result_paths:
            rank = result_paths.index(file_path) + 1
            score = next(
                (r.get("score", 0) for r in resp["results"] if r["path"] == file_path),
                0,
            )
            print(f"  ✓ Test file found at rank {rank} with score {score:.4f}")
            return True

        remaining = int(deadline - time.monotonic())
        tier = resp.get("tier_used", "?")
        count = resp.get("result_count", 0)
        print(f"  [attempt {attempt}] Not yet indexed — "
              f"{count} results via {tier}, {remaining}s remaining …")
        time.sleep(5)

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelAI smoke test")
    parser.add_argument(
        "--socket",
        default="/run/sentinel/orchestrator.sock",
        help="Path to the orchestrator Unix socket",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds to wait for the test file to be indexed (default: 120)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not delete the test file after the test",
    )
    args = parser.parse_args()

    print("\n━━━ SentinelAI Smoke Test ━━━\n")

    # ── Step 1: Check services reachable ──────────────────────────────────
    print("Step 1: Checking services …")
    check_services_reachable(args.socket)
    stats = check_service_stats(args.socket)
    print(f"  Orchestrator responded ✓  (tier_used={stats.get('tier_used', '?')})")

    # ── Step 2: Create test file ──────────────────────────────────────────
    print("\nStep 2: Creating test file …")
    unique_token = f"sentinelai_smoke_{int(time.time())}_xq9z"
    content = (
        f"SentinelAI smoke test file\n"
        f"Unique token: {unique_token}\n"
        f"The quick brown fox jumps over the lazy dog.\n"
        f"This file was created by the installer smoke test.\n"
    )
    # Write to the user's home dir so the bulk indexer watches it
    home = Path(os.path.expanduser("~"))
    test_path = home / f".sentinel_smoke_{int(time.time())}.txt"
    test_path.write_text(content)
    print(f"  Test file: {test_path}")
    print(f"  Unique token: {unique_token}")

    # ── Step 3: Wait for indexing ─────────────────────────────────────────
    print("\nStep 3: Waiting for file to appear in the index …")
    found = wait_for_file_in_index(
        socket_path=args.socket,
        file_path=str(test_path),
        search_phrase=f"sentinelai smoke test {unique_token}",
        timeout=args.timeout,
    )

    # ── Step 4: Cleanup ───────────────────────────────────────────────────
    if not args.no_cleanup:
        test_path.unlink(missing_ok=True)
        print(f"\n  Test file removed.")

    # ── Step 5: Result ────────────────────────────────────────────────────
    print()
    if found:
        print("━━━ SMOKE TEST PASSED ✓ ━━━")
        print()
        print("SentinelAI is working correctly:")
        print("  • Filesystem monitoring detected the test file")
        print("  • Content was extracted and embedded")
        print("  • The orchestrator returned it in search results")
        sys.exit(0)
    else:
        print("━━━ SMOKE TEST INCONCLUSIVE ━━━", file=sys.stderr)
        print(file=sys.stderr)
        print("The test file was not found within the timeout.", file=sys.stderr)
        print("This is normal during the first-run bulk indexing.", file=sys.stderr)
        print("Monitor progress with:", file=sys.stderr)
        print("  journalctl -u sentinel-daemon -f", file=sys.stderr)
        print("Then re-run this script:", file=sys.stderr)
        print(f"  python3 {sys.argv[0]}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
