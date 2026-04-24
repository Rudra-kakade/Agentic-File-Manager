#!/usr/bin/env python3
"""
sentinel-query: one-time model download script.

Downloads the Llama-3 8B Instruct model in GGUF format (Q4_K_M quantization)
from HuggingFace and saves it to /var/sentinel/models/.

Q4_K_M is chosen because:
  • ~4.5 GB on disk — fits comfortably alongside the 1.5 GB vector store
  • Excellent quality/size tradeoff for structured JSON generation tasks
  • Query translation uses deterministic greedy decoding (temp=0), so
    quantisation noise is negligible

This script requires an internet connection.  It runs once during installation
(Phase 9) and is never called at query time.

Usage
─────
    python scripts/download_model.py
    python scripts/download_model.py --output-dir /var/sentinel/models
    python scripts/download_model.py --repo-id bartowski/Meta-Llama-3-8B-Instruct-GGUF
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/var/sentinel/models"

# Primary source: bartowski's well-maintained GGUF repo.
DEFAULT_REPO_ID = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
DEFAULT_FILENAME = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

# Alternative: TheBloke's repo (fallback if primary is unavailable).
FALLBACK_REPO_ID = "TheBloke/Meta-Llama-3-8B-Instruct-GGUF"
FALLBACK_FILENAME = "meta-llama-3-8b-instruct.Q4_K_M.gguf"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(
    output_dir: str,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """
    Download *filename* from HuggingFace *repo_id* into *output_dir*.

    Returns the path to the downloaded file.
    Raises SystemExit on unrecoverable errors.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError:
        print(
            "Error: huggingface_hub is not installed.\n"
            "Run: pip install huggingface-hub --break-system-packages",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dest = output_path / filename
    if dest.exists():
        print(f"Model already exists at {dest} — skipping download.")
        return dest

    print(f"Downloading {filename} from {repo_id} …")
    print(f"  Destination: {output_path}")
    print(f"  Approximate size: 4.5 GB — this may take several minutes.")
    print()

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        if repo_id != FALLBACK_REPO_ID:
            print(f"Retrying with fallback repo {FALLBACK_REPO_ID} …", file=sys.stderr)
            return download(output_dir, FALLBACK_REPO_ID, FALLBACK_FILENAME)
        sys.exit(1)

    actual = Path(local_path)
    print(f"\nModel saved to: {actual}")
    print(f"File size: {actual.stat().st_size / 1e9:.2f} GB")
    return actual


def verify(path: Path) -> bool:
    """
    Quick sanity check: open the first 4 bytes and confirm the GGUF magic.
    GGUF files start with the ASCII bytes "GGUF" (0x47 0x47 0x55 0x46).
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"GGUF":
            print("File integrity check passed (GGUF magic OK).")
            return True
        else:
            print(
                f"Warning: unexpected file magic {magic!r} — "
                "file may be corrupted.",
                file=sys.stderr,
            )
            return False
    except OSError as exc:
        print(f"Could not verify file: {exc}", file=sys.stderr)
        return False


def write_env_hint(model_path: Path) -> None:
    """
    Print a hint about the environment variable the query server uses.
    The installer (Phase 9) will write this to /etc/sentinel/sentinel.env.
    """
    print()
    print("─" * 60)
    print("Add the following to /etc/sentinel/sentinel.env:")
    print(f"  SENTINEL_LLM_MODEL_PATH={model_path}")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Llama-3 8B GGUF model for sentinel-query."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="GGUF filename within the repository",
    )
    args = parser.parse_args()

    model_path = download(args.output_dir, args.repo_id, args.filename)
    ok = verify(model_path)
    write_env_hint(model_path)

    if not ok:
        sys.exit(1)

    print("\nModel ready.  You can now start sentinel-query.")


if __name__ == "__main__":
    main()
