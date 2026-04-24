#!/usr/bin/env python3
"""
scripts/download_model.py

One-time setup: downloads all-MiniLM-L6-v2 from HuggingFace and exports it
to ONNX format.  Run this during installation, not at runtime.

Usage:
    python scripts/download_model.py [--output /var/sentinel/models/minilm]

Requirements (install separately before running):
    pip install transformers optimum[exporters] onnxruntime tokenizers torch

The output directory will contain:
    model.onnx              — ONNX model (used at runtime)
    tokenizer.json          — HuggingFace fast tokenizer
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
"""

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("download_model")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def download_and_export(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading tokenizer from HuggingFace: %s", MODEL_ID)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.error("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(str(output_dir))
    log.info("Tokenizer saved to %s", output_dir)

    log.info("Exporting model to ONNX …")
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        log.error("optimum not installed. Run: pip install optimum[exporters]")
        sys.exit(1)

    main_export(
        model_name_or_path = MODEL_ID,
        output             = output_dir,
        task               = "feature-extraction",
        opset              = 14,
        optimize           = "O2",     # ONNX graph optimisations
        monolith           = True,     # single model.onnx file
    )

    # Verify the exported file exists
    onnx_path = output_dir / "model.onnx"
    if not onnx_path.exists():
        # optimum may name it differently
        candidates = list(output_dir.glob("*.onnx"))
        if candidates:
            candidates[0].rename(onnx_path)
            log.info("Renamed %s → model.onnx", candidates[0].name)
        else:
            log.error("ONNX export failed — no .onnx file found in %s", output_dir)
            sys.exit(1)

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    log.info("Export complete: %s (%.1f MB)", onnx_path, size_mb)
    log.info("")
    log.info("✓ Model ready at: %s", output_dir)
    log.info("  Set SENTINEL_MODEL_PATH=%s before starting the engine.", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and export MiniLM to ONNX")
    parser.add_argument(
        "--output",
        type    = Path,
        default = Path("/var/sentinel/models/minilm"),
        help    = "Output directory for model files",
    )
    args = parser.parse_args()
    download_and_export(args.output)


if __name__ == "__main__":
    main()
