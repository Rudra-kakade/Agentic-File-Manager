"""
sentinel-embedding/src/model.py

ONNX Runtime wrapper for all-MiniLM-L6-v2.

Key design points:
  - Uses ONNX Runtime (not PyTorch) — ~22 MB model, no CUDA dependency.
  - Mean-pooling over token embeddings, then L2 normalisation → 384-dim unit vector.
  - Tokeniser is loaded from the same directory as the ONNX model file.
  - Thread-safe: ONNX Runtime sessions are safe to call from multiple threads
    as long as each call is from a single thread (we use a ThreadPoolExecutor,
    so this is guaranteed).

Directory layout expected (produced by download_model.py):
  <model_path>/
    model.onnx
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

log = logging.getLogger("sentinel.model")

# Model constants for all-MiniLM-L6-v2
EMBEDDING_DIM = 384
MAX_SEQ_LEN   = 256    # MiniLM was trained on max 256 tokens; truncate beyond that


class EmbeddingModel:
    """
    CPU-optimised ONNX inference for all-MiniLM-L6-v2.

    Usage (from a thread):
        model = EmbeddingModel("/var/sentinel/models/minilm")
        vecs  = model.encode(["HDFS distributed filesystem", "another doc"])
        # vecs.shape == (2, 384), float32, L2-normalised
    """

    def __init__(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)
        onnx_path = model_dir / "model.onnx"
        tok_path  = model_dir / "tokenizer.json"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                "Run scripts/download_model.py first."
            )

        # ── ONNX session options ──────────────────────────────────────────────
        opts = ort.SessionOptions()
        opts.intra_op_num_threads  = 4    # tune per machine
        opts.inter_op_num_threads  = 1
        opts.execution_mode        = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Cache the optimised model so subsequent starts are faster
        opts.optimized_model_filepath = str(model_dir / "model_optimised.onnx")

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options   = opts,
            providers      = ["CPUExecutionProvider"],
        )
        log.info("ONNX session loaded: %s", onnx_path)

        # ── Tokeniser ─────────────────────────────────────────────────────────
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_padding(
            pad_id    = 0,
            pad_token = "[PAD]",
            length    = None,   # pad to longest in batch
        )
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        log.info("Tokeniser loaded: %s", tok_path)

    def warmup(self) -> None:
        """Run a single dummy inference to initialise ONNX runtime caches."""
        self.encode(["warmup"])
        log.debug("Model warmup complete")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings → (N, 384) float32 array.
        All vectors are L2-normalised (unit vectors).
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        # Tokenise
        encodings   = self._tokenizer.encode_batch(texts)
        input_ids   = np.array([e.ids           for e in encodings], dtype=np.int64)
        attention   = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_types = np.zeros_like(input_ids,                         dtype=np.int64)

        # ONNX inference
        outputs = self._session.run(
            output_names = None,
            input_feed   = {
                "input_ids":      input_ids,
                "attention_mask": attention,
                "token_type_ids": token_types,
            },
        )

        # outputs[0] shape: (batch, seq_len, hidden_dim)
        token_embeddings = outputs[0]

        # Mean pooling — ignore padding tokens
        mask_expanded = attention[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = (token_embeddings * mask_expanded).sum(axis=1)
        sum_mask       = mask_expanded.sum(axis=1).clip(min=1e-9)
        embeddings     = sum_embeddings / sum_mask   # (N, 384)

        # L2 normalisation
        norms        = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
        embeddings  /= norms

        return embeddings.astype(np.float32)
