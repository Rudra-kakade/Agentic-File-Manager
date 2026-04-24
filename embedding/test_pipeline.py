"""
tests/test_pipeline.py

Integration test for the full embedding engine + vector store pipeline.

Run with:
    pytest tests/test_pipeline.py -v

Requires the model to be downloaded first:
    python scripts/download_model.py --output /tmp/sentinel-test-model
    SENTINEL_MODEL_PATH=/tmp/sentinel-test-model pytest tests/test_pipeline.py -v

Without the model, model-dependent tests are skipped automatically.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def model_dir() -> Optional[Path]:
    p = os.environ.get("SENTINEL_MODEL_PATH")
    if p and (Path(p) / "model.onnx").exists():
        return Path(p)
    return None


# ── Unit tests: HnswIndex ─────────────────────────────────────────────────────

class TestHnswIndex:
    def _make_index(self, tmp_dir):
        from src.config import VectorStoreConfig
        from src.vector_store import HnswIndex
        cfg = VectorStoreConfig(
            index_path         = tmp_dir / "hnsw",
            listen_socket      = tmp_dir / "vs.sock",
            max_elements       = 10_000,
            persist_interval_s = 999.0,
        )
        return HnswIndex(cfg)

    def test_upsert_and_search(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        vec = np.random.rand(384).astype(np.float32)
        vec /= np.linalg.norm(vec)

        idx.upsert("/home/user/file.txt", 1000, vec)
        results = idx.search(vec, k=5)

        assert len(results) == 1
        assert results[0][0] == "/home/user/file.txt"
        assert results[0][1] > 0.99   # should be near 1.0 (same vector)

    def test_upsert_overwrites(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        v1 = np.ones(384, dtype=np.float32); v1 /= np.linalg.norm(v1)
        v2 = np.zeros(384, dtype=np.float32); v2[0] = 1.0

        idx.upsert("/file.txt", 1000, v1)
        idx.upsert("/file.txt", 2000, v2)  # same path, new vector

        # Should return only 1 result (the updated one)
        results = idx.search(v2, k=10)
        paths = [r[0] for r in results]
        assert paths.count("/file.txt") == 1

    def test_filter_paths(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        query = np.ones(384, dtype=np.float32); query /= np.linalg.norm(query)

        for i in range(5):
            v = np.random.rand(384).astype(np.float32)
            v /= np.linalg.norm(v)
            idx.upsert(f"/file{i}.txt", i, v)

        # Restrict to only /file0.txt and /file1.txt
        results = idx.search(query, k=10, filter_paths=["/file0.txt", "/file1.txt"])
        paths   = {r[0] for r in results}
        assert paths.issubset({"/file0.txt", "/file1.txt"})

    def test_delete(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        vec = np.ones(384, dtype=np.float32); vec /= np.linalg.norm(vec)
        idx.upsert("/del.txt", 1000, vec)
        assert idx.delete("/del.txt") is True
        results = idx.search(vec, k=5)
        assert all(r[0] != "/del.txt" for r in results)

    def test_persist_and_reload(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        vec = np.ones(384, dtype=np.float32); vec /= np.linalg.norm(vec)
        idx.upsert("/persist.txt", 1234, vec)
        idx.persist()

        # Reload
        idx2 = self._make_index(tmp_dir)
        results = idx2.search(vec, k=5)
        assert any(r[0] == "/persist.txt" for r in results)

    def test_stats(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        s   = idx.stats()
        assert "total_vectors" in s
        assert "deleted" in s

    def test_compaction(self, tmp_dir):
        idx = self._make_index(tmp_dir)
        vecs = {}
        for i in range(20):
            v = np.random.rand(384).astype(np.float32); v /= np.linalg.norm(v)
            idx.upsert(f"/file{i}.txt", i, v)
            vecs[f"/file{i}.txt"] = v

        # Delete 80% to trigger compaction threshold
        for i in range(16):
            idx.delete(f"/file{i}.txt")

        idx.maybe_compact()
        assert len(idx._deleted) == 0
        assert idx.stats()["total_vectors"] == 4


# ── Unit tests: EmbeddingModel ─────────────────────────────────────────────────

class TestEmbeddingModel:
    @pytest.mark.skipif(
        not os.environ.get("SENTINEL_MODEL_PATH"),
        reason="Set SENTINEL_MODEL_PATH to run model tests",
    )
    def test_encode_single(self, model_dir):
        from src.model import EmbeddingModel
        m   = EmbeddingModel(model_dir)
        out = m.encode(["Hello world"])
        assert out.shape == (1, 384)
        # Check L2-normalised
        norm = np.linalg.norm(out[0])
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.skipif(
        not os.environ.get("SENTINEL_MODEL_PATH"),
        reason="Set SENTINEL_MODEL_PATH to run model tests",
    )
    def test_encode_batch(self, model_dir):
        from src.model import EmbeddingModel
        texts = ["HDFS distributed file system", "Python programming", "neural network training"]
        m     = EmbeddingModel(model_dir)
        out   = m.encode(texts)
        assert out.shape == (3, 384)

    @pytest.mark.skipif(
        not os.environ.get("SENTINEL_MODEL_PATH"),
        reason="Set SENTINEL_MODEL_PATH to run model tests",
    )
    def test_semantic_similarity(self, model_dir):
        """Similar texts should have higher cosine similarity than dissimilar ones."""
        from src.model import EmbeddingModel
        m    = EmbeddingModel(model_dir)
        vecs = m.encode([
            "HDFS Hadoop distributed file system",
            "distributed file system storage",
            "chocolate chip cookie recipe",
        ])
        sim_related    = float(np.dot(vecs[0], vecs[1]))
        sim_unrelated  = float(np.dot(vecs[0], vecs[2]))
        assert sim_related > sim_unrelated, (
            f"Expected HDFS texts to be more similar to each other "
            f"({sim_related:.3f}) than to cookie recipe ({sim_unrelated:.3f})"
        )


# ── Integration test: full vector store server over socket ────────────────────

class TestVectorStoreServer:
    @pytest.fixture
    def server_task(self, tmp_dir):
        """Spin up a real VectorStoreServer in a background asyncio task."""
        from src.config import VectorStoreConfig
        from src.vector_store import VectorStoreServer

        cfg = VectorStoreConfig(
            index_path         = tmp_dir / "hnsw",
            listen_socket      = tmp_dir / "vs.sock",
            max_elements       = 1000,
            persist_interval_s = 9999.0,
        )

        async def _run():
            srv = VectorStoreServer(cfg)
            await srv.run()

        loop = asyncio.new_event_loop()

        import threading
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        fut = asyncio.run_coroutine_threadsafe(_run(), loop)

        import time; time.sleep(0.3)  # allow server to bind

        yield str(tmp_dir / "vs.sock")

        loop.call_soon_threadsafe(loop.stop)

    def _send_recv(self, sock_path: str, req: dict) -> dict:
        import socket, json, struct
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(sock_path)
            data   = json.dumps(req).encode()
            header = struct.pack("<I", len(data))
            s.sendall(header + data)
            raw_len = s.recv(4)
            length  = struct.unpack("<I", raw_len)[0]
            resp    = b""
            while len(resp) < length:
                resp += s.recv(length - len(resp))
        return json.loads(resp)

    def test_upsert_and_search_over_socket(self, server_task):
        vec = np.ones(384, dtype=np.float32)
        vec /= np.linalg.norm(vec)

        r1 = self._send_recv(server_task, {
            "op": "upsert", "path": "/test.txt", "timestamp": 1000,
            "vector": vec.tolist(),
        })
        assert r1["ok"] is True

        r2 = self._send_recv(server_task, {
            "op": "search", "vector": vec.tolist(), "k": 5,
        })
        assert r2["ok"] is True
        assert any(r["path"] == "/test.txt" for r in r2["results"])

    def test_stats_over_socket(self, server_task):
        r = self._send_recv(server_task, {"op": "stats"})
        assert r["ok"] is True
        assert "total_vectors" in r
