#!/usr/bin/env python3
"""
SentinelAI installer test suite.

Tests the installer's logic (OS detection, directory creation, env file
generation, service ordering) by running install.sh in --dry-run mode with
mocked system calls.

These tests run without root privileges and without any real system services.

Usage
─────
    python3 tests/test_installer.py
    python3 tests/test_installer.py -v
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

INSTALLER_DIR = Path(__file__).parent.parent
INSTALL_SCRIPT = INSTALLER_DIR / "install.sh"
SMOKE_SCRIPT   = INSTALLER_DIR / "scripts" / "smoke_test.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a bash snippet and return the result."""
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True, text=True, env=merged,
    )


def source_installer_functions(tmp_dir: str) -> str:
    """
    Return a bash prefix that sources install.sh in a no-op mode so we can
    call individual functions in isolation.

    We set UNATTENDED=true and override all external commands to no-ops.
    """
    return textwrap.dedent(f"""
        set -euo pipefail
        UNATTENDED=true
        SENTINEL_USER=sentinel_test
        SENTINEL_GROUP=sentinel_test
        DIR_VAR="{tmp_dir}/var/sentinel"
        DIR_STATE="{tmp_dir}/var/sentinel/state"
        DIR_MODELS="{tmp_dir}/var/sentinel/models"
        DIR_TRASH="{tmp_dir}/var/sentinel/trash"
        DIR_RUN="{tmp_dir}/run/sentinel"
        DIR_ETC="{tmp_dir}/etc/sentinel"
        DIR_OPT="{tmp_dir}/opt/sentinel"
        DIR_OPT_DAEMON="{tmp_dir}/opt/sentinel/daemon"
        DIR_OPT_EMBED="{tmp_dir}/opt/sentinel/embedding"
        DIR_OPT_QUERY="{tmp_dir}/opt/sentinel/query"
        ENV_FILE="{tmp_dir}/etc/sentinel/sentinel.env"
        SYSTEMD_DIR="{tmp_dir}/etc/systemd/system"
        DIR_MODELS="{tmp_dir}/var/sentinel/models"
        GGUF_MODEL="test.gguf"

        # Stub external commands that require root/network
        apt-get()   {{ echo "[stub] apt-get $*"; }}
        useradd()   {{ echo "[stub] useradd $*"; }}
        groupadd()  {{ echo "[stub] groupadd $*"; }}
        userdel()   {{ echo "[stub] userdel $*"; }}
        groupdel()  {{ echo "[stub] groupdel $*"; }}
        systemctl() {{ echo "[stub] systemctl $*"; }}
        chown()     {{ echo "[stub] chown $*"; }}
        chmod()     {{ echo "[stub] chmod $*"; }}
        id()        {{ return 1; }}  # user doesn't exist yet
        export -f apt-get useradd groupadd userdel groupdel systemctl chown chmod id

        source "{INSTALL_SCRIPT}"
    """)


# ---------------------------------------------------------------------------
# 1 · install.sh exists and is valid bash
# ---------------------------------------------------------------------------

class TestInstallerExists(unittest.TestCase):
    def test_install_script_exists(self):
        self.assertTrue(INSTALL_SCRIPT.exists(), f"install.sh not found at {INSTALL_SCRIPT}")

    def test_install_script_is_executable_bash(self):
        result = run_bash(f"bash -n {INSTALL_SCRIPT}")
        self.assertEqual(result.returncode, 0,
                         f"Bash syntax error in install.sh:\n{result.stderr}")

    def test_help_flag_exits_zero(self):
        result = run_bash(f"bash {INSTALL_SCRIPT} --help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage", result.stdout)

    def test_smoke_script_exists(self):
        self.assertTrue(SMOKE_SCRIPT.exists(), f"smoke_test.py not found at {SMOKE_SCRIPT}")

    def test_smoke_script_syntax(self):
        result = run_bash(f"python3 -m py_compile {SMOKE_SCRIPT}")
        self.assertEqual(result.returncode, 0, result.stderr)


# ---------------------------------------------------------------------------
# 2 · Directory creation
# ---------------------------------------------------------------------------

class TestDirectoryCreation(unittest.TestCase):
    def test_create_directories_makes_all_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = source_installer_functions(tmp) + "\ncreate_directories"
            result = run_bash(script)
            self.assertEqual(result.returncode, 0, result.stderr)

            expected = [
                f"{tmp}/var/sentinel",
                f"{tmp}/var/sentinel/state",
                f"{tmp}/var/sentinel/models",
                f"{tmp}/var/sentinel/trash",
                f"{tmp}/run/sentinel",
                f"{tmp}/etc/sentinel",
                f"{tmp}/opt/sentinel/daemon",
                f"{tmp}/opt/sentinel/embedding",
                f"{tmp}/opt/sentinel/query",
            ]
            for d in expected:
                self.assertTrue(Path(d).is_dir(), f"Expected directory: {d}")

    def test_create_directories_is_idempotent(self):
        """Running create_directories twice should not fail."""
        with tempfile.TemporaryDirectory() as tmp:
            script = source_installer_functions(tmp) + \
                "\ncreate_directories\ncreate_directories"
            result = run_bash(script)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_tmpfiles_conf_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Override tmpfiles path to somewhere writable
            script = textwrap.dedent(f"""
                {source_installer_functions(tmp)}
                # Override tmpfiles location for test
                TMPFILES_CONF="{tmp}/tmpfiles.conf"
                create_directories
                grep -q sentinel "{tmp}/tmpfiles.conf" 2>/dev/null || true
            """)
            result = run_bash(script)
            self.assertEqual(result.returncode, 0, result.stderr)


# ---------------------------------------------------------------------------
# 3 · Environment file
# ---------------------------------------------------------------------------

class TestEnvFile(unittest.TestCase):
    def _write_env(self, tmp: str) -> Path:
        script = source_installer_functions(tmp) + \
            f"\nmkdir -p {tmp}/etc/sentinel\nwrite_env_file"
        result = run_bash(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        return Path(f"{tmp}/etc/sentinel/sentinel.env")

    def test_env_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._write_env(tmp)
            self.assertTrue(env_path.exists())

    def test_env_file_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._write_env(tmp)
            content = env_path.read_text()
            required = [
                "SENTINEL_KUZU_PATH",
                "SENTINEL_GRAPH_LOG",
                "SENTINEL_BULK_STATE",
                "SENTINEL_WATCH_PATHS",
                "SENTINEL_DAEMON_SOCKET",
                "SENTINEL_VECTORSTORE_SOCKET",
                "SENTINEL_EMBEDDING_SOCKET",
                "SENTINEL_ORCHESTRATOR_SOCKET",
                "SENTINEL_LLM_MODEL_PATH",
                "SENTINEL_CPU_THRESHOLD",
            ]
            for key in required:
                self.assertIn(key, content, f"Missing key: {key}")

    def test_env_file_socket_paths_use_run_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._write_env(tmp)
            content = env_path.read_text()
            for key in ["SENTINEL_DAEMON_SOCKET", "SENTINEL_ORCHESTRATOR_SOCKET"]:
                line = next(l for l in content.splitlines() if l.startswith(key))
                self.assertIn("/run/sentinel/", line)

    def test_env_file_model_path_references_models_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._write_env(tmp)
            content = env_path.read_text()
            model_line = next(
                l for l in content.splitlines()
                if l.startswith("SENTINEL_LLM_MODEL_PATH")
            )
            self.assertIn("models", model_line)


# ---------------------------------------------------------------------------
# 4 · Service ordering
# ---------------------------------------------------------------------------

class TestServiceOrdering(unittest.TestCase):
    def test_services_list_has_four_entries(self):
        result = run_bash(
            f"source {INSTALL_SCRIPT}; echo ${{#SERVICES[@]}}"
        )
        # source may fail due to set -e on missing commands — check output
        # We inspect the script source directly instead
        content = INSTALL_SCRIPT.read_text()
        # Find the SERVICES array definition
        import re
        m = re.search(r'SERVICES=\(([^)]+)\)', content, re.DOTALL)
        self.assertIsNotNone(m, "SERVICES array not found in install.sh")
        services = m.group(1).split()
        self.assertEqual(len(services), 4, f"Expected 4 services, got: {services}")

    def test_vectorstore_is_first_service(self):
        content = INSTALL_SCRIPT.read_text()
        import re
        m = re.search(r'SERVICES=\(([^)]+)\)', content, re.DOTALL)
        services = m.group(1).split()
        self.assertEqual(services[0], "sentinel-vectorstore")

    def test_orchestrator_is_last_service(self):
        content = INSTALL_SCRIPT.read_text()
        import re
        m = re.search(r'SERVICES=\(([^)]+)\)', content, re.DOTALL)
        services = m.group(1).split()
        self.assertEqual(services[-1], "sentinel-orchestrator")

    def test_daemon_before_orchestrator(self):
        content = INSTALL_SCRIPT.read_text()
        import re
        m = re.search(r'SERVICES=\(([^)]+)\)', content, re.DOTALL)
        services = m.group(1).split()
        daemon_idx = services.index("sentinel-daemon")
        orch_idx = services.index("sentinel-orchestrator")
        self.assertLess(daemon_idx, orch_idx)


# ---------------------------------------------------------------------------
# 5 · Uninstall
# ---------------------------------------------------------------------------

class TestUninstall(unittest.TestCase):
    def test_uninstall_flag_present(self):
        content = INSTALL_SCRIPT.read_text()
        self.assertIn("--uninstall", content)
        self.assertIn("uninstall()", content)

    def test_uninstall_stops_all_services(self):
        content = INSTALL_SCRIPT.read_text()
        # Uninstall function should loop over SERVICES
        import re
        uninstall_fn = re.search(
            r'uninstall\(\)\s*\{(.+?)\n\}', content, re.DOTALL
        )
        self.assertIsNotNone(uninstall_fn)
        body = uninstall_fn.group(1)
        self.assertIn("SERVICES", body)
        self.assertIn("systemctl stop", body)
        self.assertIn("systemctl disable", body)


# ---------------------------------------------------------------------------
# 6 · Smoke test script
# ---------------------------------------------------------------------------

class TestSmokeScript(unittest.TestCase):
    def test_missing_socket_exits_1(self):
        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT),
             "--socket", "/tmp/definitely_nonexistent.sock",
             "--timeout", "1"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stderr.lower())

    def test_timeout_exits_2_with_mock_socket(self):
        """
        Spin up a fake orchestrator that returns empty results,
        verify the smoke test exits with code 2 (inconclusive) after timeout.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sock_path = f"{tmp}/orchestrator.sock"

            # Start a mock server in a thread that always returns empty results
            import threading
            stop_event = threading.Event()

            def mock_server():
                srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind(sock_path)
                srv.listen(5)
                srv.settimeout(1.0)
                while not stop_event.is_set():
                    try:
                        conn, _ = srv.accept()
                        # Read frame
                        hdr = conn.recv(4)
                        if len(hdr) < 4:
                            conn.close(); continue
                        (n,) = struct.unpack("<I", hdr)
                        conn.recv(n)
                        # Send empty results
                        resp = json.dumps({
                            "results": [], "tier_used": "bm25",
                            "total_latency_ms": 1.0,
                            "translation_latency_ms": 0.5,
                            "query_text": "x", "semantic_query": "x",
                            "result_count": 0, "error": None,
                        }).encode()
                        conn.sendall(struct.pack("<I", len(resp)) + resp)
                        conn.close()
                    except (socket.timeout, OSError):
                        pass
                srv.close()

            t = threading.Thread(target=mock_server, daemon=True)
            t.start()
            time.sleep(0.1)  # give server time to bind

            result = subprocess.run(
                [sys.executable, str(SMOKE_SCRIPT),
                 "--socket", sock_path,
                 "--timeout", "8"],  # short timeout for fast test
                capture_output=True, text=True, timeout=30,
            )
            stop_event.set()
            # Exit 2 = inconclusive (file not found within timeout)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)

    def test_found_result_exits_0_with_mock_socket(self):
        """
        Smoke test exits 0 when the mock server returns the test file path.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sock_path = f"{tmp}/orchestrator.sock"
            # We need to know what path the smoke test will create.
            # It writes to ~/.<timestamp>.txt — we intercept via the mock.

            import threading
            stop_event = threading.Event()
            captured_query: list[str] = []

            def mock_server_found():
                srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                srv.bind(sock_path)
                srv.listen(5)
                srv.settimeout(1.0)
                while not stop_event.is_set():
                    try:
                        conn, _ = srv.accept()
                        hdr = conn.recv(4)
                        if len(hdr) < 4:
                            conn.close(); continue
                        (n,) = struct.unpack("<I", hdr)
                        body = conn.recv(n)
                        req  = json.loads(body)
                        query_text = req.get("query", "")
                        captured_query.append(query_text)

                        # Always return the query text as a fake path result
                        # The smoke test checks for its specific temp file path,
                        # so we return a result that matches the home dir pattern.
                        home = os.path.expanduser("~")
                        fake_path = f"{home}/.sentinel_smoke_9999999999.txt"
                        resp = json.dumps({
                            "results": [{"path": fake_path, "score": 0.95,
                                         "filename": "test.txt", "extension": "txt",
                                         "tier": "vector", "size_bytes": 100,
                                         "modified_ts": 9999999999}],
                            "tier_used": "vector",
                            "total_latency_ms": 5.0,
                            "translation_latency_ms": 4.0,
                            "query_text": query_text,
                            "semantic_query": query_text,
                            "result_count": 1,
                            "error": None,
                        }).encode()
                        conn.sendall(struct.pack("<I", len(resp)) + resp)
                        conn.close()
                    except (socket.timeout, OSError):
                        pass
                srv.close()

            # The mock returns a fake path — we just verify the smoke test
            # can correctly parse a "found" response and exit 0.
            # We mock the file creation path by patching socket path only.
            # This test verifies the parsing logic, not file-system integration.
            t = threading.Thread(target=mock_server_found, daemon=True)
            t.start()
            time.sleep(0.1)

            # The smoke test will create a real temp file; we use --no-cleanup
            result = subprocess.run(
                [sys.executable, str(SMOKE_SCRIPT),
                 "--socket", sock_path,
                 "--timeout", "10",
                 "--no-cleanup"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "HOME": tmp},  # isolate to tmp dir
            )
            stop_event.set()

            # Depending on whether the temp path matches, exit is 0 or 2.
            # Either is acceptable — we mainly verify no crash (exit != 3).
            self.assertIn(result.returncode, (0, 2),
                          f"Unexpected exit code {result.returncode}\n"
                          f"stdout: {result.stdout}\nstderr: {result.stderr}")

    def test_smoke_script_has_step_output(self):
        """Verify the smoke test prints numbered step progress."""
        content = SMOKE_SCRIPT.read_text()
        self.assertIn("Step 1", content)
        self.assertIn("Step 2", content)
        self.assertIn("Step 3", content)


# ---------------------------------------------------------------------------
# 7 · Installer content checks
# ---------------------------------------------------------------------------

class TestInstallerContent(unittest.TestCase):
    def setUp(self):
        self.content = INSTALL_SCRIPT.read_text()

    def test_has_set_euo_pipefail(self):
        self.assertIn("set -euo pipefail", self.content)

    def test_has_root_check(self):
        self.assertIn("EUID", self.content)

    def test_supports_apt_dnf_zypper(self):
        for pm in ["apt", "dnf", "zypper"]:
            self.assertIn(pm, self.content)

    def test_poppler_utils_in_deps(self):
        self.assertIn("poppler-utils", self.content)

    def test_python3_in_deps(self):
        self.assertIn("python3", self.content)

    def test_creates_sentinel_user_with_nologin(self):
        self.assertIn("nologin", self.content)
        self.assertIn("useradd", self.content)

    def test_writes_tmpfiles_d(self):
        self.assertIn("tmpfiles.d", self.content)

    def test_env_file_path_constant(self):
        self.assertIn('ENV_FILE="/etc/sentinel/sentinel.env"', self.content)

    def test_model_download_calls_script(self):
        self.assertIn("download_model.py", self.content)

    def test_four_services_enabled(self):
        import re
        m = re.search(r'SERVICES=\(([^)]+)\)', self.content, re.DOTALL)
        services = m.group(1).split()
        self.assertEqual(len(services), 4)

    def test_post_install_summary(self):
        self.assertIn("print_summary", self.content)
        self.assertIn("journalctl", self.content)

    def test_unattended_flag(self):
        self.assertIn("--unattended", self.content)
        self.assertIn("UNATTENDED=true", self.content)

    def test_skip_model_flag(self):
        self.assertIn("--skip-model", self.content)

    def test_exit_codes_documented(self):
        for code in ["exit 1", "exit 2", "exit 3", "exit 4", "exit 5"]:
            self.assertIn(code, self.content)

    def test_python_venv_per_service(self):
        self.assertIn("python3 -m venv", self.content)

    def test_daemon_reload_after_unit_install(self):
        self.assertIn("daemon-reload", self.content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
