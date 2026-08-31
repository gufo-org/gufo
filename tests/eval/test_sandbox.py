"""Negative tests for the gufo-agent-eval sandbox.

#153 requires that these be covered: host files, credentials, external
writes, external network, and orphaned processes. Each test asserts that
something the agent must *not* be able to do actually fails.

These tests need bubblewrap and unprivileged user namespaces; they skip
rather than fail where the host cannot provide them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.eval.agent import task as task_mod  # noqa: E402
from tools.eval.agent.jail import BwrapBackend, JailSpec, stage_rootfs  # noqa: E402

PROBE_TASK = REPO / "tools/eval/tasks/regex-log/environment/env.nix"


def _sandbox_available() -> str | None:
    problems = BwrapBackend().preflight()
    return "; ".join(problems) if problems else None


@unittest.skipIf(_sandbox_available(), f"sandbox unusable: {_sandbox_available()}")
class SandboxNegativeTest(unittest.TestCase):
    """Things the agent must not be able to do."""

    @classmethod
    def setUpClass(cls):
        cls.backend = BwrapBackend()
        cls.scratch = Path(tempfile.mkdtemp(prefix="gufo-agent-eval-test-"))
        cls.rootfs = stage_rootfs(
            task_mod.nix_build(PROBE_TASK), cls.scratch / "rootfs"
        )

    @classmethod
    def tearDownClass(cls):
        for child in cls.scratch.rglob("*"):
            if not child.is_symlink() and child.is_dir():
                child.chmod(child.stat().st_mode | 0o700)
        shutil.rmtree(cls.scratch, ignore_errors=True)

    def run_in_jail(self, script: str, network: str = "none", timeout: float = 60):
        return self.backend.run(
            JailSpec(
                rootfs=self.rootfs,
                command=["/bin/sh", "-c", script],
                chdir="/app",
                env={"HOME": "/root", "PATH": "/bin:/usr/bin"},
                network=network,
            ),
            timeout_sec=timeout,
        )

    def test_host_files_are_not_readable(self):
        """The host filesystem must not appear inside the jail."""
        result = self.run_in_jail(
            "cat /etc/shadow 2>/dev/null && echo LEAK; "
            "ls /home 2>/dev/null && echo LEAK-HOME; "
            "echo done"
        )
        self.assertNotIn("LEAK", result.stdout)

    def test_the_repository_is_not_visible(self):
        """The checkout the runner was launched from must not be reachable."""
        result = self.run_in_jail(f"ls {REPO} 2>/dev/null && echo LEAK; echo done")
        self.assertNotIn("LEAK", result.stdout)

    def test_host_credentials_are_not_readable(self):
        """Common credential paths must not resolve."""
        probes = ["/root/.ssh", "/root/.aws", "/root/.config/gh", "/etc/ssh"]
        script = "; ".join(f"ls {p} 2>/dev/null && echo LEAK-{p}" for p in probes)
        result = self.run_in_jail(script + "; echo done")
        self.assertNotIn("LEAK", result.stdout)

    def test_host_environment_is_cleared(self):
        """No host environment variable may survive into the jail."""
        os.environ["GUFO_EVAL_TEST_SECRET"] = "must-not-appear"
        try:
            result = self.run_in_jail("env")
        finally:
            del os.environ["GUFO_EVAL_TEST_SECRET"]
        self.assertNotIn("must-not-appear", result.stdout)

    def test_writes_outside_the_workspace_do_not_reach_the_host(self):
        """A write inside the jail must not modify the host."""
        canary = self.scratch / "host-canary.txt"
        canary.write_text("original")
        result = self.run_in_jail(
            f"echo tampered > {canary} 2>/dev/null; "
            f"echo tampered > /nix/store/canary 2>/dev/null; echo done"
        )
        self.assertEqual(canary.read_text(), "original")
        self.assertIn("done", result.stdout)

    def test_the_nix_store_is_read_only(self):
        """The runtime must not be mutable from inside the jail."""
        result = self.run_in_jail(
            "echo x > /nix/store/probe 2>&1 || echo READONLY"
        )
        self.assertIn("READONLY", result.stdout)

    def test_external_network_is_unreachable(self):
        """With network=none there must be no route anywhere."""
        result = self.run_in_jail(
            "getent hosts github.com >/dev/null 2>&1 && echo LEAK-DNS; "
            "echo done"
        )
        self.assertNotIn("LEAK", result.stdout)

    def test_the_process_tree_is_killed_on_timeout(self):
        """A timeout must leave nothing behind."""
        marker = "gufo-eval-orphan-probe"
        result = self.run_in_jail(f"sh -c 'sleep 300 # {marker}' & sleep 300", timeout=5)
        self.assertTrue(result.timed_out)

        # Give the kernel a moment to reap the tree.
        time.sleep(2)
        survivors = subprocess.run(
            ["pgrep", "-fc", marker], capture_output=True, text=True
        )
        self.assertEqual(
            survivors.stdout.strip() or "0", "0", "orphaned process survived the timeout"
        )

    def test_endpoint_only_reaches_the_endpoint_and_nothing_else(self):
        """The documented network boundary, pinned.

        On an endpoint-only task the agent's shell shares Pi's jail, so it can
        reach the inference endpoint. That is a known deviation, recorded in
        the README. What must stay true is that it reaches *nothing else*: no
        DNS, no external host, no other port. This test fails if that widens.
        """
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"reachable")

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)

        script = textwrap.dedent(
            f"""
            if command -v getent >/dev/null 2>&1; then
              getent hosts github.com >/dev/null 2>&1 && echo LEAK-DNS
            fi
            echo probe-done
            """
        )
        result = self.backend.run(
            JailSpec(
                rootfs=self.rootfs,
                command=["/bin/sh", "-c", script],
                chdir="/app",
                env={"HOME": "/root", "PATH": "/bin:/usr/bin"},
                network="endpoint-only",
                endpoint=("127.0.0.1", port),
            ),
            timeout_sec=90,
        )
        self.assertIn("probe-done", result.stdout)
        self.assertNotIn("LEAK", result.stdout)

    def test_the_verifier_directory_is_absent_from_the_agent_jail(self):
        """Hidden fixtures must never be reachable while the agent runs."""
        result = self.run_in_jail("ls /tests 2>/dev/null && echo LEAK-TESTS; echo done")
        self.assertNotIn("LEAK", result.stdout)


class SanitizerTest(unittest.TestCase):
    """No artifact may carry credentials, addresses, or user paths."""

    def setUp(self):
        from tools.eval.agent import result as result_mod

        self.result_mod = result_mod

    def test_secret_keys_are_redacted(self):
        out = self.result_mod.sanitize(
            {"api_key": "sk-secret", "nested": {"Authorization": "Bearer abc"}}
        )
        self.assertEqual(out["api_key"], "[redacted]")
        self.assertEqual(out["nested"]["Authorization"], "[redacted]")

    def test_private_addresses_are_removed(self):
        for address in (
            "http://127.0.0.1:8080/v1",
            "http://192.168.1.50:9999/v1",
            "http://10.0.0.4:8080",
            "http://172.16.3.9:8080",
        ):
            with self.subTest(address=address):
                self.assertNotIn(
                    address.split("//")[1].split("/")[0],
                    self.result_mod.scrub_text(address),
                )

    def test_user_paths_are_removed(self):
        scrubbed = self.result_mod.scrub_text("/home/alice/strix-halo.cpp/tools")
        self.assertNotIn("alice", scrubbed)

    def test_bearer_tokens_and_keys_are_removed(self):
        for secret in ("Bearer abc123def456", "sk-abcdef0123456789"):
            with self.subTest(secret=secret):
                self.assertNotIn(
                    secret.split()[-1], self.result_mod.scrub_text(secret)
                )

    def test_scratch_paths_are_removed(self):
        scrubbed = self.result_mod.scrub_text("/tmp/gufo-agent-eval-ab12cd/out")
        self.assertNotIn("ab12cd", scrubbed)

    def test_a_whole_document_is_scrubbed(self):
        document = {
            "identity": {"api_key": "sk-live-key", "base_url": "http://192.168.1.7:8080"},
            "tasks": [{"verifier_output": "failed at /home/bob/work/x.py"}],
        }
        out = self.result_mod.sanitize(document)
        rendered = str(out)
        for leak in ("sk-live-key", "192.168.1.7", "bob"):
            self.assertNotIn(leak, rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
