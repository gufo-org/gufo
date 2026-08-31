"""Every task manifest must parse and carry its provenance.

A generator produced these manifests, and a quoting bug in it shipped five
unparseable files. This test is the guard against that recurring.
"""

import sys
import tomllib
import unittest
from pathlib import Path

TASKS = Path(__file__).resolve().parents[2] / "tools" / "eval" / "tasks"

REQUIRED_PROVENANCE = (
    "upstream",
    "upstream_revision",
    "upstream_authors",
    "license",
)


class TaskManifestTest(unittest.TestCase):
    def task_dirs(self):
        return sorted(p for p in TASKS.iterdir() if (p / "task.toml").is_file())

    def test_suite_is_not_empty(self):
        self.assertTrue(self.task_dirs(), f"no tasks under {TASKS}")

    def test_manifests_parse_and_declare_provenance(self):
        for directory in self.task_dirs():
            with self.subTest(task=directory.name):
                manifest = tomllib.loads((directory / "task.toml").read_text())

                provenance = manifest.get("provenance", {})
                for key in REQUIRED_PROVENANCE:
                    self.assertIn(key, provenance, f"{directory.name}: missing {key}")

                # Apache-2.0 section 4b: these are modified, not verbatim.
                self.assertTrue(
                    provenance.get("modified_from_upstream"),
                    f"{directory.name}: must declare modified_from_upstream",
                )

                network = manifest.get("environment", {}).get("network")
                self.assertIn(network, ("none", "endpoint-only"), directory.name)

    def test_required_files_exist(self):
        for directory in self.task_dirs():
            with self.subTest(task=directory.name):
                for relative in (
                    "instruction.md",
                    "PROVENANCE.md",
                    "environment/env.nix",
                    "tests/verifier.nix",
                ):
                    self.assertTrue(
                        (directory / relative).is_file(),
                        f"{directory.name}: missing {relative}",
                    )

    def test_canary_is_preserved(self):
        """Ported files must keep the upstream benchmark-data canary."""
        for directory in self.task_dirs():
            with self.subTest(task=directory.name):
                for relative in ("task.toml", "environment/env.nix", "tests/verifier.nix"):
                    text = (directory / relative).read_text()
                    self.assertIn(
                        "terminal-bench-canary",
                        text,
                        f"{directory.name}/{relative}: canary string removed",
                    )

    def test_tests_are_not_reachable_from_the_agent_workdir(self):
        """Hidden fixtures must not sit inside the agent-visible environment."""
        for directory in self.task_dirs():
            with self.subTest(task=directory.name):
                staged = (directory / "environment").rglob("test_outputs.py")
                for path in staged:
                    # break-filter-js-from-html deliberately exposes its tests;
                    # upstream does the same and PROVENANCE records it.
                    self.assertEqual(
                        directory.name,
                        "break-filter-js-from-html",
                        f"{directory.name}: verifier reachable at {path}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
