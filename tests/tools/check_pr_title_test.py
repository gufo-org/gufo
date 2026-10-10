#!/usr/bin/env python3
"""CPU-only regressions for the pull request title check."""

import contextlib
import importlib.util
import io
from pathlib import Path
import re
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_pr_title", ROOT / "tools/ci/check-pr-title.py")
TITLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TITLE)


class CheckPrTitleTest(unittest.TestCase):
    def test_release_policy_examples_pass(self):
        # The examples in docs/RELEASING.md must stay valid titles.
        for title in (
                "fix(cache): preserve image prefixes after cancellation",
                "perf(qwen): make prompt tokenization linear",
                "feat(server): support constrained JSON output",
                "docs: explain the release policy",
                "feat(server)!: remove the legacy completion endpoint",
                "fix(qwen38_flash_next): keep a dotted.scope/path-like scope"):
            with self.subTest(title=title):
                self.assertIsNone(TITLE.validate(title))

    def test_types_match_release_policy(self):
        # The type table in docs/RELEASING.md is the policy the check enforces.
        policy = (ROOT / "docs/RELEASING.md").read_text()
        table = policy.split("The accepted types are:", 1)[1].split("\n\n", 2)[1]
        documented = re.findall(r"^\| `([a-z]+)` \|", table, re.MULTILINE)
        self.assertTrue(documented, "the release policy type table moved")
        self.assertEqual(sorted(documented), sorted(TITLE.TYPES))
        for kind in documented:
            with self.subTest(kind=kind):
                self.assertIsNone(TITLE.validate(f"{kind}: describe it"))

    def test_invalid_titles_fail(self):
        for title in (
                "",
                "Fix: capitalized type",
                "feature: unknown type",
                "fix:missing space",
                "fix: ",
                "fix:  leading space in the description",
                "fix(): empty scope",
                "fix(Cache): uppercase scope",
                "fix(-cache): scope starts with punctuation",
                "fix(cache) : space before colon",
                "fix!(cache): breaking marker before the scope",
                "fix(cache)!!: doubled breaking marker",
                "Merge branch 'main' into feature"):
            with self.subTest(title=title):
                self.assertIsNotNone(TITLE.validate(title))

    def test_multiline_titles_fail(self):
        for title in ("fix: one\ntwo", "fix: one\rtwo", "fix: one\n"):
            with self.subTest(title=title):
                self.assertEqual(TITLE.validate(title),
                                 "the title must be a single line")

    def test_main_reports_status(self):
        for title, status, prefix in (
                ("fix(ci): cover the title check", 0, "PASS:"),
                ("Fix the title check", 1, "FAIL:")):
            with self.subTest(title=title):
                output = io.StringIO()
                with mock.patch.object(sys, "argv", ["check-pr-title.py", title]), \
                        contextlib.redirect_stdout(output):
                    self.assertEqual(TITLE.main(), status)
                self.assertTrue(output.getvalue().startswith(prefix),
                                output.getvalue())


if __name__ == "__main__":
    unittest.main()
