#!/usr/bin/env python3
"""CPU-only regressions for local Markdown link and anchor checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("check_docs", ROOT / "tools/ci/check-docs.py")
DOCS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCS)


class CheckDocsTest(unittest.TestCase):
    def test_slugs_follow_github(self):
        for title, slug in (
            ("`max_tokens`", "max_tokens"),
            ("Foo & Bar", "foo--bar"),
            ("**Bold** _emphasis_ snake_case", "bold-emphasis-snake_case"),
            ("[Linked](other.md) title ##", "linked-title"),
            ("Qwen3.8 27B", "qwen38-27b"),
        ):
            with self.subTest(title=title):
                self.assertEqual(DOCS.slugify_heading(title), slug)

    def test_anchors_number_repeats_and_skip_code(self):
        anchors = DOCS.extract_anchors(
            "# Dup\n## Dup\n```sh\n# Ghost heading\n```\n~~~\n# Tilde ghost\n~~~\n# After\n")
        self.assertEqual(anchors, {"dup", "dup-1", "after"})

    def check(self, files):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text)
            paths = sorted(root.rglob("*.md"))
            return DOCS.check_links_and_anchors(root, paths)[1]

    def test_links_resolve_from_the_linking_file(self):
        files = {"README.md": "# Top\n", "sub/x.md": "[x](README.md)\n"}
        self.assertEqual(len(self.check(files)), 1)
        files["sub/x.md"] = "[x](../README.md#top) [y](/README.md#top)\n"
        self.assertEqual(self.check(files), [])

    def test_anchor_errors(self):
        files = {"a.md": "## max_tokens\n## Dup\n## Dup\n"
                         "[ok](#max_tokens) [ok](#dup-1) [bad](#maxtokens) [bad](#dup-2)\n"}
        errors = self.check(files)
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(all("#maxtokens" in e or "#dup-2" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
