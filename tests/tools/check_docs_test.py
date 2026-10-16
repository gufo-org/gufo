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

    def test_numbered_headings_do_not_collide(self):
        for headings, expected in (
            ("## Dup\n## Dup-1\n## Dup\n", {"dup", "dup-1", "dup-2"}),
            ("## Dup\n## Dup\n## Dup-1\n", {"dup", "dup-1", "dup-1-1"}),
            ("## Dup\n## Dup-1\n## Dup\n## Dup\n## Dup-1\n",
             {"dup", "dup-1", "dup-2", "dup-3", "dup-1-1"}),
        ):
            with self.subTest(headings=headings):
                self.assertEqual(DOCS.extract_anchors(headings), expected)
                links = "\n".join(f"[jump](#{anchor})" for anchor in sorted(expected))
                self.assertEqual(self.check({"a.md": headings + links}), [])

    def test_fenced_examples_do_not_validate_literal_links(self):
        for opening, closing in (
            ("```markdown", "```"),
            ("~~~markdown", "~~~"),
            ("````markdown", "````"),
            ("   ~~~`example`", "   ~~~~\t"),
        ):
            with self.subTest(opening=opening):
                content = (
                    f"# Guide\n\n{opening}\n## Example\n"
                    f"[Jump](#example) [File](missing.md)\n{closing}\n\n"
                    "[Guide](#guide)\n"
                )
                self.assertEqual(self.check({"a.md": content}), [])
                self.assertEqual(DOCS.extract_anchors(content), {"guide"})

    def test_backticks_in_info_do_not_open_a_fence(self):
        for opening in ("```foo`bar", "   ````foo`bar"):
            with self.subTest(opening=opening):
                content = f"# Guide\n\n{opening}\n## Visible\n\n[Visible](#visible)\n"
                self.assertEqual(self.check({"a.md": content}), [])
                self.assertEqual(DOCS.extract_anchors(content), {"guide", "visible"})

    def test_fences_require_a_matching_closing_line(self):
        for opening, false_closing, closing in (
            ("````markdown", "```", "````"),
            ("```markdown", "~~~", "```"),
            ("~~~markdown", "```", "~~~"),
            ("~~~markdown", "~~~ trailing text", "~~~"),
        ):
            with self.subTest(opening=opening, false_closing=false_closing):
                content = (
                    f"# Guide\n\n{opening}\n{false_closing}\n"
                    f"## Hidden\n[Hidden](#hidden) [File](missing.md)\n{closing}\n"
                    "## Visible\n[Visible](#visible)\n"
                )
                self.assertEqual(self.check({"a.md": content}), [])
                self.assertEqual(DOCS.extract_anchors(content), {"guide", "visible"})

    def test_unclosed_fence_hides_literal_links(self):
        for opening in ("```markdown", "~~~markdown"):
            with self.subTest(opening=opening):
                content = f"# Guide\n\n{opening}\n## Hidden\n[Hidden](missing.md#hidden)\n"
                self.assertEqual(self.check({"a.md": content}), [])
                self.assertEqual(DOCS.extract_anchors(content), {"guide"})

    def test_link_errors_after_fences_keep_line_numbers(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                content = newline.join((
                    "# Guide", "", "```markdown", "## Example",
                    "[Jump](#example)", "```", "[Broken](#example)", "",
                ))
                errors = self.check({"a.md": content})
                self.assertEqual(len(errors), 1, errors)
                self.assertIn("a.md:7: Broken anchor '#example'", errors[0])

    def test_json_blocks_still_validate_syntax(self):
        valid = '# Guide\n\n```json\n{"example": "[Jump](missing.md#example)"}\n```\n'
        self.assertEqual(self.check({"a.md": valid}), [])
        invalid = '# Guide\n\n```json\n{"values": [1, 2,]}\n```\n'
        errors = self.check({"a.md": invalid})
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("a.md:3: Invalid JSON in fenced code block", errors[0])


if __name__ == "__main__":
    unittest.main()
