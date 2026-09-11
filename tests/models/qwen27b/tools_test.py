#!/usr/bin/env python3
"""The production sweep must never turn partial results into a speed claim."""
import copy
import importlib.util
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "qwen27b_drafts", ROOT / "tools/qwen27b/drafts.py")
drafts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drafts)

gguf_spec = importlib.util.spec_from_file_location(
    "gufo_gguf", ROOT / "tools/quant/gufo-gguf.py")
gguf = importlib.util.module_from_spec(gguf_spec)
gguf_spec.loader.exec_module(gguf)


class QualificationTest(unittest.TestCase):
    def test_bf16_reference_reader_is_little_endian(self):
        payload = struct.pack("<5H", 0x3F80, 0xC000, 0x3F00, 0x3B80, 0x0000)
        self.assertEqual(gguf.DEQUANT[30](payload).tolist(),
                         [1.0, -2.0, 0.5, 0.00390625, 0.0])

    def setUp(self):
        self.report = {
            "prompt_mode": "chat",
            "aggregate": {"prompts": 2, "completed": 2, "exact": 2, "skipped": []},
            "cases": [{"id": "one"}, {"id": "two"}],
        }

    def test_complete(self):
        self.assertEqual(drafts.qualified(self.report)["completed"], 2)

    def test_partial_mismatch_empty_or_skipped_fails(self):
        for field, value in (("completed", 1), ("exact", 1), ("prompts", 0),
                             ("skipped", ["timeout"])):
            report = copy.deepcopy(self.report)
            report["aggregate"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                drafts.qualified(report)

    def test_missing_or_duplicate_case_fails(self):
        for cases in ([{"id": "one"}], [{"id": "one"}, {"id": "one"}]):
            self.report["cases"] = cases
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                drafts.qualified(self.report)

    def test_raw_fails(self):
        self.report["prompt_mode"] = "raw"
        with self.assertRaises(ValueError):
            drafts.qualified(self.report)

    def test_release_comparison_uses_token_count_and_digest(self):
        report = {"cases": [{"id": "one", "reference": {
            "tokens": 128, "token_sha256": "a" * 64}}]}
        original = drafts.reference_tokens(report)
        self.assertEqual(original, {"one": (128, "a" * 64)})
        for field, value in (("tokens", 127), ("token_sha256", "b" * 64)):
            changed = copy.deepcopy(report)
            changed["cases"][0]["reference"][field] = value
            with self.subTest(field=field):
                self.assertNotEqual(original, drafts.reference_tokens(changed))
        report["cases"][0]["reference"]["token_sha256"] = ""
        with self.assertRaises(ValueError):
            drafts.reference_tokens(report)


if __name__ == "__main__":
    unittest.main()
