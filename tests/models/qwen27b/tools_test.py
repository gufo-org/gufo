#!/usr/bin/env python3
"""The production sweep must never turn partial results into a speed claim."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "qwen27b_drafts", ROOT / "tools/qwen27b/drafts.py")
drafts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drafts)


class QualificationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
