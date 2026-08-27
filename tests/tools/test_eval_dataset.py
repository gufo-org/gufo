#!/usr/bin/env python3
"""Validate the pinned DS4 evaluation data and its row-level audit records."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest


class EvalDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            Path(__file__).resolve().parent.parent
            / "quality"
            / "antirez-ds4.json"
        )
        cls.document = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_source_identity_and_counts(self) -> None:
        self.assertEqual(self.document["schema"], "gufo.eval-cases.v2")
        self.assertEqual(
            self.document["source_revision"],
            "84cc882352757baf628a1776badf7cc54d584e28",
        )
        self.assertEqual(
            self.document["source_blob"],
            "7aed5d5c5b5cdc74d1b3aa0310161e2b437c1d08",
        )
        self.assertEqual(len(self.document["cases"]), 75)
        self.assertEqual(len(self.document["excluded"]["cases"]), 17)
        self.assertEqual(
            Counter(case["dataset"] for case in self.document["cases"]),
            Counter({"GPQA Diamond": 25, "SuperGPQA": 25, "AIME 2025": 25}),
        )

    def test_ds4_order_and_first_four_keys(self) -> None:
        cases = self.document["cases"]
        self.assertEqual(
            [(case["id"], case["answer"]) for case in cases[:4]],
            [
                ("recNu3MXkvWUzHZr9", "B"),
                ("001b51d76b4d422988f2c11f104a2c6c", "C"),
                ("aime2025-01", "70"),
                ("recoiTJPGUmzAkief", "C"),
            ],
        )
        self.assertEqual(
            [case["ds4_index"] for case in cases], list(range(75))
        )
        for index, case in enumerate(cases):
            expected_source = ("GPQA", "SuperGPQA", "AIME2025")[index % 3]
            if expected_source == "GPQA":
                self.assertTrue(case["source"].startswith("GPQA Diamond"))
            else:
                self.assertEqual(case["source"], expected_source)

    def test_row_audit_hashes(self) -> None:
        for case in self.document["cases"]:
            audited = {
                key: case[key]
                for key in (
                    "ds4_index",
                    "source",
                    "id",
                    "domain",
                    "title",
                    "kind",
                    "question",
                    "choices",
                    "answer",
                )
            }
            serialized = json.dumps(
                audited,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.assertEqual(
                hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                case["source_record_sha256"],
            )
            self.assertTrue(case["license"])
            self.assertTrue(case["dataset_url"])
            self.assertTrue(case["audit_note"])

    def test_compsec_metadata_only_policy(self) -> None:
        excluded = self.document["excluded"]
        self.assertEqual(excluded["source"], "COMPSEC")
        self.assertEqual(excluded["commit_policy"], "metadata-only")
        self.assertEqual(
            [case["ds4_index"] for case in excluded["cases"]],
            list(range(75, 92)),
        )
        self.assertEqual(
            [case["answer"] for case in excluded["cases"]],
            [
                "17-20",
                "18-20",
                "11",
                "18-19",
                "5-6",
                "10-15",
                "9-10",
                "9-11",
                "6-7",
                "5",
                "3,13-15",
                "8,20-22",
                "11",
                "10",
                "12-13",
                "3",
                "10-14",
            ],
        )
        for case in excluded["cases"]:
            self.assertNotIn("question", case)
            self.assertNotIn("title", case)


if __name__ == "__main__":
    unittest.main()
