#!/usr/bin/env python3
"""Validate pinned DS4 evaluation data, provenance, and comparison metrics."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/ds4"))
from reference import compare_logits


class ReferenceMetricsTest(unittest.TestCase):
    @staticmethod
    def compare(a, b):
        return compare_logits(struct.pack(f"<{len(a)}f", *a),
                              struct.pack(f"<{len(b)}f", *b), len(a))

    def test_common_shift_preserves_probabilities(self):
        result = self.compare([1, 2, 3], [17, 18, 19])
        self.assertFalse(result["within_existing_vector_bounds"])
        self.assertEqual(result["mean_logit_shift"], -16)
        self.assertEqual(result["centered_rmse"], 0)
        self.assertEqual(result["jensen_shannon_nats"], 0)
        self.assertEqual(result["max_probability_difference"], 0)
        self.assertTrue(result["same_top1"])

    def test_distribution_difference_is_symmetric_and_bounded(self):
        a, b = [1000, -1000, 0], [-1000, 1000, 0]
        forward, backward = self.compare(a, b), self.compare(b, a)
        self.assertAlmostEqual(forward["jensen_shannon_nats"], math.log(2))
        self.assertEqual(forward["jensen_shannon_nats"],
                         backward["jensen_shannon_nats"])
        self.assertEqual(forward["max_probability_difference"], 1)
        self.assertFalse(forward["same_top1"])
        self.assertEqual(self.compare(a, a)["jensen_shannon_nats"], 0)
        # One probability is a subnormal, the other underflows to zero.
        self.assertTrue(math.isfinite(
            self.compare([0, -745], [0, -1000])["jensen_shannon_nats"]))

    def test_nonfinite_and_incomplete_logits_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(RuntimeError):
                self.compare([value, 1], [1, 2])
        with self.assertRaises(RuntimeError):
            self.compare([1, 2], [1])


class EvalDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).parent / "fixtures" / "antirez-ds4.json"
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

    def test_official_continuation_fixture(self) -> None:
        data = self.path.with_name("official-0731.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "0ec57ef24cc0cb5838a4c0ccacf806edc12cbe09f27f0c13efdf90eb33591a4d",
        )
        document = json.loads(data)
        self.assertEqual(document["checkpoint"], "0731")
        self.assertEqual(
            document["source_revision"],
            "6289c516273979173abbc062209a81dd3706b804",
        )
        self.assertIn("MIT License", document["license"])
        self.assertEqual(len(document["source_files_sha256"]), 313)
        cases = document["cases"]
        self.assertEqual(len({case["id"] for case in cases}), 105)
        self.assertEqual(
            Counter(case["group"] for case in cases),
            Counter({"continuation-100": 100, "smoke-5": 5}),
        )
        tokens = Counter()
        for case in cases:
            self.assertRegex(case["id"], r"^[a-zA-Z0-9_-]+$")
            self.assertTrue(case["prompt"])
            pieces = [bytes.fromhex(value)
                      for value in case["token_bytes_hex"]]
            self.assertTrue(all(pieces))
            self.assertEqual(b"".join(pieces),
                             case["continuation"].encode("utf-8"))
            tokens[case["group"]] += len(pieces)
        self.assertEqual(tokens,
                         Counter({"continuation-100": 2313, "smoke-5": 14}))


if __name__ == "__main__":
    unittest.main()
