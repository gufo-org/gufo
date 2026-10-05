"""Keep the real-agent oracle independent of what the model claims it did."""

import json
from pathlib import Path
import tempfile
import unittest

import agent_catalog as catalog


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        _, self.expected = catalog.prepare(self.root, 1)
        self.change = {
            "id": 42, "expected_revision": 0,
            "changes": [{"op": "set", "field": "body", "value": self.expected["body"]}],
            "dry_run": False, "metadata": {"comment": None, "attributes": {"enabled": True}},
        }

    def state(self):
        return json.loads((self.root / "catalog-state.json").read_text())

    def test_dry_run_does_not_mutate(self):
        before = self.state()
        result = catalog.execute(self.root, "catalog_apply", {**self.change, "dry_run": True})
        self.assertEqual(self.state(), before)
        self.assertFalse(result["committed"])
        self.assertEqual(result["record"]["body"], self.expected["body"])

    def test_rejected_later_operation_is_atomic(self):
        before = self.state()
        invalid = {**self.change, "changes": self.change["changes"] + [
            {"op": "set", "field": "unknown", "value": "bad"}]}
        with self.assertRaises(Exception):
            catalog.execute(self.root, "catalog_apply", invalid)
        self.assertEqual(self.state(), before)

    def test_stale_replay_is_rejected(self):
        catalog.execute(self.root, "catalog_apply", self.change)
        before = self.state()
        with self.assertRaisesRegex(AssertionError, "stale revision"):
            catalog.execute(self.root, "catalog_apply", self.change)
        self.assertEqual(self.state(), before)

    def test_nested_literal_roundtrip_and_receipt(self):
        catalog.execute(self.root, "catalog_apply", self.change)
        checked = catalog.execute(self.root, "catalog_verify", {
            "id": 42, "mode": "exact",
            "checks": [{"field": "body", "expected": self.expected["body"]}]})
        args = {"id": 42, "digest": checked["digest"], "status": "42", "confirmed": "true",
                "details": {"revision": checked["revision"], "note": None}}
        catalog.execute(self.root, "catalog_receipt", args)
        self.assertEqual(json.loads((self.root / "catalog-receipt-1.json").read_text()), args)
        for key, bad in (("status", 42), ("confirmed", True), ("digest", "a" * 64)):
            with self.assertRaises(Exception):
                catalog.execute(self.root, "catalog_receipt", {**args, key: bad})

    def test_incorrect_verification_fails(self):
        with self.assertRaisesRegex(AssertionError, "field mismatch"):
            catalog.execute(self.root, "catalog_verify", {
                "id": 42, "mode": "exact", "checks": [{"field": "body", "expected": "wrong"}]})


if __name__ == "__main__":
    unittest.main()
