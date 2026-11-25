#!/usr/bin/env python3
"""CPU-only regressions for the pull request compiler cache save decision."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "cache_growth", ROOT / "tools/ci/cache_growth.py")
GROWTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GROWTH)


class CacheGrowthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "cache"
        self.state = Path(self.tmp.name) / "state" / "snapshot.json"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, size):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def snapshot(self):
        self.assertEqual(GROWTH.main(["snapshot", str(self.root), str(self.state)]), 0)

    def test_touched_hits_do_not_count(self):
        hit = self.write("a/hit.o", 4096)
        self.snapshot()
        os.utime(hit, (1, 1))
        save, reason = GROWTH.decide(self.root, self.state, 1024)
        self.assertFalse(save, reason)

    def test_new_and_resized_files_count(self):
        self.write("a/manifest", 100)
        self.snapshot()
        self.write("a/manifest", 700)
        self.write("b/new.o", 600)
        self.assertEqual(GROWTH.growth(
            {"a/manifest": 100}, GROWTH.listing(self.root)), (2, 1300))
        self.assertTrue(GROWTH.decide(self.root, self.state, 1300)[0])
        self.assertFalse(GROWTH.decide(self.root, self.state, 1301)[0])

    def test_cold_or_missing_snapshot_saves(self):
        self.assertTrue(GROWTH.decide(self.root, self.state, 1)[0])
        self.snapshot()  # Empty: the restore found nothing.
        self.write("new.o", 1)
        self.assertTrue(GROWTH.decide(self.root, self.state, 2**30)[0])

    def test_missing_cache_directory_is_empty(self):
        self.assertEqual(GROWTH.listing(self.root / "absent"), {})


if __name__ == "__main__":
    sys.exit(unittest.main())
