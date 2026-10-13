"""Selection invariants; these tests never load weights or start Gufo."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cluster
import impact


HERE = Path(__file__).resolve().parent


class ImpactTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((HERE / "impact.json").read_text())

    def test_registry_must_be_complete(self):
        names = impact.suite_names(HERE / "run.py")
        impact.validate(self.manifest, names)
        with self.assertRaises(ValueError):
            impact.validate(self.manifest, [*names, "new-suite"])

    def test_private_continuation_overrides_runtime_scope(self):
        result = impact.plan(self.manifest, ["src/models/qwen38_flash_next/continuation_adapter.cpp"])
        self.assertEqual(result["paths"][0]["rules"], ["flash-next-continuation"])
        self.assertEqual({row["profile"] for row in result["selected"]},
                         {"flash-next-q4/off", "flash-next-q4/mtp"})
        names = {row["suite"] for row in result["selected"]}
        self.assertLessEqual({"cache", "long-context", "cache-edits", "discovery", "state-edges"}, names)

    def test_shared_qwen_is_not_private_qwen27(self):
        result = impact.plan(self.manifest, ["src/models/qwen/tokenizer.cpp"])
        profiles = {row["profile"] for row in result["selected"]}
        self.assertIn("flash-next-q4/mtp", profiles)
        self.assertIn("qwen27-q8/dflash2", profiles)
        self.assertNotIn("deepseek4/off", profiles)
        self.assertTrue(result["obligations"])

    def test_multiple_changes_preserve_independent_reasons(self):
        result = impact.plan(self.manifest, ["src/models/qwen38_flash_next/engine.cpp",
                                             "src/models/deepseek_v4_flash/model.cpp"])
        for row in result["selected"]:
            if row["profile"].startswith("flash-next"):
                self.assertTrue(all("qwen38_flash_next" in reason["path"] for reason in row["reasons"]))

    def test_unknown_paths_expand_and_deferred_are_visible(self):
        result = impact.plan(self.manifest, ["src/new_component.cpp"], ["flash-next-q4"])
        self.assertEqual(result["paths"][0]["rules"], ["fallback"])
        self.assertTrue(result["selected"])
        self.assertTrue(result["out_of_scope"])
        self.assertIn("cache-pressure", {row["suite"] for row in result["deferred"]})
        self.assertNotIn("cache-pressure", {row["suite"] for row in result["selected"]})
        self.assertEqual(result["timing_qualification"], "not performed; candidate correctness only")

    def test_docs_select_nothing_but_graph_changes_expand(self):
        self.assertFalse(impact.plan(self.manifest, ["docs/SERVER.md"])["groups"])
        self.assertTrue(impact.plan(self.manifest, ["tests/functional/impact.json"])["groups"])

    def test_deleted_and_renamed_paths_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args], stderr=subprocess.DEVNULL)
            git("init")
            git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Test")
            (root / "old.cpp").write_text("old\n")
            (root / "deleted.cpp").write_text("deleted\n")
            git("add", ".")
            git("commit", "-m", "initial")
            base = git("rev-parse", "HEAD").decode().strip()
            (root / "old.cpp").rename(root / "new.cpp")
            (root / "deleted.cpp").unlink()
            git("add", "-A")
            git("commit", "-m", "rename and delete")
            self.assertEqual(set(impact.changed_paths(root, base, "HEAD")),
                             {"old.cpp", "new.cpp", "deleted.cpp"})

    def test_bad_tags_and_profile_filters_are_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["rules"][0]["affects"] = ["typo"]
        with self.assertRaises(ValueError):
            impact.validate(bad, impact.suite_names(HERE / "run.py"))
        with self.assertRaises(ValueError):
            impact.plan(self.manifest, [], ["typo"])

    def test_executor_retains_failed_groups_and_runs_both_modes(self):
        plan = impact.plan(self.manifest, ["src/models/qwen38_flash_next/engine.cpp"], ["flash-next-q4"])
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True):
            output = Path(directory) / "results"
            codes = [subprocess.CompletedProcess([], 1)] + [
                subprocess.CompletedProcess([], 0) for _ in plan["groups"][1:]]
            with patch("cluster.subprocess.run", side_effect=codes) as run:
                self.assertEqual(cluster.execute(plan, Path("/binary"), Path("/weights"), output), 1)
            self.assertEqual(run.call_count, len(plan["groups"]))
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual({row["profile"] for row in report["groups"]},
                             {"flash-next-q4/off", "flash-next-q4/mtp"})
            for row in report["groups"]:
                args = row["command"]
                self.assertIn("--mtp-model", args)
                self.assertIn("--mmproj", args)
                self.assertIn("--record-baseline", args)


if __name__ == "__main__":
    unittest.main()
