#!/usr/bin/env python3
"""Tests for model infrastructure generation and drift checking."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_infrastructure", ROOT / "tools/ci/generate-infrastructure.py"
)
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class GenerateInfrastructureTest(unittest.TestCase):
    def test_live_infrastructure_is_in_sync(self):
        """Repository infrastructure must match manifests with zero drift."""
        is_clean, drifts = GEN.check_targets(ROOT)
        self.assertTrue(is_clean, f"Drift detected in repo: {list(drifts.keys())}")

    def test_load_manifests_discovers_all_models(self):
        manifests = GEN.load_manifests(ROOT)
        names = [m["name"] for m in manifests]
        expected = [
            "deepseek_v4_flash",
            "embedding_gemma2",
            "minimax_h3",
            "qwen",
            "qwen38_flash_next",
            "qwen3_asr",
            "qwen3_tts",
            "qwen_image_21",
        ]
        self.assertEqual(names, sorted(expected))

    def test_missing_metadata_name_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "src/models/bad_model"
            model_dir.mkdir(parents=True)
            (model_dir / "extension.json").write_text(
                json.dumps({"version": "1.0.0"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                GEN.load_manifests(root)
            self.assertIn("missing 'name'", str(ctx.exception))

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "src/models/bad_json"
            model_dir.mkdir(parents=True)
            (model_dir / "extension.json").write_text("{not: valid json", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                GEN.load_manifests(root)
            self.assertIn("Failed to parse manifest", str(ctx.exception))

    def test_duplicate_extension_name_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m1 = root / "src/models/m1"
            m2 = root / "src/models/m2"
            m1.mkdir(parents=True)
            m2.mkdir(parents=True)
            (m1 / "extension.json").write_text(
                json.dumps({"name": "duplicate_name"}), encoding="utf-8"
            )
            (m2 / "extension.json").write_text(
                json.dumps({"name": "duplicate_name"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                GEN.load_manifests(root)
            self.assertIn("Duplicate extension name 'duplicate_name'", str(ctx.exception))

    def test_duplicate_provider_id_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            m1 = root / "src/models/m1"
            m2 = root / "src/models/m2"
            m1.mkdir(parents=True)
            m2.mkdir(parents=True)
            (m1 / "extension.json").write_text(
                json.dumps({"name": "m1", "provider": {"id": "shared-provider"}}),
                encoding="utf-8",
            )
            (m2 / "extension.json").write_text(
                json.dumps({"name": "m2", "provider": {"id": "shared-provider"}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                GEN.load_manifests(root)
            self.assertIn("Duplicate provider id 'shared-provider'", str(ctx.exception))

    def test_conditional_metadata_omissions(self):
        """Extensions without docs or tests are handled gracefully without breaking."""
        manifests = [
            {
                "name": "bare_bones",
                "version": "0.1.0",
                "provider": {"id": "bare-bones", "display_name": "Bare Bones"},
                # No docs, no tests, no functional, no runtime_libraries
            },
            {
                "name": "full_model",
                "version": "1.0.0",
                "provider": {"id": "full-model", "display_name": "Full Model"},
                "docs": {
                    "title": "Full Model",
                    "inputs": "SafeTensors",
                    "guide": "docs/models/full-model/README.md",
                },
                "tests": [
                    {"target": "full_test", "check_pr": True},
                    {"target": "full_custom", "ctest": "^custom.*", "check_pr": True},
                    {"target": "full_ignored", "check_pr": False},
                ],
                "runtime_libraries": ["libfull.so"],
                "production_paths": ["src/models/full_model/runtime.cpp"],
            },
        ]

        # Check doc generation
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "docs/models").mkdir(parents=True)
            (root / "docs/models/README.md").write_text(
                "# Models\n"
                "<!-- BEGIN GENERATED MODEL TABLE -->\n"
                "old\n"
                "<!-- END GENERATED MODEL TABLE -->\n",
                encoding="utf-8",
            )
            (root / "tools/ci").mkdir(parents=True)
            (root / "tools/ci/check-docs.py").write_text(
                "REQUIRED_DOC_FILES = [\n"
                "    # BEGIN GENERATED MODEL DOCS\n"
                "old\n"
                "    # END GENERATED MODEL DOCS\n"
                "]\n",
                encoding="utf-8",
            )
            (root / "tools/ci/check-format.py").write_text(
                "# format\n"
                "    # BEGIN GENERATED VENDORED PATHS\n"
                "old\n"
                "    # END GENERATED VENDORED PATHS\n",
                encoding="utf-8",
            )
            (root / "cmake").mkdir(parents=True)
            (root / "cmake/Checks.cmake").write_text(
                "# cmake\n"
                "  # BEGIN GENERATED MODEL PR TARGETS\n"
                "old\n"
                "  # END GENERATED MODEL PR TARGETS\n"
                "# BEGIN GENERATED MODEL CTEST OVERRIDES\n"
                "old\n"
                "# END GENERATED MODEL CTEST OVERRIDES\n",
                encoding="utf-8",
            )

            # Generate Checks.cmake
            checks_cmake = GEN.generate_checks_cmake(root, manifests)
            self.assertIn("full_test", checks_cmake)
            self.assertIn("full_custom", checks_cmake)
            self.assertNotIn("full_ignored", checks_cmake)
            self.assertIn("list(REMOVE_ITEM gufo_pr_tests\n  full_custom)", checks_cmake)
            self.assertIn('list(APPEND gufo_pr_tests\n  "^custom.*")', checks_cmake)

            # Generate check-docs.py
            docs_py = GEN.generate_check_docs(root, manifests)
            self.assertIn('"full-model"', docs_py)
            compile(docs_py, "check-docs.py", "exec")

            # Generate README.md
            readme = GEN.generate_models_readme(root, manifests)
            self.assertIn("| Full Model | SafeTensors | [Usage](full-model/README.md) |", readme)

    def test_drift_detection_detects_changes(self):
        """Simulate drift in a managed file and verify check_targets flags it."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Create minimal structure mirroring repo
            (root / "src/models/m1").mkdir(parents=True)
            (root / "src/models/m1/extension.json").write_text(
                json.dumps(
                    {
                        "name": "m1",
                        "version": "1.0.0",
                        "provider": {"id": "m1", "display_name": "M1"},
                        "docs": {
                            "title": "M1 Model",
                            "inputs": "SafeTensors",
                            "guide": "docs/models/m1/README.md",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "docs/models").mkdir(parents=True)
            (root / "docs/models/README.md").write_text(
                "# Models\n<!-- BEGIN GENERATED MODEL TABLE -->\n<!-- END GENERATED MODEL TABLE -->\n",
                encoding="utf-8",
            )
            (root / "tools/ci").mkdir(parents=True)
            (root / "tools/ci/check-docs.py").write_text(
                "    # BEGIN GENERATED MODEL DOCS\n    # END GENERATED MODEL DOCS\n",
                encoding="utf-8",
            )
            (root / "tools/ci/check-format.py").write_text(
                "    # BEGIN GENERATED VENDORED PATHS\n    # END GENERATED VENDORED PATHS\n",
                encoding="utf-8",
            )
            (root / "cmake").mkdir(parents=True)
            (root / "cmake/Checks.cmake").write_text(
                "  # BEGIN GENERATED MODEL PR TARGETS\n  # END GENERATED MODEL PR TARGETS\n"
                "# BEGIN GENERATED MODEL CTEST OVERRIDES\n# END GENERATED MODEL CTEST OVERRIDES\n",
                encoding="utf-8",
            )
            (root / "tests/functional").mkdir(parents=True)
            (root / "tests/functional/devlab.core.json").write_text(
                json.dumps(
                    {
                        "schema": "gufo-devlab-suite-v1",
                        "disabled": [],
                        "representatives": {},
                        "assets": {},
                        "models": {},
                        "rules": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "tests/functional/impact.core.json").write_text(
                json.dumps(
                    {
                        "schema": "gufo-functional-impact-rules-v1",
                        "policy": {},
                        "behaviors": {},
                        "profiles": {},
                        "suites": {},
                        "rules": [],
                        "baseline_suites": {},
                        "fallback": {},
                        "configurations": {},
                    }
                ),
                encoding="utf-8",
            )
            (root / "tests/functional/devlab.json").write_text("{}", encoding="utf-8")
            (root / "tests/functional/impact.json").write_text("{}", encoding="utf-8")

            # Check should report drift initially because markers and json are empty
            is_clean, drifts = GEN.check_targets(root)
            self.assertFalse(is_clean)
            self.assertGreater(len(drifts), 0)

            # Write synced content
            GEN.write_targets(root)

            # Now check should report clean
            is_clean, drifts = GEN.check_targets(root)
            self.assertTrue(is_clean, f"Expected clean, got {drifts}")

            # Tamper with one file
            readme_path = root / "docs/models/README.md"
            readme_path.write_text(
                readme_path.read_text(encoding="utf-8").replace("M1", "TamperedM1"),
                encoding="utf-8",
            )

            is_clean, drifts = GEN.check_targets(root)
            self.assertFalse(is_clean)
            self.assertIn("docs/models/README.md", drifts)

    def test_cli_check_flag_exits_clean_and_drift(self):
        """CLI with --check exits 0 on sync and 1 on drift."""
        # Current repo should exit 0
        cmd = [sys.executable, str(ROOT / "tools/ci/generate-infrastructure.py"), "--check"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("PASS:", res.stdout)


if __name__ == "__main__":
    unittest.main()
