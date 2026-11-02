#!/usr/bin/env python3
"""Distribution checks must reject incomplete installs and dependency recipes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install = load("check_install", "tools/extensions/check_install.py")
packages = load("extension_packages", "tools/windows/check_extension_packages.py")


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "gufo"
        self.binary.write_bytes(b"binary")
        self.license = self.root / "LICENSE"
        self.license.write_text("License notice")
        self.provenance = {"extensionApiVersion": "0.1.0", "selectionMode": "lock",
                           "builtExtensions": [{"name": "example", "version": "1.0.0", "source": "external"}]}

    def audit(self, mode="lock", licenses=None, data=(), runtime=(), extensions=("example",)):
        def run(command, **kwargs):
            return SimpleNamespace(returncode=0, stderr="", stdout=(json.dumps(self.provenance)
                                   if "diagnose" in command else "gufo version test"))
        with patch.object(install.subprocess, "run", side_effect=run):
            return install.audit(self.binary, mode, extensions,
                                 [self.license] if licenses is None else licenses, data, runtime)

    def test_install_inventory_and_hashes(self):
        report = self.audit()
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(len(item["sha256"]) == 64 for item in report["artifacts"]))

    def test_incomplete_install_is_not_a_pass(self):
        for kwargs in ({"licenses": []}, {"licenses": [self.root / "absent"]},
                       {"data": [self.root / "missing-data"]}, {"mode": "discovery"},
                       {"extensions": ["absent"]}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.audit(**kwargs)["status"], "FAIL")

    def test_failed_binary_is_not_a_pass(self):
        with patch.object(install.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="missing runtime")):
            report = install.audit(self.binary, "lock", [], [self.license], [], [])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["version"]["stderr"], "missing runtime")

    def test_lazy_runtime_must_be_shipped(self):
        manifest = self.root / "runtime.txt"
        manifest.write_text("decoder.dll\n")
        self.assertEqual(self.audit(runtime=[manifest])["status"], "FAIL")
        (self.root / "decoder.dll").write_bytes(b"runtime")
        self.assertEqual(self.audit(runtime=[manifest])["status"], "PASS")
        manifest.write_text("../outside.dll\n")
        self.assertEqual(self.audit(runtime=[manifest])["status"], "FAIL")

    def test_incomplete_or_duplicate_provenance_is_rejected(self):
        self.provenance["builtExtensions"].append(self.provenance["builtExtensions"][0])
        self.assertEqual(self.audit()["status"], "FAIL")
        self.provenance["builtExtensions"] = [{"name": "example"}]
        self.assertEqual(self.audit()["status"], "FAIL")

    def test_windows_ports_and_features(self):
        (self.root / "extensions.lock").write_text('{"extensions":["example"]}')
        directory = self.root / "src/models/example"
        directory.mkdir(parents=True)
        (directory / "extension.json").write_text(json.dumps({"name": "example", "platforms": ["windows"],
            "packages": [{"windows": "ffmpeg[avcodec,avformat]"}, {"windows": "soxr"}]}))
        manifest = {"dependencies": ["soxr", {"name": "ffmpeg", "features": ["avcodec", "avformat"]}]}
        self.assertFalse(packages.check(self.root, manifest)["errors"])
        manifest["dependencies"][1]["features"].remove("avcodec")
        self.assertIn("avcodec", " ".join(packages.check(self.root, manifest)["errors"]))
        manifest["dependencies"] = []
        self.assertIn("missing port", " ".join(packages.check(self.root, manifest)["errors"]))
        (self.root / "extensions.lock").write_text('{"extensions":[]}')
        self.assertFalse(packages.check(self.root, manifest, from_lock=True)["errors"])

    def test_external_windows_dependencies_are_checked(self):
        external = self.root / "external"
        external.mkdir()
        (external / "extension.json").write_text(json.dumps({"name": "external", "platforms": ["windows"],
            "packages": [{"windows": "new-port"}]}))
        report = packages.check(self.root, {"dependencies": []}, [external])
        self.assertEqual(report["checked_extensions"], ["external"])
        self.assertIn("new-port", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
