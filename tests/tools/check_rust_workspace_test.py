#!/usr/bin/env python3
"""Registration errors must fail before Cargo silently omits a component."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rust_workspace', ROOT / 'tools/ci/check-rust-workspace.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        paths = ['Cargo.toml', 'src/core/rust/src/lib.rs', 'src/core/rust/CMakeLists.txt']
        paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/core').glob('*/Cargo.toml')]
        for name in paths:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        for path in (ROOT / 'src/core').glob('*/include'):
            (self.root / path.relative_to(ROOT)).mkdir(parents=True)

    def test_complete_registration(self):
        self.assertEqual(checker.check(self.root), [])

    def test_unlisted_manifest(self):
        target = self.root / 'src/core/unlisted/Cargo.toml'
        target.parent.mkdir()
        target.write_text('[package]\nname = "unlisted"\n')
        self.assertTrue(any('workspace members differ' in error for error in checker.check(self.root)))

    def test_unlinked_crate(self):
        path = self.root / 'src/core/rust/Cargo.toml'
        path.write_text(path.read_text().replace('gufo-base64 = { path = "../base64" }', ''))
        self.assertTrue(any('not linked' in error for error in checker.check(self.root)))

    def test_missing_extern(self):
        path = self.root / 'src/core/rust/src/lib.rs'
        path.write_text(path.read_text().replace('extern crate gufo_http;', ''))
        self.assertTrue(any('extern crates differ' in error for error in checker.check(self.root)))

    def test_missing_header_directory(self):
        path = self.root / 'src/core/rust/CMakeLists.txt'
        path.write_text(path.read_text().replace('${PROJECT_SOURCE_DIR}/src/core/http/include;', ''))
        self.assertTrue(any('include directories differ' in error for error in checker.check(self.root)))


if __name__ == '__main__':
    unittest.main()
