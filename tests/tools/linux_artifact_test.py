#!/usr/bin/env python3
"""Validate portable artifact provenance before executing model checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'linux_artifact', Path(__file__).resolve().parents[2] / 'tools/ci/linux_artifact.py')
artifact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifact)


class ArtifactTest(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / 'candidate'
        (root / 'libexec').mkdir(parents=True)
        (root / 'libexec/gufo').write_bytes(b'ELF fixture')
        manifest = {'schema': 2, 'revision': 'a' * 40, 'version': 'gufo version development (aaaaaaa)',
                    'sha256': hashlib.sha256(b'ELF fixture').hexdigest(),
                    'loader': 'runtime/glibc/lib/ld-linux-x86-64.so.2', 'libraries': ['runtime/glibc/lib']}
        return root, manifest

    def test_wrong_revision_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(artifact, 'output') as execute:
                with self.assertRaisesRegex(ValueError, 'authorized revision'):
                    artifact.verify(root, 'b' * 40)
                execute.assert_not_called()

    def test_tampered_binary_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'libexec/gufo').write_bytes(b'tampered')
            with patch.object(artifact, 'output') as execute:
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    artifact.verify(root, 'a' * 40)
                execute.assert_not_called()

    def test_unbundled_runtime_is_rejected_even_on_a_nix_host(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(artifact, 'output', side_effect=[manifest['version'],
                    'libc.so.6 => /nix/store/host-glibc/lib/libc.so.6']):
                with self.assertRaisesRegex(ValueError, 'unbundled runtime'):
                    artifact.verify(root, 'a' * 40)

    def test_missing_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(artifact, 'output', side_effect=[manifest['version'],
                    'libhsa-runtime64.so.1 => not found']):
                with self.assertRaisesRegex(ValueError, 'unbundled runtime'):
                    artifact.verify(root, 'a' * 40)

    def test_multiline_extension_provenance_retains_revision_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            manifest['version'] += '\nExtension API: 0.1.1\nExtension selection: discovery\nBuilt extensions: none'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(artifact, 'output', side_effect=[manifest['version'], 'libc.so.6 => /runtime/libc.so.6']):
                artifact.verify(root, 'a' * 40)
            # A later provenance line must not hide an incorrect first-line revision.
            wrong = manifest['version'].replace('(aaaaaaa)', '(bbbbbbb)')
            with patch.object(artifact, 'output', return_value=wrong):
                with self.assertRaisesRegex(ValueError, 'wrong revision'):
                    artifact.verify(root, 'a' * 40)

    def test_share_copy_is_removable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'store/share'
            (source / 'licenses/gufo').mkdir(parents=True)
            (source / 'licenses/gufo/LICENSE').write_text('license')
            for path in [source / 'licenses/gufo/LICENSE', source / 'licenses/gufo',
                         source / 'licenses', source]:
                path.chmod(0o444 if path.is_file() else 0o555)
            try:
                destination = Path(directory) / 'candidate/share'
                artifact.copy_share(source, destination)
                self.assertEqual((destination / 'licenses/gufo/LICENSE').read_text(), 'license')
                shutil.rmtree(destination)
                self.assertFalse(destination.exists())
            finally:
                for path in [source, *source.rglob('*')]:
                    if path.is_dir():
                        path.chmod(0o755)


if __name__ == '__main__':
    unittest.main()
