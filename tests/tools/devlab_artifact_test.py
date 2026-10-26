#!/usr/bin/env python3
"""Exercise artifact integrity, adaptive routing and persistent runtime reuse."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
import zipfile

SPEC = importlib.util.spec_from_file_location(
    'devlab_artifact', Path(__file__).resolve().parents[2] / 'tools/ci/devlab_artifact.py')
cdn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cdn)


class DownloadTest(unittest.TestCase):
    def test_missing_digest_rejected(self):
        for value in (None, '', 'sha256:abc', 'md5:' + 'a' * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cdn.checked_digest(value)

    def test_tampered_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'bundle'
            def transfer(url, path):
                path.write_bytes(b'tampered')
            artifact = {'id': 1, 'digest': 'sha256:' + hashlib.sha256(b'original').hexdigest()}
            with patch.object(cdn, 'transfer', side_effect=transfer):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    cdn.fetch(Mock(), artifact, destination)

    def test_partial_probe_still_measures_rate(self):
        result = Mock(stdout='206 1048576 12.0', returncode=28)
        with patch.object(cdn.subprocess, 'run', return_value=result):
            self.assertAlmostEqual(cdn.probe_route('https://example.org/bundle'), 1048576 / 12)

    def test_fast_direct_route_does_not_start_warp(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bundle'
            def fetch(command, **kwargs):
                output.write_bytes(b'fixture')
                self.assertNotIn('--proxy', command)
            with patch.object(cdn, 'probe_route', return_value=10 * 1024 * 1024), \
                 patch.object(cdn, 'warp_proxy') as warp, \
                 patch.object(cdn.subprocess, 'run', side_effect=fetch):
                cdn.transfer(lambda: 'https://example.org/bundle', output)
            warp.assert_not_called()

    def test_slow_direct_route_uses_faster_warp_and_stops_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bundle'
            proxy = MagicMock()
            proxy.__enter__.return_value = 'http://127.0.0.1:18080'
            def fetch(command, **kwargs):
                output.write_bytes(b'fixture')
                self.assertIn('--proxy', command)
                self.assertIn('http://127.0.0.1:18080', command)
            with patch.object(cdn, 'probe_route', side_effect=[50 * 1024, 5 * 1024 * 1024]), \
                 patch.object(cdn, 'warp_proxy', return_value=proxy), \
                 patch.object(cdn.subprocess, 'run', side_effect=fetch):
                cdn.transfer(lambda: 'https://example.org/bundle', output)
            proxy.__exit__.assert_called_once()

    def test_warp_unavailable_falls_back_to_direct(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bundle'
            def fetch(command, **kwargs):
                output.write_bytes(b'fixture')
                self.assertNotIn('--proxy', command)
            with patch.object(cdn, 'probe_route', return_value=1024 * 1024), \
                 patch.object(cdn, 'warp_proxy', side_effect=RuntimeError('UDP blocked')), \
                 patch.object(cdn.subprocess, 'run', side_effect=fetch):
                cdn.transfer(lambda: 'https://example.org/bundle', output)

    def test_extract_only_expected_member(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'bundle.zip'
            output = Path(directory) / 'bundle.tar.gz'
            for name in ('../../escape', 'linux-artifacts.tar.gz'):
                with zipfile.ZipFile(archive, 'w') as source:
                    source.writestr(name, b'payload')
                if name.startswith('..'):
                    with self.assertRaisesRegex(ValueError, 'Unexpected'):
                        cdn.extract_member(archive, 'linux-artifacts.tar.gz', output)
                    self.assertFalse(output.exists())
                else:
                    cdn.extract_member(archive, 'linux-artifacts.tar.gz', output)
                    self.assertEqual(output.read_bytes(), b'payload')


    def test_superseded_run_stops_waiting(self):
        github = Mock()
        github.artifacts.return_value = []
        github.api.return_value = {'run_attempt': 3, 'status': 'in_progress'}
        with self.assertRaisesRegex(RuntimeError, 'superseded'):
            cdn.wait_artifact(github, 123, 2, 'linux-devlab-2')

    def test_runtime_cache_is_reused_and_tampering_is_repaired(self):
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'source.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                entry = tarfile.TarInfo('runtime/lib/example.so')
                entry.size = 7
                tar.addfile(entry, io.BytesIO(b'library'))
            payload = archive.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            manifest = root / 'runtime.json'
            manifest.write_text(cdn.json.dumps({'artifact': 'linux-devlab-1-runtime',
                                               'digest': 'sha256:' + digest}))
            cache = root / 'cache'
            output = root / 'output'
            output.mkdir()
            def fetch(github, artifact, destination):
                with zipfile.ZipFile(destination, 'w') as source:
                    source.writestr('linux-runtime.tar.gz', payload)
            with patch.object(cdn, 'wait_artifact', return_value={}), \
                 patch.object(cdn, 'fetch', side_effect=fetch) as downloader:
                cdn.runtime(Mock(), 123, 1, manifest, cache, output)
                cdn.runtime(Mock(), 123, 1, manifest, cache, output)
                self.assertEqual(downloader.call_count, 1)
                (cache / (digest + '.tar.gz')).write_bytes(b'tampered')
                cdn.runtime(Mock(), 123, 1, manifest, cache, output)
                self.assertEqual(downloader.call_count, 2)
            self.assertEqual((output / 'runtime/lib/example.so').read_bytes(), b'library')


if __name__ == '__main__':
    unittest.main()
