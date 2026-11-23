#!/usr/bin/env python3
"""Exercise artifact integrity, adaptive routing and persistent runtime reuse."""

from email.message import Message
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError
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

    def test_failed_build_job_stops_waiting_for_its_package_only(self):
        github = Mock()
        github.artifacts.return_value = []
        jobs = {'jobs': [{'name': 'Build / Linux checked package', 'conclusion': 'failure'},
                         {'name': 'Build / Linux package', 'conclusion': None}]}
        github.api.side_effect = lambda path, **_: (
            jobs if '/jobs' in path else {'run_attempt': 2, 'status': 'in_progress'})
        with self.assertRaisesRegex(RuntimeError, 'Linux checked package failed'):
            cdn.wait_artifact(github, 123, 2, 'linux-devlab-2-checked', cdn.BUILD_JOBS['checked'])
        jobs['jobs'][1]['conclusion'] = 'failure'
        jobs['jobs'][0]['conclusion'] = None
        with patch.object(cdn.time, 'sleep', side_effect=TimeoutError('waited')):
            with self.assertRaisesRegex(TimeoutError, 'waited'):
                cdn.wait_artifact(github, 123, 2, 'linux-devlab-2-checked', cdn.BUILD_JOBS['checked'])

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


DIGEST = 'sha256:' + '0' * 64


def http_error(code, body=b'', **headers):
    message = Message()
    for key, value in headers.items():
        message[key.replace('_', '-')] = value
    return HTTPError('https://api.github.com/x', code, 'error', message, io.BytesIO(body))


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeClock:
    """Monotonic and wall time that advance only when the code under test sleeps."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def patch(self):
        return patch.multiple(cdn.time, monotonic=lambda: self.now, time=lambda: self.now,
                              sleep=self.sleep)


class RateLimitTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.github = cdn.GitHub('gufo-org/gufo')
        patches = [self.clock.patch(), patch.dict(cdn.os.environ, {'GH_TOKEN': 'token'}),
                   patch.object(cdn.random, 'uniform', return_value=2.0)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def call(self, *responses, **kwargs):
        with patch.object(cdn, 'urlopen', side_effect=list(responses)) as opener, \
             redirect_stdout(io.StringIO()):
            result = self.github.api('/actions/runs/1', **kwargs)
        return result, opener.call_count

    def test_primary_limit_waits_until_reset(self):
        error = http_error(403, x_ratelimit_remaining='0', x_ratelimit_reset=str(int(self.clock.now) + 300))
        result, calls = self.call(error, Response(b'{"ok": true}'))
        self.assertEqual(result, {'ok': True})
        self.assertEqual((calls, self.clock.sleeps), (2, [302.0]))

    def test_retry_after_is_honoured_for_403_and_429(self):
        for code in (403, 429):
            with self.subTest(code=code):
                self.clock.sleeps.clear()
                result, calls = self.call(http_error(code, retry_after='45'), Response(b'{}'))
                self.assertEqual((result, calls, self.clock.sleeps), ({}, 2, [47.0]))

    def test_secondary_limit_without_headers_waits_a_minute(self):
        body = b'{"message": "You have exceeded a secondary rate limit."}'
        result, calls = self.call(http_error(403, body), Response(b'{}'))
        self.assertEqual((calls, self.clock.sleeps), (2, [62.0]))

    def test_authorization_failures_are_not_retried(self):
        for error in (http_error(403, b'{"message": "Resource not accessible by integration"}',
                                 x_ratelimit_remaining='4000'),
                      http_error(401, b'{"message": "Bad credentials"}', retry_after='5'),
                      http_error(500, retry_after='5')):
            with self.subTest(code=error.code), self.assertRaises(HTTPError):
                self.call(error)
        self.assertEqual(self.clock.sleeps, [])

    def test_wait_never_outlasts_the_deadline(self):
        error = http_error(403, x_ratelimit_remaining='0', x_ratelimit_reset=str(int(self.clock.now) + 900))
        with self.assertRaises(HTTPError):
            self.call(error, deadline=self.clock.now + 600)
        self.assertEqual(self.clock.sleeps, [])

    def test_past_reset_still_waits_a_floor(self):
        error = http_error(403, x_ratelimit_remaining='0', x_ratelimit_reset=str(int(self.clock.now) - 30))
        self.call(error, Response(b'{}'))
        self.assertEqual(self.clock.sleeps, [7.0])

    def test_reset_is_measured_against_github_clock(self):
        # The runner clock runs 4000 s ahead of GitHub's Date header.
        from email.utils import formatdate
        error = http_error(403, x_ratelimit_remaining='0', x_ratelimit_reset='5300',
                           date=formatdate(5000, usegmt=True))
        self.clock.now = 9000.0
        self.call(error, Response(b'{}'))
        self.assertEqual(self.clock.sleeps, [302.0])

    def test_header_less_secondary_limit_backs_off(self):
        body = b'{"message": "You have exceeded a secondary rate limit."}'
        errors = [http_error(403, body) for _ in range(3)]
        result, calls = self.call(*errors, Response(b'{}'))
        self.assertEqual((calls, self.clock.sleeps), (4, [62.0, 122.0, 242.0]))

    def test_429_with_rate_limit_body_is_retried(self):
        result, calls = self.call(http_error(429, b'{"message": "API rate limit exceeded"}'),
                                  Response(b'{}'))
        self.assertEqual((calls, self.clock.sleeps), (2, [62.0]))

    def test_signed_url_waits_out_rate_limits(self):
        opener = Mock()
        opener.open.side_effect = [http_error(403, retry_after='20'),
                                   http_error(302, location='https://example.org/blob')]
        with patch.object(cdn, 'build_opener', return_value=opener), redirect_stdout(io.StringIO()):
            self.assertEqual(cdn.signed_url(self.github, 7), 'https://example.org/blob')
        self.assertEqual(self.clock.sleeps, [22.0])
        opener.open.side_effect = [http_error(403, b'{"message": "Must have admin rights"}')]
        with patch.object(cdn, 'build_opener', return_value=opener), \
             self.assertRaisesRegex(RuntimeError, 'HTTP 403'):
            cdn.signed_url(self.github, 7)


class FakeRun:
    """Attempt `attempt` of a run whose build job finishes at `done`.

    `carried` marks a build job taken over from an earlier attempt, which
    started before this attempt; `published` holds every listed artifact.
    """

    def __init__(self, attempt=1, done=1200.0, published=None, carried=False,
                 conclusion='success', job=cdn.BUILD_JOBS['release']):
        self.attempt, self.done, self.conclusion, self.job = attempt, done, conclusion, job
        self.published = [f'linux-devlab-{attempt}'] if published is None else published
        self.carried = carried
        self.clock = FakeClock()
        self.requests = []

    def artifacts(self, run, name, deadline=None):
        self.requests.append('artifacts')
        rows = []
        for item in self.published:
            entry = {'id': 7, 'name': item, 'expired': False, 'digest': DIGEST}
            if isinstance(item, dict):
                entry.update(item)
            fresh = entry['name'].startswith(f'linux-devlab-{self.attempt}') and not self.carried
            if entry['name'] == name and (self.clock.now >= self.done or not fresh):
                rows.append(entry)
        return rows

    def api(self, path, deadline=None):
        self.requests.append(path)
        if path.endswith('/jobs?per_page=100'):
            if f'/attempts/{self.attempt}/' not in path:
                raise AssertionError(f'unexpected attempt in {path}')
            finished = self.clock.now >= self.done
            began = '2026-11-23T18:45:15Z' if self.carried else '2026-11-23T20:01:30Z'
            return {'jobs': [{'name': self.job, 'started_at': began,
                              'conclusion': self.conclusion if finished else None}]}
        return {'run_attempt': self.attempt, 'status': 'in_progress',
                'run_started_at': '2026-11-23T20:01:14Z'}


def earlier(previous):
    return f'linux-devlab-{previous}'


class WaitArtifactTest(unittest.TestCase):
    def wait(self, run, **kwargs):
        with run.clock.patch(), redirect_stdout(io.StringIO()):
            return cdn.wait_artifact(run, 123, run.attempt, f'linux-devlab-{run.attempt}', **kwargs)

    def test_polling_stays_within_the_shared_token_budget(self):
        # The old 10 s loop made ~900 requests an hour per waiting job.
        run = FakeRun(done=float('inf'))
        with self.assertRaises(TimeoutError):
            self.wait(run)
        hours = (run.clock.now - 1000.0) / 3600
        self.assertLessEqual(len(run.requests) / hours, 30)
        self.assertNotIn('artifacts', run.requests)
        self.assertLessEqual(max(run.clock.sleeps), 150)

    def test_finished_build_is_found_on_the_first_poll(self):
        run = FakeRun(done=0.0)
        self.assertEqual(self.wait(run)['name'], 'linux-devlab-1')
        self.assertEqual((len(run.requests), run.clock.sleeps), (3, []))

    def test_deadline_is_two_hours(self):
        run = FakeRun(done=float('inf'))
        with self.assertRaises(TimeoutError):
            self.wait(run)
        self.assertGreater(run.clock.now - 1000.0, 7200 - 150)
        self.assertLessEqual(run.clock.now - 1000.0, 7200)

    def test_rerun_of_failed_jobs_uses_the_carried_build_package(self):
        run = FakeRun(attempt=2, done=0.0, published=['linux-devlab-1'], carried=True)
        self.assertEqual(self.wait(run, earlier=earlier)['name'], 'linux-devlab-1')
        self.assertEqual(run.clock.sleeps, [])

    def test_newest_earlier_package_wins(self):
        run = FakeRun(attempt=3, done=0.0, published=['linux-devlab-1', 'linux-devlab-2'],
                      carried=True)
        self.assertEqual(self.wait(run, earlier=earlier)['name'], 'linux-devlab-2')

    def test_expired_earlier_package_is_skipped(self):
        run = FakeRun(attempt=3, done=0.0, carried=True,
                      published=['linux-devlab-1', {'name': 'linux-devlab-2', 'expired': True}])
        self.assertEqual(self.wait(run, earlier=earlier)['name'], 'linux-devlab-1')

    def test_rebuilt_attempt_never_falls_back(self):
        # This attempt ran its own build, so an earlier package is stale.
        run = FakeRun(attempt=2, done=1000.0 + 600, published=['linux-devlab-1'])
        with self.assertRaisesRegex(RuntimeError, 'without publishing linux-devlab-2'):
            self.wait(run, earlier=earlier)

    def test_earlier_package_is_ignored_while_the_build_reruns(self):
        run = FakeRun(attempt=2, done=1000.0 + 600, published=['linux-devlab-1', 'linux-devlab-2'])
        self.assertEqual(self.wait(run, earlier=earlier)['name'], 'linux-devlab-2')

    def test_duplicate_current_package_is_rejected(self):
        run = FakeRun(attempt=2, done=0.0, carried=True,
                      published=['linux-devlab-1', 'linux-devlab-2', {'name': 'linux-devlab-2', 'id': 8}])
        with self.assertRaisesRegex(RuntimeError, 'more than once'):
            self.wait(run, earlier=earlier)

    def test_successful_build_without_package_fails_fast(self):
        run = FakeRun(attempt=2, done=0.0, published=[], carried=True)
        with self.assertRaisesRegex(RuntimeError, 'without publishing linux-devlab-2'):
            self.wait(run, earlier=earlier)
        self.assertEqual(run.clock.sleeps, [])

    def test_checked_package_falls_back_to_its_own_earlier_name(self):
        captured = {}
        def wait(github, run, attempt, name, job, fallback):
            captured.update(name=name, job=job, earlier=fallback(1))
            return {'id': 7, 'digest': DIGEST}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cdn, 'wait_artifact', side_effect=wait), \
             patch.object(cdn, 'fetch'), patch.object(cdn, 'extract_member'), \
             patch('sys.argv', ['devlab_artifact.py', 'linux', '--variant', 'checked',
                                            '--repository', 'gufo-org/gufo', '--run', '123',
                                            '--attempt', '2', '--output', directory + '/out']):
            cdn.main()
        self.assertEqual(captured, {'name': 'linux-devlab-2-checked',
                                    'job': cdn.BUILD_JOBS['checked'],
                                    'earlier': 'linux-devlab-1-checked'})

if __name__ == '__main__':
    unittest.main()
