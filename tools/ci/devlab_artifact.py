#!/usr/bin/env python3
"""Download verified CI artifacts using bounded direct/WARP probes and cache pinned runtimes."""

import argparse
from contextlib import contextmanager
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import zipfile


class GitHub:
    def __init__(self, repository):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid repository")
        self.repository = repository
        self.root = f"https://api.github.com/repos/{repository}"

    def api(self, path, data=None, method=None, missing=False, deadline=None):
        # Rate-limit waits never outlast the caller's deadline (an hour by default).
        deadline = time.monotonic() + 3600 if deadline is None else deadline
        retries = 0
        while True:
            request = Request(self.root + path, method=method,
                              data=None if data is None else json.dumps(data).encode(),
                              headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"],
                                       "Accept": "application/vnd.github+json",
                                       "Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=60) as response:
                    body = response.read()
                    return json.loads(body) if body else None
            except HTTPError as error:
                if missing and error.code == 404:
                    return None
                if not wait_out_rate_limit(error, retries, deadline):
                    raise
                retries += 1

    def artifacts(self, run, name, deadline=None):
        # Filtering by name avoids pagination through unrelated model results.
        return self.api(f"/actions/runs/{run}/artifacts?name={name}", deadline=deadline)["artifacts"]


def rate_limit_delay(error, retries=0):
    """Seconds to wait before retrying a rate-limited response, or None to fail.

    GitHub answers both primary and secondary rate limits with 403 or 429; only
    responses that carry a rate-limit signal are retried, so authorization
    failures still fail immediately. `retries` counts earlier rate-limited
    responses to the same request.
    """
    if error.code not in (403, 429):
        return None
    headers = error.headers or {}
    jitter = random.uniform(1, 10)
    retry_after = headers.get('retry-after')
    if retry_after and retry_after.strip().isdigit():
        return min(int(retry_after), 3600) + jitter
    reset = headers.get('x-ratelimit-reset')
    if headers.get('x-ratelimit-remaining') == '0' and reset and reset.strip().isdigit():
        # Measure against GitHub's clock when it is given: runner clock skew
        # must not turn the wait into a burst of immediate retries.
        try:
            now = parsedate_to_datetime(headers['date']).timestamp()
        except (KeyError, TypeError, ValueError, IndexError):
            now = time.time()
        return min(max(int(reset) - now, 5), 3600) + jitter
    try:
        message = error.read().decode(errors='replace')
    except (OSError, ValueError):
        message = ''
    # Secondary limits may omit both headers; GitHub asks for at least a minute
    # and an increasing wait while the limit persists.
    if 'rate limit' in message.lower():
        return min(60 * 2 ** retries, 3600) + jitter
    return None


def wait_out_rate_limit(error, retries, deadline):
    """Sleep through a rate limit and return True, or return False to fail now."""
    delay = rate_limit_delay(error, retries)
    if delay is None or time.monotonic() + delay > deadline:
        return False
    print(f"GitHub API rate limited (HTTP {error.code}); retrying in {delay:.0f}s", flush=True)
    time.sleep(delay)
    return True


def checked_digest(value):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value or ""):
        raise ValueError("Missing or invalid GitHub artifact SHA-256")
    return value[7:]


def verify(path, digest):
    expected = checked_digest(digest)
    with Path(path).open("rb") as source:
        checksum = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
        actual = checksum.hexdigest()
    if actual != expected:
        raise ValueError("Artifact checksum mismatch")


def probe_route(url, proxy=None):
    command = ["curl", "--silent", "--location", "--fail", "--max-time", "12",
               "--range", "0-2097151", "--output", os.devnull,
               "--proto", "=https", "--proto-redir", "=https",
               "--write-out", "%{http_code} %{size_download} %{time_total}"]
    if proxy:
        command += ["--proxy", proxy]
    result = subprocess.run(command + [url], capture_output=True, text=True, timeout=20)
    try:
        status, size, elapsed = result.stdout.split()
        speed = int(size) / max(float(elapsed), 0.001) if status == "206" else 0
    except (ValueError, ZeroDivisionError):
        speed = 0
    # Partial ranges are useful measurements even when curl's time limit expires.
    print(f"{'WARP' if proxy else 'Direct'} route: {speed / 1024:.0f} KiB/s", flush=True)
    return speed


@contextmanager
def warp_proxy(directory):
    version = "4.2.1"
    archive = directory / "usque.zip"
    url = ("https://github.com/Diniboy1123/usque/releases/download/"
           f"v{version}/usque_{version}_{'windows' if os.name == 'nt' else 'linux'}_amd64.zip")
    subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location",
                    "--max-time", "120", "--proto", "=https", "--proto-redir", "=https",
                    "--output", str(archive), url], check=True, timeout=130)
    digest = ("f6f7f0a1a2bc9bcc15cf563ec1f892d00690a92c086b23ed3211b802209099e7"
              if os.name == "nt" else
              "4117e20695078af9c11edecd1a826c009bbc7ea0b7f64458612b4198910bc313")
    verify(archive, "sha256:" + digest)
    executable = directory / ("usque.exe" if os.name == "nt" else "usque")
    with zipfile.ZipFile(archive) as source, source.open(executable.name) as data, executable.open("wb") as output:
        shutil.copyfileobj(data, output)
    executable.chmod(0o700)
    config = directory / "warp.json"
    command = [str(executable), "--config", str(config)]
    with (directory / "warp.log").open("wb") as log:
        # Credentials remain in the private temporary directory, never in CI logs.
        subprocess.run(command + ["register", "--accept-tos"], stdout=log, stderr=log,
                       check=True, timeout=30)
        config.chmod(0o600)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen(command + ["l4-http-proxy", "--bind", "127.0.0.1",
                                             "--port", str(port), "--system-dns"],
                                   stdout=log, stderr=log)
        try:
            proxy = f"http://127.0.0.1:{port}"
            ready = False
            for _ in range(5):
                trace = subprocess.run(["curl", "--silent", "--fail", "--max-time", "5",
                                        "--proxy", proxy, "https://www.cloudflare.com/cdn-cgi/trace"],
                                       capture_output=True, text=True, timeout=10)
                if trace.returncode == 0 and "warp=on" in trace.stdout.splitlines():
                    ready = True
                    break
                if process.poll() is not None:
                    break
                time.sleep(1)
            if not ready:
                raise RuntimeError("WARP proxy did not become ready")
            yield proxy
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def transfer(url, bundle):
    def fetch(proxy=None):
        command = ["curl", "--fail", "--location", "--retry", "2",
                   "--connect-timeout", "20", "--max-time", "900",
                   "--speed-limit", "131072", "--speed-time", "60",
                   "--proto", "=https", "--proto-redir", "=https",
                   "--output", str(bundle)]
        if proxy:
            command += ["--proxy", proxy]
        started = time.monotonic()
        try:
            subprocess.run(command + [url()], check=True)
        except subprocess.SubprocessError:
            raise RuntimeError("Artifact transfer failed; signed URL withheld") from None
        print(f"Binary transfer: {bundle.stat().st_size / 1024 / 1024:.1f} MiB in "
              f"{time.monotonic() - started:.1f}s ({'WARP' if proxy else 'direct'})", flush=True)

    direct = probe_route(url())
    # Avoid registration and tunnel setup when the direct route is already fast.
    if direct >= 5 * 1024 * 1024:
        fetch()
        return
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        directory.chmod(0o700)
        try:
            with warp_proxy(directory) as proxy:
                tunneled = probe_route(url(), proxy)
                if tunneled > max(direct * 1.25, 128 * 1024):
                    print("Selected WARP for binary download", flush=True)
                    fetch(proxy)
                    return
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            print(f"WARP unavailable ({type(error).__name__}); using direct route", flush=True)
    print("Selected direct route for binary download", flush=True)
    fetch()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def signed_url(github, artifact, deadline=None):
    deadline = time.monotonic() + 300 if deadline is None else deadline
    retries = 0
    while True:
        request = Request(github.root + f"/actions/artifacts/{artifact}/zip", headers={
            'Authorization': 'Bearer ' + os.environ['GH_TOKEN']})
        try:
            build_opener(NoRedirect).open(request, timeout=30)
        except HTTPError as error:
            if error.code == 302 and error.headers['Location'].startswith('https://'):
                return error.headers['Location']
            if wait_out_rate_limit(error, retries, deadline):
                retries += 1
                continue
            raise RuntimeError(f'Cannot obtain artifact download URL: HTTP {error.code}') from None
        raise RuntimeError('Expected an artifact download redirect')


# The hosted job that produces each Devlab package, as the caller's "Build"
# job names the reusable linux-build.yml job.
BUILD_JOBS = {'release': 'Build / Linux package', 'checked': 'Build / Linux checked package'}
PACKAGES = {'release': ('', 'linux-artifacts.tar.gz'), 'checked': ('-checked', 'linux-checked.tar.gz')}


def started(stamp):
    """Seconds since the epoch for a GitHub timestamp, or None when absent."""
    return datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp() if stamp else None


def wait_artifact(github, run, attempt, name, job=BUILD_JOBS['release'], earlier=None):
    """Wait for this run's artifact `name`, published by the hosted job `job`.

    `earlier(k)` names the same package as attempt k published it: re-running only
    failed jobs carries the successful build job into the new attempt without
    rebuilding, so its package keeps the attempt that produced it.

    The GITHUB_TOKEN request budget is shared by every job of the repository,
    so each poll reads only this attempt's job list; the artifact list is read
    once the build job has succeeded, which happens after its upload.
    """
    deadline = time.monotonic() + 7200

    def listed(candidate):
        return [item for item in github.artifacts(run, candidate, deadline=deadline)
                if item['name'] == candidate and not item['expired']]

    def single(found, candidate):
        if len(found) > 1:
            raise RuntimeError(f'Run {run} lists {candidate} more than once')
        checked_digest(found[0].get('digest'))
        return found[0]

    def state():
        current = github.api(f'/actions/runs/{run}', deadline=deadline)
        if current['run_attempt'] != attempt or current['status'] == 'completed':
            raise RuntimeError('Source run ended or was superseded before artifact delivery')
        return current

    state()
    checked = time.monotonic()
    interval = 60
    while True:
        jobs = github.api(f'/actions/runs/{run}/attempts/{attempt}/jobs?per_page=100',
                          deadline=deadline)['jobs']
        rows = [row for row in jobs if row['name'] == job]
        if any(row['conclusion'] in ('failure', 'cancelled', 'timed_out', 'skipped')
               for row in rows):
            raise RuntimeError(f'Hosted job {job} failed')
        if any(row['conclusion'] == 'success' for row in rows):
            found = listed(name)
            if found:
                return single(found, name)
            # A build job carried from an earlier attempt started before this one.
            began = started(state().get('run_started_at'))
            carried = began is not None and all(
                started(row.get('started_at')) is not None and started(row['started_at']) < began
                for row in rows)
            if earlier and carried:
                for previous in range(attempt - 1, 0, -1):
                    candidate = earlier(previous)
                    found = listed(candidate)
                    if found:
                        print(f'Using {candidate} from the build carried into attempt {attempt}',
                              flush=True)
                        return single(found, candidate)
            raise RuntimeError(f'Hosted job {job} succeeded without publishing {name}; '
                               're-run all jobs')
        if time.monotonic() - checked >= 1800:
            state()
            checked = time.monotonic()
        if time.monotonic() + interval > deadline:
            raise TimeoutError('Hosted artifact unavailable')
        print(f'Waiting for {name}; model downloads continue', flush=True)
        time.sleep(interval)
        interval = min(interval * 1.5, 150)


def fetch(github, artifact, destination):
    checked_digest(artifact.get('digest'))
    transfer(lambda: signed_url(github, artifact['id']), destination)
    verify(destination, artifact['digest'])
    print('Artifact SHA-256 verified', flush=True)


def extract_member(source, member, destination):
    with zipfile.ZipFile(source) as archive:
        if archive.namelist() != [member]:
            raise ValueError('Unexpected artifact ZIP contents')
        with archive.open(member) as data, Path(destination).open('wb') as output:
            shutil.copyfileobj(data, output)


def runtime(github, run, attempt, manifest, cache, output):
    descriptor = json.loads(manifest.read_text())
    digest = checked_digest(descriptor['digest'])
    name = descriptor['artifact']
    if not re.fullmatch(r'linux-[A-Za-z0-9_-]+-runtime', name):
        raise ValueError('Invalid runtime artifact name')
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (digest + '.tar.gz')
    if archive.exists():
        try:
            verify(archive, descriptor['digest'])
        except ValueError:
            archive.unlink()
    if not archive.exists():
        artifact = wait_artifact(github, run, attempt, name)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            source = Path(directory) / 'runtime.zip'
            fetch(github, artifact, source)
            pending = Path(directory) / 'runtime.tar.gz'
            extract_member(source, 'linux-runtime.tar.gz', pending)
            verify(pending, descriptor['digest'])
            pending.replace(archive)
    else:
        print('Using verified persistent Linux runtime cache', flush=True)
    # Only the pinned runtime tree is installed; tar's data filter rejects escaping links.
    import tarfile
    with tarfile.open(archive) as source:
        if any(not item.name.startswith('runtime/') and item.name != 'runtime' for item in source):
            raise ValueError('Unexpected runtime archive path')
        source.extractall(output, filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('linux', 'fetch', 'runtime'))
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY'))
    parser.add_argument('--run', type=int)
    parser.add_argument('--attempt', type=int)
    parser.add_argument('--artifact', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--variant', choices=sorted(PACKAGES), default='release',
                        help='linux: the release comparison packages or the assertion-enabled candidate')
    args = parser.parse_args()
    github = GitHub(args.repository)
    if args.operation == 'fetch':
        if not args.artifact:
            parser.error('--artifact is required')
        # The calling step has a 20-minute budget; keep rate-limit waits short.
        fetch(github, github.api(f'/actions/artifacts/{args.artifact}',
                                 deadline=time.monotonic() + 300), args.output)
    else:
        if not args.run or not args.attempt:
            parser.error('--run and --attempt are required')
        if args.operation == 'runtime':
            if not args.manifest or not args.cache:
                parser.error('--manifest and --cache are required')
            runtime(github, args.run, args.attempt, args.manifest, args.cache, args.output)
        else:
            suffix, member = PACKAGES[args.variant]
            artifact = wait_artifact(github, args.run, args.attempt, f'linux-devlab-{args.attempt}{suffix}',
                                     BUILD_JOBS[args.variant],
                                     lambda previous: f'linux-devlab-{previous}{suffix}')
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'binaries.zip'
                fetch(github, artifact, source)
                extract_member(source, member, args.output)


if __name__ == '__main__':
    main()
