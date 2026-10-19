"""Shared setup and retained evidence for Qwen deep-history qualification."""
import argparse
import hashlib
import json
import math
import re
import shutil
import socket
from pathlib import Path


def setup(kind):
    parser = argparse.ArgumentParser(description=f'Qwen27B {kind} workload; AR or DFlash2')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--draft', type=Path)
    parser.add_argument('--mode', choices=['ar', 'dflash2'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True, help='Revision used to build --binary')
    parser.add_argument('--context', type=int, default=196608 if kind == 'deep-peers' else 131072)
    parser.add_argument('--sessions', type=int, default=3)
    parser.add_argument('--ram-bytes', default='2147483648')
    parser.add_argument('--disk-bytes', type=int, default=48 << 30)
    parser.add_argument('--expect-component', action='store_true')
    args = parser.parse_args()
    if args.mode == 'dflash2' and args.draft is None:
        parser.error('--draft is required for DFlash2')
    if not args.binary.is_file() or not args.model.is_file():
        parser.error('Binary and model must exist before functional execution')
    if args.mode == 'dflash2' and not args.draft.is_file():
        parser.error('DFlash weights must exist')
    if args.context < 1 or args.sessions < 3 or args.disk_bytes < 1:
        parser.error('Context/disk must be positive and these peer workloads require at least three sessions')
    if args.ram_bytes != 'auto' and (not args.ram_bytes.isdecimal() or int(args.ram_bytes) < 1):
        parser.error('--ram-bytes must be a positive integer or auto')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    disk = out / 'disk'
    with socket.socket() as bound:
        bound.bind(('127.0.0.1', 0))
        port = bound.getsockname()[1]
    command = [str(args.binary.resolve()), 'serve', '--host', '127.0.0.1', '--port', str(port),
               'llm', '--model', str(args.model.resolve()), '--sessions', str(args.sessions),
               '--context', str(args.context), '--served-model-name', 'qwen27b-qualification',
               '--think', 'off', '--log-progress', '--cache-disk', str(disk), '--cache-disk-bytes', str(args.disk_bytes)]
    if args.ram_bytes != 'auto':
        command += ['--cache-ram-bytes', args.ram_bytes]
    if args.mode == 'ar':
        command += ['--speculative', 'off']
    if args.mode == 'dflash2':
        command += ['--speculative', 'dflash2', '--dflash-model', str(args.draft.resolve())]
    binary_sha = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    (out / 'configuration.json').write_text(json.dumps({
        'argv': command, 'binary_sha256': binary_sha, 'mode': args.mode,
        'expect_component': args.expect_component, 'ram_bytes': args.ram_bytes,
        'source_revision': args.source_revision,
        'context': args.context, 'sessions': args.sessions,
        'disk_bytes': args.disk_bytes,
        'qualification': 'Only executed assertions qualify this run; retained phase times are raw observations, not a performance comparison.'
    }, indent=2) + '\n')
    return args, str(out), str(disk), port, command


def retain_response(out, label, messages, result, elapsed):
    directory = Path(out)
    destination = directory / (label + '.response.json')
    if destination.exists():
        raise RuntimeError('Refusing to overwrite a request observation')
    choices = result.get('choices', [])
    content = choices[0].get('message', {}).get('content') if choices else None
    destination.write_text(json.dumps({
        'label': label, 'messages': messages, 'response': result,
        'elapsed_seconds': elapsed,
        'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
        'completion_sha256': hashlib.sha256(content.encode()).hexdigest() if isinstance(content, str) else None,
    }, indent=2) + '\n')
    if not isinstance(content, str):
        raise RuntimeError('Missing text completion; raw response retained')
    metrics = result['usage']['gufo']
    for phase in ['prefill_ms', 'decode_ms']:
        value = metrics.get(phase)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f'Missing/invalid required raw request phase {phase}; response retained')



def verify_loaded(out, log_name, mode):
    path = Path(out) / log_name
    loaded = [match.group(1) for line in path.read_text().splitlines()
              if 'event=load_completed' in line and 'kind=text' in line
              for match in [re.search(r'\bspeculative=(\w+)', line)] if match]
    expected = 'off' if mode == 'ar' else 'dflash2'
    if loaded != [expected]:
        raise RuntimeError(f'Expected loaded {expected}, observed {loaded}')


def cleanup_disk(out, disk):
    root, directory = Path(out).resolve(), Path(disk)
    if directory.is_symlink() or directory.parent.resolve() != root or directory.name != 'disk':
        raise RuntimeError('Refusing cleanup outside the owned disk directory')
    if directory.exists():
        inventory = [{'path': str(path.relative_to(directory)), 'bytes': path.stat().st_size}
                     for path in directory.rglob('*') if path.is_file()]
        (root / 'disk-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        shutil.rmtree(directory)


def stop_process(process, *, kill=False, timeout=120):
    import os
    import signal
    import subprocess
    if process is None:
        return None
    if process.poll() is None:
        selected = signal.SIGKILL if kill else signal.SIGTERM
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, selected)
        else:
            process.send_signal(selected)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)
            raise RuntimeError('Server did not terminate within the bounded grace period')
    return process.returncode


class OwnedRunCleanup:
    def __init__(self, out, disk, process):
        import atexit
        self.out, self.disk, self.process = out, disk, process
        self.finished = False
        self.samplers = []
        atexit.register(self.abort)

    def track_sampler(self, sampler):
        self.samplers.append(sampler)

    def finish(self, *, workload_completed=False):
        if self.finished:
            return
        failure = None
        code = None
        try:
            code = stop_process(self.process())
        except Exception as error:
            failure = repr(error)
        for sampler in self.samplers:
            try:
                sampler.stop()
            except Exception as error:
                failure = repr(error) if failure is None else failure + '; ' + repr(error)
        try:
            cleanup_disk(self.out, self.disk)
        except Exception as error:
            failure = repr(error) if failure is None else failure + '; ' + repr(error)
        Path(self.out, 'lifecycle.json').write_text(json.dumps({
            'status': 'workload_recorded' if workload_completed and failure is None else 'aborted',
            'qualified': False,
            'server_returncode': code, 'cleanup_error': failure,
            'reason': 'Scenario assertions and matched controls remain necessary.'
        }, indent=2) + '\n')
        self.finished = True
        if failure is not None:
            raise RuntimeError(failure)

    def abort(self):
        if not self.finished:
            self.finish()


def assert_checkpoint_selection(out, label, result, *, restart=False, minimum=1):
    """Assert reported selection, retaining failures independently of raw responses."""
    usage = result['usage']
    metrics = usage.get('gufo', {}).get('component_cache')
    checks = []
    def check(name, passed, actual):
        checks.append({'name': name, 'passed': bool(passed), 'actual': actual})
    check('component_schema', isinstance(metrics, dict) and metrics.get('schema') == 'component-cache-v1', metrics)
    if isinstance(metrics, dict):
        names = ['selected_tokens', 'deepest_live_tokens', 'deepest_ram_tokens', 'deepest_disk_tokens']
        values = [metrics.get(name) for name in names]
        valid = all(type(value) is int and value >= 0 for value in values)
        check('integer_boundaries', valid, dict(zip(names, values)))
        if valid:
            selected, live, ram, disk = values
            check('newest_available_boundary', selected == max(live, ram, disk), values)
            check('required_depth', selected >= minimum, {'selected': selected, 'minimum': minimum})
            cached = usage.get('prompt_tokens_details', {}).get('cached_tokens')
            check('cached_matches_selection', cached == selected, {'cached': cached, 'selected': selected})
            if restart:
                check('restart_uses_disk', metrics.get('reuse') == 'disk' and selected == disk and live == 0 and ram == 0,
                      {'reuse': metrics.get('reuse'), 'boundaries': values})
    destination = Path(out) / (label + '.checkpoint-assertions.json')
    if destination.exists():
        raise RuntimeError('Refusing to overwrite checkpoint assertions')
    destination.write_text(json.dumps({'label': label, 'checks': checks}, indent=2) + '\n')
    failures = [entry['name'] for entry in checks if not entry['passed']]
    if failures:
        raise RuntimeError(f'{label}: checkpoint assertions failed: {failures}')
    return metrics['selected_tokens']


def completed_partial_prefill(text):
    # A concurrently appended log may end midway through a count.
    text = ''.join(line for line in text.splitlines(keepends=True) if line.endswith('\n'))
    observations = [{'request': int(match[1]), 'processed': int(match[2]), 'total': int(match[3])}
                    for match in re.finditer(r'\brequest=(\d+) phase=prefill tokens=(\d+)/(\d+)\b', text)]
    active = [row for row in observations if row['processed'] > 0]
    if len({row['request'] for row in active}) != 1:
        raise ValueError('Expected one observed cancelled prefill request')
    row = max(active, key=lambda value: value['processed'])
    if not 0 < row['processed'] < row['total']:
        raise ValueError('Cancellation did not demonstrate a partial completed prefill step')
    if len({value['total'] for value in active}) != 1:
        raise ValueError('Prefill total changed within one request')
    return row


def assert_cancelled_frontier(out, label, result, observation, *, component):
    usage = result['usage']
    frontier = usage['prompt_tokens'] - observation['total'] + observation['processed']
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens')
    metrics = usage.get('gufo', {}).get('component_cache', {})
    passed = type(cached) is int and 0 < frontier <= cached <= usage['prompt_tokens']
    if component:
        passed = passed and metrics.get('reuse') == 'live' and metrics.get('selected_tokens') == cached
    destination = Path(out) / (label + '.cancel-frontier.json')
    if destination.exists():
        raise RuntimeError('Refusing to overwrite cancelled-frontier evidence')
    destination.write_text(json.dumps({
        'observation': observation, 'expected_minimum_frontier': frontier,
        'cached_tokens': cached, 'component_metrics': metrics, 'passed': bool(passed),
        'scope': 'Completed prefill frontier reuse; exact numerical cold/main comparison still required'
    }, indent=2) + '\n')
    if not passed:
        raise RuntimeError('Resend did not retain the observed completed prefill frontier')


class ServerResourceSampler:
    """Retain process/host samples and an explicit HTTP-ready load observation."""
    def __init__(self, out, label, process):
        import threading
        self.process = process
        self.out = Path(out)
        self.label = label
        self.destination = self.out / (label + '.resources.jsonl')
        self.stream = self.destination.open('x')
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.count = 0
        self.minimum_available = None
        self.maximum_rss = 0
        self.after_load = None
        self.finished = False
        self.identity = self._identity()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _identity(self):
        # Linux comm may contain spaces or parentheses; split after its closing ')'.
        fields = Path(f'/proc/{self.process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[19])  # Field22: process start time in clock ticks.

    def _observe(self, event):
        import datetime
        import time
        host = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in {'MemAvailable', 'MemFree', 'Cached', 'SReclaimable'}:
                host[key + '_bytes'] = int(value.split()[0]) * 1024
        process = {}
        identity_status = 'exited'
        try:
            if self._identity() != self.identity:
                identity_status = 'pid_reused'
            else:
                identity_status = 'same_process'
                for line in Path(f'/proc/{self.process.pid}/status').read_text().splitlines():
                    key, _, value = line.partition(':')
                    if key in {'VmRSS', 'VmHWM', 'VmSize'}:
                        process[key + '_bytes'] = int(value.split()[0]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            pass
        return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'monotonic_seconds': time.monotonic(), 'event': event,
                'pid': self.process.pid, 'start_ticks': self.identity,
                'process_identity': identity_status, 'host': host, 'process': process}

    def _record(self, event):
        with self.lock:
            row = self._observe(event)
            self.stream.write(json.dumps(row) + '\n')
            self.stream.flush()
            self.count += 1
            available = row['host']['MemAvailable_bytes']
            self.minimum_available = available if self.minimum_available is None else min(self.minimum_available, available)
            self.maximum_rss = max(self.maximum_rss, row['process'].get('VmRSS_bytes', 0))
            return row

    def _run(self):
        try:
            while not self.done.is_set():
                self._record('sample')
                self.done.wait(1)
        except Exception as error:
            self.error = repr(error)
            self.done.set()

    def mark_loaded(self):
        if self.after_load is not None:
            raise RuntimeError('Refusing a second loaded resource observation')
        self.after_load = self._record('http_ready_loaded')
        if self.after_load['process_identity'] != 'same_process' or not self.after_load['process']:
            raise RuntimeError('Loaded server resource observation unavailable')

    def stop(self):
        if self.finished:
            return
        self.done.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Resource sampler did not stop')
        self._record('sampler_stopped')
        self.stream.close()
        ttm = Path('/sys/module/ttm/parameters/pages_limit')
        summary = {'scope': 'Observed server launch; sampled minima, not unsampled instantaneous minima',
                   'pid': self.process.pid, 'start_ticks': self.identity,
                   'sample_count': self.count, 'after_load': self.after_load,
                   'minimum_sampled_MemAvailable_bytes': self.minimum_available,
                   'maximum_sampled_RSS_bytes': self.maximum_rss,
                   'ttm_pages_limit_observed': ttm.read_text().strip() if ttm.exists() else None,
                   'default_ttm_limit_qualified': False,
                   'sampler_error': getattr(self, 'error', None)}
        (self.out / (self.label + '.resources-summary.json')).write_text(json.dumps(summary, indent=2) + '\n')
        self.finished = True
        if summary['sampler_error']:
            raise RuntimeError('Resource sampler failed: ' + summary['sampler_error'])


def response_signature(result):
    choices = result.get('choices', [])
    if len(choices) != 1:
        raise ValueError('Expected exactly one completion choice')
    choice = choices[0]
    message = choice.get('message', {})
    if (not isinstance(message.get('content'), str) or
            type(result['usage'].get('completion_tokens')) is not int or
            result['usage']['completion_tokens'] < 0):
        raise ValueError('Missing text or actual completion count')
    return {key: message.get(key) for key in
            ('role', 'content', 'reasoning_content', 'tool_calls')} | {
        'finish_reason': choice.get('finish_reason'),
        'completion_tokens': result['usage']['completion_tokens'],
    }


def assert_cold_control(out, label, cached, cold):
    # Store every observation before asserting; failure is evidence too.
    metrics = cold['usage']['gufo']
    cached_tokens = cold['usage'].get('prompt_tokens_details', {}).get('cached_tokens')
    record = {
        'cached_signature': response_signature(cached),
        'cold_signature': response_signature(cold),
        'cold_cached_tokens': cached_tokens,
        'cold_prefill_tokens': metrics.get('prefill_tokens'),
        'cold_prompt_tokens': cold['usage']['prompt_tokens'],
        'qualified': False,
    }
    path = Path(out) / (label + '.cold-comparison.json')
    path.write_text(json.dumps(record, indent=2) + '\n')
    if (not all(type(record[key]) is int for key in
                ('cold_cached_tokens', 'cold_prefill_tokens', 'cold_prompt_tokens')) or
            record['cold_prompt_tokens'] <= 0 or
            record['cached_signature'] != record['cold_signature'] or
            cached_tokens != 0 or
            record['cold_prefill_tokens'] != record['cold_prompt_tokens']):
        raise RuntimeError('Cold control changed output or reused cached work; raw comparison retained')
    record['qualified'] = True
    path.write_text(json.dumps(record, indent=2) + '\n')


def wait_for_published(out, log_name, minimum, server, timeout=120):
    import time
    started = time.monotonic()
    deadline = started + timeout
    path = Path(out) / log_name
    while True:
        text = path.read_text()
        published = [int(value) for value in
                     re.findall(r'event=published\s+tokens=(\d+)\b', text)]
        if published and max(published) >= minimum:
            observation = {'minimum': minimum, 'published_tokens': max(published),
                           'server_pid': server.pid, 'server_returncode': server.poll(),
                           'qualified': True}
            (Path(out) / (log_name + '.pre-kill-publication.json')).write_text(
                json.dumps(observation, indent=2) + '\n')
            if observation['server_returncode'] is not None:
                raise RuntimeError('Server exited before controlled SIGKILL')
            return observation
        if server.poll() is not None or time.monotonic() >= deadline:
            observation = {'minimum': minimum, 'published_tokens': published,
                           'server_returncode': server.poll(), 'qualified': False}
            (Path(out) / (log_name + '.pre-kill-publication.json')).write_text(
                json.dumps(observation, indent=2) + '\n')
            raise RuntimeError('No required published checkpoint before controlled SIGKILL')
        time.sleep(0.1)



def verify_execution_log(out, log_name, mode):
    verify_loaded(out, log_name, mode)
    text = (Path(out) / log_name).read_text()
    # Progress logs repeat cumulative counters. Count completed HTTP requests
    # once, so intermediate decode observations cannot inflate actual drafts.
    counts = [int(value) for line in text.splitlines()
              if '[http]' in line and 'event=completed' in line and 'method=POST' in line
              for value in re.findall(r'\bdraft_proposed=(\d+)\b', line)]
    proposed = sum(counts)
    record = {'mode': mode, 'proposed': proposed, 'completed_request_counts': counts,
              'qualified': bool(counts) and (proposed == 0 if mode == 'ar' else proposed > 0)}
    (Path(out) / (log_name + '.execution.json')).write_text(json.dumps(record, indent=2) + '\n')
    if not record['qualified']:
        raise RuntimeError('Loaded mode did not execute the required actual draft work')


def assert_publication_wait(out, label, segment, result):
    """Require a real pending job, successful publication and deeper reuse."""
    begins = list(re.finditer(
        r'event=publication_wait request=(\d+) stage=start checkpoint_id=(\d+) '
        r'pending_tokens=(\d+) available_tokens=(\d+) limit_ms=([\d.e+-]+)', segment))
    record = {'qualified': False, 'raw_log': segment, 'waits': []}
    for begin in begins:
        request, checkpoint, pending, previous = map(int, begin.groups()[:4])
        end = re.search(
            r'event=publication_wait request=' + str(request) +
            r' stage=release checkpoint_id=' + str(checkpoint) +
            r' pending_tokens=(\d+) available_tokens=(\d+) limit_ms=([\d.e+-]+)'
            r' wait_ms=([\d.e+-]+) release=(\w+)', segment[begin.end():])
        row = {'request': request, 'checkpoint_id': checkpoint,
               'pending_tokens': pending, 'previous_tokens': previous,
               'limit_ms': float(begin[5])}
        if end:
            row.update(reason=end[5], available_tokens=int(end[2]),
                       wait_ms=float(end[4]), release_limit_ms=float(end[3]))
            between = segment[begin.end():begin.end() + end.start()]
            row['published_tokens'] = [int(value) for value in re.findall(
                r'event=published\s+tokens=(\d+) checkpoint_id=' +
                str(checkpoint) + r'\b', between)]
        record['waits'].append(row)
    cached = result['usage'].get('prompt_tokens_details', {}).get('cached_tokens')
    metrics = result['usage'].get('gufo', {}).get('component_cache', {})
    record['response_cached_tokens'] = cached
    record['response_component_cache'] = metrics
    valid = len(record['waits']) == 1
    if valid:
        row = record['waits'][0]
        valid = (row['pending_tokens'] > row['previous_tokens'] + 4096 and
                 row.get('reason') in ('not_pending', 'available') and
                 0 <= row.get('wait_ms', -1) <= 120000 and
                 0 < row.get('release_limit_ms', 0) <= row['limit_ms'] and
                 any(value >= row['pending_tokens'] for value in row.get('published_tokens', [])) and
                 type(cached) is int and cached >= row['pending_tokens'] and
                 metrics.get('selected_tokens') == cached)
    record['qualified'] = bool(valid)
    Path(out, label + '.publication-wait.json').write_text(json.dumps(record, indent=2) + '\n')
    if not valid:
        raise RuntimeError('Rewritten request did not prove successful pending publication and deeper reuse')
