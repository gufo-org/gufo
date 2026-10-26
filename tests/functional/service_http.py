"""Shared evidence and strict HTTP/SSE checks for non-chat model services."""

import base64
from contextlib import contextmanager, nullcontext
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import threading
import time
import wave

import httpx


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def word_error_rate(actual, expected):
    words = lambda text: re.findall(r"[^\W_]+", text.casefold())
    reference, hypothesis = words(expected), words(actual)
    if not reference:
        raise ValueError('expected transcript must contain words')
    previous = list(range(len(reference) + 1))
    for i, word in enumerate(hypothesis, 1):
        current = [i]
        for j, target in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != target)))
        previous = current
    return previous[-1] / len(reference)


def pcm_wav(pcm, rate=24000):
    result = io.BytesIO()
    with wave.open(result, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(pcm)
    return result.getvalue()


def audio_check(data, wav=True):
    if wav:
        with wave.open(io.BytesIO(data)) as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
            frames = audio.getnframes()
            pcm = audio.readframes(frames)
            assert len(pcm) == frames * 2, 'truncated WAV payload'
    else:
        pcm = data
    assert len(pcm) % 2 == 0 and pcm, 'empty or partial PCM samples'
    samples = struct.unpack('<' + 'h' * (len(pcm) // 2), pcm)
    duration = len(samples) / 24000
    rms = math.sqrt(sum((sample / 32768) ** 2 for sample in samples) / len(samples))
    assert .3 <= duration <= 20, ('audio duration', duration)
    assert rms > .001, ('silent audio', rms)
    return pcm, {'duration_seconds': duration, 'rms': rms,
                 'sha256': hashlib.sha256(pcm).hexdigest()}


def events(response, observation=None, path=None):
    """Parse SSE blocks; reject broken JSON and mismatched event types."""
    with path.open('w') if path else nullcontext(None) as capture:
        event, data = None, []
        for line in response.iter_lines():
            if capture:
                capture.write(line + '\n')
                capture.flush()
            if not line:
                if data:
                    value = json.loads('\n'.join(data))
                    assert value.get('type') == event, (event, value)
                    assert event != 'error', value
                    if observation is not None and 'first_event_ms' not in observation:
                        observation['first_event_ms'] = (time.perf_counter() - response.extensions['functional_started']) * 1000
                    yield value
                event, data = None, []
            elif line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                data.append(line[5:].lstrip())
        assert not data, 'unterminated SSE event'


def transcript_events(items):
    assert items and items[-1]['type'] == 'transcript.text.done', 'missing ASR terminal event'
    assert all(item['type'] == 'transcript.text.delta' for item in items[:-1]), items
    text = ''.join(item['delta'] for item in items[:-1])
    assert text and text == items[-1]['text'], items
    return text


def speech_events(items):
    assert items and items[-1]['type'] == 'speech.audio.done', 'missing TTS terminal event'
    assert all(item['type'] == 'speech.audio.delta' for item in items[:-1]), items
    audio = b''.join(base64.b64decode(item['audio'], validate=True) for item in items[:-1])
    assert audio, 'no streamed audio'
    return audio


class Evidence:
    def __init__(self, output, report, timeout=180):
        self.output, self.report, self.timeout = output, report, timeout
        self.lock = threading.Lock()
        self.report.update(cases=[], requests=[])

    def save(self):
        (self.output / 'report.json').write_text(json.dumps(self.report, indent=2) + '\n')

    @contextmanager
    def case(self, name):
        row = {'name': name, 'status': 'running'}
        self.report['cases'].append(row)
        start = time.perf_counter()
        self.save()
        print('CHECK', name, flush=True)
        try:
            yield row
            row['status'] = 'passed'
        except BaseException as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row['elapsed_ms'] = (time.perf_counter() - start) * 1000
            self.save()

    @contextmanager
    def request(self, url, *, label, method='POST', cancel=False, **kwargs):
        with self.lock:
            index = len(self.report['requests'])
            row = {'label': label, 'method': method, 'url': url, 'status': 'running',
                   'request': kwargs.get('json', kwargs.get('data'))}
            if 'files' in kwargs:
                row['uploads'] = {name: {'filename': item[0], 'bytes': len(item[1]),
                    'sha256': hashlib.sha256(item[1]).hexdigest()} for name, item in kwargs['files'].items()}
            self.report['requests'].append(row)
        start = time.perf_counter()
        path = self.output / f'request-{index:03d}.body'
        row['body'] = path.name
        # No retries: a transient server failure must remain visible.
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                with client.stream(method, url, **kwargs) as response:
                    response.extensions['functional_started'] = start
                    row.update(http_status=response.status_code, headers=dict(response.headers),
                               headers_ms=(time.perf_counter() - start) * 1000)
                    yield response, row, path
                    if not cancel:
                        assert response.is_stream_consumed, 'response was not completely consumed'
                    row['status'] = 'cancelled-by-test' if cancel else 'completed'
        except BaseException as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row['elapsed_ms'] = (time.perf_counter() - start) * 1000
            # Worker requests are persisted after their case joins every peer.
            # Serializing while another worker mutates a row would race.
            if threading.current_thread() is threading.main_thread():
                self.save()

    def call(self, url, *, label, method='POST', **kwargs):
        with self.request(url, label=label, method=method, **kwargs) as (response, row, path):
            chunks = []
            for chunk in response.iter_bytes():
                if not chunks:
                    row['first_body_ms'] = (time.perf_counter() - response.extensions['functional_started']) * 1000
                chunks.append(chunk)
            body = b''.join(chunks)
            path.write_bytes(body)
            return response.status_code, body, row

    def json(self, url, **kwargs):
        status, body, row = self.call(url, **kwargs)
        assert status == 200, (status, body[:1000])
        return json.loads(body)

    def reject(self, url, code, **kwargs):
        status, body, _ = self.call(url, **kwargs)
        parsed = json.loads(body)
        assert status == 400 and parsed['error']['code'] == code, (status, parsed)
        assert parsed['error']['message'] and parsed['error']['type'], parsed
