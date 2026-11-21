"""Real-model music jobs: discovery, errors, plan-only, score edits, audio, replay and cancellation."""

import hashlib
import io
import json
import math
import struct
import time
import wave


STYLE = 'English, upbeat synth pop, bright female voice, drums and bass, 120 BPM'
LYRICS = '[Verse]\nCity lights are calling me\n[Chorus]\nDance until the morning'
# Short songs: `seconds` is a maximum, so 8 s stops the semantic stage at 200
# codec frames (the model's minimum) and keeps every song under a minute.
SECONDS = 8
SEED = 5
TERMINAL = ('completed', 'failed', 'cancelled')


def music_check(data, maximum_seconds):
    """Check a 48 kHz stereo PCM16 song: duration, level and no hard clipping."""
    with wave.open(io.BytesIO(data)) as audio:
        shape = audio.getnchannels(), audio.getsampwidth(), audio.getframerate()
        assert shape == (2, 2, 48000), ('WAV format', shape)
        frames = audio.getnframes()
        pcm = audio.readframes(frames)
    assert len(pcm) == frames * 4 and frames, 'empty or truncated WAV payload'
    samples = struct.unpack('<' + 'h' * (len(pcm) // 2), pcm)
    duration = frames / 48000
    rms = math.sqrt(sum((sample / 32768) ** 2 for sample in samples) / len(samples))
    clipped = sum(abs(sample) >= 32767 for sample in samples) / len(samples)
    assert 1 <= duration <= maximum_seconds + 1e-6, ('song duration', duration)
    assert rms > .005, ('silent song', rms)
    assert clipped < .01, ('clipped song', clipped)
    return {'duration_seconds': duration, 'rms': rms, 'clipped_fraction': clipped,
            'pcm_sha256': hashlib.sha256(pcm).hexdigest()}


def run(evidence, url, model, log_path, timeout=900):
    endpoint = url + '/v1/audio/music'

    def create(label, body):
        status, data, _ = evidence.call(endpoint, label=label, json={'model': model, **body})
        assert status == 202, (status, data[:1000])
        job = json.loads(data)
        assert job['object'] == 'audio.music' and job['status'] in ('queued', 'in_progress'), job
        return job['id']

    def status(label, job_id):
        return evidence.json(f'{endpoint}/{job_id}', label=label, method='GET')

    def wait(label, job_id, until=TERMINAL):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = status(label + '-poll', job_id)
            if job['status'] in until:
                return job
            time.sleep(2)
        raise AssertionError(f'{label}: job {job_id} did not reach {until} in {timeout} s')

    def download(label, job_id, resource, **kwargs):
        status_code, data, row = evidence.call(f'{endpoint}/{job_id}/{resource}', label=label,
                                               method='GET', **kwargs)
        return status_code, data, row

    with evidence.case('music-capabilities'):
        models = evidence.json(url + '/v1/models', label='models-music', method='GET')
        entry = next(item for item in models['data'] if item['id'] == model)
        music = entry.get('gufo', {}).get('music')
        assert music, entry
        assert music['duration'] == 'maximum' and music['max_seconds'] == 360, music
        assert 'abc' in music['score_formats'] and music['score_output'], music
        assert music['plan_only'] and music['tempo_key_meter'] == 'unsupported', music
        evidence.report['music_capabilities'] = music

    with evidence.case('rejected-requests'):
        evidence.reject(endpoint, 'unsupported_parameter', label='bpm-field',
                        json={'model': model, 'prompt': STYLE, 'gufo': {'bpm': 120}})
        evidence.reject(endpoint, 'unsupported_parameter', label='unknown-field',
                        json={'model': model, 'prompt': STYLE, 'colour': 1})
        evidence.reject(endpoint, 'unsupported_parameter', label='seconds-over-maximum',
                        json={'model': model, 'prompt': STYLE, 'seconds': 400})
        evidence.reject(endpoint, 'invalid_request', label='instrumental-with-lyrics',
                        json={'model': model, 'prompt': STYLE, 'instrumental': True, 'lyrics': 'x'})
        evidence.reject(endpoint, 'invalid_request', label='provider-option',
                        json={'model': model, 'prompt': STYLE,
                              'gufo': {'options': {'planning': 'sideways'}}})
        status_code, data, _ = evidence.call(endpoint, label='unknown-model',
                                             json={'model': 'nope', 'prompt': STYLE})
        assert status_code == 404 and json.loads(data)['error']['code'] == 'model_not_found', data
        status_code, _, _ = evidence.call(f'{endpoint}/music_{"0" * 24}', label='unknown-job',
                                          method='GET')
        assert status_code == 404

    with evidence.case('plan-only'):
        plan = create('plan-only', {'prompt': STYLE, 'lyrics': LYRICS,
                                    'gufo': {'plan_only': True, 'seed': SEED,
                                             'options': {'abc_sampling': {'max_tokens': 300}}}})
        job = wait('plan-only', plan)
        assert job['status'] == 'completed', job
        assert job['gufo']['has_score'] and not job['gufo']['has_audio'], job
        assert job['seconds'] is None, job
        status_code, score, row = download('plan-only-score', plan, 'score')
        assert status_code == 200 and row['headers']['content-type'].startswith('text/vnd.abc'), row
        score = score.decode()
        assert score.startswith('X:') and 'V:' in score, score[:200]
        status_code, _, _ = download('plan-only-content', plan, 'content')
        assert status_code == 404, 'a plan-only job has no audio'

    song_request = {'prompt': STYLE, 'lyrics': LYRICS, 'seconds': SECONDS,
                    'gufo': {'seed': SEED, 'score': {'format': 'abc', 'text': score}}}
    with evidence.case('song-from-edited-score'):
        song = create('song', song_request)
        job = wait('song', song)
        assert job['status'] == 'completed' and job['gufo']['has_audio'], job
        assert job['gufo']['finish'] in ('truncated', 'complete'), job
        assert 0 < job['seconds'] <= SECONDS, job
        status_code, wav, row = download('song-content', song, 'content')
        assert status_code == 200 and row['headers']['content-type'] == 'audio/wav', row
        measures = music_check(wav, SECONDS)
        assert abs(measures['duration_seconds'] - job['seconds']) < 1e-3, (measures, job)
        row['audio'] = measures
        status_code, played, _ = download('song-score', song, 'score')
        assert status_code == 200 and played.decode() == score, 'the song played another score'
        evidence.report['song'] = {'job': job, 'audio': measures}

    with evidence.case('range-requests'):
        status_code, head, row = download('range-head', song, 'content',
                                          headers={'Range': 'bytes=0-43'})
        assert status_code == 206 and head == wav[:44], row
        assert row['headers']['content-range'] == f'bytes 0-43/{len(wav)}', row
        status_code, tail, _ = download('range-tail', song, 'content',
                                        headers={'Range': 'bytes=-100'})
        assert status_code == 206 and tail == wav[-100:]
        status_code, _, row = download('range-unsatisfiable', song, 'content',
                                       headers={'Range': f'bytes={len(wav)}-'})
        assert status_code == 416 and row['headers']['content-range'] == f'bytes */{len(wav)}', row

    with evidence.case('seeded-replay'):
        replay = create('replay', song_request)
        job = wait('replay', replay)
        assert job['status'] == 'completed', job
        status_code, again, _ = download('replay-content', replay, 'content')
        assert status_code == 200 and again == wav, 'same request and seed changed the song'

    with evidence.case('cancel-by-delete'):
        cancelled = create('cancel', {'prompt': STYLE, 'lyrics': LYRICS,
                                      'gufo': {'seed': SEED, 'options': {'planning': 'off'}}})
        wait('cancel', cancelled, until=('in_progress',) + TERMINAL)
        # Let the job enter generation, then delete it mid-stage.
        time.sleep(3)
        deleted = evidence.json(f'{endpoint}/{cancelled}', label='cancel-delete', method='DELETE')
        assert deleted['deleted'] and deleted['id'] == cancelled, deleted
        status_code, _, _ = evidence.call(f'{endpoint}/{cancelled}', label='cancel-gone',
                                          method='GET')
        assert status_code == 404
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            lines = log_path.read_text(errors='replace').splitlines()
            if any(f'job={cancelled} ' in line and 'event=finished state=deleted' in line
                   for line in lines):
                break
            time.sleep(1)
        else:
            raise AssertionError(f'the server never stopped job {cancelled}')

    with evidence.case('recovery-after-cancel'):
        # A short instrumental after the cancellation: the worker is free again.
        after = create('after-cancel', {'prompt': STYLE, 'instrumental': True, 'seconds': SECONDS,
                                        'gufo': {'seed': SEED + 1,
                                                 'options': {'planning': 'off'}}})
        job = wait('after-cancel', after)
        assert job['status'] == 'completed', job
        status_code, data, _ = download('after-cancel-content', after, 'content')
        assert status_code == 200
        music_check(data, SECONDS)
        evidence.json(f'{endpoint}/{after}', label='after-cancel-delete', method='DELETE')
