#!/usr/bin/env python3
"""Paired main/PR server timings for embedding, ASR and TTS workloads."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

from devlab_models import HERE, registry, service_command
from phases import Server, cool_down, gpu_temperature, verdict
from metrics import NOISE_MS, SLOWDOWN


REQUESTS = {'short': 'Short input', 'second': 'Second input', 'repeat': 'Warm replay'}


def completed_log(path, request_id):
    for line in path.read_text().splitlines():
        if f'request={request_id} ' in line and 'event=completed ' in line:
            return dict(re.findall(r'(\w+)=([^\s]+)', line))
    raise ValueError('Missing server completion metrics: ' + request_id)


def number(value, name):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid server metric: ' + name)
    return result


def measure_arm(model_id, binary, assets, output, audio_root):
    from service_http import Evidence, audio_check
    output.mkdir(parents=True, exist_ok=False)
    model = registry()['models'][model_id]
    report = {'status': 'running', 'model': model_id}
    evidence = Evidence(output, report)
    launch = service_command(model_id, binary, assets, audio_root, 'phases')
    server = Server(binary, None, output / 'server.log', launch)
    rows = {}
    try:
        server.start(600)
        startup = server.startup()
        url = f'http://127.0.0.1:{server.port}/v1'
        for label in REQUESTS:
            if model['category'] == 'embedding':
                text = 'The river flows past the garden.' if label != 'second' else ' '.join(['river stone garden'] * 100)
                payload = evidence.json(url + '/embeddings', label=label,
                                        json={'model': 'phases', 'input': [text]})
                vector = payload['data'][0]['embedding']
                assert len(vector) == 768 and all(math.isfinite(x) for x in vector)
                work = {'prompt_tokens': payload['usage']['prompt_tokens'], 'inputs': len(payload['data'])}
            elif model['category'] == 'asr':
                payload = evidence.json(url + '/audio/transcriptions', label=label,
                    data={'model': 'phases', 'language': 'English', 'max_tokens': '128'},
                    files={'file': ('speech.wav', (audio_root / 'clear-english-voice.wav').read_bytes(), 'audio/wav')})
                assert payload['text'].strip(), payload
                work = {'text_sha256': hashlib.sha256(payload['text'].encode()).hexdigest()}
            else:
                text = ('The little brown dog ran across the garden.' if label != 'second'
                        else 'Please bring a glass of water to the table.')
                body = {'model': 'phases', 'voice': {'base': 'narrator', 'custom': 'ryan',
                    'design': 'voice-design'}[model['variant']], 'input': text, 'seed': 42,
                    'language': 'English', 'max_new_tokens': 180}
                if model['variant'] == 'design':
                    body['instructions'] = 'A clear English voice speaking calmly and naturally.'
                status, payload, _ = evidence.call(url + '/audio/speech', label=label, json=body)
                assert status == 200, payload[:1000]
                audio_check(payload)
                work = {}
            request = report['requests'][-1]
            rows[label] = {'request_id': request['headers']['x-request-id'],
                           'headers': request['headers'], 'work': work}
        report['status'] = 'passed'
    except BaseException:
        report['status'] = 'failed'
        raise
    finally:
        server.stop()
        evidence.save()
    for label, row in rows.items():
        log = completed_log(output / 'server.log', row['request_id'])
        row['metrics'] = {'server_ms': number(log['duration_ms'], 'duration_ms')}
        category = model['category']
        if category == 'asr':
            timing = dict(re.findall(r'([\w_]+);dur=([0-9.]+)', row['headers'].get('server-timing', '')))
            for phase in ('decode_audio', 'features', 'audio_encoder', 'text_decoder'):
                row['metrics'][phase + '_ms'] = number(timing[phase], phase)
            for key in ('audio_tokens', 'generated_tokens', 'chunks', 'audio_ms'):
                row['work'][key] = number(log[key], key)
        elif category == 'tts':
            row['metrics']['synthesis_ms'] = number(log['synthesis_ms'], 'synthesis_ms')
            for key in ('codec_steps', 'audio_ms'):
                row['work'][key] = number(log[key], key)
        else:
            for key in ('prompt_tokens', 'inputs', 'dimensions'):
                row['work'][key] = number(log[key], key)
        del row['headers']
    return {'startup': startup, 'requests': rows}


def summarize(arms):
    by = {side: [arm for arm in arms if arm['revision'] == side] for side in ('base', 'head')}
    if len(by['base']) != len(by['head']) or len(by['base']) < 2:
        raise ValueError('Missing paired repeats')
    phases, diverged = [], []
    for label in REQUESTS:
        comparable = all(base['requests'][label]['work'] == head['requests'][label]['work']
                         for base, head in zip(by['base'], by['head']))
        if not comparable:
            diverged.append(label)
        for metric in by['base'][0]['requests'][label]['metrics']:
            values = {side: [arm['requests'][label]['metrics'][metric] for arm in by[side]]
                      for side in by}
            phases.append({'group': label, 'key': label + '.' + metric, 'label': metric,
                           **values, 'verdict': verdict(list(zip(values['base'], values['head'])))
                           if comparable else 'noisy', 'comparable_work': comparable})
    startup = []
    for metric in ('init', 'weights', 'vision', 'structured', 'listen'):
        values = {side: [arm['startup'][metric] for arm in by[side]] for side in by}
        pairs = list(zip(values['base'], values['head']))
        startup.append({'group': 'startup', 'key': 'startup.' + metric, 'label': metric,
                         **values, 'verdict': verdict(pairs) if all(a is not None and b is not None
                                                                for a, b in pairs) else 'missing'})
    return startup + phases, diverged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('base-binary', 'head-binary', 'assets', 'logs', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    for option in ('model-id', 'base-revision', 'head-revision'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--audio-root', type=Path, default=HERE / 'fixtures/audio')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('at least two paired repeats are required')
    assets = json.loads(args.assets.read_text())
    args.logs.mkdir(parents=True, exist_ok=False)
    arms = []
    started = time.monotonic()
    idle = gpu_temperature()
    # Same thermal control as Flash-Next, with every reading retained.
    if idle is not None:
        time.sleep(30)
        idle = gpu_temperature()
    for repeat in range(args.repeats):
        for side in ('base', 'head'):
            temperature = cool_down(idle + 3) if idle is not None else None
            arm = measure_arm(args.model_id, getattr(args, side + '_binary').resolve(), assets,
                              args.logs / f'{side}-{repeat}', args.audio_root.resolve())
            arms.append({'revision': side, 'repeat': repeat, 'gpu_temp_c': temperature, **arm})
            # Retain completed arms even if a later arm fails.
            args.output.write_text(json.dumps({'status': 'incomplete', 'arms': arms}, indent=2) + '\n')
    phases, diverged = summarize(arms)
    result = {'schema': 1, 'status': 'passed', 'model': args.model_id,
              'revisions': {'base': args.base_revision, 'head': args.head_revision},
              'settings': {'repeats': args.repeats, 'slowdown': SLOWDOWN, 'noise_ms': NOISE_MS,
                           'requests': REQUESTS, 'idle_temp_c': idle},
              'wall_s': time.monotonic() - started, 'arms': arms,
              'phases': phases, 'diverged': diverged}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
