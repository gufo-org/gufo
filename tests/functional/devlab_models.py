#!/usr/bin/env python3
"""Pinned Devlab model assets and production commands; downloads are plan-scoped."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import time

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / 'devlab.json'


def registry():
    data = json.loads(REGISTRY.read_text())
    assert data['schema'] == 1
    owners = set(data['models']) | set(data['representatives'])
    for rule in data['rules']:
        if not set(rule['owners']) <= owners:
            raise ValueError('Unknown owner in ' + rule['id'])
    for key, reason in data.get('disabled', {}).items():
        if key not in data['models'] or not reason:
            raise ValueError('Disabled models need a known id and a reason: ' + key)
    for name, asset in data['assets'].items():
        if not re.fullmatch('[0-9a-f]{40}', asset['revision']):
            raise ValueError('Unpinned asset: ' + name)
        for path in asset['required']:
            if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
                raise ValueError('Invalid asset path: ' + path)
    return data


def asset_keys(plan, data):
    keys = set()
    for selection in plan['models']:
        model = data['models'][selection['id']]
        keys.add(model['asset'])
        if 'draft_asset' in model:
            keys.add(model['draft_asset'])
        if model['category'] == 'tts':
            keys.add('asr')  # The main binary is the fixed intelligibility evaluator.
    return sorted(keys)


def download(plan, cache, data):
    from huggingface_hub import snapshot_download
    paths = {}
    for key in asset_keys(plan, data):
        asset = data['assets'][key]
        destination = cache / key / asset['revision']
        print('Download', key, asset['revision'], 'to', destination, flush=True)
        existing = sum((destination / name).is_file() for name in asset['required'])
        print(f'{key}: {existing}/{len(asset["required"])} required files already present', flush=True)
        started = time.monotonic()
        for attempt in range(3):
            try:
                snapshot_download(repo_id=asset['repo'], revision=asset['revision'],
                                  allow_patterns=asset['patterns'], local_dir=destination)
                break
            except Exception as error:
                print(f'{key}: attempt {attempt + 1} failed: {error}', flush=True)
                if attempt == 2:
                    raise
                time.sleep(2)
        for name in asset['required']:
            path = destination / name
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(path)
        paths[key] = str(destination.resolve())
        print(f'{key}: ready in {time.monotonic() - started:.1f} s', flush=True)
    return paths


def llm_command(model_id, binary, assets, *, mode=None, context=32768, sessions=1):
    model = registry()['models'][model_id]
    if model['category'] != 'llm':
        raise ValueError('Not an LLM: ' + model_id)
    root = Path(assets[model['asset']])
    command = [str(binary), 'serve', 'llm', '--model', str(root / model['target']),
               '--speculative', mode or model['mode'], '--context', str(context),
               '--sessions', str(sessions), model['draft_option'],
               str(Path(assets[model['draft_asset']]) / model['draft'])]
    if model.get('projector'):
        command += ['--mmproj', str(root / model['projector'])]
    return command


def service_command(model_id, binary, assets, audio_root, public_model):
    model = registry()['models'][model_id]
    command = [str(binary), 'serve', model['service'], '--model', str(assets[model['asset']]),
               '--served-model-name', public_model]
    if model.get('variant') == 'base':
        command += ['--voice', f'narrator={audio_root / "clear-english-voice.wav"}',
                    '--voice-text', f'narrator={audio_root / "clear-english-voice.txt"}']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = download(json.loads(args.plan.read_text()), args.cache.resolve(), registry())
    args.output.write_text(json.dumps(paths, indent=2) + '\n')


if __name__ == '__main__':
    main()
