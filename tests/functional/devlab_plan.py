#!/usr/bin/env python3
"""Select model representatives per changed path, then plan their functional suites."""

import argparse
import fnmatch
import json
from pathlib import Path, PurePosixPath

import impact
from devlab_models import HERE, registry


def timing_checks(model, reasons):
    paths = [reason['path'] for reason in reasons if reason['path']]
    build = any(path.startswith(('cmake/', '.devops/')) or path in
                ('CMakeLists.txt', 'CMakePresets.json', 'flake.nix', 'flake.lock') for path in paths)
    production = [path for path in paths if path.startswith('src/')]
    unknown = any(reason['rule'] == 'unmapped-conservative-fallback' for reason in reasons)
    if not production and not build and not unknown:
        return []  # Harness-only changes still measure every phase, without gating unrelated noise.
    category = model['category']
    if category == 'llm':
        checks = ['*.cache_restore_ms', '*.prompt_ms', '*.predicted_ms', '*.cache_snapshot_ms']
    elif category == 'asr':
        checks = ['*.server_ms', '*.decode_audio_ms', '*.features_ms', '*.audio_encoder_ms', '*.text_decoder_ms']
    elif category == 'tts':
        checks = ['*.server_ms', '*.synthesis_ms']
    else:
        checks = ['*.server_ms']
    if build or unknown or any(any(part in path for part in ('loader', 'config', 'tensor', 'serve.cpp'))
                               for path in production):
        checks += ['startup.*']
    return checks


def select(paths, data, tier='pr', ar=False, benchmark=False):
    manifest = json.loads((HERE / 'impact.json').read_text())
    impact.validate(manifest, impact.suite_names(HERE / 'run.py'))
    selected, resolutions = {}, []
    for path in sorted(set(paths)):
        if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
            raise ValueError('Expected repository-relative path: ' + path)
        matches = [rule for rule in data['rules'] if any(
            fnmatch.fnmatchcase(path, pattern) for pattern in rule['paths'])]
        priority = max((rule['priority'] for rule in matches), default=-1)
        rules = [rule for rule in matches if rule['priority'] == priority] if matches else [
            {'id': 'unmapped-conservative-fallback', 'owners': list(data['representatives'])}]
        resolutions.append({'path': path, 'rules': [rule['id'] for rule in rules]})
        for rule in rules:
            for owner in rule['owners']:
                models = [data['representatives'].get(owner, owner)]
                if owner == 'tts' and tier == 'full':
                    models = [key for key, model in data['models'].items() if model['category'] == 'tts']
                for key in models:
                    selected.setdefault(key, []).append({'path': path, 'rule': rule['id'],
                        'representative': owner in data['representatives']})
    if benchmark:
        selected.setdefault('flash-next-q4', []).append(
            {'path': None, 'rule': 'manual-ar-benchmark', 'representative': False})
    # Models disabled in devlab.json stay out of the run; the plan names them
    # with the reason so the report shows the gap.
    disabled = [{'id': key, 'reason': data.get('disabled', {})[key], 'reasons': selected.pop(key)}
                for key in sorted(selected) if key in data.get('disabled', {})]
    result = {'schema': 2, 'tier': tier, 'paths': resolutions, 'models': [], 'groups': [],
              'deferred': [], 'release_only': [], 'out_of_scope': [], 'obligations': [],
              'disabled': disabled, 'scope': sorted(selected), 'benchmark': benchmark,
              'timing_qualification': 'paired main/candidate server phases; no averaging across requests'}
    for key, reasons in sorted(selected.items()):
        model = data['models'][key]
        result['models'].append({'id': key, 'category': model['category'], 'reasons': reasons,
                                 'timing_checks': timing_checks(model, reasons)})
        if model['category'] == 'llm':
            modes = [model['mode']] + (['off'] if ar else [])
            profiles = [key + '/' + mode for mode in modes]
            affected = [reason['path'] for reason in reasons if reason['path']]
            # Benchmark-only dispatches still receive a bounded correctness plan.
            if not affected:
                affected = ['src/models/qwen38_flash_next/engine.cpp']
            planned = impact.plan(manifest, affected, profiles, tier)
            for field in ('groups', 'deferred', 'release_only', 'out_of_scope', 'obligations'):
                result[field].extend(planned[field])
        else:
            suites = {'embedding': ['embedding-text', 'embedding-images'],
                      'asr': ['asr-http'], 'tts': ['tts-http']}[model['category']]
            result['groups'].append({'profile': key + '/native', 'configuration': 'standard',
                                     'settings': {}, 'suites': suites})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--base')
    parser.add_argument('--head')
    parser.add_argument('--tier', choices=('pr', 'full'), default='pr')
    parser.add_argument('--ar', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any((args.repo, args.base, args.head)):
        if not all((args.repo, args.base, args.head)) or args.paths:
            parser.error('Use either paths or --repo/--base/--head')
        paths = impact.changed_paths(args.repo, args.base, args.head)
    else:
        paths = args.paths
    planned = select(paths, registry(), args.tier, args.ar, args.benchmark)
    planned.update(base=args.base, head=args.head)
    args.output.write_text(json.dumps(planned, indent=2) + '\n')


if __name__ == '__main__':
    main()
