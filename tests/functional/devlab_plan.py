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
    elif category == 'image':
        checks = ['*.server_ms', '*.prompt_ms', '*.denoise_ms', '*.vae_ms']
    else:
        checks = ['*.server_ms']
    if build or unknown or any(any(part in path for part in ('loader', 'config', 'tensor', 'serve.cpp'))
                               for path in production):
        checks += ['startup.*']
    return checks


def curated(planned, suites):
    rows = {'groups': [], 'out_of_scope': list(planned['out_of_scope']), 'obligations': planned['obligations']}
    for group in planned['groups']:
        kept = [suite for suite in group['suites'] if suite in suites]
        rows['out_of_scope'].extend({'profile': group['profile'], 'suite': suite,
                                     'configuration': group['configuration']}
                                    for suite in group['suites'] if suite not in suites)
        if kept:
            rows['groups'].append({**group, 'suites': kept})
    for field in ('deferred', 'release_only'):
        rows[field] = [row for row in planned[field] if row['suite'] in suites]
        rows['out_of_scope'].extend(row for row in planned[field] if row['suite'] not in suites)
    return rows


def select(paths, data, tier='pr', ar=False, benchmark=False):
    manifest = json.loads((HERE / 'impact.json').read_text())
    names = impact.suite_names(HERE / 'suites.py')
    impact.validate(manifest, names)
    for key, model in data['models'].items():
        if not set(model.get('suites', names)) <= set(names):
            raise ValueError('Unknown Devlab suite for ' + key)
        if not set(model.get('settings', {})) <= set(manifest['configurations']):
            raise ValueError('Unknown Devlab configuration for ' + key)
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
                # A category owner covers every model in it, including NPU and
                # speculative variants; runner shards keep the wall time flat.
                # Image generation keeps one representative outside the full tier.
                if owner in data['representatives'] and (owner != 'image' or tier == 'full'):
                    models = [key for key, model in data['models'].items() if model['category'] == owner]
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
    # Every PR runs the release suite set, sharded across devices; the tier
    # only keeps image generation to one smoke-tested representative on PRs.
    result = {'schema': 2, 'tier': tier, 'suite_tier': 'full', 'paths': resolutions, 'models': [], 'groups': [],
              'deferred': [], 'release_only': [], 'out_of_scope': [], 'obligations': [],
              'disabled': disabled, 'scope': sorted(selected), 'benchmark': benchmark,
              'timing_qualification': 'paired main/candidate server phases; no averaging across requests'}
    for key, reasons in sorted(selected.items()):
        model = data['models'][key]
        result['models'].append({'id': key, 'category': model['category'], 'reasons': reasons,
                                 'timing_checks': timing_checks(model, reasons)})
        # The model's CTest labels (GPU tests from the checked package) run when
        # its private paths change, and on release runs.
        config = model.get('ctest')
        if config and (tier == 'full' or any(reason['rule'] in config['rules'] for reason in reasons)):
            result['models'][-1]['ctest'] = {'labels': config['labels'], 'environment': config['environment']}
        if model['category'] == 'llm':
            modes = [model['mode']] + (['off'] if ar else [])
            profiles = [key + '/' + mode for mode in modes]
            affected = [reason['path'] for reason in reasons if reason['path']]
            # Benchmark-only dispatches still receive a bounded correctness plan.
            if not affected:
                affected = ['src/models/qwen38_flash_next/engine.cpp']
            planned = impact.plan(manifest, affected, profiles, result['suite_tier'])
            if 'suites' in model:
                planned = curated(planned, set(model['suites']))
            # A model overrides a configuration's server settings where the
            # shared value does not create the condition a suite tests, such
            # as a RAM budget sized for another model's checkpoints.
            overrides = model.get('settings', {})
            planned['groups'] = [{**group, 'settings': {**group['settings'],
                                                        **overrides.get(group['configuration'], {})}}
                                 for group in planned['groups']]
            for field in ('groups', 'deferred', 'release_only', 'out_of_scope', 'obligations'):
                result[field].extend(planned[field])
        else:
            suites = {'embedding': ['embedding-text', 'embedding-images', 'embedding-audio', 'embedding-video'],
                      'asr': ['asr-http'], 'tts': ['tts-http'],
                      'image': ['image-smoke' if tier == 'pr' else 'image-http'],
                      'loom': ['loom-smoke']}[model['category']]
            result['groups'].append({'profile': key + ('/npu' if model.get('npu') else '/native'),
                                     'configuration': 'standard', 'settings': {}, 'suites': suites})
    return result


COSTS = HERE / 'devlab_costs.json'
# Wall time one runner job may take, excluding package and weight setup. Jobs
# split by model first; a model splits further only past this budget.
SHARD_BUDGET_S = 30 * 60


def costs():
    data = json.loads(COSTS.read_text())
    assert data['schema'] == 1
    return data


def suite_cost(table, model, suite):
    model_costs = table['models'].get(model, {})
    return model_costs.get('suites', {}).get(suite, table['default_suite_s'])


def pack(suites, cost, capacity):
    """First-fit decreasing; each bin keeps the plan's suite order."""
    bins = []
    for suite in sorted(suites, key=lambda name: -cost(name)):
        target = next((row for row in bins if row['seconds'] + cost(suite) <= capacity), None)
        if target is None:
            target = {'suites': [], 'seconds': 0}
            bins.append(target)
        target['suites'].append(suite)
        target['seconds'] += cost(suite)
    for row in bins:
        row['suites'].sort(key=suites.index)
    return bins


def units(plan, table, budget, isolation):
    """Indivisible pieces of work: one model's servers for one job, or its timing arms."""
    rows = []
    for selection in plan['models']:
        key = selection['id']
        model_costs = table['models'].get(key, {})
        timing = model_costs.get('timing_s', table['default_timing_s'])
        if selection.get('ctest'):
            # GPU tests run first, on the device that then serves the model's suites.
            rows.append({'label': key + '-ctest', 'pool': key if selection['category'] == 'llm' else 'services',
                         'seconds': model_costs.get('ctest_s', table.get('default_ctest_s', 900)),
                         'part': {'model': key, 'groups': [], 'timing': False, 'ctest': True}})
        groups = [group for group in plan['groups'] if group['profile'].split('/')[0] == key]
        if selection['category'] != 'llm':
            # One service command runs every suite; its timing arms follow on the same device.
            seconds = sum(suite_cost(table, key, suite) for group in groups for suite in group['suites'])
            # Loom packages have a smoke check only, no timing arms.
            timed = selection['category'] != 'loom'
            rows.append({'label': key, 'pool': 'services', 'seconds': seconds + (timing if timed else 0),
                         'part': {'model': key, 'groups': groups, 'timing': timed}})
            continue
        # A job's first server reads cold weights; the rest load from the page cache.
        load = model_costs.get('warm_load_s', model_costs.get('load_s', table['default_load_s']))
        cost = lambda suite: suite_cost(table, key, suite)  # noqa: E731
        for group in groups:
            mode = group['profile'].split('/')[1]
            shared = [suite for suite in group['suites'] if suite not in isolation]
            servers = pack(shared, cost, max(budget - load, 1)) + [
                {'suites': [suite], 'seconds': cost(suite)} for suite in group['suites']
                if isolation.get(suite) == 'server']
            # Suites in a named pool share one server apart from the rest.
            for pool in sorted({isolation[suite] for suite in group['suites']
                                if isolation.get(suite, 'server') != 'server'}):
                members = [suite for suite in group['suites'] if isolation.get(suite) == pool]
                servers.append({'suites': members, 'seconds': sum(cost(suite) for suite in members)})
            for number, server in enumerate(servers, 1):
                server['seconds'] += load
                rows.append({'label': f"{key}-{mode}-{group['configuration']}", 'pool': key,
                             'cold': model_costs.get('load_s', table['default_load_s']) - load,
                             'seconds': server['seconds'],
                             'part': {'model': key, 'timing': False, 'groups': [
                                 {**group, 'suites': server['suites'],
                                  **({'server': number} if len(servers) > 1 else {})}]}})
        rows.append({'label': key + '-timing', 'pool': key, 'seconds': timing,
                     'part': {'model': key, 'groups': [], 'timing': True}})
    return rows


def shard(plan, table=None, budget=SHARD_BUDGET_S):
    """Split a plan into independent runner jobs, one device each.

    LLM groups become servers whose suites fit the budget; suites that read
    server-wide cache state get a server of their own. Each LLM's paired
    timing arms stay together on one device. Work then packs into jobs up to
    the budget: one LLM's pieces share a job, and service models share theirs.
    """
    table = table or costs()
    manifest = json.loads((HERE / 'impact.json').read_text())
    # These suites read server-wide cache state (RAM budget, disk turnover,
    # shared-prefix leaders) that earlier suites change; each gets a new server.
    # "server": a server of its own; another name: one server shared by that pool.
    isolation = {name: suite['isolation'] for name, suite in manifest['suites'].items()
                 if suite.get('isolation', 'shared') != 'shared'}
    models = registry()['models']
    pools = {}
    for row in units(plan, table, budget, isolation):
        pools.setdefault(row['pool'], []).append(row)
    shards = []
    for pool, rows in pools.items():
        # A model with max_jobs 1 keeps all its work on one device, past the budget.
        limit = budget if models.get(pool, {}).get('max_jobs') != 1 else float('inf')
        jobs = []
        for row in sorted(rows, key=lambda row: -row['seconds']):
            job = next((job for job in jobs if job['seconds'] + row['seconds'] <= limit), None)
            if job is None:
                # The job's first server pays the cold load once.
                job = {'rows': [], 'seconds': max((r.get('cold', 0) for r in rows), default=0)}
                jobs.append(job)
            job['rows'].append(row)
            job['seconds'] += row['seconds']
        for index, job in enumerate(jobs, 1):
            name = pool if len(jobs) == 1 else f'{pool}-{index}'
            parts = []
            # A model's parts merge into one, so it loads its assets and runs its timing once.
            for row in job['rows']:
                part = next((part for part in parts if part['model'] == row['part']['model']), None)
                if part is None:
                    parts.append({**row['part'], 'groups': list(row['part']['groups'])})
                else:
                    part['groups'] += row['part']['groups']
                    part['timing'] = part['timing'] or row['part']['timing']
                    part['ctest'] = part.get('ctest', False) or row['part'].get('ctest', False)
            shards.append({'id': name, 'parts': parts, 'seconds': job['seconds']})
    ids = [row['id'] for row in shards]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate shard ids: ' + ', '.join(ids))
    return shards


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
    planned.update(base=args.base, head=args.head, shards=shard(planned),
                   # The hosted checked build bundles these labels' test executables.
                   ctest_labels=sorted({label for model in planned['models']
                                        for label in model.get('ctest', {}).get('labels', [])}))
    # Suites run in a per-revision order: an order dependence surfaces as a
    # failure that reproduces from the seed in the plan and each report.
    planned['order_seed'] = int(args.head[:8], 16) if args.head and all(
        char in '0123456789abcdef' for char in args.head[:8]) else None
    args.output.write_text(json.dumps(planned, indent=2) + '\n')


if __name__ == '__main__':
    main()
