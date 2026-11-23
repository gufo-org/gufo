#!/usr/bin/env python3
"""Run selected Devlab models sequentially; retain partial correctness and timing results."""

import argparse
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import cluster
from devlab_models import HERE, registry


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def service_args(key, binary, baseline, assets, output):
    model = registry()['models'][key]
    args = [sys.executable, str(HERE / 'model_services.py'), model['service'], '--binary', str(binary),
            '--model', assets[model['asset']], '--output', str(output)]
    if model['category'] == 'tts':
        args += ['--variant', model['variant'], '--asr-binary', str(baseline), '--asr-model', assets['asr']]
    if model.get('npu'):
        args += ['--npu']
    return args


BENCH_DEPTHS = '0,4096,8192,12288,16384,32768,65536,131072'


def jobs(plan, shard_id=None):
    """(selection, correctness groups, run timing, run GPU tests) per model; a shard runs only its part."""
    if shard_id is None:
        return [(selection, [group for group in plan['groups'] if group['profile'].split('/')[0] == selection['id']],
                 selection['category'] != 'loom', bool(selection.get('ctest'))) for selection in plan['models']]
    shard = next((row for row in plan.get('shards', []) if row['id'] == shard_id), None)
    if shard is None:
        raise ValueError('Unknown shard: ' + shard_id)
    selections = {row['id']: row for row in plan['models']}
    return [(selections[part['model']], part['groups'], part['timing'], part.get('ctest', False))
            for part in shard['parts']]


CTEST = HERE.parents[1] / 'tools/ci/ctest_bundle.py'


def ctest_args(selection, bundle, assets, output):
    """ctest_bundle.py run for the model's labels; model paths come from its pinned assets."""
    model = registry()['models'][selection['id']]
    paths = {'target': lambda: Path(assets[model['asset']]) / model['target'],
             'draft': lambda: Path(assets[model['draft_asset']]) / model['draft'],
             'asset': lambda: Path(assets[model['asset']])}
    args = [sys.executable, str(CTEST), 'run', '--bundle', str(bundle), '--output', str(output)]
    for label in selection['ctest']['labels']:
        args += ['--label', label]
    for key, value in selection['ctest']['environment'].items():
        args += ['--env', f'{key}={paths[value]()}']
    return args


def benchmark(binary, baseline, assets, report):
    """Main first, then the PR, on the same device: native AR through 128K depth."""
    model = registry()['models']['flash-next-q4']
    weights = Path(assets[model['asset']]) / model['target']
    for revision, executable in (('base', baseline), ('head', binary)):
        with (report / f'bench-{revision}.md').open('w') as out, \
                (report / f'bench-{revision}.stderr').open('w') as err:
            subprocess.run([str(executable), 'bench', '--model', str(weights), '--speculative', 'off',
                            '--n-prompt', '2048', '--n-gen', '128', '--n-depth', BENCH_DEPTHS,
                            '--repetitions', '1'], stdout=out, stderr=err, timeout=3600)


def execute(plan, binary, baseline, assets_path, output, report, logs, shard_id=None, check_binary=None,
            ctest_bundle=None):
    """Correctness runs on check_binary (assertions on) when given; timing compares release builds."""
    checked = check_binary or binary
    assets = json.loads(assets_path.read_text())
    data = registry()
    output.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    summary = {'schema': 2, 'groups': [], 'models': [], 'status': 'running',
               'correctness_binary': 'checked' if check_binary else 'release'}
    if shard_id is not None:
        summary['shard'] = shard_id

    def save():
        write(output / 'summary.json', summary)
        write(report / 'summary.json', summary)

    save()
    for selection, groups, timing, ctest in jobs(plan, shard_id):
        key = selection['id']
        model = data['models'][key]
        item = {'id': key, 'correctness': 'running' if groups or ctest else 'skipped',
                'timing': 'pending' if timing else 'skipped', 'reasons': selection['reasons']}
        summary['models'].append(item)
        save()
        if ctest:
            # The model's GPU tests (the checked package's CTest labels) count as
            # correctness. They run first, before servers churn the page cache
            # that demand-mapped weights read through.
            started = time.monotonic()
            try:
                if ctest_bundle is None:
                    raise ValueError('The plan selects GPU tests but no --ctest-bundle was given')
                destination = output / (key + '.ctest')
                args = ctest_args(selection, ctest_bundle, assets, destination)
                with (output / (key + '.ctest.log')).open('w') as log:
                    subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=3 * 3600)
                path = destination / 'results.json'
                observed = json.loads(path.read_text()) if path.exists() else {
                    'status': 'error', 'tests': [], 'error': cluster.tail(output / (key + '.ctest.log'))}
                item['ctest'] = {'labels': selection['ctest']['labels'], 'status': observed['status'],
                                 'tests': [{field: row[field] for field in ('name', 'status', 'seconds', 'log_tail')
                                            if field in row} for row in observed['tests']]}
                if observed.get('error'):
                    item['ctest']['error'] = observed['error']
                write(report / (key + '.ctest.json'), observed)
            except Exception:
                item['ctest'] = {'labels': selection['ctest']['labels'], 'status': 'error',
                                 'tests': [], 'error': traceback.format_exc()}
            item['ctest_s'] = round(time.monotonic() - started, 1)
            if item['ctest']['status'] != 'passed':
                item['correctness'] = 'failed' if item['ctest']['status'] == 'failed' else 'error'
            save()
        started = time.perf_counter()
        try:
            if not groups:
                pass
            elif model['category'] == 'loom':
                # One greedy answer and one forced tool call (host sampling from
                # the package's logits) through `gufo serve llm`.
                command = [sys.executable, str(HERE / 'family_smoke.py'), 'loom', '--binary', str(checked),
                           '--model', assets[model['asset']],
                           '--checkpoint', assets[model['checkpoint_asset']],
                           '--log-dir', str(output / key)]
                with (output / (key + '.log')).open('w') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                row = {**groups[0], 'status': 'passed' if result.returncode == 0 else 'failed',
                       'command': command, 'exit_code': result.returncode}
                if result.returncode != 0:
                    row.update(error='Loom smoke check failed', log_tail=cluster.tail(output / (key + '.log')))
                rows = [row]
            elif model['category'] == 'llm':
                subplan = {**plan, 'groups': groups}
                cluster.execute(subplan, checked, None, output / key, assets)
                rows = json.loads((output / key / 'summary.json').read_text())['groups']
            else:
                destination = output / key
                command = service_args(key, checked, baseline, assets, destination)
                if model['category'] == 'embedding':
                    command += ['--embedding-suite', 'smoke' if 'embedding-smoke' in groups[0]['suites'] else 'full']
                if model['category'] == 'image':
                    command += ['--image-suite', 'smoke' if 'image-smoke' in groups[0]['suites'] else 'full']
                with (output / (key + '.log')).open('w') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                path = destination / 'report.json'
                observed = json.loads(path.read_text()) if path.exists() else {}
                passed = result.returncode == 0 and observed.get('status') == 'passed'
                row = {**groups[0], 'status': 'passed' if passed else 'failed',
                       'command': command, 'exit_code': result.returncode}
                if not passed:
                    row.update(error=observed.get('error', 'Service runner did not produce a passing report'),
                               log_tail=cluster.tail(output / (key + '.log')))
                item['known_limitations'] = observed.get('known_limitations', [])
                rows = [row]
            if groups:
                summary['groups'].extend(rows)
                if not all(row['status'] == 'passed' for row in rows):
                    item['correctness'] = 'failed'
        except Exception:
            item.update(correctness='error', error=traceback.format_exc())
        if item['correctness'] == 'running':
            item['correctness'] = 'passed'
        item['correctness_s'] = round(time.perf_counter() - started, 1)
        save()
        if not timing:
            continue
        # Correctness failure must not hide the speed controls or other models.
        started = time.perf_counter()
        try:
            script = 'phases.py' if model['category'] == 'llm' else 'service_phases.py'
            destination = report / (key + '.phases.json')
            args = [sys.executable, str(HERE / script), '--model-id', key, '--assets', str(assets_path),
                    '--base-binary', str(baseline), '--head-binary', str(binary),
                    '--base-revision', plan['base'], '--head-revision', plan['head'],
                    '--output', str(destination), '--logs', str(logs / key)]
            with (output / (key + '.phases.log')).open('w') as log:
                result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
            observed = json.loads(destination.read_text()) if destination.exists() else {}
            repeats = observed.get('settings', {}).get('repeats', 0)
            passed = (result.returncode == 0 and bool(observed.get('phases')) and repeats >= 2
                      and observed.get('revisions') == {'base': plan['base'], 'head': plan['head']}
                      and all(sum(arm.get('revision') == side for arm in observed.get('arms', [])) == repeats
                              for side in ('base', 'head')))
            if passed:
                for phase in observed['phases']:
                    phase['checked'] = phase.get('checked', True) and any(fnmatch.fnmatchcase(phase['key'], pattern)
                                           for pattern in selection['timing_checks'])
                observed['timing_checks'] = selection['timing_checks']
                write(destination, observed)
            item['timing'] = 'passed' if passed else 'error'
            if not passed:
                item['timing_error'] = cluster.tail(output / (key + '.phases.log'))
        except Exception:
            item.update(timing='error', timing_error=traceback.format_exc())
        item['timing_s'] = round(time.perf_counter() - started, 1)
        save()
        # The manual AR depth benchmark runs on the device that timed Flash-Next.
        if plan.get('benchmark') and key == 'flash-next-q4':
            benchmark(binary, baseline, assets, report)
    failed = any(item[field] not in ('passed', 'skipped') for item in summary['models']
                 for field in ('correctness', 'timing'))
    summary['status'] = 'failed' if failed else 'passed'
    save()
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'binary', 'base-binary', 'assets', 'output', 'report', 'logs'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--shard', help="Run only this entry of the plan's shards")
    parser.add_argument('--check-binary', type=Path,
                        help='Assertion-enabled candidate for correctness; --binary then only times')
    parser.add_argument('--ctest-bundle', type=Path,
                        help="The checked package's portable bundle, for models whose plan selects GPU tests")
    args = parser.parse_args()
    return execute(json.loads(args.plan.read_text()), args.binary.resolve(), args.base_binary.resolve(),
                   args.assets.resolve(), args.output.resolve(), args.report.resolve(), args.logs.resolve(),
                   args.shard, args.check_binary.resolve() if args.check_binary else None,
                   args.ctest_bundle.resolve() if args.ctest_bundle else None)


if __name__ == '__main__':
    sys.exit(main())
