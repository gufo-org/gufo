#!/usr/bin/env python3
"""Run selected Devlab models sequentially; retain partial correctness and timing results."""

import argparse
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
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
    return args


def execute(plan, binary, baseline, assets_path, output, report, logs):
    assets = json.loads(assets_path.read_text())
    data = registry()
    output.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    summary = {'schema': 2, 'groups': [], 'models': [], 'status': 'running'}

    def save():
        write(output / 'summary.json', summary)
        write(report / 'summary.json', summary)

    save()
    for selection in plan['models']:
        key = selection['id']
        model = data['models'][key]
        groups = [group for group in plan['groups'] if group['profile'].split('/')[0] == key]
        item = {'id': key, 'correctness': 'running', 'timing': 'pending', 'reasons': selection['reasons']}
        summary['models'].append(item)
        save()
        try:
            if model['category'] == 'llm':
                subplan = {**plan, 'groups': groups}
                cluster.execute(subplan, binary, None, output / key, assets)
                rows = json.loads((output / key / 'summary.json').read_text())['groups']
            else:
                destination = output / key
                command = service_args(key, binary, baseline, assets, destination)
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
            summary['groups'].extend(rows)
            item['correctness'] = 'passed' if all(row['status'] == 'passed' for row in rows) else 'failed'
        except Exception:
            item.update(correctness='error', error=traceback.format_exc())
        save()
        # Correctness failure must not hide the speed controls or other models.
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
                    phase['checked'] = any(fnmatch.fnmatchcase(phase['key'], pattern)
                                           for pattern in selection['timing_checks'])
                observed['timing_checks'] = selection['timing_checks']
                write(destination, observed)
            item['timing'] = 'passed' if passed else 'error'
            if not passed:
                item['timing_error'] = cluster.tail(output / (key + '.phases.log'))
        except Exception:
            item.update(timing='error', timing_error=traceback.format_exc())
        save()
    failed = any(item['correctness'] != 'passed' or item['timing'] != 'passed' for item in summary['models'])
    summary['status'] = 'failed' if failed else 'passed'
    save()
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'binary', 'base-binary', 'assets', 'output', 'report', 'logs'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    return execute(json.loads(args.plan.read_text()), args.binary.resolve(), args.base_binary.resolve(),
                   args.assets.resolve(), args.output.resolve(), args.report.resolve(), args.logs.resolve())


if __name__ == '__main__':
    sys.exit(main())
