#!/usr/bin/env python3
"""Merge sharded Devlab job results into the one summary report.py renders.

Each runner job writes `<results>/shards/<shard id>/summary.json` plus its
model's `<model>.phases.json` when it ran the paired timing arms. A shard
without a summary (the job died, was cancelled or never started) marks its
model's correctness or timing as an error, so missing evidence never passes.
"""

import argparse
import json
from pathlib import Path
import re
import shutil

ORDER = ('error', 'failed', 'running', 'pending', 'passed', 'skipped')


def worst(values):
    """The least successful state; skipped only when every part skipped."""
    values = list(values) or ['skipped']
    return min(values, key=lambda value: ORDER.index(value) if value in ORDER else 0)


def collect(plan, artifacts, results):
    """Copy each shard's newest attempt from functional-report-<shard>-<attempt>/."""
    for row in plan.get('shards', []):
        pattern = re.compile(re.escape('functional-report-' + row['id']) + r'-(\d+)')
        attempts = sorted((int(match[1]), path) for path in artifacts.glob('functional-report-*')
                          if (match := pattern.fullmatch(path.name)))
        if attempts:
            source = attempts[-1][1] / 'shards' / row['id']
            if source.is_dir():
                shutil.copytree(source, results / 'shards' / row['id'], dirs_exist_ok=True)


def merge(plan, results, output):
    shards = plan.get('shards', [])
    found = {}
    for row in shards:
        path = results / 'shards' / row['id'] / 'summary.json'
        found[row['id']] = json.loads(path.read_text()) if path.is_file() else None
    output.mkdir(parents=True, exist_ok=True)
    models = {}
    rows = {}
    for row in shards:
        summary = found[row['id']]
        directory = results / 'shards' / row['id']
        if summary is None:
            for part in row['parts']:
                models.setdefault(part['model'], {'parts': []})['parts'].append(
                    {'correctness': 'error' if part['groups'] or part.get('ctest') else 'skipped',
                     'timing': 'error' if part['timing'] else 'skipped',
                     'error': f"Shard {row['id']} produced no summary"})
                for group in part['groups']:
                    rows.setdefault((group['profile'], group['configuration']), []).append(
                        {**group, 'status': 'failed', 'exit_code': None,
                         'error': f"Shard {row['id']} produced no summary"})
            continue
        for observed in summary['models']:
            models.setdefault(observed['id'], {'parts': []})['parts'].append(observed)
        for group in summary['groups']:
            rows.setdefault((group['profile'], group['configuration']), []).append(group)
        for part in row['parts']:
            source = directory / (part['model'] + '.phases.json')
            if part['timing'] and source.is_file():
                shutil.copyfile(source, output / source.name)
        for source in directory.glob('bench-*'):
            shutil.copyfile(source, output / source.name)
    merged = {'schema': 2, 'groups': [], 'models': [], 'status': 'passed',
              'shards': {row['id']: 'missing' if found[row['id']] is None else found[row['id']]['status']
                         for row in shards}}
    # Which candidate build the correctness checks ran on (assertions or release).
    builds = {summary.get('correctness_binary', 'release') for summary in found.values() if summary}
    if builds:
        merged['correctness_binary'] = 'checked' if builds == {'checked'} else 'release'
    for selection in plan['models']:
        key = selection['id']
        parts = models.get(key, {'parts': []})['parts']
        item = {'id': key, 'reasons': selection['reasons'],
                'correctness': worst(part['correctness'] for part in parts
                                     if part['correctness'] != 'skipped'),
                'timing': worst(part['timing'] for part in parts if part['timing'] != 'skipped')}
        for field in ('error', 'timing_error'):
            text = '\n'.join(part[field] for part in parts if part.get(field))
            if text:
                item[field] = text
        ctest = [part['ctest'] for part in parts if part.get('ctest')]
        if ctest:
            item['ctest'] = ctest[0]
        elif selection.get('ctest'):
            item['ctest'] = {'labels': selection['ctest']['labels'], 'status': 'error', 'tests': [],
                             'error': 'GPU test results are missing'}
            item['correctness'] = 'error' if item['correctness'] == 'passed' else item['correctness']
        limitations = [text for part in parts for text in part.get('known_limitations', [])]
        if limitations:
            item['known_limitations'] = list(dict.fromkeys(limitations))
        merged['models'].append(item)
    for group in plan['groups']:
        observed = rows.get((group['profile'], group['configuration']), [])
        suites = [suite for row in observed for suite in row['suites']]
        failures = {name: error for row in observed for name, error in (row.get('failures') or {}).items()}
        row = {'profile': group['profile'], 'configuration': group['configuration'],
               'suites': group['suites'],
               'status': 'passed' if observed and sorted(suites) == sorted(group['suites'])
               and all(part['status'] == 'passed' for part in observed) else 'failed',
               'exit_code': next((part.get('exit_code') for part in observed if part.get('exit_code')), 0),
               'command': [part.get('command') for part in observed]}
        if row['status'] != 'passed':
            errors = [part['error'] for part in observed if part.get('error')]
            if sorted(suites) != sorted(group['suites']):
                errors.append('Missing suites: ' + ', '.join(sorted(set(group['suites']) - set(suites))))
            row.update(error='\n'.join(errors) or None, failures=failures,
                       log_tail='\n'.join(part['log_tail'] for part in observed if part.get('log_tail')))
        merged['groups'].append(row)
    # Timing is skipped only for a model the plan leaves untimed: a timed part
    # passes or errors, and a missing shard errors.
    if any(item['correctness'] != 'passed' or item['timing'] not in ('passed', 'skipped')
           for item in merged['models']):
        merged['status'] = 'failed'
    (output / 'summary.json').write_text(json.dumps(merged, indent=2) + '\n')
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--results', type=Path, help='Directory holding shards/<id>/')
    source.add_argument('--artifacts', type=Path, help='Downloaded functional-report-<id>-<attempt>/ artifacts')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    results = args.results
    if args.artifacts:
        results = args.output / 'shard-inputs'
        collect(plan, args.artifacts, results)
    merge(plan, results, args.output)


if __name__ == '__main__':
    main()
