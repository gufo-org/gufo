#!/usr/bin/env python3
"""Refresh devlab_costs.json, the shard planner's per-suite seconds, from Devlab results.

Pass result directories written by devlab_run.py (a job's functional-results/
or a downloaded model-functional artifact). Observations replace the
matching entries; models and suites they do not cover keep their values.
"""

import argparse
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
COSTS = HERE / 'devlab_costs.json'
# run.py labels that run on behalf of one selected suite.
CACHE_PARTS = ('text-cancel', 'thinking-tool-cancel', 'legacy-tool-cancel', 'image-cancel',
               'image-thinking-cancel', 'cache-persist', 'disk-spacing')


def owner(label):
    """The selected suite a run.py label belongs to."""
    base = label[:-len('-disk')] if label.endswith('-disk') else label
    if base == 'image-count-cancel':
        return 'image-count'
    if base in CACHE_PARTS:
        return 'cache'
    return base


def observe(results):
    """{model: {'load_s': [...], 'suites': {suite: [...]}, 'timing_s': [...], 'ctest_s': [...]}} from one results tree."""
    observed = {}
    for report in sorted(results.rglob('report.json')):
        data = json.loads(report.read_text())
        if 'suites' not in data or 'startup_ms' not in data:
            continue
        model = report.parent.parent.name
        row = observed.setdefault(model, {'load_s': [], 'suites': {}, 'timing_s': [], 'ctest_s': []})
        row['load_s'].append(data['startup_ms'] / 1000)
        seconds = {}
        for label, suite in data['suites'].items():
            if suite.get('status') == 'passed' and isinstance(suite.get('seconds'), (int, float)):
                seconds[owner(label)] = seconds.get(owner(label), 0) + suite['seconds']
        if data.get('restart_ms') and 'cache' in seconds:
            seconds['cache'] += data['restart_ms'] / 1000
        for suite, value in seconds.items():
            row['suites'].setdefault(suite, []).append(value)
    for summary in sorted(results.rglob('summary.json')):
        data = json.loads(summary.read_text())
        for item in data.get('models', []) if isinstance(data, dict) else []:
            row = observed.setdefault(item['id'], {'load_s': [], 'suites': {}, 'timing_s': [], 'ctest_s': []})
            if item.get('timing') == 'passed' and isinstance(item.get('timing_s'), (int, float)):
                row['timing_s'].append(item['timing_s'])
            if item.get('ctest', {}).get('status') == 'passed' and isinstance(item.get('ctest_s'), (int, float)):
                row['ctest_s'].append(item['ctest_s'])
            # Service models run as one command; their suites share its time.
            if item.get('correctness') == 'passed' and not row['load_s'] and item.get('correctness_s'):
                groups = [group for group in data.get('groups', [])
                          if group['profile'].split('/')[0] == item['id']]
                suites = [suite for group in groups for suite in group['suites']]
                for suite in suites:
                    row['suites'].setdefault(suite, []).append(item['correctness_s'] / len(suites))
    return observed


def update(table, observed):
    for model, row in observed.items():
        entry = table['models'].setdefault(model, {})
        if row['load_s']:
            # A fresh device reads cold weights; a warm local rerun loads faster,
            # so keep the slowest load seen.
            entry['load_s'] = round(max(row['load_s'] + [entry.get('load_s', 0)]))
            entry['warm_load_s'] = round(min(row['load_s']))
        if row['timing_s']:
            entry['timing_s'] = round(statistics.median(row['timing_s']))
        if row.get('ctest_s'):
            entry['ctest_s'] = round(statistics.median(row['ctest_s']))
        suites = entry.setdefault('suites', {})
        for suite, values in row['suites'].items():
            suites[suite] = round(statistics.median(values), 1)
        entry['suites'] = dict(sorted(suites.items()))
    table['models'] = dict(sorted(table['models'].items()))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--costs', type=Path, default=COSTS)
    args = parser.parse_args()
    table = json.loads(args.costs.read_text())
    for results in args.results:
        table = update(table, observe(results))
    args.costs.write_text(json.dumps(table, indent=2) + '\n')


if __name__ == '__main__':
    main()
