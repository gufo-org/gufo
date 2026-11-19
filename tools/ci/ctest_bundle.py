#!/usr/bin/env python3
"""Bundle CTest-registered GPU tests off their build tree, then run them on Devlab.

`targets` and `collect` run in the checked package build (.devops/nix/package.nix):
they select the tests of the requested labels whose command is an executable
built in the tree, copy those executables to libexec/gufo-tests and write
share/gufo/ctest/manifest.json with each test's arguments and CTest properties.
Fixture files that the tests' compile definitions name under the source tree
are copied to share/gufo/ctest/source; tests open them through
tests/support/source_path.hpp, which maps the compiled source root to
GUFO_TEST_SOURCE_DIR. Tests whose command is an interpreter (repository checks)
stay with the CPU checks. `run` executes a bundled label like `ctest -L`, from
the portable bundle that tools/ci/linux_artifact.py packs.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

MANIFEST = Path('share/gufo/ctest/manifest.json')
EXECUTABLES = Path('libexec/gufo-tests')
SOURCE = Path('share/gufo/ctest/source')
# CTest properties the runner reproduces. Any other property would change a
# test's verdict or setup, so collection refuses it instead of ignoring it.
SUPPORTED = {'LABELS', 'RESOURCE_LOCK', 'RUN_SERIAL', 'SKIP_RETURN_CODE', 'TIMEOUT',
             'WORKING_DIRECTORY', 'ENVIRONMENT'}
DEFAULT_TIMEOUT_S = 1500  # CTest's default


def label_pattern(labels):
    if not labels or any(not re.fullmatch(r'[A-Za-z0-9_.+-]+', label) for label in labels):
        raise ValueError('Expected plain CTest labels')
    return '^(' + '|'.join(re.escape(label) for label in labels) + ')$'


def listed(build, labels):
    output = subprocess.check_output(['ctest', '--show-only=json-v1', '-L', label_pattern(labels)],
                                     cwd=build, text=True)
    return json.loads(output)['tests']


TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|\[(=*)\[(.*?)\]\2\]', re.S)
ADD_TEST = re.compile(r'^add_test\((.*)\)\s*$', re.M)


def registered(build):
    """{name: command} from the generated CTestTestfile.cmake files.

    CTest's JSON omits the command of a test whose executable is not built
    yet, so the build phase reads the registration itself.
    """
    commands = {}
    for path in Path(build).rglob('CTestTestfile.cmake'):
        for line in ADD_TEST.findall(path.read_text()):
            tokens = [match[0] if match[2] == '' and match[1] == '' else match[2]
                      for match in TOKEN.findall(line)]
            if len(tokens) >= 2:
                commands[tokens[0]] = tokens[1:]
    return commands


def selected(build, labels):
    """(test, executable) for each labelled test that runs an executable built in the tree."""
    build = Path(build).resolve()
    commands = registered(build)
    rows = []
    for test in listed(build, labels):
        command = test.get('command') or commands.get(test['name']) or []
        test = {**test, 'command': command}
        executable = Path(command[0]).resolve() if command else None
        if executable is None or not executable.is_relative_to(build):
            continue
        rows.append((test, executable))
    return rows


def properties(test):
    values = {row['name']: row['value'] for row in test.get('properties', [])}
    unsupported = sorted(set(values) - SUPPORTED)
    if unsupported:
        raise ValueError(f"{test['name']}: unsupported CTest properties {', '.join(unsupported)}")
    return values


def fixtures(build, source, targets):
    """Source-tree paths named by the compile definitions of the targets' own objects."""
    entries = json.loads((Path(build) / 'compile_commands.json').read_text())
    pattern = re.compile(r'-D\w+=\\*"(' + re.escape(str(source)) + r'/[^"\\]+)\\*"')
    found = set()
    for entry in entries:
        output = entry.get('output', '')
        if not any(f'CMakeFiles/{target}.dir/' in output for target in targets):
            continue
        command = entry.get('command') or ' '.join(entry.get('arguments', []))
        found.update(pattern.findall(command))
    return sorted(Path(path) for path in found)


def collect(build, source, labels, output):
    build, source = Path(build).resolve(), Path(source).resolve()
    rows = selected(build, labels)
    if not rows:
        raise ValueError('No built test executables carry the labels ' + ', '.join(labels))
    (output / EXECUTABLES).mkdir(parents=True, exist_ok=True)
    tests = []
    for test, executable in rows:
        values = properties(test)
        environment = {}
        for item in values.get('ENVIRONMENT', []):
            key, _, value = item.partition('=')
            if str(build) in value:
                raise ValueError(f"{test['name']}: environment names the build tree")
            environment[key] = value
        destination = output / EXECUTABLES / executable.name
        if not destination.exists():
            shutil.copy2(executable, destination)
        tests.append({'name': test['name'], 'executable': (EXECUTABLES / executable.name).as_posix(),
                      'args': test['command'][1:], 'labels': values.get('LABELS', []),
                      'timeout_s': values.get('TIMEOUT', DEFAULT_TIMEOUT_S),
                      'skip_return_code': values.get('SKIP_RETURN_CODE'),
                      'environment': environment})
    copied = []
    for path in fixtures(build, source, sorted({executable.name for _, executable in rows})):
        if not path.exists():
            raise FileNotFoundError(path)
        relative = path.relative_to(source)
        destination = output / SOURCE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)
        copied.append(relative.as_posix())
    manifest = {'schema': 1, 'labels': sorted(labels), 'source_dir': str(source),
                'fixtures': copied, 'tests': tests}
    (output / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Bundled {len(tests)} tests, {len({t['executable'] for t in tests})} executables, "
          f"{len(copied)} fixtures for {', '.join(labels)}", flush=True)
    return manifest


def run(bundle, labels, environment, output, launchers='tests'):
    """Run the bundle's tests of these labels in CTest order; skips are reported, not passed over."""
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / MANIFEST).read_text())
    if manifest['schema'] != 1:
        raise ValueError('Unknown test bundle schema')
    wanted = set(labels)
    tests = [test for test in manifest['tests'] if wanted & set(test['labels'])]
    if not tests:
        raise ValueError('The bundle has no tests labelled ' + ', '.join(labels))
    output.mkdir(parents=True, exist_ok=True)
    results = {'schema': 1, 'labels': sorted(labels), 'tests': [], 'status': 'running'}
    for test in tests:
        name = test['name']
        log = output / (re.sub(r'[^A-Za-z0-9_.-]', '_', name) + '.log')
        launcher = bundle / launchers / Path(test['executable']).name
        env = {**os.environ, **test['environment'], **environment,
               'GUFO_TEST_SOURCE_DIR': str(bundle / SOURCE)}
        started = time.monotonic()
        row = {'name': name, 'command': [str(launcher), *test['args']]}
        print(f'[{name}] start', flush=True)
        with tempfile.TemporaryDirectory(dir=output) as directory, log.open('w') as stream:
            try:
                result = subprocess.run(row['command'], cwd=directory, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=test['timeout_s'])
                row['exit_code'] = result.returncode
                if result.returncode == 0:
                    row['status'] = 'passed'
                elif test.get('skip_return_code') is not None and result.returncode == test['skip_return_code']:
                    row['status'] = 'skipped'
                else:
                    row['status'] = 'failed'
            except subprocess.TimeoutExpired:
                row.update(status='failed', exit_code=None, error=f"timed out after {test['timeout_s']} s")
        row['seconds'] = round(time.monotonic() - started, 1)
        if row['status'] != 'passed':
            row['log_tail'] = tail(log)
        print(f"[{name}] {row['status']} in {row['seconds']} s", flush=True)
        results['tests'].append(row)
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    results['status'] = 'failed' if any(row['status'] == 'failed' for row in results['tests']) else 'passed'
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


def tail(path, lines=40):
    text = path.read_text(errors='replace').splitlines() if path.exists() else []
    return '\n'.join(text[-lines:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    targets = commands.add_parser('targets', help='Print the build targets of the labelled tests')
    targets.add_argument('--build', type=Path, required=True)
    targets.add_argument('--label', action='append', required=True)
    gather = commands.add_parser('collect', help='Copy the labelled tests and their fixtures')
    gather.add_argument('--build', type=Path, required=True)
    gather.add_argument('--source', type=Path, required=True)
    gather.add_argument('--label', action='append', required=True)
    gather.add_argument('--output', type=Path, required=True)
    execute = commands.add_parser('run', help='Run the labelled tests of a portable bundle')
    execute.add_argument('--bundle', type=Path, required=True)
    execute.add_argument('--label', action='append', required=True)
    execute.add_argument('--env', action='append', default=[], metavar='KEY=VALUE')
    execute.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'targets':
        print('\n'.join(sorted({executable.name for _, executable in selected(args.build, args.label)})))
    elif args.command == 'collect':
        collect(args.build, args.source, args.label, args.output)
    else:
        environment = dict(item.split('=', 1) for item in args.env)
        return int(run(args.bundle, args.label, environment, args.output)['status'] != 'passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
