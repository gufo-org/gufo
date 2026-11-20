#!/usr/bin/env python3
"""Bundled GPU tests keep their CTest selection, properties, fixtures and verdicts."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/ci'))
SPEC = importlib.util.spec_from_file_location(
    'ctest_bundle', Path(__file__).resolve().parents[2] / 'tools/ci/ctest_bundle.py')
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def prop(name, value):
    return {'name': name, 'value': value}


class CollectTest(unittest.TestCase):
    def tree(self, root):
        source, build = root / 'source', root / 'source/build'
        (source / 'tests/models/ds4/fixtures').mkdir(parents=True)
        (source / 'tests/models/ds4/fixtures/official.json').write_text('{}')
        (source / 'tests/fixtures').mkdir(parents=True)
        (source / 'tests/fixtures/goldens.json').write_text('[]')
        (source / 'tests/fixtures/unrelated.json').write_text('[]')
        (build / 'tests/models/ds4').mkdir(parents=True)
        for name in ('ds4_quality_test', 'ds4_cli_test'):
            (build / 'tests/models/ds4' / name).write_bytes(b'ELF ' + name.encode())
        (build / 'compile_commands.json').write_text(json.dumps([
            {'output': 'tests/models/ds4/CMakeFiles/ds4_quality_test.dir/quality_test.cpp.o',
             'command': f'g++ -DGUFO_DS4_OFFICIAL_FIXTURE=\\"{source}/tests/models/ds4/fixtures/official.json\\" '
                        f'-DGUFO_CHAT_TEMPLATE_HF_GOLDENS=\\"{source}/tests/fixtures/goldens.json\\" -c x.cpp'},
            {'output': 'tests/other/CMakeFiles/other_test.dir/other.cpp.o',
             'command': f'g++ -DOTHER=\\"{source}/tests/fixtures/unrelated.json\\" -c y.cpp'}]))
        executable = str(build / 'tests/models/ds4')
        (build / 'tests/models/ds4/CTestTestfile.cmake').write_text(
            f'add_test([=[ds4.chat]=] "{executable}/ds4_cli_test" "--sampled-chat")\n'
            f'set_tests_properties([=[ds4.chat]=] PROPERTIES LABELS "ds4")\n')
        tests = [
            {'name': 'ds4.target', 'command': [executable + '/ds4_quality_test'],
             'properties': [prop('LABELS', ['gpu', 'ds4']), prop('SKIP_RETURN_CODE', 77),
                            prop('RESOURCE_LOCK', ['ds4_model']), prop('WORKING_DIRECTORY', executable)]},
            {'name': 'ds4.dspark', 'command': [executable + '/ds4_quality_test', '--dspark'],
             'properties': [prop('LABELS', ['gpu', 'ds4']), prop('TIMEOUT', 1800.0)]},
            # CTest omits the command while the executable is not built yet.
            {'name': 'ds4.chat', 'properties': [prop('LABELS', ['ds4'])]},
            # Repository checks run an interpreter from the source tree; CPU CI keeps them.
            {'name': 'ds4.dataset', 'command': ['/nix/store/python3/bin/python3', str(source / 'dataset_test.py')],
             'properties': [prop('LABELS', ['cpu', 'ds4'])]}]
        return source, build, tests

    def test_collects_built_executables_properties_and_their_fixtures_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, build, tests = self.tree(root)
            with patch.object(bundle.subprocess, 'check_output', return_value=json.dumps({'tests': tests})) as ctest:
                manifest = bundle.collect(build, source, ['ds4'], root / 'out')
            self.assertIn('^(ds4)$', ctest.call_args.args[0])
            self.assertEqual([test['name'] for test in manifest['tests']], ['ds4.target', 'ds4.dspark', 'ds4.chat'])
            target, dspark, chat = manifest['tests']
            self.assertEqual((target['skip_return_code'], target['timeout_s']), (77, bundle.DEFAULT_TIMEOUT_S))
            self.assertEqual((dspark['args'], dspark['timeout_s']), (['--dspark'], 1800.0))
            self.assertEqual((chat['executable'], chat['args']), ('libexec/gufo-tests/ds4_cli_test', ['--sampled-chat']))
            self.assertEqual(sorted(manifest['fixtures']),
                             ['tests/fixtures/goldens.json', 'tests/models/ds4/fixtures/official.json'])
            self.assertTrue((root / 'out/share/gufo/ctest/source/tests/fixtures/goldens.json').is_file())
            self.assertFalse((root / 'out/share/gufo/ctest/source/tests/fixtures/unrelated.json').exists())
            self.assertEqual(sorted(path.name for path in (root / 'out/libexec/gufo-tests').iterdir()),
                             ['ds4_cli_test', 'ds4_quality_test'])
            self.assertEqual(json.loads((root / 'out' / bundle.MANIFEST).read_text())['source_dir'], str(source))

    def test_unsupported_properties_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, build, tests = self.tree(root)
            tests[0]['properties'].append(prop('WILL_FAIL', True))
            with patch.object(bundle.subprocess, 'check_output', return_value=json.dumps({'tests': tests})):
                with self.assertRaisesRegex(ValueError, 'ds4.target: unsupported CTest properties WILL_FAIL'):
                    bundle.collect(build, source, ['ds4'], root / 'out')


class RunTest(unittest.TestCase):
    def bundle(self, root, script):
        (root / 'tests').mkdir(parents=True)
        launcher = root / 'tests/fake_test'
        launcher.write_text('#!/bin/sh\n' + script)
        launcher.chmod(0o755)
        (root / bundle.SOURCE).mkdir(parents=True)
        tests = [{'name': name, 'executable': 'libexec/gufo-tests/fake_test', 'args': [name],
                  'labels': labels, 'timeout_s': 5, 'skip_return_code': 77, 'environment': {'FROM_CTEST': '1'}}
                 for name, labels in (('ds4.pass', ['ds4']), ('ds4.skip', ['ds4']), ('ds4.fail', ['ds4', 'gpu']),
                                      ('other.fail', ['other']))]
        (root / bundle.MANIFEST).write_text(json.dumps({'schema': 1, 'tests': tests}))

    def test_verdicts_follow_ctest_and_the_environment_reaches_the_test(self):
        script = ('test "$FROM_CTEST$MODEL" = 1/weights || exit 3\n'
                  'test -d "$GUFO_TEST_SOURCE_DIR" || exit 4\n'
                  'case "$1" in ds4.pass) exit 0;; ds4.skip) exit 77;; *) echo broken; exit 1;; esac\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root / 'bundle', script)
            results = bundle.run(root / 'bundle', ['ds4'], {'MODEL': '/weights'}, root / 'out')
            saved = json.loads((root / 'out/results.json').read_text())
        self.assertEqual(results['status'], 'failed')
        self.assertEqual(saved, results)
        self.assertEqual([(row['name'], row['status']) for row in results['tests']],
                         [('ds4.pass', 'passed'), ('ds4.skip', 'skipped'), ('ds4.fail', 'failed')])
        self.assertIn('broken', results['tests'][2]['log_tail'])

    def test_timeouts_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root / 'bundle', 'exec sleep 30\n')
            manifest = json.loads((root / 'bundle' / bundle.MANIFEST).read_text())
            manifest['tests'] = manifest['tests'][:1]
            manifest['tests'][0]['timeout_s'] = 0.5
            (root / 'bundle' / bundle.MANIFEST).write_text(json.dumps(manifest))
            results = bundle.run(root / 'bundle', ['ds4'], {}, root / 'out')
        self.assertEqual(results['tests'][0]['status'], 'failed')
        self.assertIn('timed out', results['tests'][0]['error'])

    def test_a_label_without_tests_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root / 'bundle', 'exit 0\n')
            with self.assertRaisesRegex(ValueError, 'no tests labelled qwen27b'):
                bundle.run(root / 'bundle', ['qwen27b'], {}, root / 'out')


if __name__ == '__main__':
    unittest.main()
