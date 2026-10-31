"""Hosted CPU contracts for Devlab ownership, pinned assets, speed evidence and reports."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cluster
import devlab_models
import devlab_plan
import devlab_run
import report
import phases
import service_phases


class PlanTest(unittest.TestCase):
    def setUp(self):
        # Ownership contracts hold for every model, disabled ones included.
        self.registry = devlab_models.registry()
        self.data = {**self.registry, 'disabled': {}}

    def plan(self, *paths, **kwargs):
        return devlab_plan.select(paths, self.data, **kwargs)

    def test_disabled_models_are_named_and_not_planned(self):
        self.assertIn('deepseek4', self.registry['disabled'])
        plan = devlab_plan.select(['src/models/deepseek_v4_flash/model.cpp', 'src/core/sampling.cpp'],
                                  self.registry)
        self.assertEqual({item['id'] for item in plan['models']}, {'flash-next-q4'})
        self.assertNotIn('deepseek4', {group['profile'].split('/')[0] for group in plan['groups']})
        self.assertEqual([row['id'] for row in plan['disabled']], ['deepseek4'])
        self.assertTrue(plan['disabled'][0]['reason'])
        self.assertEqual(devlab_models.asset_keys(plan, self.registry), ['flash-next'])
        broken = {**self.registry, 'disabled': {'deepseek-typo': 'reason'}}
        with patch('devlab_models.json.loads', return_value=broken):
            with self.assertRaises(ValueError):
                devlab_models.registry()

    def models(self, *paths, **kwargs):
        return {item['id'] for item in self.plan(*paths, **kwargs)['models']}

    def test_shared_llm_uses_only_flash_next(self):
        for path in ('src/core/sampling.cpp', 'src/cache/cache.cpp',
                     'src/families/qwen/text/tokenizer.cpp', 'src/families/qwen/vision/encoder.hip',
                     'src/cli/serve/text_model_runner.cpp'):
            with self.subTest(path=path):
                self.assertEqual(self.models(path), {'flash-next-q4'})

    def test_private_model_and_mixed_paths_preserve_both(self):
        private = 'src/models/deepseek_v4_flash/model.cpp'
        self.assertEqual(self.models(private), {'deepseek4'})
        self.assertEqual(self.models(private, 'src/core/sampling.cpp'), {'deepseek4', 'flash-next-q4'})
        self.assertEqual(self.models('src/models/qwen/hip/attention.hip'), {'qwen27-q4'})
        self.assertEqual(self.models('src/models/qwen38_flash_next/engine.cpp'), {'flash-next-q4'})

    def test_global_changes_cover_one_representative_per_category(self):
        self.assertEqual(self.models('src/cli/serve/serve.cpp'),
                         {'flash-next-q4', 'embedding-gemma2', 'qwen3-asr', 'qwen3-tts-base'})
        self.assertEqual(self.models('new-unclassified-file'), self.models('CMakeLists.txt'))

    def test_hosted_check_list_selects_no_model(self):
        # cmake/Checks.cmake lists the hosted CPU suite, which CI runs itself.
        self.assertEqual(self.models('cmake/Checks.cmake'), set())
        self.assertEqual(self.models('cmake/Checks.cmake', 'src/models/embedding_gemma2/hip/ops.hip'),
                         {'embedding-gemma2'})
        self.assertEqual(self.models('cmake/toolchains/windows-therock.cmake'), self.models('CMakeLists.txt'))
        # Its CPU test sources need no model either; the code they test still selects.
        self.assertEqual(self.models('tests/cli/openai_chat_test.cpp', 'tests/cli/http_server_test.cpp'), set())
        self.assertEqual(self.models('tests/cli/openai_chat_test.cpp', 'src/cli/serve/openai_chat.cpp'),
                         {'flash-next-q4'})
        self.assertEqual(self.models('tests/cli/audio_websocket_test.cpp'), {'qwen3-asr', 'qwen3-tts-base'})
        self.assertEqual(self.models('tests/cli/continuation_cache_test.cpp'), self.models('CMakeLists.txt'))

    def test_category_and_fixture_ownership(self):
        for path, expected in [
            ('src/models/embedding_gemma2/hip/ops.hip', {'embedding-gemma2'}),
            ('src/cli/serve/audio_asr_api.cpp', {'qwen3-asr'}),
            ('src/models/qwen3_tts/hip/synthesis_runtime.cpp', {'qwen3-tts-base'}),
            ('tests/functional/fixtures/audio/clear-english-voice.wav', {'qwen3-asr', 'qwen3-tts-base'})]:
            self.assertEqual(self.models(path), expected)
        self.assertEqual(self.models('docs/SERVER.md'), set())
        self.assertEqual(self.models('src/models/qwen_image_21/model.cpp'), set())
        self.assertEqual(self.models('src/models/minimax_h3/runtime.cpp'), set())

    def test_embedding_smoke_and_full_coverage(self):
        pr = self.plan('src/models/embedding_gemma2/video_processor.cpp')
        self.assertEqual(pr['groups'][0]['suites'], ['embedding-smoke'])
        full = self.plan('src/models/embedding_gemma2/video_processor.cpp', tier='full')
        self.assertEqual(full['groups'][0]['suites'],
                         ['embedding-text', 'embedding-images', 'embedding-audio', 'embedding-video'])

    def test_full_tts_and_optional_ar(self):
        self.assertEqual(self.models('src/models/qwen3_tts/config.cpp', tier='full'),
                         {'qwen3-tts-base', 'qwen3-tts-custom', 'qwen3-tts-design'})
        plan = self.plan('src/models/deepseek_v4_flash/model.cpp', ar=True)
        self.assertEqual({g['profile'] for g in plan['groups']}, {'deepseek4/off', 'deepseek4/dspark'})
        self.assertEqual(self.models('docs/README.md', benchmark=True), {'flash-next-q4'})

    def test_timing_gates_follow_production_changes(self):
        metadata = self.plan('tests/functional/devlab_run.py')
        self.assertTrue(all(not model['timing_checks'] for model in metadata['models']))
        production = self.plan('src/core/sampling.cpp')
        self.assertIn('*.predicted_ms', production['models'][0]['timing_checks'])
        self.assertNotIn('startup.*', production['models'][0]['timing_checks'])

    def test_dependency_downloads_are_plan_scoped(self):
        self.assertEqual(devlab_models.asset_keys(self.plan('src/cli/serve/audio_tts_api.cpp'), self.data),
                         ['asr', 'tts-base'])
        self.assertEqual(devlab_models.asset_keys(self.plan('src/models/deepseek_v4_flash/model.cpp'), self.data),
                         ['deepseek4', 'dspark'])
        self.assertEqual(devlab_models.asset_keys(self.plan('docs/SERVER.md'), self.data), [])
        for asset in self.data['assets'].values():
            self.assertRegex(asset['revision'], r'^[0-9a-f]{40}$')

    def test_generic_llm_commands_keep_drafts_and_capabilities(self):
        assets = {key: '/weights/' + key for key in self.data['assets']}
        for key, draft, preset in [('qwen27-q4', '--dflash-model', 'qwen38'),
                                    ('deepseek4', '--dspark-model', 'deepseek4')]:
            plan = self.plan('src/models/qwen/engine.cpp' if key.startswith('qwen') else
                             'src/models/deepseek_v4_flash/model.cpp')
            args = cluster.command(plan['groups'][0], Path('/gufo'), None, Path('/out'), 'pr', assets)
            self.assertIn(draft, args)
            self.assertEqual(args[args.index('--sampling-preset')+1], preset)
            self.assertEqual('--mmproj' in args, key.startswith('qwen'))
            self.assertIn('--record-baseline', args)

    def test_tts_evaluator_is_merge_base_and_fixture_is_packaged(self):
        assets = {key: '/weights/' + key for key in self.data['assets']}
        command = devlab_run.service_args('qwen3-tts-base', Path('/head'), Path('/base'), assets, Path('/out'))
        self.assertEqual(command[command.index('--asr-binary')+1], '/base')
        fixture = devlab_models.HERE / 'fixtures/audio'
        self.assertEqual({p.name for p in fixture.iterdir() if p.suffix in ('.wav', '.txt')},
                         {'clear-english-voice.wav', 'clear-english-voice.txt'})
        self.assertEqual(hashlib.sha256((fixture / 'clear-english-voice.wav').read_bytes()).hexdigest(),
                         '41ff28fbea935f45dae8a760100609bf39a64dccbedc742edb4f0711d516ef5a')


class ExecutionTest(unittest.TestCase):
    def test_ephemeral_download_fetches_only_selected_assets_and_checks_required_files(self):
        data = devlab_models.registry()
        plan = devlab_plan.select(['src/cli/serve/audio_asr_api.cpp'], data)
        fetched = []
        def snapshot(**kwargs):
            fetched.append(kwargs)
            for name in data['assets']['asr']['required']:
                path = kwargs['local_dir'] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                'huggingface_hub': SimpleNamespace(snapshot_download=snapshot)}):
            paths = devlab_models.download(plan, Path(directory), data)
            self.assertEqual(set(paths), {'asr'})
            self.assertEqual(len(fetched), 1)
            self.assertEqual(fetched[0]['revision'], data['assets']['asr']['revision'])
            (Path(paths['asr']) / 'config.json').unlink()
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(snapshot_download=lambda **kw: None)}):
                with self.assertRaises(FileNotFoundError):
                    devlab_models.download(plan, Path(directory), data)

    def test_model_failure_does_not_hide_other_models_or_speed(self):
        plan = devlab_plan.select(['src/cli/serve/audio_asr_api.cpp', 'src/cli/serve/audio_tts_api.cpp'],
                                  devlab_models.registry())
        plan.update(base='a'*40, head='b'*40)
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            output = Path(args[args.index('--output')+1])
            if args[1].endswith('model_services.py'):
                output.mkdir()
                failed = args[2] == 'asr'
                (output/'report.json').write_text(json.dumps({'status': 'failed' if failed else 'passed'}))
                return subprocess.CompletedProcess(args, int(failed))
            output.write_text(json.dumps({'arms': [{'revision': side} for side in ('base','head','base','head')],
                'settings': {'repeats': 2}, 'revisions': {'base': plan['base'], 'head': plan['head']},
                'phases': [{'key': 'short.server_ms'}, {'key': 'startup.listen', 'checked': False}]}))
            return subprocess.CompletedProcess(args, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root/'assets.json'
            assets.write_text(json.dumps({'asr': '/asr', 'tts-base': '/tts'}))
            with patch('devlab_run.subprocess.run', side_effect=run):
                code = devlab_run.execute(plan, Path('/head'), Path('/base'), assets,
                                          root/'results', root/'report', root/'logs')
            observed = json.loads((root/'report/summary.json').read_text())
            timing = json.loads((root/'report/qwen3-asr.phases.json').read_text())
            self.assertTrue(timing['phases'][0]['checked'])
            self.assertFalse(timing['phases'][1]['checked'])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 4)
        self.assertEqual([item['correctness'] for item in observed['models']], ['failed', 'passed'])
        self.assertTrue(all(item['timing'] == 'passed' for item in observed['models']))
        self.assertEqual(len(observed['groups']), 2)



class TimingTest(unittest.TestCase):
    def arms(self):
        return [{'revision': side, 'startup': {'init': 1, 'weights': 100,
                 'vision': None, 'structured': None, 'listen': 1},
                 'requests': {label: {'work': {'tokens': 10}, 'metrics': {'server_ms': value}}
                              for label in service_phases.REQUESTS}}
                for side, value in [('base', 100), ('head', 120), ('base', 100), ('head', 120)]]

    def test_generic_phase_server_does_not_require_flash_weights(self):
        server = phases.Server(Path('/gufo'), None, Path('/log'),
                               ['/gufo', 'serve', 'asr', '--model', '/asr'])
        self.assertEqual(server.command[:5], ['/gufo', 'serve', 'asr', '--model', '/asr'])
        self.assertIn('--port', server.command)
        self.assertNotIn('--mtp-model', server.command)

    def test_startup_uses_server_timestamps_and_ignores_loading_listener(self):
        server = phases.Server(Path('/gufo'), None, Path('/log'), ['/gufo'])
        server.spawned, server.ready = 10, 12
        server.events = [
            (10.1, '[server] event=listening monotonic_us=1000000'),
            (10.2, '[loader] event=load_started'),
            (11.1, '[loader] event=load_completed elapsed_ms=900 monotonic_us=2000000'),
            # Reader delivery and the health probe are deliberately much later.
            (11.5, '[server] event=listening monotonic_us=2001250'),
        ]
        measured = server.startup()
        self.assertEqual(measured['listen'], 1.25)
        self.assertEqual(measured['listen_source'], 'server')
        server.events = [(stamp, line.split(' monotonic_us=')[0]) for stamp, line in server.events]
        self.assertAlmostEqual(server.startup()['listen'], 900)
        self.assertEqual(server.startup()['listen_source'], 'health_poll')
        found, _ = service_phases.summarize(self.arms())
        listen = next(phase for phase in found if phase['key'] == 'startup.listen')
        self.assertFalse(listen['checked'])
        arms = self.arms()
        for arm in arms:
            arm['startup']['listen_source'] = 'server'
        found, _ = service_phases.summarize(arms)
        self.assertTrue(next(phase for phase in found if phase['key'] == 'startup.listen')['checked'])

    def test_server_work_must_match_and_repeats_must_be_complete(self):
        arms = self.arms()
        phases, diverged = service_phases.summarize(arms)
        self.assertFalse(diverged)
        self.assertEqual([p['key'] for p in phases if p['group'] == 'startup'],
                         ['startup.init', 'startup.weights', 'startup.vision', 'startup.structured', 'startup.listen'])
        self.assertEqual(next(p for p in phases if p['key'] == 'short.server_ms')['verdict'], 'slower')
        arms[1]['requests']['short']['work']['tokens'] = 11
        phases, diverged = service_phases.summarize(arms)
        self.assertEqual(diverged, ['short'])
        self.assertEqual(next(p for p in phases if p['key'] == 'short.server_ms')['verdict'], 'noisy')
        with self.assertRaises(ValueError):
            service_phases.summarize(arms[:-1])
        for value in ('nan', '-1', 'inf'):
            with self.assertRaises(ValueError):
                service_phases.number(value, 'duration_ms')

    def test_report_does_not_pass_missing_models_or_speed(self):
        plan = devlab_plan.select(['src/cli/serve/audio_asr_api.cpp'], devlab_models.registry())
        plan.update(base='a'*40, head='b'*40)
        row = {**plan['groups'][0], 'status': 'passed'}
        summary = {'groups': [row], 'models': [{'id': 'qwen3-asr', 'correctness': 'passed', 'timing': 'passed'}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = type('Args', (), {'authorize': 'success', 'smoke': 'success', 'benchmark': False,
                                    'results': root, 'run_url': 'https://run'})()
            state, description, comment = report.render_models(args, plan, summary)
            self.assertEqual(state, 'error')
            self.assertIn('incomplete', description)
            self.assertIn('Qwen3-ASR', comment)
            phases, diverged = service_phases.summarize(self.arms())
            data = {'revisions': {'base': plan['base'], 'head': plan['head']}, 'arms': self.arms(), 'phases': phases, 'diverged': diverged,
                    'settings': {'repeats': 2, 'slowdown': .05, 'noise_ms': 10, 'requests': service_phases.REQUESTS},
                    'wall_s': 1}
            (root/'qwen3-asr.phases.json').write_text(json.dumps(data))
            state, _, comment = report.render_models(args, plan, summary)
            self.assertEqual(state, 'failure')
            self.assertIn('Every repeat', comment)
            for phase in data['phases']:
                if phase['verdict'] == 'slower':
                    phase['verdict'] = 'same'
            (root/'qwen3-asr.phases.json').write_text(json.dumps(data))
            self.assertEqual(report.render_models(args, plan, summary)[0], 'success')
            checked = next(phase for phase in data['phases'] if phase['key'] == 'short.server_ms')
            checked['verdict'] = 'noisy'
            (root/'qwen3-asr.phases.json').write_text(json.dumps(data))
            state, description, comment = report.render_models(args, plan, summary)
            self.assertEqual(state, 'success')
            self.assertIn('timing warning', description)
            self.assertIn('⚠️', comment)
            state, description, comment = report.render_models(args, {**plan, 'tier': 'full'}, summary)
            self.assertEqual(state, 'success')
            self.assertIn('timing warning', description)
            self.assertIn('⚠️', comment)
            checked['comparable_work'] = False
            (root/'qwen3-asr.phases.json').write_text(json.dumps(data))
            self.assertEqual(report.render_models(args, plan, summary)[0], 'error')
            checked['comparable_work'] = True
            checked['verdict'] = 'missing'
            (root/'qwen3-asr.phases.json').write_text(json.dumps(data))
            self.assertEqual(report.render_models(args, plan, summary)[0], 'error')
            self.assertEqual(report.render_models(args, plan, {'groups': [], 'models': []})[0], 'error')


if __name__ == '__main__':
    unittest.main()
