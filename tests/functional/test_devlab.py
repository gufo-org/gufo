"""Hosted CPU contracts for Devlab ownership, pinned assets, speed evidence and reports."""

import hashlib
import json
from pathlib import Path
import shutil
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

NPU_MODELS = {'embedding-gemma2-npu', 'qwen3-asr-npu', 'qwen3-tts-custom-npu'}


def without_npu(data):
    """The registry without its NPU variants, for checks of generic behavior."""
    data = {**data, 'models': {key: model for key, model in data['models'].items() if not model.get('npu')}}
    data['rules'] = [{**rule, 'owners': [owner for owner in rule['owners'] if owner not in NPU_MODELS]}
                     for rule in data['rules']]
    return data


class PlanTest(unittest.TestCase):
    def setUp(self):
        # Ownership contracts hold for every model, disabled ones included.
        self.registry = devlab_models.registry()
        self.data = {**self.registry, 'disabled': {}}

    def plan(self, *paths, **kwargs):
        return devlab_plan.select(paths, self.data, **kwargs)

    def test_disabled_models_are_named_and_not_planned(self):
        data = {**self.registry, 'disabled': {'deepseek4': 'reason'}}
        plan = devlab_plan.select(['src/models/deepseek_v4_flash/model.cpp', 'src/core/sampling.cpp'], data)
        self.assertEqual({item['id'] for item in plan['models']}, {'flash-next-q4', 'qwen27-q4'})
        self.assertNotIn('deepseek4', {group['profile'].split('/')[0] for group in plan['groups']})
        self.assertEqual([row['id'] for row in plan['disabled']], ['deepseek4'])
        self.assertTrue(plan['disabled'][0]['reason'])
        self.assertEqual(devlab_models.asset_keys(plan, data), ['dflash2', 'flash-next', 'qwen27'])
        broken = {**self.registry, 'disabled': {'deepseek-typo': 'reason'}}
        with patch('devlab_models.json.loads', return_value=broken):
            with self.assertRaises(ValueError):
                devlab_models.registry()

    def test_curated_suites_apply_at_every_tier(self):
        suites = self.registry['models']['deepseek4']['suites']
        for tier in ('pr', 'full'):
            with self.subTest(tier=tier):
                plan = self.plan('src/models/deepseek_v4_flash/engine.cpp', tier=tier)
                planned = [suite for group in plan['groups'] for suite in group['suites']]
                self.assertEqual(planned, suites)
                self.assertFalse(plan['release_only'] or plan['deferred'])
                self.assertIn('cache-growth', {row['suite'] for row in plan['out_of_scope']})
        unknown = {**self.data, 'models': {**self.data['models'], 'deepseek4': {
            **self.data['models']['deepseek4'], 'suites': ['cache', 'no-such-suite']}}}
        with self.assertRaises(ValueError):
            devlab_plan.select(['src/models/deepseek_v4_flash/engine.cpp'], unknown)

    def models(self, *paths, **kwargs):
        return {item['id'] for item in self.plan(*paths, **kwargs)['models']}

    def test_shared_llm_covers_every_llm(self):
        for path in ('src/core/sampling.cpp', 'src/cache/cache.cpp',
                     'src/families/qwen/text/tokenizer.cpp', 'src/families/qwen/vision/encoder.hip',
                     'src/cli/serve/text_model_runner.cpp'):
            with self.subTest(path=path):
                self.assertEqual(self.models(path), {'flash-next-q4', 'qwen27-q4', 'deepseek4'})

    def test_private_model_and_mixed_paths_preserve_both(self):
        private = 'src/models/deepseek_v4_flash/model.cpp'
        self.assertEqual(self.models(private), {'deepseek4'})
        self.assertEqual(self.models(private, 'src/core/sampling.cpp'), {'flash-next-q4', 'qwen27-q4', 'deepseek4'})
        self.assertEqual(self.models('src/models/qwen/hip/attention.hip'), {'qwen27-q4'})
        self.assertEqual(self.models('src/models/qwen38_flash_next/engine.cpp'), {'flash-next-q4'})

    def test_global_changes_cover_every_model_but_one_image_representative(self):
        self.assertEqual(self.models('src/cli/serve/serve.cpp'),
                         {'flash-next-q4', 'qwen27-q4', 'deepseek4'} | {'qwen3-tts-base', 'qwen3-tts-custom', 'qwen3-tts-design', 'qwen3-tts-custom-npu'} | NPU_MODELS |
                         {'embedding-gemma2', 'qwen3-asr', 'qwen-image-turbo'})
        self.assertIn('qwen-image', self.models('src/cli/serve/serve.cpp', tier='full'))
        self.assertEqual(self.models('new-unclassified-file'), self.models('CMakeLists.txt'))

    def test_hosted_check_list_selects_no_model(self):
        # cmake/Checks.cmake lists the hosted CPU suite, which CI runs itself.
        self.assertEqual(self.models('cmake/Checks.cmake'), set())
        self.assertEqual(self.models('cmake/Checks.cmake', 'src/models/embedding_gemma2/hip/ops.hip'),
                         {'embedding-gemma2', 'embedding-gemma2-npu'})
        self.assertEqual(self.models('cmake/toolchains/windows-therock.cmake'), self.models('CMakeLists.txt'))
        # Its CPU test sources need no model either; the code they test still selects.
        self.assertEqual(self.models('tests/cli/openai_chat_test.cpp', 'tests/cli/http_server_test.cpp'), set())
        self.assertEqual(self.models('tests/cli/openai_chat_test.cpp', 'src/cli/serve/openai_chat.cpp'),
                         {'flash-next-q4', 'qwen27-q4', 'deepseek4'})
        self.assertEqual(self.models('tests/cli/audio_websocket_test.cpp'),
                         {'qwen3-asr', 'qwen3-asr-npu'} | {'qwen3-tts-base', 'qwen3-tts-custom', 'qwen3-tts-design', 'qwen3-tts-custom-npu'})
        self.assertEqual(self.models('tests/cli/unlisted_test.cpp'), self.models('CMakeLists.txt'))

    def test_category_and_fixture_ownership(self):
        for path, expected in [
            ('src/models/embedding_gemma2/hip/ops.hip', {'embedding-gemma2', 'embedding-gemma2-npu'}),
            ('src/cli/serve/audio_asr_api.cpp', {'qwen3-asr', 'qwen3-asr-npu'}),
            ('src/models/qwen3_tts/hip/synthesis_runtime.cpp', {'qwen3-tts-base', 'qwen3-tts-custom', 'qwen3-tts-design', 'qwen3-tts-custom-npu'}),
            ('tests/functional/fixtures/audio/clear-english-voice.wav',
             {'qwen3-asr', 'qwen3-asr-npu'} | {'qwen3-tts-base', 'qwen3-tts-custom', 'qwen3-tts-design', 'qwen3-tts-custom-npu'}),
            ('src/core/npu/bfp_gemm.cpp', NPU_MODELS),
            ('src/models/qwen_image_21/hip/denoise.hip', {'qwen-image-turbo'}),
            ('src/cli/serve/image_api.cpp', {'qwen-image-turbo'})]:
            self.assertEqual(self.models(path), expected)
        self.assertEqual(self.models('docs/server.md'), set())
        self.assertEqual(self.models('src/models/minimax_h3/runtime.cpp'), set())

    def test_npu_variants_run_beside_their_models(self):
        plan = self.plan('src/core/npu/host_ops.cpp')
        self.assertEqual({group['profile'] for group in plan['groups']},
                         {'embedding-gemma2-npu/npu', 'qwen3-asr-npu/npu', 'qwen3-tts-custom-npu/npu'})
        embedding = [g for g in plan['groups'] if g['profile'].startswith('embedding')][0]
        self.assertEqual(embedding['suites'], ['embedding-text', 'embedding-images', 'embedding-audio',
                                               'embedding-video'])
        args = devlab_run.service_args('qwen3-asr-npu', Path('/head'), Path('/base'),
                                       {'asr': '/asr'}, Path('/out'))
        self.assertEqual(args[-1], '--npu')
        self.assertNotIn('--npu', devlab_run.service_args('qwen3-asr', Path('/head'), Path('/base'),
                                                          {'asr': '/asr'}, Path('/out')))
        # The speed arms serve the same NPU configuration as the correctness run.
        assets = {'asr': '/asr'}
        self.assertEqual(devlab_models.service_command('qwen3-asr-npu', Path('/gufo'), assets,
                                                       Path('/audio'), 'phases')[-1], '--npu')
        self.assertNotIn('--npu', devlab_models.service_command('qwen3-asr', Path('/gufo'), assets,
                                                                Path('/audio'), 'phases'))

    def test_prs_run_the_release_suite_set(self):
        for tier in ('pr', 'full'):
            plan = self.plan('src/models/embedding_gemma2/video_processor.cpp', 'src/core/sampling.cpp', tier=tier)
            embedding = next(group for group in plan['groups'] if group['profile'].startswith('embedding'))
            self.assertEqual(embedding['suites'],
                             ['embedding-text', 'embedding-images', 'embedding-audio', 'embedding-video'])
            self.assertFalse(plan['release_only'])
            self.assertEqual(plan['suite_tier'], 'full')
            self.assertIn('reasoning-budget', {suite for group in plan['groups'] for suite in group['suites']})
        # Image generation alone keeps a PR smoke representative.
        self.assertEqual(self.plan('src/models/qwen_image_21/image.cpp')['groups'][0]['suites'], ['image-smoke'])

    def test_full_tts_and_optional_ar(self):
        self.assertEqual(self.models('src/models/qwen3_tts/config.cpp', tier='full'),
                         {'qwen3-tts-base', 'qwen3-tts-custom', 'qwen3-tts-design', 'qwen3-tts-custom-npu'})
        self.assertEqual(self.plan('src/models/qwen_image_21/image.cpp')['groups'][0]['suites'], ['image-smoke'])
        image = self.plan('src/models/qwen_image_21/image.cpp', tier='full')
        self.assertEqual({model['id'] for model in image['models']}, {'qwen-image-turbo', 'qwen-image'})
        self.assertEqual({group['suites'][0] for group in image['groups']}, {'image-http'})
        self.assertIn('*.denoise_ms', image['models'][0]['timing_checks'])
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
                         ['asr', 'tts-base', 'tts-custom', 'tts-design'])
        self.assertEqual(devlab_models.asset_keys(self.plan('src/models/deepseek_v4_flash/model.cpp'), self.data),
                         ['deepseek4', 'dspark'])
        self.assertEqual(devlab_models.asset_keys(self.plan('src/cli/serve/image_service.cpp'), self.data),
                         ['image-turbo'])
        self.assertEqual(devlab_models.asset_keys(self.plan('docs/server.md'), self.data), [])
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
        plan = devlab_plan.select(['src/cli/serve/audio_asr_api.cpp', 'src/models/embedding_gemma2/model.cpp'],
                                  without_npu(devlab_models.registry()))
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
            assets.write_text(json.dumps({'asr': '/asr', 'embedding': '/embedding'}))
            with patch('devlab_run.subprocess.run', side_effect=run):
                code = devlab_run.execute(plan, Path('/head'), Path('/base'), assets,
                                          root/'results', root/'report', root/'logs')
            observed = json.loads((root/'report/summary.json').read_text())
            timing = json.loads((root/'report/qwen3-asr.phases.json').read_text())
            self.assertTrue(timing['phases'][0]['checked'])
            self.assertFalse(timing['phases'][1]['checked'])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 4)
        self.assertEqual([item['correctness'] for item in observed['models']], ['passed', 'failed'])
        self.assertTrue(all(item['timing'] == 'passed' for item in observed['models']))
        self.assertEqual(len(observed['groups']), 2)


class ShardTest(unittest.TestCase):
    TABLE = {'schema': 1, 'default_suite_s': 60, 'default_load_s': 30, 'default_timing_s': 300,
             'models': {'flash-next-q4': {'load_s': 20, 'timing_s': 200,
                                          'suites': {'cache': 300, 'tools': 200, 'discovery': 1}},
                        'qwen3-asr': {'timing_s': 100, 'suites': {'asr-http': 60}},
                        'embedding-gemma2': {'timing_s': 100, 'suites': {'embedding-smoke': 30}}}}

    def plan(self, *paths, tier='full'):
        plan = devlab_plan.select(paths, without_npu(devlab_models.registry()), tier)
        plan.update(base='a' * 40, head='b' * 40)
        return plan

    @staticmethod
    def parts(shards, model):
        return [part for row in shards for part in row['parts'] if part['model'] == model]

    def test_every_suite_runs_once_within_budget(self):
        plan = self.plan('src/models/qwen38_flash_next/engine.cpp', 'src/models/qwen3_asr/model.cpp',
                         'src/models/embedding_gemma2/model.cpp', tier='pr')
        shards = devlab_plan.shard(plan, self.TABLE, budget=720)
        for group in plan['groups']:
            planned = [suite for row in shards for part in row['parts'] for server in part['groups']
                       if (server['profile'], server['configuration']) == (group['profile'], group['configuration'])
                       for suite in server['suites']]
            self.assertEqual(sorted(planned), sorted(group['suites']))
        # Each model times once, on one device; a shard names each model once.
        for model in ('flash-next-q4', 'qwen3-asr', 'embedding-gemma2'):
            self.assertEqual(sum(part['timing'] for part in self.parts(shards, model)), 1)
        for row in shards:
            self.assertEqual(len(row['parts']), len({part['model'] for part in row['parts']}))
            # A suite costlier than the budget still runs, alone.
            self.assertTrue(row['seconds'] <= 720 or sum(len(server['suites']) for part in row['parts']
                                                         for server in part['groups']) == 1)
        # Short service models share a device; an LLM never shares with another model.
        services = [row for row in shards if {'qwen3-asr', 'embedding-gemma2'} & {p['model'] for p in row['parts']}]
        self.assertEqual(len(services), 1)
        self.assertEqual({p['model'] for p in services[0]['parts']}, {'qwen3-asr', 'embedding-gemma2'})
        llm = [row for row in shards if any(part['model'] == 'flash-next-q4' for part in row['parts'])]
        self.assertTrue(all(len(row['parts']) == 1 for row in llm))
        self.assertGreater(len(llm), 1)
        # Suites that read server-wide cache state get a server of their own.
        servers = [server for part in self.parts(shards, 'flash-next-q4') for server in part['groups']]
        for suite in ('cache', 'batch'):
            self.assertIn([suite], [server['suites'] for server in servers])
        names = [(server['configuration'], server.get('server')) for server in servers]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len({row['id'] for row in shards}), len(shards))

    def test_order_seed_reaches_the_runner(self):
        plan = self.plan('src/models/qwen38_flash_next/engine.cpp')
        assets = {key: '/weights/' + key for key in devlab_models.registry()['assets']}
        args = cluster.command(plan['groups'][0], Path('/gufo'), None, Path('/out'), 'full', assets, 1234)
        self.assertEqual(args[args.index('--order-seed') + 1], '1234')
        self.assertLess(args.index('--order-seed'), args.index('--'))
        self.assertNotIn('--order-seed', cluster.command(plan['groups'][0], Path('/gufo'), None,
                                                         Path('/out'), 'full', assets))

    def test_shard_runs_only_its_part(self):
        plan = self.plan('src/models/qwen38_flash_next/engine.cpp', 'src/models/qwen3_asr/model.cpp',
                         'src/models/embedding_gemma2/model.cpp', tier='pr')
        plan['shards'] = devlab_plan.shard(plan, self.TABLE, budget=400)
        for row in plan['shards']:
            observed = devlab_run.jobs(plan, row['id'])
            self.assertEqual([(selection['id'], groups, timing) for selection, groups, timing in observed],
                             [(part['model'], part['groups'], part['timing']) for part in row['parts']])
        with self.assertRaises(ValueError):
            devlab_run.jobs(plan, 'no-such-shard')

    def write_shard(self, root, row, status='passed', phases=True):
        directory = root / 'shards' / row['id']
        directory.mkdir(parents=True)
        groups, items = [], []
        for part in row['parts']:
            groups += [{**group, 'status': status, 'exit_code': int(status != 'passed'), 'command': ['run'],
                        **({'failures': {group['suites'][0]: 'broken'}, 'log_tail': 'tail'}
                           if status != 'passed' else {})} for group in part['groups']]
            items.append({'id': part['model'], 'reasons': [],
                          'correctness': status if part['groups'] else 'skipped',
                          'timing': 'passed' if part['timing'] else 'skipped'})
            if part['timing'] and phases:
                (directory / (part['model'] + '.phases.json')).write_text('{"phases": []}')
        (directory / 'summary.json').write_text(json.dumps({'schema': 2, 'shard': row['id'], 'groups': groups,
                                                            'models': items, 'status': status}))

    def test_merge_recombines_groups_and_flags_missing_shards(self):
        import devlab_merge
        plan = self.plan('src/models/qwen38_flash_next/engine.cpp', 'src/models/qwen3_asr/model.cpp')
        plan['shards'] = devlab_plan.shard(plan, self.TABLE, budget=400)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in plan['shards']:
                self.write_shard(root / 'in', row)
            merged = devlab_merge.merge(plan, root / 'in', root / 'out')
            self.assertEqual(merged['status'], 'passed')
            self.assertEqual([(row['profile'], row['configuration'], row['suites']) for row in merged['groups']],
                             [(row['profile'], row['configuration'], row['suites']) for row in plan['groups']])
            self.assertTrue((root / 'out' / 'flash-next-q4.phases.json').is_file())
            self.assertEqual([(item['correctness'], item['timing']) for item in merged['models']],
                             [('passed', 'passed')] * 2)
            # One failed and one missing correctness shard; the ASR shard (with its timing) missing.
            llm = [row for row in plan['shards'] if row['parts'][0]['model'] == 'flash-next-q4'
                   and not any(part['timing'] for part in row['parts'])]
            asr = next(row for row in plan['shards'] if row['parts'][0]['model'] == 'qwen3-asr')
            for row in (llm[0], llm[1], asr):
                shutil.rmtree(root / 'in' / 'shards' / row['id'])
            self.write_shard(root / 'in', llm[0], status='failed')
            merged = devlab_merge.merge(plan, root / 'in', root / 'out2')
        flash, asr_item = merged['models']
        self.assertEqual((flash['correctness'], flash['timing']), ('error', 'passed'))
        self.assertIn(llm[1]['id'], flash['error'])
        self.assertEqual((asr_item['correctness'], asr_item['timing']), ('error', 'error'))
        failed = [row for row in merged['groups'] if row['status'] != 'passed']
        self.assertTrue(any('broken' in row['failures'].values() for row in failed))
        self.assertTrue(any('Missing suites' in (row['error'] or '') or 'no summary' in (row['error'] or '')
                            for row in failed))
        self.assertEqual(merged['status'], 'failed')

    def test_rerun_attempts_replace_only_their_shards(self):
        import devlab_merge
        plan = self.plan('src/models/qwen38_flash_next/engine.cpp', tier='pr')
        plan['shards'] = devlab_plan.shard(plan, self.TABLE, budget=400)
        first, second = plan['shards'][:2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in plan['shards']:
                attempts = ((1, 'failed'), (2, 'passed')) if row is second else ((1, 'passed'),)
                for attempt, status in attempts:
                    self.write_shard(root / 'artifacts' / f"functional-report-{row['id']}-{attempt}", row, status)
            devlab_merge.collect(plan, root / 'artifacts', root / 'results')
            merged = devlab_merge.merge(plan, root / 'results', root / 'out')
        self.assertEqual(merged['status'], 'passed')
        self.assertEqual(set(merged['shards'].values()), {'passed'})


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

    def test_npu_fallback_fails_the_timing_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            log.write_text('[INFO] npu: qwen3dec ready\n')
            self.assertEqual(service_phases.npu_fallbacks(log), [])
            log.write_text('[WARN] npu: qwen3dec unavailable (busy); decoding on the CPU\n')
            self.assertEqual(len(service_phases.npu_fallbacks(log)), 1)

    def test_base_without_npu_yields_pr_only_observations(self):
        reason = 'base binary has no --npu; PR-only NPU timings'
        arms = [arm if arm['revision'] == 'head' else {'revision': 'base', 'unsupported': reason}
                for arm in self.arms()]
        found, diverged = service_phases.summarize(arms)
        self.assertFalse(diverged)
        self.assertTrue(found)
        for phase in found:
            self.assertEqual(phase['base'], [None, None])
            self.assertEqual((phase['verdict'], phase['checked'], phase['advisory_reason']),
                             ('missing', False, reason))
        short = next(phase for phase in found if phase['key'] == 'short.server_ms')
        self.assertEqual(short['head'], [120, 120])
        plan = devlab_plan.select(['src/core/npu/host_ops.cpp'], devlab_models.registry())
        plan.update(base='a'*40, head='b'*40, models=[m for m in plan['models'] if m['id'] == 'qwen3-asr-npu'],
                    groups=[g for g in plan['groups'] if g['profile'].startswith('qwen3-asr-npu')])
        row = {**plan['groups'][0], 'status': 'passed'}
        summary = {'groups': [row], 'models': [{'id': 'qwen3-asr-npu', 'correctness': 'passed', 'timing': 'passed'}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {'revisions': {'base': plan['base'], 'head': plan['head']}, 'arms': arms, 'phases': found,
                    'diverged': diverged, 'settings': {'repeats': 2, 'slowdown': .05, 'noise_ms': 10,
                    'requests': service_phases.REQUESTS}, 'wall_s': 1}
            (root/'qwen3-asr-npu.phases.json').write_text(json.dumps(data))
            args = type('Args', (), {'authorize': 'success', 'smoke': 'success', 'benchmark': False,
                                    'results': root, 'run_url': 'https://run'})()
            self.assertEqual(report.timing_state(data), 'observations')
            state, description, comment = report.render_models(args, plan, summary)
            self.assertEqual(state, 'success')
            self.assertIn('timing observations retained', description)
            self.assertIn(reason, comment)
            self.assertIn('— → 120 ms', comment)

    def test_report_does_not_pass_missing_models_or_speed(self):
        plan = devlab_plan.select(['src/cli/serve/audio_asr_api.cpp'], without_npu(devlab_models.registry()))
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
