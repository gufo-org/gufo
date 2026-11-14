#!/usr/bin/env python3
"""NPU output contracts: install only runtime files and reject partial bundles."""

from pathlib import Path
import subprocess
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ci'))
from npu_artifacts import check_artifacts, read_manifest


class NpuArtifactsTest(unittest.TestCase):
    def test_executable_alias_honors_install_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for platform, binary, alias in [('Linux', 'gufo', 'gufo-server'),
                                             ('Windows', 'gufo.exe', 'gufo-server.exe')]:
                source = root / platform
                source.mkdir()
                (source / binary).write_bytes(b'executable')
                (source / 'CMakeLists.txt').write_text(
                    'cmake_minimum_required(VERSION 3.25)\nproject(test NONE)\n'
                    f'install(FILES "{binary}" DESTINATION bin)\n'
                    f'include("{ROOT.as_posix()}/cmake/GufoInstallExecutableAlias.cmake")\n')
                build, prefix = source / 'build', source / 'relocated'
                subprocess.run(['cmake', '-S', str(source), '-B', str(build),
                                f'-DCMAKE_SYSTEM_NAME={platform}',
                                f'-DCMAKE_INSTALL_PREFIX={source.as_posix()}/unused'],
                               check=True, capture_output=True)
                subprocess.run(['cmake', '--install', str(build), '--prefix', str(prefix)],
                               check=True, capture_output=True)
                self.assertEqual((prefix / 'bin' / alias).read_bytes(), b'executable')
                self.assertFalse((source / 'unused').exists())

    def test_manifest_rejects_invalid_paths_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.txt'
            for text in ('', '../escape', '/absolute/path', 'design/../escape',
                         'design/a\ndesign/a\n', 'C:/escape', 'design\\file'):
                with self.subTest(text=text):
                    manifest.write_text(text)
                    with self.assertRaises(ValueError):
                        read_manifest(manifest)

    def test_missing_and_empty_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / 'design/program.bin'
            artifact.parent.mkdir()
            for exists in (False, True):
                if exists:
                    artifact.touch()
                with self.assertRaises(ValueError):
                    check_artifacts(root, ['design/program.bin'])
            artifact.write_bytes(b'instructions')
            check_artifacts(root, ['design/program.bin'])

    def test_deleted_variant_output_rebuilds_only_that_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'aie').mkdir()
            (root / 'aie/__init__.py').touch()
            (root / 'aie/iron.py').touch()
            kernels = root / 'kernels/qwen3pre'
            kernels.mkdir(parents=True)
            (kernels / 'qwen3pre.py').write_text(
                'from pathlib import Path\nimport sys\n'
                'out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n'
                'for suffix in (".xclbin", ".bin", "_plan.txt"):\n'
                '    (out / ("qwen3pre_" + sys.argv[2] + suffix)).write_text("runtime")\n')
            (root / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(test NONE)\n'
                f'include("{ROOT.as_posix()}/cmake/GufoNpuDesigns.cmake")\n'
                f'set(GUFO_NPU_KERNELS "{root.as_posix()}/kernels")\n'
                f'include("{ROOT.as_posix()}/cmake/GufoNpuDesignList.cmake")\n')
            build = root / 'build'
            with patch.dict(os.environ, {'PEANO_INSTALL_DIR': str(root), 'PYTHONPATH': str(root)}):
                subprocess.run(['cmake', '-S', str(root), '-B', str(build), '-G', 'Ninja',
                                '-DGUFO_NPU_DESIGN=qwen3pre',
                                f'-DGUFO_IRON_PYTHON={sys.executable}'], check=True, capture_output=True)
                command = ['cmake', '--build', str(build), '--target', 'npu_design_qwen3pre']
                subprocess.run(command, check=True, capture_output=True)
                unchanged = build / 'npu/qwen3pre/qwen3pre_256.bin'
                before = unchanged.stat().st_mtime_ns
                missing = build / 'npu/qwen3pre/qwen3pre_512_plan.txt'
                missing.unlink()
                subprocess.run(command, check=True, capture_output=True)
                self.assertTrue(missing.is_file())
                self.assertEqual(unchanged.stat().st_mtime_ns, before)

    def test_deleted_decoder_layout_rebuilds_only_its_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'aie').mkdir()
            (root / 'aie/__init__.py').touch()
            (root / 'aie/iron.py').touch()
            kernels = root / 'kernels/qwen3dec'
            kernels.mkdir(parents=True)
            (kernels / 'qwen3dec.py').write_text(
                'from pathlib import Path\nimport os, sys\n'
                'assert os.path.samefile(os.environ["CXX"], sys.executable)\n'
                'out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n'
                'for suffix in (".xclbin", "_dispatch.so", "_layout.txt"):\n'
                '    (out / (sys.argv[2] + suffix)).write_text("runtime")\n')
            (root / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(test NONE)\n'
                f'include("{ROOT.as_posix()}/cmake/GufoNpuDesigns.cmake")\n'
                f'set(GUFO_NPU_KERNELS "{root.as_posix()}/kernels")\n'
                f'include("{ROOT.as_posix()}/cmake/GufoNpuDesignList.cmake")\n')
            build = root / 'build'
            with patch.dict(os.environ, {'PEANO_INSTALL_DIR': str(root), 'PYTHONPATH': str(root)}):
                subprocess.run(['cmake', '-S', str(root), '-B', str(build), '-G', 'Ninja',
                                '-DGUFO_NPU_DESIGN=qwen3dec', '-DCMAKE_SYSTEM_NAME=Windows',
                                f'-DGUFO_NPU_HOST_CXX={sys.executable}',
                                f'-DGUFO_IRON_PYTHON={sys.executable}'], check=True, capture_output=True)
                command = ['cmake', '--build', str(build), '--target', 'npu_design_qwen3dec']
                subprocess.run(command, check=True, capture_output=True)
                unchanged = build / 'npu/qwen3dec/qwen3dec_dispatch.so'
                before = unchanged.stat().st_mtime_ns
                missing = build / 'npu/qwen3dec/qwen3dec_p_layout.txt'
                missing.unlink()
                subprocess.run(command, check=True, capture_output=True)
                self.assertTrue(missing.is_file())
                self.assertEqual(unchanged.stat().st_mtime_ns, before)

    def test_prebuilt_install_and_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            designs = root / 'designs'
            # A small standalone consumer uses the production declarations.
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(test NONE)\n'
                f'include("{ROOT.as_posix()}/cmake/GufoNpuDesigns.cmake")\n'
                f'include("{ROOT.as_posix()}/cmake/GufoNpuDesignList.cmake")\n')
            # Generate the expected manifest without invoking a compiler.
            (root / 'aie').mkdir()
            (root / 'aie/__init__.py').touch()
            (root / 'aie/iron.py').touch()
            build = root / 'build'
            def configure(*args, success=True):
                result = subprocess.run(
                    ['cmake', '-S', str(source), '-B', str(build), *args],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
            with patch.dict(os.environ, {'PEANO_INSTALL_DIR': str(root), 'PYTHONPATH': str(root)}):
                configure('-DCMAKE_SYSTEM_NAME=Linux', f'-DGUFO_IRON_PYTHON={sys.executable}')
            names = read_manifest(build / 'npu-artifacts.txt')
            self.assertIn('qwen3dec/qwen3dec_p_layout.txt', names)
            self.assertEqual(len(names), 21)
            for name in names:
                path = designs / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'runtime')
            (designs / 'qwen3dec/debug.mlir').write_text('compiler intermediate')
            configure(f'-DGUFO_NPU_PREBUILT_DESIGNS={designs}')
            prefix = root / 'installed'
            subprocess.run(['cmake', '--install', str(build), '--prefix', str(prefix)],
                           check=True, capture_output=True)
            installed = prefix / 'share/gufo/npu'
            check_artifacts(installed, names)
            self.assertEqual(read_manifest(installed / 'npu-artifacts.txt'), names)
            self.assertFalse((installed / 'qwen3dec/debug.mlir').exists())
            build = root / 'windows-build'
            configure('-DCMAKE_SYSTEM_NAME=Windows', f'-DGUFO_NPU_PREBUILT_DESIGNS={designs}')
            prefix = root / 'windows-installed'
            subprocess.run(['cmake', '--install', str(build), '--prefix', str(prefix)],
                           check=True, capture_output=True)
            check_artifacts(prefix / 'bin/npu', names)
            self.assertEqual(read_manifest(prefix / 'bin/npu/npu-artifacts.txt'), names)
            (designs / names[-1]).unlink()
            configure(f'-DGUFO_NPU_PREBUILT_DESIGNS={designs}', success=False)


if __name__ == '__main__':
    unittest.main()
