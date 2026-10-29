#!/usr/bin/env python3
"""Windows packaging regressions; runs on Linux and Windows with pefile."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "check_bundle", Path(__file__).resolve().parents[2] / "tools/windows/check_bundle.py"
)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class WindowsBundleTest(unittest.TestCase):
    def test_missing_transitive_openmp_fails_even_if_installed_on_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gufo.exe").touch()
            (root / "rocalution.dll").touch()
            imports = {
                "gufo.exe": {"rocalution.dll", "kernel32.dll"},
                "rocalution.dll": {"libomp140.x86_64.dll"},
                "LIBOMP140.x86_64.DLL": {"api-ms-win-crt-runtime-l1-1-0.dll"},
            }
            with patch.object(bundle, "imported_dlls", side_effect=lambda p: imports[p.name]):
                self.assertEqual(
                    bundle.check_bundle(root),
                    ["rocalution.dll requires unbundled libomp140.x86_64.dll"],
                )
                (root / "LIBOMP140.x86_64.DLL").touch()
                self.assertEqual(bundle.check_bundle(root), [])

    def test_audio_imports_allow_winmm_but_require_bundled_codecs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("gufo.exe", "sndfile.dll", "out123.dll"):
                (root / name).touch()
            imports = {
                "gufo.exe": {"sndfile.dll"},
                "sndfile.dll": {"libmpg123.dll"},
                "out123.dll": {"WINMM.DLL"},
                "libmpg123.dll": {"kernel32.dll"},
            }
            with patch.object(bundle, "imported_dlls", side_effect=lambda p: imports[p.name]):
                self.assertEqual(
                    bundle.check_bundle(root),
                    ["sndfile.dll requires unbundled libmpg123.dll"],
                )
                (root / "libmpg123.dll").touch()
                self.assertEqual(bundle.check_bundle(root), [])

    def test_video_imports_allow_ncrypt_but_require_bundled_codecs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("gufo.exe", "avformat-63.dll"):
                (root / name).touch()
            imports = {
                "gufo.exe": {"avformat-63.dll"},
                "avformat-63.dll": {"NCRYPT.DLL", "avcodec-63.dll"},
                "avcodec-63.dll": {"kernel32.dll"},
            }
            with patch.object(bundle, "imported_dlls", side_effect=lambda p: imports[p.name]):
                self.assertEqual(
                    bundle.check_bundle(root),
                    ["avformat-63.dll requires unbundled avcodec-63.dll"],
                )
                (root / "avcodec-63.dll").touch()
                self.assertEqual(bundle.check_bundle(root), [])

    def test_lazy_video_libraries_are_required_without_pe_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gufo.exe").touch()
            (root / "gufo-video-libraries.txt").write_text("avformat-63.dll\n", encoding="utf-8")
            with patch.object(bundle, "imported_dlls", return_value={"kernel32.dll"}):
                self.assertEqual(bundle.check_bundle(root),
                                 ["video runtime requires unbundled avformat-63.dll"])
                (root / "AVFORMAT-63.DLL").touch()
                self.assertEqual(bundle.check_bundle(root), [])

    def test_developer_and_crt_libraries_are_not_system_dependencies(self):
        for name in ("libomp140.x86_64.dll", "libomp.dll", "msvcp140.dll", "vcruntime140.dll"):
            with self.subTest(name=name):
                self.assertFalse(bundle.is_system_dll(name))
        self.assertTrue(bundle.is_system_dll("KERNEL32.DLL"))

    def test_normal_and_delay_imports_are_checked(self):
        from types import SimpleNamespace as Object

        pe = Object(
            FILE_HEADER=Object(Machine=0x8664),
            DIRECTORY_ENTRY_IMPORT=[Object(dll=b"KERNEL32.dll")],
            DIRECTORY_ENTRY_DELAY_IMPORT=[Object(dll=b"missing.dll")],
        )
        with patch.object(bundle.pefile, "PE") as reader:
            reader.return_value.__enter__.return_value = pe
            with patch.object(pe, "parse_data_directories", create=True):
                self.assertEqual(bundle.imported_dlls(Path("gufo.exe")), {"kernel32.dll", "missing.dll"})

    def test_missing_executable_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing gufo.exe"):
                bundle.check_bundle(Path(tmp))


if __name__ == "__main__":
    unittest.main()
