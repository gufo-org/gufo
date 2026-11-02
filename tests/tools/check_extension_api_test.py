#!/usr/bin/env python3
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("extension_api", ROOT / "tools/ci/check-extension-api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class ApiPolicyTests(unittest.TestCase):
    def fixture(self):
        return {api.SURFACE: "include/gufo/ext/version.hpp\nsrc/cache/forward.hpp\n",
                api.SERIES: json.dumps({"major": 0, "minor": 1, "revision_base": 0}),
                "src/cache/forward.hpp": "struct State { int bytes; };"}

    def test_forwarded_declaration_requires_all_policy_parts(self):
        old = self.fixture(); new = old.copy()
        new["src/cache/forward.hpp"] = "struct State { long bytes; };"
        self.assertEqual(len(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), [])), 4)
        records = {f"{api.RECORDS}/0.1/test-change.md": "Describe the API change.\n"}
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), ["ext-api"], {}, records), [])

    def test_whitespace_comments_and_strings(self):
        self.assertEqual(api.tokens('int x; // old\n'), api.tokens('int  x; /* new */'))
        self.assertNotEqual(api.tokens('auto s = "//old";'), api.tokens('auto s = "//new";'))
        self.assertEqual(api.tokens('set(X "#value") # old', 'x.cmake'),
                         api.tokens('set(X "#value") #[=[ new ]=]', 'x.cmake'))
        self.assertNotEqual(api.tokens('auto s = R"x(//old)x";'),
                            api.tokens('auto s = R"x(//new)x";'))
        old = self.fixture(); new = old.copy(); new["src/cache/forward.hpp"] += "\n// comment"
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), []), [])

    def test_deleted_and_cmake_contracts(self):
        old = self.fixture(); new = old.copy(); del new["src/cache/forward.hpp"]
        new[api.SURFACE] = "include/gufo/ext/version.hpp\n"
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), [])
        self.assertEqual(len(errors), 4)
        self.assertTrue(any("ext-api" in e for e in errors))
        records = {f"{api.RECORDS}/0.1/test-change.md": "Describe the API change.\n"}
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), ["ext-api"], {}, records), [])
        new = old.copy(); new["cmake/GufoExtension.cmake"] = "function(new_contract)\nendfunction()"
        self.assertTrue(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), []))

    def test_transitive_addition_requires_policy(self):
        old = self.fixture(); new = old.copy()
        new[api.SURFACE] += "include/gufo/ext/platform/fs.hpp\nsrc/core/platform/fs.hpp\n"
        new["include/gufo/ext/platform/fs.hpp"] = '#pragma once\n#include "src/core/platform/fs.hpp"\n'
        new["src/core/platform/fs.hpp"] = 'namespace gufo::platform { bool Exists(const char*); }\n'
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), [])
        self.assertEqual(len(errors), 4)
        records = {f"{api.RECORDS}/0.1/test-change.md": "Describe the API change.\n"}
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), ["ext-api"], {}, records), [])

    def test_subdirectories_declaration_modification_requires_policy(self):
        old = self.fixture()
        old[api.SURFACE] += "include/gufo/ext/experimental/hip_utils.hpp\nsrc/core/hip/hip_utils.hpp\n"
        old["include/gufo/ext/experimental/hip_utils.hpp"] = '#pragma once\n#include "src/core/hip/hip_utils.hpp"\n'
        old["src/core/hip/hip_utils.hpp"] = 'namespace gufo::hip { void Sync(); }\n'
        new = old.copy()
        new["src/core/hip/hip_utils.hpp"] = 'namespace gufo::hip { void Sync(int stream); }\n'
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), [])
        self.assertEqual(len(errors), 4)
        records = {f"{api.RECORDS}/0.1/test-change.md": "Describe the API change.\n"}
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), ["ext-api"], {}, records), [])

    def test_parallel_records_advance_after_merge_without_shared_edits(self):
        series = self.fixture()[api.SERIES]
        a = {f"{api.RECORDS}/0.1/pr-a.md": "Add A"}
        b = {f"{api.RECORDS}/0.1/pr-b.md": "Add B"}
        self.assertEqual(api.computed_version(series, a), "0.1.1")
        self.assertEqual(api.computed_version(series, b), "0.1.1")
        self.assertEqual(api.computed_version(series, a | b), "0.1.2")
        old = self.fixture(); new = old.copy()
        new["src/cache/forward.hpp"] += " void Added();"
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                        ["ext-api"], a, a | b), [])

    def test_records_cannot_be_edited_deleted_or_renamed(self):
        old = self.fixture()
        records = {f"{api.RECORDS}/0.1/original.md": "Original change"}
        for new_records in ({}, {next(iter(records)): "Edited"},
                            {f"{api.RECORDS}/0.1/renamed.md": "Original change"}):
            with self.subTest(records=new_records):
                errors = api.check_diff(lambda p: old.get(p, ""), lambda p: old.get(p, ""),
                                        ["ext-api"], records, new_records)
                self.assertTrue(any("immutable" in e for e in errors))

    def test_descriptive_record_normalization_preserves_descriptions_and_version(self):
        old = self.fixture()
        previous = {f"{api.RECORDS}/0.1/{name}.md": f"Original {name} description\n"
                    for name in api.LEGACY_RECORD_NAMES}
        normalized = {f"{api.RECORDS}/0.1/{name}.md":
                      f"# {title}\n\n{previous[f'{api.RECORDS}/0.1/{legacy}.md']}"
                      for legacy, (name, title) in api.LEGACY_RECORD_NAMES.items()}
        self.assertEqual(api.computed_version(old[api.SERIES], previous),
                         api.computed_version(old[api.SERIES], normalized))
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: old.get(p, ""),
                                        [], previous, normalized), [])
        already_normalized = normalized.copy()
        path = f"{api.RECORDS}/0.1/automatic-versioning.md"
        duplicate_heading = normalized | {path: "# Automatic API revision derivation\n\n" + normalized[path]}
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: old.get(p, ""),
                                ["ext-api"], already_normalized, duplicate_heading)
        self.assertTrue(any("immutable" in e for e in errors))
        normalized[f"{api.RECORDS}/0.1/automatic-versioning.md"] += "Edited history"
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: old.get(p, ""),
                                ["ext-api"], previous, normalized)
        self.assertTrue(any("immutable" in e for e in errors))

    def test_repository_records_have_descriptive_names_without_versions(self):
        for path, body in api.records_for_root(ROOT).items():
            with self.subTest(path=path):
                self.assertRegex(Path(path).stem, r"^[a-z][a-z0-9_-]*$")
                self.assertNotRegex(body, r"(?m)^# [0-9]+\.[0-9]+\.[0-9]+")

    def test_guidance_and_agents_symlink_do_not_advance_api_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = root / api.RECORDS
            (records / "0.1").mkdir(parents=True)
            (root / api.SERIES).write_text(self.fixture()[api.SERIES])
            (records / "0.1/baseline.md").write_text("# Source API baseline\n\nBaseline")
            (records / "README.md").write_text("Record instructions")
            for name in ("AGENTS.md", "CLAUDE.md"):
                if os.name != "nt":
                    (records / name).symlink_to("README.md")
                else:
                    (records / name).write_text("README.md")
            self.assertEqual(api.version_for_root(root), "0.1.0")
            self.assertEqual(len(api.records_for_root(root)), 1)

    def test_wrong_series_and_base_edits_rejected(self):
        old = self.fixture(); new = old.copy()
        records = {f"{api.RECORDS}/0.2/change.md": "Wrong series"}
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                ["ext-api"], {}, records)
        self.assertTrue(any("active series" in e for e in errors))
        new[api.SERIES] = json.dumps({"major": 0, "minor": 1, "revision_base": 5})
        records = {f"{api.RECORDS}/0.1/change.md": "Change"}
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                ["ext-api"], {}, records)
        self.assertTrue(any("revision_base is fixed" in e for e in errors))

    def test_compatibility_series_transition_retains_history(self):
        old = self.fixture(); new = old.copy()
        new[api.SERIES] = json.dumps({"major": 0, "minor": 2, "revision_base": 0})
        previous = {f"{api.RECORDS}/0.1/old.md": "Old API"}
        records = previous | {f"{api.RECORDS}/0.2/break.md": "Breaking API change"}
        self.assertEqual(api.computed_version(new[api.SERIES], records), "0.2.1")
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                        ["ext-api"], previous, records), [])

    def test_record_validation_and_legacy_migration(self):
        self.assertTrue(api.record_errors({f"{api.RECORDS}/0.1/empty.md": "  "}))
        self.assertTrue(api.record_errors({f"{api.RECORDS}/0.1/BAD.json": "Change"}))
        old = self.fixture(); del old[api.SERIES]
        old[api.VERSION] = 'inline constexpr auto kApiVersion = "0.1.9";'
        new = self.fixture()
        new[api.SERIES] = json.dumps({"major": 0, "minor": 1, "revision_base": 9})
        records = {f"{api.RECORDS}/0.1/migration.md": "Migrate versioning"}
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                        ["ext-api"], {}, records), [])

    def test_migration_cannot_reuse_existing_revision_numbers(self):
        old = self.fixture(); del old[api.SERIES]
        old[api.VERSION] = 'inline constexpr auto kApiVersion = "0.1.10";'
        new = self.fixture()
        records = {f"{api.RECORDS}/0.1/one.md": "First change",
                   f"{api.RECORDS}/0.1/two.md": "Second change"}
        new[api.SERIES] = json.dumps({"major": 0, "minor": 1, "revision_base": 9})
        # 9 + 2 is newer than 10, but it reuses revision 10 for a new record.
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                ["ext-api"], {}, records)
        self.assertTrue(any("preserve the existing API revision" in e for e in errors))
        new[api.SERIES] = json.dumps({"major": 0, "minor": 1, "revision_base": 10})
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                        ["ext-api"], {}, records), [])

    def test_historical_conversion_preserves_versions_and_requires_complete_history(self):
        old = self.fixture(); del old[api.SERIES]
        old[api.VERSION] = 'inline constexpr auto kApiVersion = "0.1.10";'
        new = self.fixture()
        records = {f"{api.RECORDS}/0.1/0.md": "# 0.1.0\n\nInitial contracts"}
        for revision in range(1, 11):
            records[f"{api.RECORDS}/0.1/{revision}.md"] = f"# 0.1.{revision}\n\nExisting change"
        self.assertEqual(api.computed_version(new[api.SERIES], records), "0.1.10")
        records[f"{api.RECORDS}/0.1/migration.md"] = "Automatic versioning"
        self.assertEqual(api.computed_version(new[api.SERIES], records), "0.1.11")
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                        ["ext-api"], {}, records), [])
        # An extra new record must not conceal a missing historical revision.
        records[f"{api.RECORDS}/0.1/extra.md"] = "Another change"
        del records[f"{api.RECORDS}/0.1/5.md"]
        self.assertEqual(api.computed_version(new[api.SERIES], records), "0.1.11")
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""),
                                ["ext-api"], {}, records)
        self.assertTrue(any("missing historical record for 0.1.5" in e for e in errors))

    def test_cmake_and_python_versions_match_and_build_reconfigures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cmake").mkdir()
            (root / "include/gufo/ext").mkdir(parents=True)
            for name in ("GufoExtensionVersion.cmake", "GufoExtensionVersion.hpp.in"):
                shutil.copyfile(ROOT / "cmake" / name, root / "cmake" / name)
            (root / api.SERIES).write_text(json.dumps({"major": 0, "minor": 1, "revision_base": 9}))
            records = root / api.RECORDS / "0.1"
            records.mkdir(parents=True)
            # The baseline record preserves its description but is not a bump.
            (records / "baseline.md").write_text("# Source API baseline\n\nBaseline contracts")
            (records / "first.md").write_text("First change")
            (root / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.21)
project(api_identity LANGUAGES CXX)
add_subdirectory(nested)
include(cmake/GufoExtensionVersion.cmake)
configure_file(cmake/GufoExtensionVersion.hpp.in version.hpp @ONLY)
add_executable(identity main.cpp)
target_compile_features(identity PRIVATE cxx_std_20)
target_include_directories(identity PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
''')
            (root / "main.cpp").write_text('#include "version.hpp"\n#include <iostream>\n'
                                           'int main() { std::cout << gufo::ext::kApiVersion; }\n')
            (root / "nested").mkdir()
            (root / "nested/CMakeLists.txt").write_text(
                'include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GufoExtensionVersion.cmake")\n')
            build = root / "build"
            result = subprocess.run(["cmake", "-S", str(root), "-B", str(build), "-GNinja"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for expected in ("0.1.10", "0.1.11"):
                if expected.endswith("11"):
                    (records / "parallel-pr.md").write_text("Independent API change")
                result = subprocess.run(["cmake", "--build", str(build), "--config", "Release"],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                executable = build / ("identity.exe" if os.name == "nt" else "identity")
                self.assertEqual(subprocess.check_output([str(executable)], text=True), expected)
                self.assertEqual(api.version_for_root(root), expected)

    def test_changelog_keeps_all_series_and_historical_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "include/gufo/ext").mkdir(parents=True)
            (root / api.SERIES).write_text(json.dumps({"major": 0, "minor": 2, "revision_base": 0}))
            for series, body in (("0.1", "First series change"), ("0.2", "Current series change")):
                directory = root / api.RECORDS / series
                directory.mkdir(parents=True)
                (directory / "feature.md").write_text(body)
            (root / api.RECORDS / "0.1/0.md").write_text("# 0.1.0\n\nOriginal contracts\n")
            (root / api.RECORDS / "0.1/1.md").write_text("# 0.1.1\n\nPreviously assigned version\n")
            generated = api.changelog(root)
            for text in ("Current API: 0.2.1", "First series change", "Current series change", "Original contracts", "### 0.1.0", "### 0.1.1"):
                self.assertIn(text, generated)
            self.assertLess(generated.index("## Series 0.2"), generated.index("## Series 0.1"))

    def test_transitive_forward_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "include/gufo/ext").mkdir(parents=True); (root / "src/cache").mkdir(parents=True)
            (root / "include/gufo/ext/api.hpp").write_text('#include "src/cache/forward.hpp"\n')
            (root / "src/cache/forward.hpp").write_text('#include "detail.hpp"\n')
            (root / "src/cache/detail.hpp").write_text('struct State {};')
            self.assertEqual(api.closure(root), ["include/gufo/ext/api.hpp", "src/cache/detail.hpp", "src/cache/forward.hpp"])

    def test_transitive_forward_closure_with_subdirectories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "include/gufo/ext/platform").mkdir(parents=True)
            (root / "include/gufo/ext/experimental").mkdir(parents=True)
            (root / "src/core/platform").mkdir(parents=True)
            (root / "src/core/hip/detail").mkdir(parents=True)
            (root / "include/gufo/ext/platform/fs.hpp").write_text('#include "src/core/platform/fs.hpp"\n')
            (root / "src/core/platform/fs.hpp").write_text('namespace gufo::platform {}\n')
            (root / "include/gufo/ext/experimental/hip_utils.hpp").write_text('#include "src/core/hip/hip_utils.hpp"\n')
            (root / "src/core/hip/hip_utils.hpp").write_text('#include "detail/dispatch.hpp"\n')
            (root / "src/core/hip/detail/dispatch.hpp").write_text('namespace gufo::hip {}\n')
            self.assertEqual(api.closure(root), [
                "include/gufo/ext/experimental/hip_utils.hpp",
                "include/gufo/ext/platform/fs.hpp",
                "src/core/hip/detail/dispatch.hpp",
                "src/core/hip/hip_utils.hpp",
                "src/core/platform/fs.hpp",
            ])


if __name__ == "__main__": unittest.main()
