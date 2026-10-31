#!/usr/bin/env python3
import importlib.util
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
                api.VERSION: 'inline constexpr auto kApiVersion = "0.1.0";',
                api.CHANGES: "## 0.1.0\nInitial\n",
                "src/cache/forward.hpp": "struct State { int bytes; };"}

    def test_forwarded_declaration_requires_all_policy_parts(self):
        old = self.fixture(); new = old.copy()
        new["src/cache/forward.hpp"] = "struct State { long bytes; };"
        self.assertEqual(len(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), [])), 3)
        new[api.VERSION] = new[api.VERSION].replace("0.1.0", "0.1.1")
        new[api.CHANGES] = "## 0.1.1\nAdd fields\n" + new[api.CHANGES]
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), ["ext-api"]), [])

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
        self.assertEqual(len(errors), 3)
        self.assertTrue(any("ext-api" in e for e in errors))
        new[api.VERSION] = new[api.VERSION].replace("0.1.0", "0.1.1")
        new[api.CHANGES] = "## 0.1.1\nRemove header\n" + new[api.CHANGES]
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), ["ext-api"]), [])
        new = old.copy(); new["cmake/GufoExtension.cmake"] = "function(new_contract)\nendfunction()"
        self.assertTrue(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), []))

    def test_transitive_addition_requires_policy(self):
        old = self.fixture(); new = old.copy()
        new[api.SURFACE] += "include/gufo/ext/platform/fs.hpp\nsrc/core/platform/fs.hpp\n"
        new["include/gufo/ext/platform/fs.hpp"] = '#pragma once\n#include "src/core/platform/fs.hpp"\n'
        new["src/core/platform/fs.hpp"] = 'namespace gufo::platform { bool Exists(const char*); }\n'
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), [])
        self.assertEqual(len(errors), 3)
        new[api.VERSION] = new[api.VERSION].replace("0.1.0", "0.1.1")
        new[api.CHANGES] = "## 0.1.1\nAdd platform fs\n" + new[api.CHANGES]
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), ["ext-api"]), [])

    def test_subdirectories_declaration_modification_requires_policy(self):
        old = self.fixture()
        old[api.SURFACE] += "include/gufo/ext/experimental/hip_utils.hpp\nsrc/core/hip/hip_utils.hpp\n"
        old["include/gufo/ext/experimental/hip_utils.hpp"] = '#pragma once\n#include "src/core/hip/hip_utils.hpp"\n'
        old["src/core/hip/hip_utils.hpp"] = 'namespace gufo::hip { void Sync(); }\n'
        new = old.copy()
        new["src/core/hip/hip_utils.hpp"] = 'namespace gufo::hip { void Sync(int stream); }\n'
        errors = api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), [])
        self.assertEqual(len(errors), 3)
        new[api.VERSION] = new[api.VERSION].replace("0.1.0", "0.1.1")
        new[api.CHANGES] = "## 0.1.1\nUpdate hip_utils signature\n" + new[api.CHANGES]
        self.assertEqual(api.check_diff(lambda p: old.get(p, ""), lambda p: new.get(p, ""), ["ext-api"]), [])

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
