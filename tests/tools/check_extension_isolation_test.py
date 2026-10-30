#!/usr/bin/env python3
"""Regression coverage for extension coupling inventory and header closure."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_extension_isolation", ROOT / "tools/ci/check-extension-isolation.py")
ISOLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ISOLATION)


class ExtensionIsolationTest(unittest.TestCase):
    def test_parent_relative_edges_and_transitive_closure(self):
        files = {
            "src/models/qwen/a.hpp": '#include "../shared/bridge.hpp"\n',
            "src/models/shared/bridge.hpp": '#include "../../core/types.hpp"\n',
            "src/core/types.hpp": "struct Type {};\n",
            "src/models/other/run.cpp": '#include "../qwen/a.hpp"\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            report = ISOLATION.inventory(root)
        self.assertEqual(report["cross_owner_counts"], {
            "models/other -> models/qwen": 1,
            "models/qwen -> models/shared": 1,
            "models/shared -> src/core": 1,
        })
        header = report["qwen_headers"]["src/models/qwen/a.hpp"]
        self.assertEqual(header["direct"], ["src/models/shared/bridge.hpp"])
        self.assertEqual(header["transitive"], [
            "src/core/types.hpp", "src/models/shared/bridge.hpp"])
        self.assertEqual(header["consumers"], ["src/models/other/run.cpp"])
        self.assertEqual(header["unresolved"], [])

    def test_only_cpp_directives_outside_strings_create_includes(self):
        source = '''auto raw = u8R"fixture(
#include "src/models/qwen/fake.hpp"
)fixture";
const char* escaped = "continued\\
#include <fake.hpp>";
/*
#include "comment.hpp"
*/
#if 0
  # include "src/models/qwen/real.hpp"
#endif
'''
        python_source = "fixture = '''\n#include \"src/models/qwen/fake.hpp\"\n'''\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "src/models/qwen/real.hpp": "struct Real {};\n",
                "src/core/use.cpp": source,
                "tests/fixtures/use.py": python_source,
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            report = ISOLATION.inventory(root)
        self.assertEqual(report["includes"], [{
            "path": "src/core/use.cpp", "line": 10, "owner": "src/core",
            "include": "src/models/qwen/real.hpp",
            "resolved": "src/models/qwen/real.hpp", "dependency": "models/qwen"}])

    def test_comments_do_not_create_dependencies(self):
        source = '''// qwen::Ignored\n/* #include "src/models/qwen/fake.hpp" */
#include "src/models/qwen/real.hpp" // qwen::Ignored
const char* url = "https://example/qwen::ignored";
auto raw = R"tag(/* qwen::String */)tag";
qwen::Real real;
'''
        cleaned = ISOLATION.strip_comments(source, ".cpp")
        self.assertNotIn("Ignored", cleaned)
        self.assertEqual(cleaned.count("\n"), source.count("\n"))
        self.assertIn('"https://example/qwen::ignored"', cleaned)
        self.assertEqual(ISOLATION.INCLUDE.findall(cleaned), ["src/models/qwen/real.hpp"])
        for suffix in (".cmake", ".nix", ".json"):
            source = '\"**/*.cpp\" /* qwen ignored */ \"https://example/qwen\"\n'
            cleaned = ISOLATION.strip_comments(source, suffix)
            self.assertIn('"**/*.cpp"', cleaned)
            self.assertIn('"https://example/qwen"', cleaned)
            self.assertNotIn("ignored", cleaned)
        for suffix in (".cmake", ".nix", ".py", ".sh"):
            cleaned = ISOLATION.strip_comments('set("qwen#value") # qwen ignored\n', suffix)
            self.assertIn('"qwen#value"', cleaned)
            self.assertNotIn("ignored", cleaned)

    def test_inventory_crosses_all_layers_and_closes_header_cycles(self):
        files = {
            "src/models/qwen/a.hpp": '#include "b.hpp"\n#include <vector>\n',
            "src/models/qwen/b.hpp": '#include "a.hpp"\n#include "src/core/types.hpp"\n',
            "src/core/types.hpp": "struct Type {};\n",
            "src/families/shared/tokens.hpp": "struct Tokens {};\n",
            "src/models/other/run.cpp": '#include "src/families/shared/tokens.hpp"\nqwen::Run();\n',
            "src/cache/use.cpp": '#include "src/models/qwen/a.hpp"\n',
            "src/cli/use.cpp": '#include "src/models/qwen/a.hpp"\n',
            "tests/core/use.cpp": '#include "src/models/qwen/a.hpp"\n',
            "tests/fixtures/use.cpp": '#include "src/models/qwen/a.hpp"\n',
            "tools/tune.cpp": '#include "src/models/qwen/a.hpp"\n',
            "tools/quality.py": 'MODEL = "qwen27b" # ignored qwen::Comment\n',
            ".github/workflows/check.yml": 'env:\n  GUFO_ASSETS: qwen27b\n# qwen ignored\n',
            "flake.nix": 'x = "models/qwen/a.hpp"; # qwen ignored\n',
            "tests/functional/devlab.json": '{"model": "qwen"}\n',
            "CMakeLists.txt": 'add_library(gufo_other src/models/other/run.cpp)\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            report = ISOLATION.inventory(root)
            self.assertEqual(report, ISOLATION.inventory(root))
        header = report["qwen_headers"]["src/models/qwen/a.hpp"]
        self.assertEqual(header["transitive"], ["src/core/types.hpp", "src/models/qwen/b.hpp"])
        self.assertEqual(header["unresolved"], ["vector"])
        self.assertIn("tools/tune.cpp", header["consumers"])
        edges = report["cross_owner_counts"]
        for owner in ("src/cache", "src/cli", "tests/core", "tests/fixtures", "tools/tune.cpp"):
            self.assertIn(f"{owner} -> models/qwen", edges)
        self.assertIn("models/other -> families/shared", edges)
        self.assertEqual(report["namespace_references"], [{
            "path": "src/models/other/run.cpp", "line": 2, "owner": "models/other",
            "dependency": "models/qwen", "symbol": "qwen::"}])
        self.assertTrue(any(item["path"] == "tools/quality.py"
                            for item in report["name_references"]))
        mentions = report["configuration_mentions"]
        self.assertTrue(any(item["path"].startswith(".github/") for item in mentions))
        self.assertTrue(any(item["path"] == "tests/functional/devlab.json" for item in mentions))
        self.assertFalse(any("ignored" in item["text"] for item in mentions))


if __name__ == "__main__":
    unittest.main()
