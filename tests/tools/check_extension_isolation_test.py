#!/usr/bin/env python3
"""Regression coverage for extension coupling inventory and header closure."""

import importlib.util
import json
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

    def test_core_boundary_blocks_model_and_family_includes(self):
        files = {
            "src/models/qwen/verifier.hpp": "struct Verifier {};\n",
            "src/families/qwen/speculation.hpp": "struct Spec {};\n",
            "src/core/speculative/verifier.cpp":
                '#include "src/models/qwen/verifier.hpp"\n',
            "src/core/speculative/near.cpp":
                '#include "../../families/qwen/speculation.hpp"\n',
            "src/core/sampling.hpp": '#include "src/core/json.hpp"\n',
            "src/core/json.hpp": "struct Json {};\n",
            "src/cli/serve/use.cpp": '#include "src/models/qwen/verifier.hpp"\n',
            "tests/core/use.cpp": '#include "src/models/qwen/verifier.hpp"\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            violations = ISOLATION.core_boundary_violations(
                ISOLATION.inventory(root))
        self.assertEqual(violations, [
            "src/core/speculative/near.cpp:1: ../../families/qwen/speculation.hpp",
            "src/core/speculative/verifier.cpp:1: src/models/qwen/verifier.hpp",
        ])

    @staticmethod
    def inventory(files):
        """Build a scratch tree and return its inventory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            return ISOLATION.inventory(root)

    @staticmethod
    def manifest(name, target, kind, depends):
        return json.dumps({"name": name, "target": target, "kind": kind,
                           "purpose": f"Card 07 fixture library {name}",
                           "depends": depends})

    def family_fixture(self):
        """A minimal acyclic family tree whose manifests match every include."""
        return {
            "src/families/qwen/control_tokens/library.json": self.manifest(
                "control_tokens", "gufo_qwen_control_tokens", "interface", []),
            "src/families/qwen/control_tokens/control_tokens.hpp":
                "inline constexpr unsigned kImStart = 1;\n",
            "src/families/qwen/hip_common/library.json": self.manifest(
                "hip_common", "gufo_qwen_hip_common", "interface", ["control_tokens"]),
            "src/families/qwen/hip_common/rope.hpp":
                '#include "src/families/qwen/control_tokens/control_tokens.hpp"\n',
            "src/families/qwen/text/library.json": self.manifest(
                "text", "gufo_qwen_text", "static", ["control_tokens"]),
            "src/families/qwen/text/tokenizer.hpp":
                '#include "src/families/qwen/control_tokens/control_tokens.hpp"\n',
            "src/families/qwen/kernels/library.json": self.manifest(
                "kernels", "gufo_qwen_kernels", "static", ["hip_common", "text"]),
            "src/families/qwen/kernels/kernel.hpp":
                '#include "src/families/qwen/hip_common/rope.hpp"\n'
                '#include "src/families/qwen/text/tokenizer.hpp"\n',
            "src/models/qwen/executor.hpp": "struct Executor {};\n",
            "src/models/other/run.cpp":
                '#include "src/families/qwen/kernels/kernel.hpp"\n',
        }

    def test_clean_family_graph_is_declared_and_violation_free(self):
        report = self.inventory(self.family_fixture())
        self.assertEqual(report["family_violations"], [])
        self.assertEqual(sorted(report["family_libraries"]), [
            "qwen/control_tokens", "qwen/hip_common", "qwen/kernels", "qwen/text"])
        self.assertEqual(report["family_edges"], [
            "qwen/hip_common -> qwen/control_tokens",
            "qwen/kernels -> qwen/hip_common",
            "qwen/kernels -> qwen/text",
            "qwen/text -> qwen/control_tokens"])

    def test_undeclared_and_unused_family_dependencies_fail(self):
        undeclared = self.family_fixture()
        undeclared["src/families/qwen/kernels/library.json"] = self.manifest(
            "kernels", "gufo_qwen_kernels", "static", ["hip_common"])
        self.assertIn(
            "src/families/qwen/kernels/library.json: undeclared dependency "
            "qwen/text (headers include it)",
            self.inventory(undeclared)["family_violations"])
        unused = self.family_fixture()
        unused["src/families/qwen/kernels/kernel.hpp"] = \
            '#include "src/families/qwen/hip_common/rope.hpp"\n'
        self.assertIn(
            "src/families/qwen/kernels/library.json: declared dependency qwen/text "
            "is not used by any header",
            self.inventory(unused)["family_violations"])
        unknown = self.family_fixture()
        unknown["src/families/qwen/text/library.json"] = self.manifest(
            "text", "gufo_qwen_text", "static", ["nonexistent"])
        violations = self.inventory(unknown)["family_violations"]
        self.assertIn("src/families/qwen/text/library.json: unknown dependency "
                      "nonexistent", violations)
        self.assertIn("src/families/qwen/text/library.json: declared dependency "
                      "qwen/nonexistent is not used by any header", violations)
        self.assertIn("src/families/qwen/text/library.json: undeclared dependency "
                      "qwen/control_tokens (headers include it)", violations)

    def test_family_dependency_cycle_fails(self):
        cyclic = self.family_fixture()
        cyclic["src/families/qwen/hip_common/library.json"] = self.manifest(
            "hip_common", "gufo_qwen_hip_common", "interface",
            ["control_tokens", "kernels"])
        cyclic["src/families/qwen/hip_common/rope.hpp"] += \
            '#include "src/families/qwen/kernels/kernel.hpp"\n'
        violations = self.inventory(cyclic)["family_violations"]
        self.assertIn("family dependency cycle: qwen/hip_common -> qwen/kernels -> qwen/hip_common", violations)
        self_recursive = self.family_fixture()
        self_recursive["src/families/qwen/text/library.json"] = self.manifest(
            "text", "gufo_qwen_text", "static", ["text"])
        self.assertIn("src/families/qwen/text/library.json: library depends on itself",
                      self.inventory(self_recursive)["family_violations"])

    def test_family_code_reaching_a_model_header_fails(self):
        leaking = self.family_fixture()
        leaking["src/families/qwen/kernels/kernel.hpp"] += \
            '#include "src/models/qwen/executor.hpp"\n'
        self.assertEqual(
            self.inventory(leaking)["family_violations"],
            ["src/families/qwen/kernels/kernel.hpp: family code includes model header "
             "src/models/qwen/executor.hpp"])

    def test_family_files_need_a_manifest_describing_their_library(self):
        stray = self.family_fixture()
        stray["src/families/qwen/foo.hpp"] = "struct RootPiece {};\n"
        stray["src/families/qwen/unassigned/pieces.hpp"] = "struct Piece {};\n"
        stray["src/families/qwen/text/library.json"] = self.manifest(
            "tokenz", "gufo_qwen_text", "static", ["control_tokens"])
        violations = self.inventory(stray)["family_violations"]
        self.assertIn("src/families/qwen/unassigned/pieces.hpp: file in src/families is "
                      "not inside a library with a library.json manifest", violations)
        self.assertIn("src/families/qwen/text/library.json: manifest name 'tokenz' != "
                      "directory qwen/text", violations)
        self.assertIn("src/families/qwen/foo.hpp: file in src/families is "
                      "not inside a library with a library.json manifest", violations)
        undocumented = self.family_fixture()
        undocumented["src/families/qwen/text/library.json"] = json.dumps(
            {"name": "text", "target": "", "kind": "header", "depends": ["control_tokens"]})
        violations = self.inventory(undocumented)["family_violations"]
        self.assertIn("src/families/qwen/text/library.json: manifest needs a non-empty "
                      "CMake target", violations)
        self.assertIn("src/families/qwen/text/library.json: kind must be interface or "
                      "static, got 'header'", violations)
        self.assertIn("src/families/qwen/text/library.json: manifest needs a purpose",
                      violations)

    def test_repository_satisfies_the_family_graph(self):
        # Card 07 split the Qwen family into libraries under src/families/qwen/.
        # If this fails, a family header moved or started including a model without
        # updating the owning library.json manifest, or the family graph became cyclic.
        self.assertEqual(ISOLATION.inventory(ROOT)["family_violations"], [])

    def test_repository_inventories_both_qwen_trees(self):
        # The RFC step-1 header analysis moved with the shared code: it must keep
        # covering the family libraries and the Qwen 27B model directory.
        report = ISOLATION.inventory(ROOT)
        headers = report["qwen_headers"]
        self.assertIn("src/families/qwen/control_tokens/control_tokens.hpp", headers)
        self.assertIn("src/families/qwen/hip_common/rope.hpp", headers)
        self.assertIn("src/models/qwen/hip/executor.hpp", headers)
        self.assertEqual(
            [h for h in headers if not h.startswith(("src/models/qwen/",
                                                    "src/families/qwen/"))], [])

    def test_repository_control_token_mirror_matches_the_header(self):
        # tools/gufo/control_tokens.py mirrors src/families/qwen/control_tokens
        # independently; a renamed or retargeted header breaks the mirror link.
        header = (ROOT / "src/families/qwen/control_tokens/control_tokens.hpp").read_text()
        mirror = (ROOT / "tools/gufo/control_tokens.py").read_text()
        for constant in ("kEndOfText", "kImStart", "kImEnd", "kVisionPad", "kImagePad",
                         "kVideoPad"):
            self.assertIn(constant, header)
            self.assertIn(constant, mirror)

    def test_target_ownership_needs_a_claim_for_every_dependent(self):
        files = {
            "src/models/demo/demo.cpp": "int demo() { return 1; }\n",
            "src/models/demo/library.json": json.dumps({
                "schema": 1, "name": "demo", "target": "gufo_demo", "kind": "static",
                "purpose": "Demo model."}),
            "src/models/demo/CMakeLists.txt":
                "add_library(gufo_demo STATIC demo.cpp)\n",
            "tests/tools/probe_test.cpp": "int main() {}\n",
            "CMakeLists.txt": (
                "add_subdirectory(src/models/demo)\n"
                "set(GUFO_DEMO_LIBS gufo_demo)\n"
                "add_executable(probe_test tests/tools/probe_test.cpp)\n"
                "target_link_libraries(probe_test PRIVATE ${GUFO_DEMO_LIBS})\n"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            _, edges, violations = ISOLATION.target_ownership(root)
            self.assertTrue(any("probe_test depends on demo" in violation
                                for violation in violations), violations)
            # A manifest that claims the dependent clears the audit, and the
            # link edge shows up in the ownership graph.
            (root / "src/models/demo/library.json").write_text(json.dumps({
                "schema": 1, "name": "demo", "target": "gufo_demo", "kind": "static",
                "purpose": "Demo model.", "checks": ["probe_test"]}))
            _, edges, violations = ISOLATION.target_ownership(root)
            self.assertEqual(violations, [])
            self.assertIn("demo <- probe_test (links gufo_demo)", edges)

    def test_target_ownership_rejects_unrelated_claim(self):
        with tempfile.TemporaryDirectory(prefix="card08-") as directory:
            root = Path(directory)
            model = root / "src/models/demo"
            model.mkdir(parents=True)
            (model / "library.json").write_text(json.dumps({
                "schema": 1, "name": "demo", "target": "gufo_demo", "kind": "static",
                "purpose": "Demo model.", "checks": ["unrelated_test"]}))
            (model / "CMakeLists.txt").write_text(
                "add_library(gufo_demo STATIC demo.cpp)\n")
            (model / "demo.cpp").write_text("int demo() { return 1; }\n")
            (root / "CMakeLists.txt").write_text(
                "add_subdirectory(src/models/demo)\n"
                "add_executable(unrelated_test unrelated.cpp)\n")
            (root / "unrelated.cpp").write_text("int main() {}\n")
            _, _, violations = ISOLATION.target_ownership(root)
            self.assertEqual(violations, [
                "src/models/demo/library.json: claimed target unrelated_test "
                "does not depend on demo"])

    def test_target_ownership_tracks_compiled_sources_elsewhere(self):
        manifest = {
            "schema": 1, "name": "demo", "target": "gufo_demo", "kind": "static",
            "purpose": "Demo model.", "consumers": ["demo_tool"],
        }
        files = {
            "src/models/demo/demo.cpp": "int demo() { return 1; }\n",
            "src/models/demo/library.json": json.dumps(manifest),
            "src/models/demo/CMakeLists.txt": "add_library(gufo_demo STATIC demo.cpp)\n",
            "tools/demo.cpp": "int main() {}\n",
            "CMakeLists.txt": (
                "add_subdirectory(src/models/demo)\n"
                "add_executable(demo_tool tools/demo.cpp "
                "src/models/demo/demo.cpp)\n"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            _, _, violations = ISOLATION.target_ownership(root)
            self.assertTrue(any("demo_tool compiles src/models/demo/demo.cpp outside"
                                in violation for violation in violations), violations)
            # The justification needs a reason and the source it explains.
            manifest["compiled_elsewhere"] = [{"target": "demo_tool",
                                               "source": "src/models/demo/other.cpp",
                                               "reason": "Stays with the tool."}]
            (root / "src/models/demo/library.json").write_text(json.dumps(manifest))
            _, _, violations = ISOLATION.target_ownership(root)
            self.assertTrue(any("names demo_tool for src/models/demo/other.cpp"
                                in violation for violation in violations), violations)
            manifest["compiled_elsewhere"] = [{"target": "demo_tool",
                                               "source": "src/models/demo/demo.cpp",
                                               "reason": "Stays with the tool."}]
            (root / "src/models/demo/library.json").write_text(json.dumps(manifest))
            _, edges, violations = ISOLATION.target_ownership(root)
            self.assertEqual(violations, [])
            self.assertIn("demo <- demo_tool (compiles src/models/demo/demo.cpp)", edges)

    def test_target_ownership_resolves_directory_variable_sources(self):
        """A model source reached through a CMake directory variable still counts.

        `${PROJECT_SOURCE_DIR}/…` and `${CMAKE_CURRENT_SOURCE_DIR}/…` are how a file
        outside the declaring directory is usually spelled, so an argument that still
        contains a variable must not be discarded before that resolution, or a
        model-dependent target passes without ever being inventoried.
        """
        files = {
            "src/models/demo/demo.cpp": "int demo() { return 1; }\n",
            "src/models/demo/library.json": json.dumps({
                "schema": 1, "name": "demo", "target": "gufo_demo", "kind": "static",
                "purpose": "Demo model."}),
            "src/models/demo/CMakeLists.txt": "add_library(gufo_demo STATIC demo.cpp)\n",
            "tools/demo.cpp": "int main() {}\n",
            "tools/probe/probe.cpp": "int main() {}\n",
            "CMakeLists.txt": (
                "add_subdirectory(src/models/demo)\n"
                "add_subdirectory(tools/probe)\n"
                "add_executable(demo_tool tools/demo.cpp\n"
                "  ${PROJECT_SOURCE_DIR}/src/models/demo/demo.cpp)\n"),
            "tools/probe/CMakeLists.txt": (
                "add_executable(demo_probe probe.cpp\n"
                "  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/models/demo/demo.cpp)\n"),
        }
        with tempfile.TemporaryDirectory(prefix="card08-") as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            _, edges, violations = ISOLATION.target_ownership(root)
            for target in ("demo_tool", "demo_probe"):
                self.assertTrue(
                    any(f"{target} compiles src/models/demo/demo.cpp outside" in violation
                        for violation in violations), violations)
            self.assertIn("demo <- demo_tool (compiles src/models/demo/demo.cpp)", edges)
            self.assertIn("demo <- demo_probe (compiles src/models/demo/demo.cpp)", edges)

    def test_target_ownership_exempt_trees_and_stale_claims(self):
        manifest = {
            "schema": 1, "name": "demo", "target": "gufo_demo", "kind": "static",
            "purpose": "Demo model.", "checks": ["gone_test"],
        }
        files = {
            "src/models/demo/demo.cpp": "int demo() { return 1; }\n",
            "src/models/demo/library.json": json.dumps(manifest),
            "src/models/demo/CMakeLists.txt": "add_library(gufo_demo STATIC demo.cpp)\n",
            "tests/models/demo/model_test.cpp": "int main() {}\n",
            "tests/models/demo/CMakeLists.txt": (
                "add_executable(demo_model_test model_test.cpp)\n"
                "target_link_libraries(demo_model_test PRIVATE gufo_demo)\n"),
            "CMakeLists.txt": (
                "add_subdirectory(src/models/demo)\n"
                "add_subdirectory(tests/models/demo)\n"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            _, edges, violations = ISOLATION.target_ownership(root)
            # tests/models/<model>/ stays with the model test tree, so the only
            # finding is the claim that no longer matches a real target.
            self.assertEqual(violations,
                             ["src/models/demo/library.json: claimed target gone_test "
                              "does not exist"])
            self.assertIn("demo <- demo_model_test (links gufo_demo)", edges)

    def test_repository_satisfies_target_ownership(self):
        # Card 08 gave the model, comparison and eval tiers their own targets.
        # If this fails, a target started reaching a library that never claimed
        # it: claim it in that library.json, or move the dependency out.
        _, _, violations = ISOLATION.target_ownership(ROOT)
        self.assertEqual(violations, [])

    def test_serving_boundary_blocks_cli_includes_from_production_layers(self):
        files = {
            "src/cli/serve/scheduler.hpp": "struct Scheduler {};\n",
            "src/cli/serve/http_server.hpp": "struct Http {};\n",
            "src/cli/serve/logging.hpp": "struct Logger {};\n",
            "src/models/deepseek/engine.cpp":
                '#include "src/cli/serve/logging.hpp"\n',
            "src/families/qwen/text/template.cpp":
                '#include "../http.hpp"\n',
            "src/families/qwen/http.hpp": '#include "src/cli/serve/http_server.hpp"\n',
            "src/cache/pool.cpp": '#include "src/cli/serve/scheduler.hpp"\n',
            "src/ext/runtime.cpp": '#include "src/cli/serve/scheduler.hpp"\n',
            "include/gufo/ext/serving.hpp":
                '#include "src/cli/serve/scheduler.hpp"\n',
            # Serving reaching a model is the interim step-6 edge, not this gate.
            "src/cli/serve/inference_backend.cpp":
                '#include "src/models/deepseek/engine.hpp"\n',
            # Model test executables in src/models/ are forbidden from reaching serving (step 7b).
            "src/models/qwen3_asr/tests/audio_api_test.cpp":
                '#include "src/cli/serve/http_server.hpp"\n',
            # CLI test executables under tests/ keep reaching serving directly.
            "tests/cli/scheduler_test.cpp":
                '#include "src/cli/serve/scheduler.hpp"\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            violations = ISOLATION.serving_boundary_violations(
                ISOLATION.inventory(root))
        # The family header reached through `../http.hpp` is itself reported, so
        # the parent-relative edge to it is not a serving edge.
        self.assertEqual(violations, [
            "include/gufo/ext/serving.hpp:1: src/cli/serve/scheduler.hpp",
            "src/cache/pool.cpp:1: src/cli/serve/scheduler.hpp",
            "src/ext/runtime.cpp:1: src/cli/serve/scheduler.hpp",
            "src/families/qwen/http.hpp:1: src/cli/serve/http_server.hpp",
            "src/models/deepseek/engine.cpp:1: src/cli/serve/logging.hpp",
            "src/models/qwen3_asr/tests/audio_api_test.cpp:1: src/cli/serve/http_server.hpp",
        ])

    def test_repository_satisfies_the_serving_boundary(self):
        # Card 09 moved the continuation bases, host budgets and device probe to
        # gufo_ext_runtime and the capture/backing contract to gufo_cache. If
        # this fails, production model, cache or runtime code started including
        # a serving or HTTP header again; use the neutral contract instead.
        violations = ISOLATION.serving_boundary_violations(
            ISOLATION.inventory(ROOT))
        self.assertEqual(violations, [])

    def test_repository_satisfies_the_core_boundary(self):
        # Card 06 moved Qwen-specific speculation out of src/core/speculative/.
        # If this fails, a core source started including a model or family
        # header again; move the dependency behind a core-owned contract.
        violations = ISOLATION.core_boundary_violations(
            ISOLATION.inventory(ROOT))
        self.assertEqual(violations, [])

    def test_model_core_boundary_blocks_core_and_cache_includes_from_production_models(self):
        files = {
            "src/core/json.hpp": "struct Json {};\n",
            "src/cache/types.hpp": "struct Types {};\n",
            "src/core/hip/hip_utils.hpp": "struct HipUtils {};\n",
            "src/models/demo/demo.hpp": '#include "src/cache/types.hpp"\n',
            "src/models/demo/demo.cpp": '#include "src/core/json.hpp"\n',
            "src/models/demo/hip/ops.hip": '#include "src/core/hip/hip_utils.hpp"\n',
            "src/models/demo/valid.cpp": (
                '#include <gufo/ext/json.hpp>\n'
                '#include "src/models/demo/demo.hpp"\n'
                '#include "src/families/qwen/foo.hpp"\n'),
            "src/families/qwen/foo.hpp": "struct Foo {};\n",
            "include/gufo/ext/json.hpp": "struct ExtJson {};\n",
            # Test files are exempt from this ratchet (governed by test-support in Card 12)
            "src/models/demo/tests/demo_test.cpp": '#include "src/core/json.hpp"\n',
            "tests/models/demo/other_test.cpp": '#include "src/core/json.hpp"\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            violations = ISOLATION.model_core_boundary_violations(
                ISOLATION.inventory(root))
        self.assertEqual(violations, [
            "src/models/demo/demo.cpp:1: src/core/json.hpp",
            "src/models/demo/demo.hpp:1: src/cache/types.hpp",
            "src/models/demo/hip/ops.hip:1: src/core/hip/hip_utils.hpp",
        ])

    def test_repository_satisfies_the_model_core_boundary(self):
        # Card 11 migrated all models to public extension headers under include/gufo/ext.
        # If this fails, a production model source started including a private core
        # or cache header; include <gufo/ext/...> instead.
        violations = ISOLATION.model_core_boundary_violations(
            ISOLATION.inventory(ROOT))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
