#!/usr/bin/env python3
"""Exercise production discovery/selection using actual model/family directories."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gufo-discovery-")
        self.addCleanup(self.temporary.cleanup)
        # Windows may spell TEMP with 8.3 short names, while the manifest glob
        # returns long ones; relative extension sources need one spelling.
        self.source = Path(self.temporary.name).resolve()
        self.build = self.source / "build"
        (self.source / "extensions.lock").write_text('{"schema":1,"extensions":[]}')
        (self.source / "main.cpp").write_text('''#include <iostream>
#include "gufo/ext/registry.hpp"
#include "src/cli/extension_diagnostics.hpp"
int main(int argc, char**) {
  gufo::ext::Registry registry; gufo::ext::RegisterBuiltExtensions(registry);
  if (argc > 1) std::cout << registry.ExcludedKeyDiagnostic("gguf:beta");
  else std::cout << gufo::cli::ExtensionDiagnostics(true);
}
''')
        (self.source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.21)
project(discovery_fixture LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(ENGINE_ENABLE_HIP OFF)
include("{ROOT.as_posix()}/cmake/GufoExtensionDiscovery.cmake")
gufo_discover_extensions()
set(discovery_root "${{PROJECT_SOURCE_DIR}}")
set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")
find_package(Threads REQUIRED)
add_subdirectory("{ROOT.as_posix()}/src/core/platform" platform)
add_subdirectory("{ROOT.as_posix()}/src/ext" runtime)
set(PROJECT_SOURCE_DIR "${{discovery_root}}")
add_executable(fixture main.cpp "{ROOT.as_posix()}/src/cli/extension_diagnostics.cpp")
target_include_directories(fixture PRIVATE "{ROOT.as_posix()}")
gufo_build_discovered_extensions()
get_property(selected GLOBAL PROPERTY GUFO_SELECTED_EXTENSIONS)
gufo_target_extensions(fixture EXTENSIONS ${{selected}})
''')
        self.add("base", family=True)
        self.add("alpha", depends=["base"])
        self.add("beta", hip=True)

    def add(self, name, depends=None, family=False, hip=False):
        directory = self.source / (f"src/families/test/{name}" if family else f"src/models/{name}")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": 1, "name": name, "version": "1.2.3", "api_version": ">= 0.1.0",
                    "depends": depends or [], "platforms": ["linux", "windows"],
                    "targets": {f"{name}_registration": {"requires": []}}}
        if family:
            manifest["kind"] = "library"
            body = f"int {name}_value() {{ return 1; }}"
        else:
            manifest.update(entry=f"{name}_register", registration_target=f"{name}_registration",
                            providers=[{"id": name, "capability": "embeddings", "keys": [f"gguf:{name}"],
                                        "model_ids": [f"{name}-model"], "target": f"{name}_provider", "entry": f"{name}_provide"}],
                            options=[{"name": f"--{name}-path", "type": "path", "commands": ["serve"]}])
            manifest["targets"][f"{name}_provider"] = {"requires": ["hip"] if hip else []}
            body = f'#include "gufo/ext/registry.hpp"\nvoid {name}_register(gufo::ext::Registry&) {{}}'
        (directory / "extension.json").write_text(json.dumps(manifest))
        (directory / "registration.cpp").write_text(body)
        cmake = f"gufo_extension_target(NAME {name}_registration SOURCES registration.cpp LINK_LIBRARIES gufo_ext_runtime)\n"
        if not family:
            (directory / "provider.cpp").write_text(f'''#include "gufo/ext/registry.hpp"
#include "gufo/ext/generated/{name}_constants.hpp"
void {name}_provide(gufo::ext::Registry& r) {{ r.AddProvider(gufo::ext::generated::extension_{name}::kProvider_{name}); }}
''')
            cmake += f"gufo_extension_target(NAME {name}_provider SOURCES provider.cpp LINK_LIBRARIES gufo_ext_runtime)\n"
        (directory / "extension.cmake").write_text(cmake)
        return directory

    def configure(self, *flags, error=None):
        result = subprocess.run(["cmake", "-S", str(self.source), "-B", str(self.build), "-GNinja", *flags],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if error:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(error, " ".join(result.stdout.split()))
        else:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def run_fixture(self):
        result = subprocess.run(["cmake", "--build", str(self.build), "--target", "fixture", "--parallel", "2"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        return json.loads(subprocess.check_output([str(self.build / "fixture")], text=True))

    def lock(self, names):
        (self.source / "extensions.lock").write_text(json.dumps({"schema": 1, "extensions": names}))

    def test_default_lock_excluded_runtime(self):
        self.configure()
        report = self.run_fixture()
        self.assertEqual(report["selectionMode"], "discovery")
        self.assertEqual([e["name"] for e in report["builtExtensions"]], ["base", "alpha", "beta"])
        self.assertEqual(report["builtExtensions"][1]["source"], "src/models/alpha")
        self.assertFalse(report["providers"][1]["available"])
        self.assertIn("hip", report["providers"][1]["reason"])
        self.lock(["alpha", "base"])
        self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON")
        report = self.run_fixture()
        self.assertEqual(report["selectionMode"], "lock")
        self.assertEqual([e["name"] for e in report["excludedExtensions"]], ["beta"])
        diagnostic = subprocess.check_output([str(self.build / "fixture"), "key"], text=True)
        self.assertIn("built without extension beta", diagnostic)
        self.assertIn("-DGUFO_EXTENSIONS=beta", diagnostic)
        registry = (self.build / "extensions/fixture/extension_registry.cpp").read_text()
        self.assertNotIn("::beta_register", registry)
        self.assertNotIn("::beta_provide", registry)
        # Existing metadata changes and new/removed glob matches automatically
        # reconfigure through a normal build, without another cmake -S command.
        self.add("gamma")
        report = self.run_fixture()
        self.assertEqual([e["name"] for e in report["excludedExtensions"]], ["beta", "gamma"])
        self.assertEqual(report, self.run_fixture())

    def test_explicit_empty_selector_selects_none(self):
        # An empty GUFO_EXTENSIONS is a deliberate selection of none, distinct
        # from '*' (discovery) and from the lock-mode conflict case.
        self.configure("-DGUFO_EXTENSIONS=")
        report = self.run_fixture()
        self.assertEqual(report["selectionMode"], "explicit")
        self.assertEqual(report["builtExtensions"], [])
        self.assertEqual([e["name"] for e in report["excludedExtensions"]], ["alpha", "base", "beta"])
        self.assertEqual([e["reason"] for e in report["excludedExtensions"]],
                         ["not selected in explicit mode"] * 3)

    def test_missing_lock_is_an_error_in_every_mode(self):
        (self.source / "extensions.lock").unlink()
        self.configure(error="extension lock: missing")
        self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON", error="extension lock: missing")

    def test_unshipped_reporting_at_configure_time(self):
        # Unshipped means "discovered but not in extensions.lock", independently
        # of the selection mode, so an empty lock marks all three as unshipped.
        output = self.configure()
        for name in ("alpha", "base", "beta"):
            self.assertIn(f"gufo: unshipped discovered extension {name}", output)
        self.lock(["alpha", "base"])
        output = self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON")
        self.assertIn("gufo: unshipped discovered extension beta", output)
        self.assertNotIn("gufo: unshipped discovered extension alpha", output)

    def test_collision_names_the_first_recorded_owner(self):
        # Names are visited dependency-first over sorted names: base, alpha,
        # beta, gamma. Two unshipped extensions claim the same artifact key as
        # shipped gamma, so the diagnostic must name the first recorded holder
        # (alpha) rather than whichever was seen last.
        for name in ("alpha", "beta", "gamma"):
            if name == "gamma":
                directory = self.add("gamma")
            else:
                directory = self.source / f"src/models/{name}"
            path = directory / "extension.json"
            data = json.loads(path.read_text())
            data["providers"][0]["keys"] = ["gguf:shared"]
            path.write_text(json.dumps(data))
        self.lock(["base", "gamma"])
        self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON",
                       error="keys collision 'gguf:shared' (alpha, gamma)")

    def test_false_like_name_and_production_test_isolation(self):
        self.add("off")
        # Real alpha/beta/base targets coexist with the separately configured
        # Card04 fixtures even when the root includes its test target list.
        with (self.source / "CMakeLists.txt").open("a") as out:
            out.write(f'\nadd_subdirectory("{ROOT.as_posix()}/tests/extensions" isolated-tests)\n')
        self.lock(["off"])
        self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON")
        report = self.run_fixture()
        self.assertEqual([e["name"] for e in report["builtExtensions"]], ["off"])
        self.assertEqual([e["id"] for e in report["providers"]], ["off"])
        self.assertEqual([e["name"] for e in report["excludedExtensions"]], ["alpha", "base", "beta"])

    def test_explicit_reverse_order_builds_dependencies_first(self):
        path = self.source / "src/models/alpha/extension.cmake"
        path.write_text('if(NOT TARGET base_registration)\nmessage(FATAL_ERROR "dependency build order")\nendif()\n' + path.read_text())
        self.configure("-DGUFO_EXTENSIONS=beta;alpha;base")
        self.assertEqual([e["name"] for e in self.run_fixture()["builtExtensions"]], ["base", "alpha", "beta"])

    def test_selection_failures(self):
        self.configure("-DGUFO_EXTENSIONS=missing", error="unknown extension missing")
        self.configure("-DGUFO_EXTENSIONS=alpha", error="missing selected dependency base")
        self.configure("-DGUFO_EXTENSIONS=alpha;alpha", error="duplicate names")
        # A stray separator is a malformed selector, not an unknown name.
        self.configure("-DGUFO_EXTENSIONS=alpha;;beta", error="empty entry")
        self.configure("-DGUFO_EXTENSIONS=", "-DGUFO_EXTENSIONS_FROM_LOCK=ON", error="conflicts")
        self.configure("-DGUFO_EXTENSIONS=*", "-DGUFO_EXTENSIONS_FROM_LOCK=OFF")
        self.lock(["alpha"])
        self.configure(error="missing dependency base")
        self.lock(["unknown"])
        self.configure(error="unknown extension unknown")
        for lock, error in [({"schema": 2, "extensions": []}, "unsupported schema"),
                            ({"schema": 1, "extensions": ["alpha", "alpha"]}, "duplicate value"),
                            ({"schema": 1, "extensions": [], "extra": True}, "only schema"),
                            ({"schema": 1, "extensions": "alpha"}, "must be ARRAY")]:
            (self.source / "extensions.lock").write_text(json.dumps(lock))
            self.configure(error=error)

    def test_unselected_api_and_dependency(self):
        path = self.source / "src/models/beta/extension.json"
        data = json.loads(path.read_text())
        data.update(api_version=">= 1.9.7", depends=["uninstalled"])
        path.write_text(json.dumps(data))
        self.lock(["alpha", "base"])
        self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON")
        self.assertEqual([e["name"] for e in self.run_fixture()["excludedExtensions"]], ["beta"])
        self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=OFF", error="incompatible API")

    def test_shipped_unshipped_collisions(self):
        path = self.source / "src/models/beta/extension.json"
        original = json.loads(path.read_text())
        self.lock(["alpha", "base"])
        for field in ("keys", "model_ids", "options"):
            data = copy.deepcopy(original)
            if field == "options": data["options"][0]["name"] = "--alpha-path"
            else: data["providers"][0][field] = ["gguf:alpha" if field == "keys" else "alpha-model"]
            path.write_text(json.dumps(data))
            self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON", error=f"{field} collision")

    def test_disabled_target_dependency(self):
        path = self.source / "src/models/alpha/extension.cmake"
        path.write_text(path.read_text().replace("LINK_LIBRARIES gufo_ext_runtime)",
                                                 "LINK_LIBRARIES gufo_ext_runtime beta_provider)"))
        self.configure(error="requires unavailable dependency beta_provider")

    def test_missing_build_manifest(self):
        (self.source / "src/models/beta/extension.cmake").unlink()
        self.configure()
        report = self.run_fixture()
        self.assertEqual([e["name"] for e in report["builtExtensions"]], ["base", "alpha"])
        self.assertEqual([e["name"] for e in report["excludedExtensions"]], ["beta"])
        self.assertEqual(report["excludedExtensions"][0]["reason"], "metadata only")
        self.configure("-DGUFO_EXTENSIONS=beta", error="missing")
        self.configure("-DGUFO_EXTENSIONS=alpha;base")

    def test_platform_selection(self):
        path = self.source / "src/models/beta/extension.json"
        host = "windows" if os.name == "nt" else "linux"
        data = json.loads(path.read_text()); data["platforms"] = ["linux" if host == "windows" else "windows"]
        path.write_text(json.dumps(data))
        self.configure()
        self.assertIn(f"unsupported platform {host}", self.run_fixture()["excludedExtensions"][0]["reason"])
        self.configure("-DGUFO_EXTENSIONS=beta", error=f"unavailable on {host}")
        self.lock(["beta"])
        self.configure("-DGUFO_EXTENSIONS=*", "-DGUFO_EXTENSIONS_FROM_LOCK=ON", error=f"unavailable on {host}")


if __name__ == "__main__":
    unittest.main()
