#!/usr/bin/env python3
"""Exercise production discovery/selection using actual model/family directories."""
import copy
import json
import os
import shutil
import stat
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]


PKG_CONFIG = (os.environ.get("GUFO_PKG_CONFIG_EXECUTABLE")
              or shutil.which("pkg-config") or shutil.which("pkgconf"))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.timings = {"configure": 0.0, "build": 0.0}
        self.calls = {"configure": 0, "build": 0}
        self.temporary = tempfile.TemporaryDirectory(prefix="gufo-discovery-")
        self.addCleanup(self.temporary.cleanup)
        # Windows may spell TEMP with 8.3 short names, while the manifest glob
        # returns long ones; relative extension sources need one spelling.
        self.source = Path(self.temporary.name).resolve()
        self.build = self.source / "build"
        # Each case has a fresh source tree. Normalize its temporary paths so
        # unchanged runtime objects remain reusable across cases and CI runs.
        # clang-cl also hashes the working directory without debug flags.
        # These disposable fixtures need no directory-specific debug paths;
        # keep header/content validation but share objects across temp roots.
        self.env = dict(os.environ, CCACHE_BASEDIR=str(self.source), CCACHE_NOHASHDIR="true")
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
include("{ROOT.as_posix()}/cmake/CompilerCache.cmake")
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

    def tearDown(self):
        # CTest retains this in LastTest.log even for successful cases.
        print(f"discovery timing: {self._testMethodName}: "
              f"configure={self.timings['configure']:.2f}s/{self.calls['configure']} calls, "
              f"build={self.timings['build']:.2f}s/{self.calls['build']} calls", flush=True)

    def timed_run(self, phase, command):
        start = time.monotonic()
        try:
            return subprocess.run(command, env=self.env, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        finally:
            self.timings[phase] += time.monotonic() - start
            self.calls[phase] += 1

    def add(self, name, depends=None, family=False, hip=False):
        directory = self.source / (f"src/families/test/{name}" if family else f"src/models/{name}")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": 1, "name": name, "version": "1.2.3", "api_version": ">= 0.3.0",
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
        result = self.timed_run("configure", ["cmake", "-S", str(self.source), "-B", str(self.build), "-GNinja", *flags])
        if error:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(error, " ".join(result.stdout.split()))
        else:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def run_fixture(self):
        result = self.timed_run("build", ["cmake", "--build", str(self.build), "--target", "fixture", "--parallel", "2"])
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
        for field in ("keys", "model_ids"):
            data = copy.deepcopy(original)
            data["providers"][0][field] = ["gguf:alpha" if field == "keys" else "alpha-model"]
            path.write_text(json.dumps(data))
            self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON", error=f"{field} collision")

    def test_shipped_unshipped_options_share_one_type(self):
        path = self.source / "src/models/beta/extension.json"
        data = json.loads(path.read_text())
        self.lock(["alpha", "base"])
        data["options"][0]["name"] = "--alpha-path"
        path.write_text(json.dumps(data))
        self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON")
        data["options"][0]["type"] = "string"
        path.write_text(json.dumps(data))
        self.configure("-DGUFO_EXTENSIONS_FROM_LOCK=ON", error="conflicting option types")

    # The build passes its pkg-config, which Windows finds in vcpkg's tools.
    @unittest.skipUnless(PKG_CONFIG, "needs pkg-config")
    def test_selected_extension_packages(self):
        # Only a selected extension requires its manifest's pkg-config modules,
        # which it then links as PkgConfig::<module>.
        flag = f"-DPKG_CONFIG_EXECUTABLE={PKG_CONFIG}"
        pkgconfig = self.source / "pkgconfig"
        pkgconfig.mkdir()
        (pkgconfig / "gufo-fixture-pkg.pc").write_text(
            "Name: gufo-fixture-pkg\nDescription: fixture\nVersion: 1.0\nCflags: -DGUFO_FIXTURE_PKG=1\n")
        self.env["PKG_CONFIG_PATH"] = str(pkgconfig)
        for name, module in (("alpha", "gufo-fixture-pkg"), ("beta", "gufo-missing-pkg")):
            path = self.source / f"src/models/{name}/extension.json"
            manifest = json.loads(path.read_text())
            manifest["packages"] = [{"pkg_config": module}]
            path.write_text(json.dumps(manifest))
        alpha = self.source / "src/models/alpha/extension.cmake"
        alpha.write_text(alpha.read_text().replace(
            "LINK_LIBRARIES gufo_ext_runtime)\n",
            "LINK_LIBRARIES gufo_ext_runtime PkgConfig::gufo-fixture-pkg)\n", 1)
            # Results pkg_check_modules does not cache reach the extension too.
            + 'if(NOT DEFINED gufo-fixture-pkg_LINK_LIBRARIES)\n'
              '  message(FATAL_ERROR "gufo-fixture-pkg_LINK_LIBRARIES is not visible")\n'
              'endif()\n')
        (self.source / "src/models/alpha/registration.cpp").write_text(
            '#include "gufo/ext/registry.hpp"\n'
            "static_assert(GUFO_FIXTURE_PKG == 1);\n"
            "void alpha_register(gufo::ext::Registry&) {}\n")
        self.configure(flag, error="extension beta requires pkg-config module gufo-missing-pkg")
        self.configure(flag, "-DGUFO_EXTENSIONS=alpha;base")
        self.assertEqual([e["name"] for e in self.run_fixture()["builtExtensions"]], ["base", "alpha"])
        # An extension cannot link a module only another extension declares.
        beta_cmake = self.source / "src/models/beta/extension.cmake"
        beta_cmake.write_text(beta_cmake.read_text().replace(
            "NAME beta_registration SOURCES registration.cpp LINK_LIBRARIES gufo_ext_runtime)",
            "NAME beta_registration SOURCES registration.cpp LINK_LIBRARIES gufo_ext_runtime PkgConfig::gufo-fixture-pkg)"))
        beta = self.source / "src/models/beta/extension.json"
        manifest = json.loads(beta.read_text())
        manifest["packages"] = []
        beta.write_text(json.dumps(manifest))
        self.configure(flag, "-DGUFO_EXTENSIONS=alpha;base;beta",
                       error="links PkgConfig::gufo-fixture-pkg, which extension beta does not declare")
        # Two selected extensions may declare one module; it is checked once.
        manifest["packages"] = [{"pkg_config": "gufo-fixture-pkg"}]
        beta.write_text(json.dumps(manifest))
        self.configure(flag, "-DGUFO_EXTENSIONS=alpha;base;beta")

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

    def test_external_extension_discovery(self):
        external_dir = Path(self.temporary.name) / "external_model"
        external_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": 1,
            "name": "external_model",
            "version": "1.0.0",
            "api_version": ">= 0.3.0",
            "entry": "external_model_register",
            "registration_target": "gufo_external_model",
            "depends": [],
            "platforms": ["linux", "windows"],
            "providers": [
                {
                    "id": "external-provider",
                    "capability": "embeddings",
                    "keys": ["safetensors:external"],
                    "model_ids": ["external-model"],
                    "target": "gufo_external_model",
                    "entry": "external_model_provider_register",
                }
            ],
            "options": [],
            "targets": {"gufo_external_model": {"requires": []}},
        }
        (external_dir / "extension.json").write_text(json.dumps(manifest))
        (external_dir / "src").mkdir(exist_ok=True)
        (external_dir / "src/external.cpp").write_text("""#include "gufo/ext/registry.hpp"
void external_model_provider_register(gufo::ext::Registry&) {}
void external_model_register(gufo::ext::Registry&) {}
""")
        (external_dir / "extension.cmake").write_text("""gufo_extension_target(
  NAME gufo_external_model
  SOURCES src/external.cpp
  LINK_LIBRARIES gufo_ext_runtime
)
""")
        self.configure(f"-DGUFO_EXTENSION_DIRS={external_dir.as_posix()}")
        report = self.run_fixture()
        self.assertIn("external_model", [e["name"] for e in report["builtExtensions"]])

    def external(self, name):
        directory = Path(self.temporary.name) / name
        (directory / "src").mkdir(parents=True)
        manifest = {
            "schema": 1, "name": name, "version": "1.0.0", "api_version": ">= 0.3.0",
            "entry": f"{name}_register", "registration_target": f"gufo_{name}",
            "depends": [], "platforms": ["linux", "windows"], "options": [],
            "providers": [{"id": f"{name}-provider", "capability": "embeddings",
                           "keys": [f"safetensors:{name}"], "model_ids": [f"{name}-model"],
                           "target": f"gufo_{name}", "entry": f"{name}_provider_register"}],
            "targets": {f"gufo_{name}": {"requires": []}},
        }
        (directory / "extension.json").write_text(json.dumps(manifest))
        (directory / "src/external.cpp").write_text(f"""#include "gufo/ext/registry.hpp"
void {name}_provider_register(gufo::ext::Registry&) {{}}
void {name}_register(gufo::ext::Registry&) {{}}
""")
        (directory / "extension.cmake").write_text(
            f"gufo_extension_target(NAME gufo_{name} SOURCES src/external.cpp LINK_LIBRARIES gufo_ext_runtime)\n")
        return directory

    def git(self, directory, *arguments):
        return subprocess.check_output(
            ["git", "-C", str(directory), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
             "-c", "commit.gpgsign=false", *arguments], text=True, env=self.env).strip()

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_external_extension_git_revision(self):
        tracked = self.external("tracked_ext")
        self.git(tracked, "init", "--quiet")
        self.git(tracked, "add", ".")
        self.git(tracked, "commit", "--quiet", "-m", "first")
        self.configure(f"-DGUFO_EXTENSION_DIRS={tracked.as_posix()}")

        def built():
            return {e["name"]: e for e in self.run_fixture()["builtExtensions"]}
        report = built()
        self.assertEqual(report["tracked_ext"]["revision"], self.git(tracked, "rev-parse", "HEAD"))
        self.assertIs(report["tracked_ext"]["dirty"], False)
        # In-tree extensions share the revision of the tree they live in.
        self.assertNotIn("revision", report["alpha"])
        self.assertNotIn("dirty", report["alpha"])
        # A new commit reaches the next build without another cmake -S command.
        (tracked / "src/external.cpp").write_text((tracked / "src/external.cpp").read_text() + "// second\n")
        self.git(tracked, "commit", "--quiet", "-am", "second")
        report = built()
        self.assertEqual(report["tracked_ext"]["revision"], self.git(tracked, "rev-parse", "HEAD"))
        self.assertIs(report["tracked_ext"]["dirty"], False)
        # So does staging a change, and so does an untracked file once
        # configure runs again.
        (tracked / "src/external.cpp").write_text((tracked / "src/external.cpp").read_text() + "// third\n")
        self.git(tracked, "add", "src/external.cpp")
        report = built()
        self.assertEqual(report["tracked_ext"]["revision"], self.git(tracked, "rev-parse", "HEAD"))
        self.assertIs(report["tracked_ext"]["dirty"], True)
        self.git(tracked, "commit", "--quiet", "-m", "third")
        self.assertIs(built()["tracked_ext"]["dirty"], False)
        (tracked / "untracked.txt").write_text("new")
        self.configure()
        self.assertIs(built()["tracked_ext"]["dirty"], True)

    def test_external_extension_declared_revision(self):
        # A directory without Git metadata, such as a Nix store path, reports
        # the revision the build declares for it, or "unknown".
        first = self.external("plain_ext")
        second = self.external("other_ext")
        directories = f"-DGUFO_EXTENSION_DIRS={first.as_posix()};{second.as_posix()}"
        self.configure(directories)

        def built():
            return {e["name"]: e for e in self.run_fixture()["builtExtensions"]}
        report = built()
        self.assertEqual((report["plain_ext"]["revision"], report["plain_ext"]["dirty"]), ("unknown", False))
        self.configure(directories, "-DGUFO_EXTENSION_DIR_REVISIONS=0123abc-dirty;sha256-AbC+/=")
        report = built()
        self.assertEqual((report["plain_ext"]["revision"], report["plain_ext"]["dirty"]), ("0123abc", True))
        self.assertEqual((report["other_ext"]["revision"], report["other_ext"]["dirty"]), ("sha256-AbC+/=", False))
        self.configure(directories, "-DGUFO_EXTENSION_DIR_REVISIONS=0123abc;unknown")
        self.assertEqual(built()["other_ext"]["revision"], "unknown")
        self.configure(directories, "-DGUFO_EXTENSION_DIR_REVISIONS=0123abc",
                       error="needs one entry per GUFO_EXTENSION_DIRS entry (2), got 1")
        # A declared revision is free text, such as a tag name.
        self.configure(directories, '-DGUFO_EXTENSION_DIR_REVISIONS=v1.0@a"b\\c;unknown')
        self.assertEqual(built()["plain_ext"]["revision"], 'v1.0@a"b\\c')

    def test_external_extension_allowlist_violation(self):
        external_dir = Path(self.temporary.name) / "bad_ext"
        external_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": 1,
            "name": "bad_ext",
            "version": "1.0.0",
            "api_version": ">= 0.3.0",
            "entry": "bad_ext_register",
            "registration_target": "gufo_bad_ext",
            "depends": [],
            "platforms": ["linux", "windows"],
            "providers": [
                {
                    "id": "bad-provider",
                    "capability": "embeddings",
                    "keys": ["safetensors:bad"],
                    "model_ids": ["bad-model"],
                    "target": "gufo_bad_ext",
                    "entry": "bad_ext_provider_register",
                }
            ],
            "options": [],
            "targets": {"gufo_bad_ext": {"requires": []}},
        }
        (external_dir / "extension.json").write_text(json.dumps(manifest))
        (external_dir / "bad.cpp").write_text("""extern "C" {
void bad_ext_provider_register(void*) {}
void bad_ext_register(void*) {}
}
""")
        (external_dir / "extension.cmake").write_text("""gufo_extension_target(
  NAME gufo_bad_ext
  SOURCES bad.cpp
  LINK_LIBRARIES gufo_core
)
""")
        self.configure(f"-DGUFO_EXTENSION_DIRS={external_dir.as_posix()}", error="links prohibited target 'gufo_core'")

        (external_dir / "extension.cmake").write_text("""gufo_extension_target(
  NAME gufo_bad_ext
  SOURCES bad.cpp
  LINK_LIBRARIES gufo_ext_runtime
)
target_include_directories(gufo_bad_ext PRIVATE "${PROJECT_SOURCE_DIR}/src")
""")
        self.configure(f"-DGUFO_EXTENSION_DIRS={external_dir.as_posix()}", error="violates include allowlist")

        (external_dir / "extension.cmake").write_text("""gufo_extension_target(
  NAME gufo_bad_ext
  SOURCES bad.cpp
  LINK_LIBRARIES gufo_ext_runtime
)
target_link_libraries(gufo_bad_ext PRIVATE gufo_core)
""")
        self.configure(f"-DGUFO_EXTENSION_DIRS={external_dir.as_posix()}", error="violates allowlist: links prohibited target 'gufo_core'")

        (external_dir / "extension.cmake").write_text("""gufo_extension_target(
  NAME gufo_bad_ext
  SOURCES bad.cpp
  LINK_LIBRARIES gufo_ext_runtime
)
target_compile_options(gufo_bad_ext PRIVATE "-I${PROJECT_SOURCE_DIR}/src/core")
""")
        self.configure(f"-DGUFO_EXTENSION_DIRS={external_dir.as_posix()}", error="violates include allowlist")

        (external_dir / "extension.cmake").write_text("""gufo_extension_target(
  NAME gufo_bad_ext
  SOURCES bad.cpp
  LINK_LIBRARIES gufo_ext_runtime
)
add_library(intermediate STATIC bad.cpp)
target_link_libraries(intermediate PUBLIC gufo_core)
target_link_libraries(gufo_bad_ext PRIVATE intermediate)
""")
        self.configure(f"-DGUFO_EXTENSION_DIRS={external_dir.as_posix()}", error="violates allowlist: links prohibited target 'gufo_core'")

    def template(self):
        external = self.source / "external template"
        shutil.copytree(ROOT / "examples/extension_template", external)
        # Nix store fixtures are read-only; the tests mutate their local copy.
        for path in [external, *external.rglob("*")]:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        return external

    def test_external_public_link_targets(self):
        external_dir = Path(self.temporary.name) / "linked_ext"
        external_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": 1, "name": "linked_ext", "version": "1.0.0", "api_version": ">= 0.3.0",
            "entry": "linked_ext_register", "registration_target": "gufo_linked_ext",
            "depends": [], "platforms": ["linux", "windows"], "options": [],
            "providers": [{"id": "linked-provider", "capability": "embeddings",
                           "keys": ["safetensors:linked"], "model_ids": ["linked-model"],
                           "target": "gufo_linked_ext", "entry": "linked_ext_provider_register"}],
            "targets": {"gufo_linked_ext": {"requires": []}},
        }
        (external_dir / "extension.json").write_text(json.dumps(manifest))
        (external_dir / "linked.cpp").write_text("""extern "C" {
void linked_ext_provider_register(void*) {}
void linked_ext_register(void*) {}
}
""")
        (external_dir / "extension.cmake").write_text("""gufo_extension_target(
  NAME gufo_linked_ext
  SOURCES linked.cpp
  LINK_LIBRARIES gufo_ext_runtime gufo_ext_core gufo_ext_cache
)
""")
        cmake = self.source / "CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace("gufo_build_discovered_extensions()", """add_library(gufo_core INTERFACE)
add_library(gufo_cache INTERFACE)
add_library(gufo_ext_core INTERFACE)
target_link_libraries(gufo_ext_core INTERFACE $<LINK_ONLY:gufo_core>)
add_library(gufo_ext_cache INTERFACE)
target_link_libraries(gufo_ext_cache INTERFACE $<LINK_ONLY:gufo_cache>)
gufo_build_discovered_extensions()"""))
        self.configure(f"-DGUFO_EXTENSION_DIRS={external_dir.as_posix()}")

    def test_external_template_installs_license(self):
        external = self.template()
        self.configure(f"-DGUFO_EXTENSION_DIRS={external}")
        self.run_fixture()
        prefix = self.source / "installed"
        result = subprocess.run(["cmake", "--install", str(self.build), "--prefix", str(prefix)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        notice = prefix / "share/licenses/gufo/extensions/template_model/LICENSE"
        self.assertEqual(notice.read_bytes(), (external / "LICENSE").read_bytes())

    def test_external_template_public_headers(self):
        external = self.template()
        # Exercise forwarding headers with the root include absent.
        source = external / "src/template_model.cpp"
        source.write_text('#include <gufo/ext/cache.hpp>\n'
                          '#include <gufo/ext/json.hpp>\n'
                          '#include <gufo/ext/platform/fs.hpp>\n' + source.read_text())
        with (external / "extension.cmake").open("a") as out:
            out.write('\ntarget_compile_options(gufo_template_model PRIVATE "$<$<CONFIG:Debug>:-Wall>")\n'
                      'target_include_directories(gufo_template_model PRIVATE '
                      '"$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}/src>" '
                      '"$<INSTALL_INTERFACE:${PROJECT_SOURCE_DIR}/src>")\n')
        self.configure(f"-DGUFO_EXTENSION_DIRS={external}", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        self.run_fixture()
        commands = json.loads((self.build / "compile_commands.json").read_text())
        # CMake uses forward slashes even when Python spells Windows paths with
        # backslashes. Resolve relative database paths against their build dir.
        command = next(c["command"] for c in commands
                       if (Path(c["directory"]) / c["file"]).resolve() == source.resolve())
        self.assertNotIn(f'-I{ROOT} ', command)
        self.assertNotIn(f'-I{self.source} ', command)
        # A private header outside the API SURFACE must fail to compile.
        source.write_text('#include "src/core/logging/trace.hpp"\n' + source.read_text())
        result = subprocess.run(["cmake", "--build", str(self.build), "--target", "gufo_template_model"],
                                env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("src/core/logging/trace.hpp", result.stdout)

    def test_external_settings_validation(self):
        external = self.template()
        manifest = external / "extension.cmake"
        original = manifest.read_text()
        target = "gufo_template_model"
        cases = [
            (f'target_link_libraries({target} PRIVATE "$<LINK_ONLY:gufo_core>")', "links prohibited target"),
            (f'target_link_libraries({target} PRIVATE "$<BUILD_INTERFACE:$<LINK_ONLY:gufo_core>>")', "links prohibited target"),
            (f'target_link_libraries({target} PRIVATE "$<$<CONFIG:Release>:gufo_core>")', "links prohibited target"),
            (f'target_link_libraries({target} PRIVATE "$<TARGET_NAME_IF_EXISTS:gufo_core>")', "unvalidated generator expression"),
            (f'target_include_directories({target} PRIVATE "$<BUILD_INTERFACE:${{PROJECT_SOURCE_DIR}}/src>")', "violates include allowlist"),
            (f'target_include_directories({target} PRIVATE "$<$<CONFIG:Debug>:${{PROJECT_SOURCE_DIR}}>")', "violates include allowlist"),
            (rf'target_compile_options({target} PRIVATE "SHELL:-isystem \"${{PROJECT_SOURCE_DIR}}/src\"")', "violates include allowlist"),
            (f'target_compile_options({target} PRIVATE -I "${{PROJECT_SOURCE_DIR}}/src")', "violates include allowlist"),
            (f'target_compile_options({target} PRIVATE "/external:I${{PROJECT_SOURCE_DIR}}/src")', "violates include allowlist"),
            (f'target_compile_options({target} PRIVATE "/clang:-I${{PROJECT_SOURCE_DIR}}/src")', "violates include allowlist"),
            (r'set(CMAKE_CXX_FLAGS "-I\"${PROJECT_SOURCE_DIR}/src\"")', "violates include allowlist"),
            (r'set(CMAKE_CXX_FLAGS_RELEASE "-isystem \"${PROJECT_SOURCE_DIR}/src\"")', "violates include allowlist"),
            ('set(CMAKE_HIP_FLAGS_CUSTOM "-I${PROJECT_SOURCE_DIR}/src")', "violates include allowlist"),
            ('set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/src")', "violates include allowlist"),
            (f'set_target_properties({target} PROPERTIES COMPILE_FLAGS "-Wall -I${{PROJECT_SOURCE_DIR}}/src")', "violates include allowlist"),
            ('set_source_files_properties("${GUFO_EXTENSION_BUILD_ROOT}/src/template_model.cpp" PROPERTIES COMPILE_OPTIONS "-I${PROJECT_SOURCE_DIR}/src")', "violates include allowlist"),
            ('set_source_files_properties("${GUFO_EXTENSION_BUILD_ROOT}/src/template_model.cpp" PROPERTIES COMPILE_FLAGS "-Wall -I${PROJECT_SOURCE_DIR}/src")', "violates include allowlist"),
            ('set_source_files_properties("${GUFO_EXTENSION_BUILD_ROOT}/src/template_model.cpp" PROPERTIES INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}")', "violates include allowlist"),
            (f'add_library(helper INTERFACE)\ntarget_include_directories(helper INTERFACE "${{PROJECT_SOURCE_DIR}}")\ntarget_link_libraries({target} PRIVATE helper)', "violates include allowlist"),
            (f'add_library(helper INTERFACE)\ntarget_compile_options(helper INTERFACE "-I${{PROJECT_SOURCE_DIR}}/src")\ntarget_link_libraries({target} PRIVATE helper)', "violates include allowlist"),
            (f'add_library(gufo_core INTERFACE)\nadd_library(hidden_core ALIAS gufo_core)\ntarget_link_libraries({target} PRIVATE hidden_core)', "links prohibited target"),
            ('add_library(helper STATIC "${GUFO_EXTENSION_BUILD_ROOT}/src/template_model.cpp")\ntarget_include_directories(helper PRIVATE "${PROJECT_SOURCE_DIR}/src")', "violates include allowlist"),
            (f'target_include_directories({target} PRIVATE "${{PROJECT_SOURCE_DIR}}/src/../src/core")', "violates include allowlist"),
        ]
        if os.name != "nt":
            (self.source / "private-link").symlink_to(self.source / "src", target_is_directory=True)
            cases.append((f'target_include_directories({target} PRIVATE "${{PROJECT_SOURCE_DIR}}/private-link")',
                          "violates include allowlist"))
        relative = os.path.relpath(self.source / "src", self.build).replace(os.sep, "/")
        cases.append((f'target_compile_options({target} PRIVATE "-I{relative}")', "violates include allowlist"))
        for cmake, error in cases:
            with self.subTest(cmake=cmake):
                manifest.write_text(original + "\n" + cmake + "\n")
                self.configure(f"-DGUFO_EXTENSION_DIRS={external}", error=error)

    def test_external_parent_flags_and_late_mutations(self):
        external = self.template()
        for flag in ("CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_DEBUG"):
            with self.subTest(flag=flag):
                self.configure(f"-DGUFO_EXTENSION_DIRS={external}",
                               f"-D{flag}=-I{self.source}/src", error="violates include allowlist")
                # Each flag is inherited from the cache; clear it for the next case.
                self.configure(f"-D{flag}=")
        with (self.source / "CMakeLists.txt").open("a") as out:
            out.write('\ntarget_include_directories(gufo_template_model PRIVATE "${PROJECT_SOURCE_DIR}")\n')
        self.configure(error="violates include allowlist")



if __name__ == "__main__":
    unittest.main()
