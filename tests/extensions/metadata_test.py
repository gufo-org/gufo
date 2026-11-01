#!/usr/bin/env python3
"""Bounded production-CMake metadata contracts and independent schema checks."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/extensions/fixture"
BASE = [json.loads((FIXTURES / f"{name}.json").read_text()) for name in ("alpha", "beta", "base")]
MODULE = ROOT / "cmake/GufoExtension.cmake"
DIAGNOSTICS = []


class ConfigureTests(unittest.TestCase):
    def configure(self, manifests, expected=None, extra=""):
        with tempfile.TemporaryDirectory(prefix="gufo-extension-metadata-") as temp:
            source = Path(temp)
            cmake = ["cmake_minimum_required(VERSION 3.21)", "project(extension_metadata NONE)",
                     'set(CMAKE_SYSTEM_NAME "Linux")', f'include("{MODULE.as_posix()}")']
            for i, manifest in enumerate(manifests):
                path = source / f"{i}.json"
                path.write_text(json.dumps(manifest))
                cmake.append(f'gufo_extension(MANIFEST "{path.as_posix()}")')
            cmake.extend(["gufo_extensions_validate()", extra])
            (source / "CMakeLists.txt").write_text("\n".join(cmake))
            result = subprocess.run(["cmake", "-S", str(source), "-B", str(source / "build")],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            DIAGNOSTICS.append({"test": self.id(), "expected": expected,
                                "extra": extra, "returncode": result.returncode, "output": result.stdout})
            if expected is None:
                self.assertEqual(result.returncode, 0, result.stdout)
            else:
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(expected, " ".join(result.stdout.split()))

    def invalid(self, change, expected):
        manifests = copy.deepcopy(BASE)
        change(manifests)
        self.configure(manifests, expected)

    def test_required_fields(self):
        for field in ("schema", "name", "version", "api_version", "depends", "platforms", "targets", "entry", "registration_target", "providers"):
            with self.subTest(field=field):
                self.invalid(lambda m: m[0].pop(field), field)
        for field in ("id", "capability", "keys", "target", "entry"):
            with self.subTest(provider_field=field):
                self.invalid(lambda m: m[0]["providers"][0].pop(field), field)
        for field in ("name", "type", "commands"):
            with self.subTest(option_field=field):
                self.invalid(lambda m: m[0]["options"][0].pop(field), field)

    def test_names_and_collisions(self):
        cases = [
            (lambda m: m[0].update(name="Bad-name"), "invalid extension name"),
            (lambda m: m[1].update(name="alpha"), "duplicate extension name"),
            (lambda m: m[1]["providers"][0].update(id="alpha"), "duplicate provider ID"),
            (lambda m: m[0]["providers"][0].update(id="bad/id"), "invalid provider ID"),
            (lambda m: m[0]["providers"][1].update(id="alpha_hip") or m[0]["providers"][0].update(id="alpha-hip"), "colliding generated provider identifier"),
            (lambda m: m[0].update(entry="unsafe::entry"), "invalid entry symbol"),
            (lambda m: m[0]["providers"][0].update(entry="alpha_register"), "duplicate registration symbol"),
            (lambda m: m[1]["targets"].update(alpha_cpu={"requires": []}), "duplicate target declaration"),
            (lambda m: m[0].update(schema=2), "unsupported schema"),
            (lambda m: m[0].update(version="1.2"), "invalid extension version"),
            (lambda m: m[0].update(kind="other"), "unknown kind"),
            (lambda m: m[0]["providers"][0].update(capability="new_core_kind"), "unknown capability"),
            (lambda m: m[0]["providers"][0].update(keys=["arbitrary:key"]), "unsupported artifact key"),
            (lambda m: m[0]["providers"][0].update(keys=[]), "keys cannot be empty"),
            (lambda m: m[0]["providers"][0].update(keys=["gguf:x", "gguf:x"]), "duplicate value"),
            (lambda m: m[0].update(providers=[]), "providers cannot be empty"),
            (lambda m: m[0].update(targets={}), "targets cannot be empty"),
            (lambda m: m[0].update(platforms=[]), "platforms cannot be empty"),
            (lambda m: m[0].update(platforms=["other"]), "unsupported platform"),
        ]
        for change, expected in cases:
            with self.subTest(expected=expected):
                self.invalid(change, expected)

    def test_false_like_owners(self):
        for token in ("off", "no", "false", "ignore", "notfound"):
            manifests = copy.deepcopy(BASE)
            manifests[0]["name"] = token
            manifests[2]["depends"] = []
            with self.subTest(token=token):
                self.configure(manifests)
                duplicate = copy.deepcopy(manifests)
                duplicate[1]["targets"]["alpha_cpu"] = {"requires": []}
                self.configure(duplicate, "duplicate target declaration")
                duplicate = copy.deepcopy(manifests)
                duplicate[1]["providers"][0]["id"] = "alpha"
                self.configure(duplicate, "duplicate provider ID")

    def test_reserved_entries(self):
        for keyword in ("delete", "namespace", "false", "co_await", "requires"):
            for provider in (False, True):
                with self.subTest(keyword=keyword, provider=provider):
                    self.invalid(lambda m: (m[0]["providers"][0] if provider else m[0]).update(entry=keyword),
                                 "reserved C++ entry symbol")

    def test_nul_metadata(self):
        for path in (("name",), ("targets", "bad\0key"),
                     ("providers", 0, "id"), ("providers", 0, "keys"),
                     ("options", 0, "help"), ("docs",), ("compile_definitions",)):
            manifests = copy.deepcopy(BASE)
            parent = manifests[0]
            for key in path[:-1]:
                parent = parent[key]
            if path[0] == "targets":
                parent[path[-1]] = {"requires": []}
            elif path[-1] == "keys":
                parent[path[-1]] = ["gguf:x\0y"]
            elif path[-1] == "docs":
                parent[path[-1]] = {"title": "x\0y", "inputs": "GGUF", "guide": "README.md"}
            elif path[-1] == "compile_definitions":
                parent[path[-1]] = {"BAD\0KEY": "value"}
            else:
                parent[path[-1]] = "x\0y"
            with self.subTest(path=path):
                self.configure(manifests, "U+0000 is not supported")
        manifests = copy.deepcopy(BASE)
        for backslashes in (1, 2, 3):
            manifests[0]["options"][0]["help"] = "x" + "\\" * backslashes + "u0000y"
            self.configure(manifests)

    def test_compile_edge_identifiers(self):
        # Actual production generation, compilation and execution; CMake false
        # tokens remain valid metadata, and C++ keywords are safe namespace names.
        with tempfile.TemporaryDirectory(prefix="gufo-extension-edges-") as temp:
            source = Path(temp)
            cmake = ["cmake_minimum_required(VERSION 3.21)", "project(extension_edges LANGUAGES CXX)",
                     "set(CMAKE_CXX_STANDARD 20)", "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
                     f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")', "find_package(Threads REQUIRED)",
                     f'add_subdirectory("{ROOT.as_posix()}/src/core/platform" platform)',
                     f'add_subdirectory("{ROOT.as_posix()}/src/ext" runtime)', f'include("{MODULE.as_posix()}")']
            names = ("off", "no", "false", "ignore", "notfound", "namespace")
            for name in names:
                # C++ false is a keyword, so exercise it as metadata only.
                entry = name if name not in ("false", "namespace") else name + "_entry"
                provider_entry = name + "_provider"
                manifest = copy.deepcopy(BASE[0])
                manifest.update(name=name, depends=[], entry=entry, registration_target=name,
                                targets={name: {"requires": []}}, providers=[{
                                    "id": name, "capability": "embeddings", "keys": ["gguf:" + name],
                                    "target": name, "entry": provider_entry}], options=[{
                                        "name": "--edge", "type": "string", "commands": ["off"],
                                        "providers": [name], "help": r"x\u0000y"}])
                manifest["providers"].append({"id": name + "-extra", "capability": "embeddings",
                                              "keys": ["gguf:" + name + "-extra"], "target": name,
                                              "entry": provider_entry + "_extra"})
                (source / f"{name}.json").write_text(json.dumps(manifest))
                (source / f"{name}.cpp").write_text(
                    f'#include <gufo/ext/generated/{name}_constants.hpp>\n'
                    f'void {entry}(gufo::ext::Registry&) {{}}\n'
                    f'void {provider_entry}(gufo::ext::Registry& r) {{'
                    f'r.AddProvider(gufo::ext::generated::extension_{name}::kProvider_{name});}}\n'
                    f'void {provider_entry}_extra(gufo::ext::Registry& r) {{'
                    f'r.AddProvider(gufo::ext::generated::extension_{name}::kProvider_{name}_extra);}}\n')
                cmake.extend([f'gufo_extension(MANIFEST "${{CMAKE_CURRENT_SOURCE_DIR}}/{name}.json")',
                              f'gufo_extension_target(NAME {name} SOURCES {name}.cpp LINK_LIBRARIES gufo_ext_runtime)'])
            cmake.extend(['enable_testing()', 'add_executable(edges main.cpp)',
                          'gufo_target_extensions(edges EXTENSIONS ' + ' '.join(names) + ')',
                          'add_test(NAME edges COMMAND edges)'])
            (source / "CMakeLists.txt").write_text("\n".join(cmake))
            (source / "main.cpp").write_text(
                '#include <gufo/ext/registry.hpp>\n'
                'int main() { gufo::ext::Registry r; gufo::ext::RegisterBuiltExtensions(r);'
                'if (r.Providers().size() != 12) return 1;'
                'for (const auto& p : r.Providers()) {'
                'if (p.id.ends_with("-extra")) { if (!p.options.empty()) return 3; continue; }'
                'if (p.options.size() != 1 || p.options[0].help != "x\\\\u0000y") return 2; } return 0; }\n')
            # Ninja keeps the build's compiler (CXX); a Visual Studio generator
            # would select MSVC instead.
            for command in (["cmake", "-S", str(source), "-B", str(source / "build"), "-GNinja"],
                            ["cmake", "--build", str(source / "build"), "--config", "Release", "--parallel", "4"],
                            ["ctest", "--test-dir", str(source / "build"), "-C", "Release", "--output-on-failure"]):
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                DIAGNOSTICS.append({"test": self.id(), "command": command,
                                    "returncode": result.returncode, "output": result.stdout})
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_dependencies_and_version(self):
        self.configure(BASE)
        current = copy.deepcopy(BASE)
        current[0]["api_version"] = ">= 0.1.5"
        self.configure(current)
        self.invalid(lambda m: m[0].update(depends=["absent"]), "missing dependency absent")
        self.invalid(lambda m: m[2].update(depends=["alpha"]), "cyclic dependency")
        self.invalid(lambda m: m[2].update(platforms=["linux"]), "unsatisfiable dependency platforms")
        for api in (">= 0.2.0", ">= 0.1.6", ">= 1.0.0", "0.1", ">= 0.01.0"):
            with self.subTest(api=api):
                self.invalid(lambda m: m[0].update(api_version=api), "API" if api.startswith(">= ") else "api_version")

    def test_target_conditions(self):
        self.invalid(lambda m: m[0]["targets"]["alpha_hip"].update(requires=["unknown"]), "unknown target condition")
        self.invalid(lambda m: m[0]["targets"]["alpha_hip"].update(requires=["linux", "windows"]), "unsatisfiable")
        self.invalid(lambda m: m[0].update(platforms=["windows"]) or m[0]["targets"]["alpha_hip"].update(requires=["linux"]), "unsatisfiable")
        self.invalid(lambda m: m[0]["targets"]["alpha_registration"].update(requires=["hip"]), "unconditional")
        self.invalid(lambda m: m[0]["providers"][0].update(target="absent"), "missing provider target")
        self.invalid(lambda m: m[0].update(registration_target="absent"), "missing registration target")
        self.invalid(lambda m: m[0]["targets"]["alpha_cpu"].update(role="test"), "must be a library")
        self.invalid(lambda m: m[0]["targets"]["alpha_cpu"].update(role="other"), "invalid target role")
        self.configure(BASE, "undeclared target", "gufo_extension_target(NAME undeclared SOURCES missing.cpp)")
        self.configure(BASE, "duplicate target definition", "set(ENGINE_ENABLE_HIP OFF)\ngufo_extension_target(NAME alpha_hip SOURCES missing.cpp)\ngufo_extension_target(NAME alpha_hip SOURCES missing.cpp)")

    def test_registry_target_contracts(self):
        self.configure(BASE, "missing selected dependency",
                       "add_custom_target(dummy)\ngufo_target_extensions(dummy EXTENSIONS alpha)")
        self.configure(BASE, "was not defined by gufo_extension_target",
                       "add_custom_target(dummy)\ngufo_target_extensions(dummy EXTENSIONS alpha beta base)")
        self.configure(BASE, "unknown extension",
                       "add_custom_target(dummy)\ngufo_target_extensions(dummy EXTENSIONS absent)")

    def test_options(self):
        self.invalid(lambda m: m[1]["options"][0].update(type="path"), "conflicting option types")
        self.invalid(lambda m: m[0]["options"][0].update(type="other"), "invalid option type")
        self.invalid(lambda m: m[0]["options"][0].update(name="-short"), "invalid option name")
        self.invalid(lambda m: m[0]["options"][0].update(commands=[]), "commands cannot be empty")
        self.invalid(lambda m: m[0]["options"][0].update(providers=["absent"]), "missing provider")
        self.invalid(lambda m: m[0]["options"][0].update(providers=[]), "providers cannot be empty")
        self.invalid(lambda m: m[0]["options"].append(m[0]["options"][0]), "duplicate option declaration")
        for name in ("--context", "--provider", "--speculative", "--model", "--host", "--port", "--temperature", "--cache-disk", "--seed"):
            with self.subTest(core_option=name):
                self.invalid(lambda m: m[0]["options"][0].update(name=name), "core-owned option")
        # A different command may use another type for the same long name.
        manifests = copy.deepcopy(BASE)
        manifests[1]["options"][0].update(type="string", commands=["serve other"])
        self.configure(manifests)

    def test_library_restrictions(self):
        self.invalid(lambda m: m[2].update(entry="base_register"), "library cannot declare entry")
        self.invalid(lambda m: m[2].update(providers=m[0]["providers"]), "library cannot declare providers")

    def test_deterministic_generation_and_exclusions(self):
        # Configure the actual production helper/targets twice in independent
        # directories without Python package discovery; no HIP compiler needed.
        with tempfile.TemporaryDirectory(prefix="gufo-extension-registry-") as temp:
            directories = [Path(temp) / name for name in ("first", "second", "available")]
            for directory in directories:
                result = subprocess.run(["cmake", "-S", str(FIXTURES), "-B", str(directory),
                                         f"-DENGINE_ENABLE_HIP={'ON' if directory.name == 'available' else 'OFF'}",
                                         "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE"],
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 0, result.stdout)
            relative = Path("extensions/extensions/extension_runtime_test")
            files = ["extension_registry.cpp", "gufo/ext/generated/alpha_constants.hpp", "gufo/ext/generated/beta_constants.hpp"]
            for name in files:
                self.assertEqual((directories[0] / relative / name).read_bytes(),
                                 (directories[1] / relative / name).read_bytes(), name)
            excluded = (directories[0] / relative / "extension_registry.cpp").read_text()
            self.assertNotIn("alpha_hip_register", excluded)
            self.assertNotIn("CreateUnavailableHipProvider", excluded)
            self.assertIn("registry.AddProvider", excluded)
            included = (directories[2] / relative / "extension_registry.cpp").read_text()
            self.assertIn("::alpha_hip_register(registry)", included)
            self.assertLess(excluded.index("::alpha_register(registry)"), excluded.index("::beta_register(registry)"))


class SchemaTests(unittest.TestCase):
    def test_full_schema(self):
        import jsonschema
        schema = json.loads((ROOT / "schemas/extension.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        for manifest in BASE:
            validator.validate(manifest)
        for path in sorted((ROOT / "src").glob("**/extension.json")):
            with self.subTest(manifest=str(path)):
                validator.validate(json.loads(path.read_text()))
        for path in (("name",), ("providers", 0, "keys", 0), ("options", 0, "help")):
            invalid = copy.deepcopy(BASE[0])
            parent = invalid
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = "x\0y"
            with self.subTest(nul=path), self.assertRaises(jsonschema.ValidationError):
                validator.validate(invalid)
        for field, value in (("docs", {"title": "x\0y", "inputs": "GGUF", "guide": "README.md"}),
                             ("compile_definitions", {"BAD\0KEY": "value"})):
            invalid = copy.deepcopy(BASE[0])
            invalid[field] = value
            with self.subTest(nul_optional=field), self.assertRaises(jsonschema.ValidationError):
                validator.validate(invalid)
        literal = copy.deepcopy(BASE[0])
        literal["options"][0]["help"] = r"x\u0000y"
        validator.validate(literal)
        for field in ("entry", "provider_entry"):
            invalid = copy.deepcopy(BASE[0])
            (invalid if field == "entry" else invalid["providers"][0])["entry"] = "delete"
            with self.subTest(keyword=field), self.assertRaises(jsonschema.ValidationError):
                validator.validate(invalid)
        # Every RFC packaging/CI optional field is accepted and type-checked.
        broad = copy.deepcopy(BASE[0])
        broad.update(packages=[{"pkg_config": "libavformat", "nix": "ffmpeg", "windows": "ffmpeg[avformat]"}],
                     compile_definitions={"DATA": "install_data:example.json"}, runtime_libraries=["runtime.txt"],
                     install_data=[], licenses=[], production_paths={"include": ["**/*.cpp"], "exclude": ["tests/"]},
                     vendored_paths=[], kernel_resources="kernels.json", docs={"title": "Example", "inputs": "GGUF", "guide": "README.md"},
                     tests=[{"target": "test_target", "ctest": "example.*", "check_pr": True, "windows_build": True, "exclude_os": []}],
                     test_labels=["example"], fixtures="tests/fixtures/", bundled_tests=[], functional="functional/", tools="tools/",
                     python_checks=[{"name": "quality", "tests": "tests/tools/", "sources": ["tools/"]}])
        validator.validate(broad)
        for field, value in (("unknown", True), ("packages", "wrong"), ("tests", [{"target": "test_target"}]),
                             ("compile_definitions", {"INVALID": False}), ("production_paths", {"include": []})):
            invalid = copy.deepcopy(broad)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(jsonschema.ValidationError):
                validator.validate(invalid)
        # Every artifact key carries the key kind configure requires
        # (gguf|safetensors|manifest|diffusers in cmake/GufoExtension.cmake).
        for key in ("arbitrary:key", "gguf", "GGUF:x"):
            invalid = copy.deepcopy(broad)
            invalid["providers"][0]["keys"] = [key]
            with self.subTest(artifact_key=key), self.assertRaises(jsonschema.ValidationError):
                validator.validate(invalid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SchemaTests if args.schema else ConfigureTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.diagnostics:
        args.diagnostics.write_text(json.dumps(DIAGNOSTICS, indent=2) + "\n")
    sys.exit(not result.wasSuccessful())
