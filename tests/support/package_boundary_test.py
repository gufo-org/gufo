"""Check project includes and reviewed CPU dependencies of a common package.

The component cache (src/cache) and the scheduler policy (src/sched) are built
without serving, model or HIP code. Usage:

    package_boundary_test.py SOURCE_DIR LINK_METADATA [ALLOWED_LINK ...]
"""

from pathlib import Path
import posixpath
import re
import sys
import tempfile
import unittest


SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".cpp", ".cc", ".cxx", ".inl", ".ipp", ".inc", ".tpp", ".hip"}
FORBIDDEN = re.compile(r"(?:^|/)(?:cli|models|(?:hip|roc)\w*)/|(?:^|/)text_model_runner\.hpp$")
INCLUDES = re.compile(r"^\s*#\s*include\s*([^\n]+)", re.MULTILINE)
LITERAL = re.compile(r'^[<"]([^>"\n]+)[>"]')
COMMENTS_AND_LITERALS = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)")'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\\n])*\''
    r'|(?P<comment>/\*.*?\*/|//[^\n]*)', re.DOTALL)


def strip_comments(source: str) -> str:
    def replace(match: re.Match) -> str:
        # Preserve quoted include paths and strings containing comment markers.
        # Raw-string contents cannot contain actual preprocessing directives.
        if match.group("comment") is not None or match.group("raw") is not None:
            return re.sub(r"[^\n]", " ", match[0])
        return match[0]

    return COMMENTS_AND_LITERALS.sub(replace, source)


def violations(root: Path, project_root: Path | None = None) -> list[str]:
    if not root.is_dir():
        return [f"{root}: package source directory is missing"]
    project_root = (project_root or root.parents[1]).resolve()
    failures = []
    visited = set()

    def visit(path: Path, chain: tuple[Path, ...]):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        source = re.sub(r"\\\r?\n", "", path.read_text())
        source = strip_comments(source)
        for directive in INCLUDES.findall(source):
            match = LITERAL.match(directive)
            if not match:
                failures.append(f"{path}: computed include cannot be checked: {directive}")
                continue
            include = posixpath.normpath(match[1])
            candidates = [path.parent / include, project_root / include]
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            relative = resolved.relative_to(project_root).as_posix() if resolved and resolved.is_relative_to(project_root) else ""
            if FORBIDDEN.search(include) or FORBIDDEN.search(relative):
                route = " -> ".join(str(p) for p in (*chain, path))
                failures.append(f"{route}: forbidden include {include}")
            elif resolved and resolved.is_relative_to(project_root):
                visit(resolved, (*chain, path))

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            visit(path, ())
    return failures


def link_violations(metadata: Path, allowed: set[str], target: str = "gufo_cache") -> list[str]:
    if not metadata.is_file():
        return [f"{metadata}: {target} link metadata is missing"]
    # Only the reviewed CPU dependencies may be linked. Arbitrary relay targets
    # are rejected too, so they cannot hide a transitive model/HIP link.
    return [f"{target}: unapproved dependency {library}"
            for line in metadata.read_text().splitlines()
            for library in line.split(";") if library and library not in allowed]


class BoundaryTests(unittest.TestCase):
    def test_forbidden_direct_and_transitive_includes(self):
        for suffix in SOURCE_SUFFIXES:
            for include in ("src/cli/serve/logging.hpp", "src/models/qwen/config.hpp",
                            "text_model_runner.hpp", "hip/hip_runtime.h", "src/core/hip/stream.hpp",
                            "hipblas/hipblas.h", "rocblas/rocblas.h", "hipcub/hipcub.hpp"):
                with self.subTest(suffix=suffix, include=include), tempfile.TemporaryDirectory() as tmp:
                    project = Path(tmp)
                    root = project / "src/cache"
                    root.mkdir(parents=True)
                    (project / "src/core").mkdir()
                    (root / ("adapter" + suffix)).write_text('#include "src/core/relay.hpp"\n')
                    (project / "src/core/relay.hpp").write_text('# include \\\n"' + include + '"\n')
                    self.assertEqual(len(violations(root, project)), 1)

    def test_forbidden_system_headers(self):
        for include in ("hip/hip_runtime.h", "hipblas/hipblas.h", "rocblas/rocblas.h", "hipcub/hipcub.hpp"):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp)
                root = project / "src/cache"
                root.mkdir(parents=True)
                (root / "adapter.hpp").write_text(f"#include <{include}>\n")
                self.assertEqual(len(violations(root, project)), 1)

    def test_comment_markers_in_literals(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root = project / "src/cache"
            root.mkdir(parents=True)
            source = ('const char* text = "/*";\n'
                      'const char* url = "https://example.com";\n'
                      'const char* raw = R"fixture(\n#include <hip/unused.h>\n)fixture";\n'
                      '#include <span>\n')
            (root / "adapter.cpp").write_text(source)
            self.assertEqual(violations(root, project), [])
            (root / "adapter.cpp").write_text(source + '#include <hipblas/hipblas.h>\n')
            self.assertEqual(len(violations(root, project)), 1)

    def test_allowed_comments_cycles_and_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root = project / "src/cache"
            root.mkdir(parents=True)
            (root / "adapter.hpp").write_text('#include <span>\n#include "types.ipp"\n/*\n#include "src/models/unused.hpp"\n*/\n')
            (root / "types.ipp").write_text('#include "adapter.hpp"\n')
            self.assertEqual(violations(root, project), [])
            self.assertTrue(violations(root / "missing", project))
            (root / "macro.hpp").write_text('#include MODEL_HEADER\n')
            self.assertTrue(violations(root, project))

    def test_linked_targets_and_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "links.txt"
            allowed = {"Threads::Threads", "gufo_platform"}
            self.assertTrue(link_violations(metadata, allowed))
            metadata.write_text("\nThreads::Threads\ngufo_platform\n")
            self.assertEqual(link_violations(metadata, allowed), [])
            self.assertEqual(len(link_violations(metadata, set())), 2)
            for dependency in ("gufo_core", "gufo_http", "gufo_model_relay", "hip::host", "/opt/rocm/lib/libamdhip64.so"):
                with self.subTest(dependency=dependency):
                    metadata.write_text(dependency + "\n")
                    self.assertEqual(len(link_violations(metadata, allowed)), 1)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    failures = violations(root)
    failures.extend(link_violations(Path(sys.argv[2]), set(sys.argv[3:]), f"gufo_{root.name}"))
    if failures:
        sys.exit("\n".join(failures))
    tests = unittest.main(argv=[sys.argv[0]], exit=False)
    if not tests.result.wasSuccessful():
        sys.exit(1)
    print(f"{root.name} package boundary passed")
