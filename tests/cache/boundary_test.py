"""Keep the common cache independent of serving and model implementations."""

from pathlib import Path
import re
import sys
import tempfile
import unittest


def violations(root: Path) -> list[str]:
    forbidden = re.compile(r"(?:^|/)(?:cli|models)/|(?:^|/)text_model_runner\.hpp$")
    includes = re.compile(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', re.MULTILINE)
    failures = []
    if not root.is_dir():
        return [f"{root}: cache source directory is missing"]
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".h", ".hpp", ".cpp", ".hip"}:
            continue
        source = re.sub(r"\\\r?\n", "", path.read_text())
        for include in includes.findall(source):
            if forbidden.search(include):
                failures.append(f"{path}: forbidden include {include}")
    return failures


class BoundaryTests(unittest.TestCase):
    def test_forbidden_includes(self):
        for include in (
            '"src/cli/serve/logging.hpp"',
            "<src/models/qwen/config.hpp>",
            '"../../cli/serve/text_model_runner.hpp"',
            '"text_model_runner.hpp"',
        ):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "nested").mkdir()
                (root / "nested" / "adapter.hpp").write_text(
                    "# include \\\n" + include + "\n"
                )
                self.assertEqual(len(violations(root)), 1)

    def test_allowed_and_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter.hpp").write_text(
                '#include <span>\n#include "src/cache/types.hpp"\n'
            )
            self.assertEqual(violations(root), [])
            self.assertTrue(violations(root / "missing"))


if __name__ == "__main__":
    failures = violations(Path(sys.argv[1]))
    if failures:
        sys.exit("\n".join(failures))
    tests = unittest.main(argv=[sys.argv[0]], exit=False)
    if not tests.result.wasSuccessful():
        sys.exit(1)
    print("cache package boundary passed")
