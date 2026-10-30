#!/usr/bin/env python3
"""Run the same C++ formatting check locally and in CI."""

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="format files in place")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    excluded = (
        "src/models/deepseek_v4_flash/runtime/",
        "src/models/deepseek_v4_flash/kernels/rocm/",
        "src/models/qwen38_flash_next/kernels/rocm/mmq/",
    )
    files = sorted(
        str(path.relative_to(root))
        for directory in ("include", "src", "tests")
        for path in (root / directory).rglob("*")
        if path.is_file()
        and path.suffix in {".cpp", ".h", ".hpp"}
        and not {"fixtures", "vendor"}.intersection(path.relative_to(root).parts)
        and not path.relative_to(root).as_posix().startswith(excluded)
    )
    flags = ["-i"] if args.fix else ["--dry-run", "--Werror"]
    result = subprocess.run(["clang-format", *flags, *files], cwd=root)
    if result.returncode == 0:
        print(f"PASS: {'Formatted' if args.fix else 'Checked'} {len(files)} C++ files")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
