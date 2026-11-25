#!/usr/bin/env python3
"""Decide whether a pull request's compiler cache is worth saving.

A pull request restores main's compiler cache, or its own from an earlier run.
Saving it again stores a full copy in the PR's cache scope, which is wasteful
when the run compiled little: a rerun recompiles the same few objects in less
time than the copy costs in the shared 10 GB budget.

  cache_growth.py snapshot DIR STATE   record DIR's files after the restore
  cache_growth.py check DIR STATE      print save=true|false for GITHUB_OUTPUT

A file counts as new when its path is absent from the snapshot or its size
changed. Modification times are ignored: ccache touches the files it hits.
Any doubt, such as a missing snapshot, saves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_MIN_MIB = 32


def listing(root: Path) -> dict[str, int]:
    files: dict[str, int] = {}
    if not root.is_dir():
        return files
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            files[path.relative_to(root).as_posix()] = size
    return files


def growth(before: dict[str, int], after: dict[str, int]) -> tuple[int, int]:
    """Count and total size of files that are new or changed in size."""
    changed = [size for path, size in after.items() if before.get(path) != size]
    return len(changed), sum(changed)


def decide(root: Path, state: Path, min_bytes: int) -> tuple[bool, str]:
    try:
        before = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return True, f"no usable snapshot ({error}); saving"
    if not isinstance(before, dict) or not before:
        return True, "the restore found no cache; saving"
    count, size = growth(before, listing(root))
    summary = f"{count} new or changed files, {size / 2**20:.1f} MiB"
    if size >= min_bytes:
        return True, f"{summary}; saving"
    return False, f"{summary}, below {min_bytes / 2**20:.0f} MiB; skipping the save"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("snapshot", "check"))
    parser.add_argument("root", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("--min-mib", type=float, default=DEFAULT_MIN_MIB)
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        files = listing(args.root)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(files), encoding="utf-8")
        print(f"{len(files)} files, {sum(files.values()) / 2**20:.1f} MiB restored")
        return 0

    save, reason = decide(args.root, args.state, int(args.min_mib * 2**20))
    print(reason, file=sys.stderr)
    print(f"save={'true' if save else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
