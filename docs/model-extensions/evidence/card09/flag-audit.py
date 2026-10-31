#!/usr/bin/env python3
"""Reproduce the Card 09 moved-source compile-line audit.

Compares every compile-command token (minus -o/-c outputs and the build
directory) for the seven serving translation units between the Card 08 tree,
where the `gufo` executable compiled them, and this tree, where `gufo_serving`
does. Run from the repository root:

  python3 docs/model-extensions/evidence/card09/flag-audit.py \
    build/card08-gpu/compile_commands.json \
    build/card09-gpu/compile_commands.json gufo gufo_serving
"""

import json
import os
import re
import sys

SRCS = [
    "continuation_cache", "continuation_disk_store", "component_text_cache",
    "text_model_runner", "text_generation_scheduler", "continuation_identity",
    "inference_backend"
]


def entries(path, repo):
    out = {}
    if not os.path.exists(path):
        return out
    for entry in json.load(open(path)):
        name = os.path.basename(entry["file"])[:-4]
        if "/src/cli/serve/" not in entry["file"] or name not in SRCS:
            continue
        match = re.search(r"CMakeFiles/([^/]+)\.dir", entry["command"])
        target = match.group(1) if match else "?"
        tokens = entry["command"].split()[1:]
        flags, skip = [], False
        for token in tokens:
            if skip:
                skip = False
                continue
            if token in ("-o", "-c"):
                skip = True
                continue
            flags.append(token.replace(entry["directory"], "@D@"))
        out.setdefault(name, {})[target] = set(flags)
    return out


def main():
    old_path, new_path, old_target, new_target = sys.argv[1:5]
    repo = os.getcwd()
    old, new = entries(old_path, repo), entries(new_path, repo)
    for name in SRCS:
        before = old.get(name, {}).get(old_target)
        after = new.get(name, {}).get(new_target)
        if not before or not after:
            print(f"== {name}: MISSING baseline={bool(before)} "
                  f"candidate={bool(after)}")
            continue
        print(f"== {name}: adds {sorted(after - before)} "
              f"removes {sorted(before - after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
