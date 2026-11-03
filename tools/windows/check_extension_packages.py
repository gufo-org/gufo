#!/usr/bin/env python3
"""Check selected extension Windows dependencies before vcpkg's project hook.

Metadata declares ports/features; it does not install them. A custom vcpkg
manifest may be supplied for an external extension without editing Gufo's.
"""
import argparse
import json
from pathlib import Path
import re


SPEC = re.compile(r"([a-z0-9][a-z0-9-]*)(?:\[([a-z0-9,-]+)\])?\Z")


def check(root, manifest, external=(), from_lock=False):
    ports = {}
    for dependency in manifest["dependencies"]:
        if isinstance(dependency, str):
            ports.setdefault(dependency, set())
        else:
            # Platform-conditional entries cannot prove Windows availability.
            if dependency.get("platform") not in (None, "windows"):
                continue
            ports.setdefault(dependency["name"], set()).update(dependency.get("features", []))
    paths = list((root / "src/models").glob("*/extension.json"))
    paths += list((root / "src/families").glob("*/*/extension.json"))
    for directory in external:
        paths += ([directory / "extension.json"] if (directory / "extension.json").is_file()
                  else list(directory.glob("*/extension.json")))
    lock_path = root / "extensions.lock"
    # Packaging recovery may target a pre-extension release with no manifests.
    locked = (set(json.loads(lock_path.read_text())["extensions"])
              if from_lock and (paths or lock_path.exists()) else None)
    errors = []
    checked = []
    for path in sorted(set(paths)):
        extension = json.loads(path.read_text())
        if "windows" not in extension["platforms"] or (locked is not None and extension["name"] not in locked):
            continue
        checked.append(extension["name"])
        for package in extension.get("packages", []):
            declaration = package.get("windows")
            if not declaration:
                continue
            match = SPEC.fullmatch(declaration)
            if match is None:
                errors.append(f"{extension['name']}: invalid vcpkg port/features '{declaration}'")
                continue
            port = match[1]
            features = set(match[2].split(",")) if match[2] else set()
            if port not in ports:
                errors.append(f"{extension['name']}: vcpkg manifest is missing port '{port}'")
            elif features - ports[port]:
                errors.append(f"{extension['name']}: port '{port}' needs explicit features "
                              + ", ".join(sorted(features - ports[port])))
    return {"checked_extensions": checked, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extension-dir", type=Path, action="append", default=[])
    parser.add_argument("--from-lock", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = check(args.root, json.loads((args.manifest or args.root / "vcpkg.json").read_text()),
                       args.extension_dir, args.from_lock)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"extension package check: {error}\n")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if report["errors"]:
        parser.exit(1, "\n".join(report["errors"]) + "\n")
    print(f"PASS: Windows dependencies for {len(report['checked_extensions'])} selected extensions")


if __name__ == "__main__":
    main()
