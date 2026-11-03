#!/usr/bin/env python3
"""Audit an installed executable, provenance and explicitly required artifacts.

Packaging checks do not establish model correctness or performance. Keep each
report with the exact binary, model conformance results and distribution digest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def file_record(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty artifact: {path}")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def validate_provenance(provenance, mode, extensions):
    if not isinstance(provenance, dict):
        raise ValueError("invalid extension provenance object")
    if provenance.get("selectionMode") != mode:
        raise ValueError(f"expected selection mode '{mode}', got '{provenance.get('selectionMode')}'")
    if not provenance.get("extensionApiVersion"):
        raise ValueError("missing extension API identity")
    built = provenance.get("builtExtensions")
    if not isinstance(built, list):
        raise ValueError("missing built extension inventory")
    names = []
    for extension in built:
        if not isinstance(extension, dict) or not all(extension.get(field) for field in ("name", "version", "source")):
            raise ValueError("incomplete extension provenance")
        names.append(extension["name"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate built extension provenance")
    missing = sorted(set(extensions) - set(names))
    if missing:
        raise ValueError("missing expected built extensions: " + ", ".join(missing))


def runtime_records(manifest, directory):
    records = [file_record(manifest)]
    names = manifest.read_text(encoding="utf-8").splitlines()
    if not names or any(not name or Path(name).name != name or "\\" in name
                        or name in (".", "..") for name in names):
        raise ValueError(f"invalid runtime file manifest: {manifest}")
    for name in names:
        records.append(file_record(directory / name))
    return records


def audit(binary, mode, extensions, licenses, data, runtime_manifests):
    report = {"status": "FAIL", "binary": str(binary), "artifacts": [], "errors": []}
    try:
        report["artifacts"].append(file_record(binary))
        for command, key in [([str(binary), "--version"], "version"),
                             ([str(binary), "diagnose", "--section", "extensions", "--json"], "provenance_output")]:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            report[key] = {"argv": command, "returncode": result.returncode,
                           "stdout": result.stdout, "stderr": result.stderr}
            if result.returncode:
                raise ValueError(f"installed executable failed: {' '.join(command)}")
        provenance = json.loads(report["provenance_output"]["stdout"])
        validate_provenance(provenance, mode, extensions)
        report["provenance"] = provenance
        if not licenses:
            raise ValueError("at least one required license artifact must be specified")
        for path in [*licenses, *data]:
            report["artifacts"].append(file_record(path))
        for manifest in runtime_manifests:
            report["artifacts"].extend(runtime_records(manifest, binary.parent))
        report["status"] = "PASS"
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report["errors"].append(str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mode", choices=("discovery", "lock", "explicit"), required=True)
    parser.add_argument("--extension", action="append", default=[])
    parser.add_argument("--license", type=Path, action="append", default=[])
    parser.add_argument("--data", type=Path, action="append", default=[])
    parser.add_argument("--runtime-manifest", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists; retain earlier attempts and use a new path")
    report = audit(args.binary, args.mode, args.extension, args.license, args.data, args.runtime_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        parser.exit(1, "\n".join(report["errors"]) + "\n")
    print(f"PASS: installed executable, provenance and {len(report['artifacts'])} required artifacts")


if __name__ == "__main__":
    main()
