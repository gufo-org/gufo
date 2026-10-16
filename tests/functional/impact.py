#!/usr/bin/env python3
"""Plan functional coverage from reviewed ownership; never starts a server."""

import argparse
import ast
import fnmatch
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def suite_names(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SUITES"
                for target in node.targets):
            return list(ast.literal_eval(node.value))
    raise ValueError(f"No SUITES registry in {path}")


def validate(manifest, names):
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported impact schema")
    if set(names) != set(manifest["suites"]):
        raise ValueError("Suite registry and impact manifest differ: " +
                         str(sorted(set(names) ^ set(manifest["suites"]))))
    tags = set(manifest["behaviors"])
    profiles = set(manifest["profiles"])
    ids = set()
    for rule in [*manifest["rules"], {"id": "fallback", **manifest["fallback"]}]:
        if rule["id"] in ids:
            raise ValueError("Duplicate rule: " + rule["id"])
        ids.add(rule["id"])
        if not set(rule["profiles"]) <= profiles:
            raise ValueError("Unknown rule profile: " + rule["id"])
        if not set(rule["affects"]) <= tags | {"*"}:
            raise ValueError("Unknown rule behavior: " + rule["id"])
    for name, suite in manifest["suites"].items():
        if not set(suite["covers"] + suite["uses"]) <= tags:
            raise ValueError("Unknown suite behavior: " + name)
        if not set(suite.get("profiles", profiles)) <= profiles:
            raise ValueError("Unknown suite profile: " + name)
        if suite["configuration"] not in manifest["configurations"]:
            raise ValueError("Unknown configuration: " + name)
        if suite["status"] not in ("active", "deferred"):
            raise ValueError("Unknown suite status: " + name)
    if not set(manifest["baseline_suites"]) <= set(names):
        raise ValueError("Unknown baseline suite")


def changed_paths(repo, base, head):
    # No rename detection: renames appear as delete/add, preserving both owners.
    output = subprocess.check_output([
        "git", "-C", str(repo), "diff", "--name-only", "--no-renames", "-z",
        base, head, "--"])
    return [path.decode("utf-8") for path in output.split(b"\0") if path]


def plan(manifest, paths, models=None):
    selected, deferred, excluded, obligations, resolutions = {}, {}, {}, [], []
    # Entries name a model (all its modes) or one profile such as flash-next-q4/mtp.
    model_filter = set(models) if models else None
    known_models = {row["model"] for row in manifest["profiles"].values()}
    if model_filter is not None and not model_filter <= known_models | set(manifest["profiles"]):
        raise ValueError("Unknown model filter")
    for path in sorted(set(paths)):
        parts = PurePosixPath(path).parts
        if not parts or path.startswith("/") or ".." in parts:
            raise ValueError("Expected repository-relative path: " + path)
        matches = [rule for rule in manifest["rules"]
                   if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule["paths"])]
        if matches:
            priority = max(rule["priority"] for rule in matches)
            rules = [rule for rule in matches if rule["priority"] == priority]
        else:
            rules = [{"id": "fallback", **manifest["fallback"]}]
        resolutions.append({"path": path, "rules": [rule["id"] for rule in rules]})
        for rule in rules:
            obligations.extend({"path": path, "rule": rule["id"], "check": check}
                               for check in rule["obligations"])
            for profile in rule["profiles"]:
                info = manifest["profiles"][profile]
                in_scope = (model_filter is None or info["model"] in model_filter
                            or profile in model_filter)
                for name, suite in manifest["suites"].items():
                    if profile not in suite.get("profiles", manifest["profiles"]):
                        continue
                    tags = set(suite["covers"] + suite["uses"])
                    intersects = "*" in rule["affects"] or bool(tags & set(rule["affects"]))
                    baseline = name in manifest["baseline_suites"]
                    if not intersects and not baseline:
                        continue
                    target = excluded if not in_scope else (
                        deferred if suite["status"] == "deferred" else selected)
                    key = (profile, name)
                    reason = {"path": path, "rule": rule["id"],
                              "behaviors": ["*"] if "*" in rule["affects"] else
                              sorted(tags & set(rule["affects"])),
                              "baseline": baseline}
                    if key not in target:
                        target[key] = {"profile": profile, "suite": name,
                                       "configuration": suite["configuration"],
                                       "vision_required": suite["vision_required"], "reasons": []}
                    target[key]["reasons"].append(reason)
    groups = {}
    for row in selected.values():
        key = (row["profile"], row["configuration"])
        groups.setdefault(key, {"profile": key[0], "configuration": key[1],
                               "settings": manifest["configurations"][key[1]], "suites": []})
        groups[key]["suites"].append(row["suite"])
    return {"schema": 1, "scope": sorted(model_filter or known_models),
            "paths": resolutions, "selected": list(selected.values()),
            "groups": list(groups.values()), "deferred": list(deferred.values()),
            "out_of_scope": list(excluded.values()), "obligations": obligations,
            "timing_qualification": "not performed; candidate correctness only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--manifest", type=Path, default=HERE / "impact.json")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--model", action="append",
                        help="Model or model/mode profile to plan; repeatable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        validate(manifest, suite_names(HERE / "run.py"))
        if any((args.repo, args.base, args.head)):
            if not all((args.repo, args.base, args.head)) or args.paths:
                raise ValueError("Use either paths or --repo/--base/--head together")
            paths = changed_paths(args.repo, args.base, args.head)
        else:
            paths = args.paths
        result = plan(manifest, paths, args.model)
        result.update(base=args.base, head=args.head)
        content = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(content)
        else:
            print(content, end="")
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
