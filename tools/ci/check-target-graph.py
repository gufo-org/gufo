#!/usr/bin/env python3
"""Check the CMake link graph written by cmake/TargetGraph.cmake.

Card 09 (RFC step 6) put the scheduler, runner pool, component text cache and
InferenceBackend in gufo_serving, and gufo_http stays a separate consumer.
Extensions, the runtime and the cache reach execution through neutral
contracts, so no target at or below gufo_ext_runtime may depend on gufo_serving
or gufo_http, directly or through an intermediate library.

The report is generated during CMake generation, so this check sees the same
edges the linker does. A protected target missing from the report is a failure:
that is how a newly declared library with no coverage gets caught.
"""

import argparse
import json
import sys
from pathlib import Path

PROTECTED = (
    "gufo_platform",
    "gufo_logging",
    "gufo_hip_host",
    "gufo_cache",
    "gufo_cache_hip",
    "gufo_core",
    "gufo_ext_runtime",
)


FORBIDDEN = {"gufo_serving", "gufo_http"}
HIP_ONLY = {"gufo_hip_host", "gufo_cache_hip", "gufo_deepseek_v4_flash",
            "gufo_qwen_kernels"}


def manifest_targets(root):
    """Every library target a model or family manifest declares."""
    targets = []
    for manifest in sorted(root.glob("src/**/library.json")):
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"check-target-graph: cannot read {manifest}: {error}")
        target = document.get("target")
        if isinstance(target, str) and target:
            targets.append(target)
    return sorted(targets)


def violations(report, protected):
    forbidden = FORBIDDEN
    problems = []
    if set(report.get("forbidden", [])) != FORBIDDEN:
        problems.append("report forbidden set disagrees with checker policy")
    if type(report.get("hip")) is not bool:
        problems.append("report must record boolean hip configuration")
    absent = set(report.get("absent", []))
    for name in sorted(absent):
        if name not in HIP_ONLY or report.get("hip") is not False:
            problems.append(f"protected target {name} cannot be absent in this configuration")
        if name in report["targets"]:
            problems.append(f"target {name} is both present and absent")
    declared = set(report["targets"]) | absent
    problems.extend(f"protected target {name} is missing from the link graph"
                    for name in protected if name not in declared)
    for name in sorted(report["targets"]):
        reached = forbidden & set(report["targets"][name])
        for dependency in sorted(reached):
            problems.append(
                f"{name} reaches {dependency} through the link graph "
                f"(closure: {', '.join(sorted(report['targets'][name]))})")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="JSON written by gufo_write_link_graph()")
    parser.add_argument("--root", type=Path, help="Repository root for manifests")
    parser.add_argument("--show", action="store_true", help="Print the whole graph")
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"check-target-graph: cannot read {args.report}: {error}")
        return 2
    protected = list(PROTECTED)
    if args.root:
        protected = sorted(set(protected) | set(manifest_targets(args.root.resolve())))
    if args.show:
        print(f"[check-target-graph] {len(protected)} protected targets")
        for name in sorted(report["targets"]):
            print(f"  {name} -> {', '.join(report['targets'][name]) or '(none)'}")
    problems = violations(report, protected)
    if problems:
        print("[check-target-graph] LINK GRAPH INVALID (nothing below "
              "gufo_ext_runtime may depend on the serving or HTTP tiers):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"[check-target-graph] PASS: link graph "
          f"({len(protected)} protected targets, no edge to "
          f"{' or '.join(sorted(report['forbidden']))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
