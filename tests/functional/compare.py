#!/usr/bin/env python3
"""Qualify retained main/PR observations without rerunning GPU work."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from metrics import qualify
from run import COMPARISON_FIELDS, compare_runs


def evidence(pairs, controls=()):
    """No duplicate runs, mismatched builds, lost history or hidden bad samples."""
    comparisons, control_comparisons, sources = [], [], []
    used, builds, reference = set(), None, None
    for control, selected in ((False, pairs), (True, controls)):
        for before, after in selected:
            paths = [Path(before).resolve(), Path(after).resolve()]
            if paths[0] == paths[1]:
                raise ValueError("a run cannot be compared with itself")
            # An unchanged-main control can reuse its main anchor, but must
            # contribute a new independent run.
            new_paths = paths[1:] if control else paths
            if any(path in used for path in new_paths):
                raise ValueError("reused observations cannot confirm a regression")
            used.update(paths)
            reports = [json.loads((path / "report.json").read_text()) for path in paths]
            binary_pair = [report["binary"] for report in reports]
            if builds is None:
                builds, reference = binary_pair, reports[0]
            if binary_pair != ([builds[0], builds[0]] if control else builds):
                raise ValueError("follow-ups must use the same main and PR production builds")
            for report in reports:
                for key in COMPARISON_FIELDS:
                    if report.get(key) != reference.get(key):
                        raise ValueError(f"follow-up changes {key}")
                suites = list(report["suites"])
                initial = list(reference["suites"])
                if not suites or initial[:len(suites)] != suites:
                    raise ValueError("follow-up must retain preceding suites/cache history")
            comparison = compare_runs(*paths)
            (control_comparisons if control else comparisons).append(comparison)
            source = {"kind": "main-control" if control else "main-pr", "runs": []}
            for path in paths:
                files = sorted([path / "report.json", *path.glob("*.requests.json")])
                source["runs"].append({
                    "directory": str(path),
                    "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                              for file in files},
                })
            sources.append(source)
    result = qualify(comparisons, control_comparisons)
    result["evidence"] = sources
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                        metavar=("MAIN", "PR"), help="Original pair first, then focused follow-ups")
    parser.add_argument("--control", nargs=2, type=Path, action="append", default=[],
                        metavar=("MAIN", "MAIN_REPEAT"), help="Optional unchanged-main control")
    parser.add_argument("--output", type=Path, required=True, help="New JSON evidence file")
    args = parser.parse_args()
    try:
        result = evidence(args.pair, args.control)
    except (ValueError, KeyError, OSError) as error:
        result = {"status": "failed", "error": str(error), "measurements": []}
    try:
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 1
    counts = {status: sum(row["status"] == status for row in result["measurements"])
              for status in ("passed", "failed", "inconclusive")}
    print(f"{result['status']}: {counts}; {args.output}")
    return {"passed": 0, "failed": 1, "inconclusive": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
