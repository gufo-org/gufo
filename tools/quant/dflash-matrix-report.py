#!/usr/bin/env python3
"""Aggregate DFlash-2 companion sweep reports into comparison tables.

Reads the JSON reports written by ``speculative-corpus.py --json`` and pivots
them into a target-by-companion matrix, so the aggregate throughput, acceptance
and per-category behaviour of every companion can be read against one target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


TARGET_LABEL = {"q8": "UD-Q8_K_L", "q4": "UD-Q4_K_XL"}
DRAFT_LABEL = {
    "dq4": "DFlash2 Q4_K_M",
    "dq8": "DFlash2 Q8_0",
    "dbf16": "DFlash2 BF16",
}
DRAFT_ORDER = ["dq4", "dq8", "dbf16"]
TARGET_ORDER = ["q8", "q4"]


def load(directory: Path, pattern: str) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(directory.glob(pattern)):
        if path.name.startswith("ar-"):
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        label = report.get("label", "")
        parts = label.split("-")
        if len(parts) < 2:
            continue
        grouped.setdefault((parts[0], parts[1]), []).append(report)
    return grouped


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/tmp/dflash-matrix")
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob selecting the reports to aggregate. One suite writes one\nfamily of names, so this is how a corpus is read on its own.",
    )
    args = parser.parse_args()

    grouped = load(Path(args.directory), args.pattern)
    if not grouped:
        print("no reports found")
        return 1

    # The token-weighted aggregate is dominated by whichever prompts run
    # longest, and those are exactly the ones the drafter does worst on -- a
    # prose case that generates the full budget contributes several times the
    # tokens of a code case that stops at EOS. The per-case median is reported
    # beside it so a companion is not ranked by one hostile prompt.
    print("## Aggregate per target and companion\n")
    print(
        "| Target | Companion | Reps | Exact | AR tok/s | Speculative tok/s | "
        "Speedup | Median case tok/s | Median acceptance | Acceptance | "
        "Avg draft |"
    )
    print(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |"
    )
    for target in TARGET_ORDER:
        for draft in DRAFT_ORDER:
            reports = grouped.get((target, draft))
            if not reports:
                continue
            aggregates = [report["aggregate"] for report in reports]
            aggregates = [item for item in aggregates if "ar_tps" in item]
            if not aggregates:
                continue
            exact = min(int(item["exact"]) for item in aggregates)
            prompts = aggregates[0].get("completed", aggregates[0]["prompts"])
            cases = [case for report in reports for case in report["cases"]]
            print(
                f"| {TARGET_LABEL.get(target, target)} | "
                f"{DRAFT_LABEL.get(draft, draft)} | {len(reports)} | "
                f"{exact}/{prompts} | "
                f"{median([item['ar_tps'] for item in aggregates]):.2f} | "
                f"{median([item['spec_tps'] for item in aggregates]):.2f} | "
                f"{median([item['speedup'] for item in aggregates]):.2f}x | "
                f"{median([case['spec_tps'] for case in cases]):.2f} | "
                f"{median([case['acceptance'] for case in cases]) * 100.0:.1f}% | "
                f"{median([item['acceptance'] for item in aggregates]) * 100.0:.1f}% | "
                f"{median([item['average_draft'] for item in aggregates]):.2f} |"
            )

    print("\n## Per case, speculative tok/s (acceptance)\n")
    case_ids: list[str] = []
    for reports in grouped.values():
        for case in reports[0]["cases"]:
            if case["id"] not in case_ids:
                case_ids.append(case["id"])

    for target in TARGET_ORDER:
        drafts = [draft for draft in DRAFT_ORDER if (target, draft) in grouped]
        if not drafts:
            continue
        print(f"\n### Target {TARGET_LABEL.get(target, target)}\n")
        header = " | ".join(DRAFT_LABEL.get(draft, draft) for draft in drafts)
        print(f"| Case | AR | {header} |")
        print("| --- | ---: |" + " ---: |" * len(drafts))
        for case_id in case_ids:
            cells: list[str] = []
            ar_values: list[float] = []
            present = False
            for draft in drafts:
                rows = [
                    case
                    for report in grouped[(target, draft)]
                    for case in report["cases"]
                    if case["id"] == case_id
                ]
                if not rows:
                    cells.append("-")
                    continue
                present = True
                ar_values.append(median([row["ar_tps"] for row in rows]))
                mark = "" if all(row["exact"] for row in rows) else " !"
                cells.append(
                    f"{median([row['spec_tps'] for row in rows]):.2f} "
                    f"({median([row['acceptance'] for row in rows]) * 100.0:.0f}%)"
                    f"{mark}"
                )
            if not present:
                continue
            print(
                f"| {case_id} | {median(ar_values):.2f} | " + " | ".join(cells) + " |"
            )

    print("\n## Per category, speculative tok/s (acceptance)\n")
    categories: list[str] = []
    for reports in grouped.values():
        for case in reports[0]["cases"]:
            if case.get("category") and case["category"] not in categories:
                categories.append(case["category"])

    for target in TARGET_ORDER:
        drafts = [draft for draft in DRAFT_ORDER if (target, draft) in grouped]
        if not drafts:
            continue
        print(f"\n### Target {TARGET_LABEL.get(target, target)}\n")
        header = " | ".join(DRAFT_LABEL.get(draft, draft) for draft in drafts)
        print(f"| Category | {header} |")
        print("| --- |" + " ---: |" * len(drafts))
        for category in categories:
            cells: list[str] = []
            for draft in drafts:
                rows = [
                    case
                    for report in grouped[(target, draft)]
                    for case in report["cases"]
                    if case.get("category") == category
                ]
                if not rows:
                    cells.append("-")
                    continue
                cells.append(
                    f"{median([row['spec_tps'] for row in rows]):.2f} "
                    f"({median([row['acceptance'] for row in rows]) * 100.0:.0f}%)"
                )
            print(f"| {category} | " + " | ".join(cells) + " |")

    print("\n`!` marks a case whose speculative completion diverged from the "
          "autoregressive reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
