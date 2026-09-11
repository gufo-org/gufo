#!/usr/bin/env python3
"""Compare DFlash2 companions against each Qwen27B target; BF16 is optional reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def qualified(report: dict) -> dict:
    """A partial or mismatching corpus is never a speed result."""
    aggregate = report["aggregate"]
    count = aggregate["prompts"]
    cases = report["cases"]
    if (count <= 0 or aggregate["completed"] != count or
            aggregate["exact"] != count or aggregate["skipped"] or
            len(cases) != count or len({case["id"] for case in cases}) != count):
        raise ValueError("incomplete, duplicate or mismatching corpus")
    if report["prompt_mode"] != "chat":
        raise ValueError("production companion comparison requires chat framing")
    return aggregate


def identity(path: Path) -> dict:
    # Once per artifact, outside every timed model invocation. stat identity in
    # the shared AR cache avoids hashing model weights again for each prompt.
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "result/bin/gufo")
    for name in ("target-q4", "target-q8", "draft-q4", "draft-q8"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--draft-bf16", type=Path,
                        help="optional BF16 reference companion; not a production recommendation")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing reports are never silently reused")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="three prompts instead of ten")
    args = parser.parse_args()
    if not os.environ.get("IN_NIX_SHELL"):
        parser.error("run inside nix develop")
    if args.max_tokens < 8 or args.repetitions < 1:
        parser.error("need at least eight tokens and one repetition")
    artifacts = {}
    names = ["binary", "target_q4", "target_q8", "draft_q4", "draft_q8"]
    if args.draft_bf16 is not None:
        names.append("draft_bf16")
    for name in names:
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, path)
        artifacts[name] = path
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {name: identity(path) for name, path in artifacts.items()}
    (args.output / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results: dict[tuple[str, str], list[dict]] = {}
    drafts = ["q4", "q8"] + (["bf16"] if args.draft_bf16 else [])
    for repetition in range(args.repetitions):
        for target in ("q4", "q8"):
            offset = repetition % len(drafts)
            order = drafts[offset:] + drafts[:offset]
            for draft in order:
                label = f"{target}-{draft}-r{repetition + 1}"
                report_path = (args.output / f"{label}.json").resolve()
                command = [
                    sys.executable, str(ROOT / "tools/quant/speculative-corpus.py"),
                    "--binary", str(args.binary),
                    "--model", str(artifacts[f"target_{target}"]),
                    "--draft-model", str(artifacts[f"draft_{draft}"]),
                    "--backend", "dflash2", "--prompt-mode", "chat",
                    "--max-tokens", str(args.max_tokens), "--draft-tokens", "7",
                    "--ar-cache", str((args.output / f"ar-{target}.json").resolve()),
                    "--json", str(report_path), "--label", label,
                ]
                if args.quick:
                    command.append("--quick")
                with (args.output / f"{label}.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                report = json.loads(report_path.read_text())
                results.setdefault((target, draft), []).append(qualified(report))
                print(f"{label}: exact corpus", flush=True)
    lines = [
        "| Target | Draft | tok/s | Acceptance |",
        "| --- | --- | ---: | ---: |",
    ]
    for (target, draft), rows in results.items():
        speed = statistics.median(row["spec_tps"] for row in rows)
        acceptance = statistics.median(row["acceptance"] for row in rows)
        lines.append(f"| {target} | {draft} | {speed:.2f} | {acceptance:.1%} |")
    text = "\n".join(lines) + "\n"
    (args.output / "README.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
