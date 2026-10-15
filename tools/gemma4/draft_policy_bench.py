#!/usr/bin/env python3
"""A/B Gemma 4 MTP draft policies over HTTP on prose, story and code workloads.

Starts one `gufo serve llm` per policy (in the order given, so repeat names in
ABBA order to cancel run-order drift), sends every workload, and records the
server's decode rate, draft counts and completion hashes. Greedy workloads
must produce identical text under every policy; sampled ones differ by
design (draft counts change the random draws), so they use several seeds.

    nix develop -c python3 tools/gemma4/draft_policy_bench.py run \\
        --model TARGET.gguf --mtp DRAFTER.gguf --out ab.json [--depth 8192] \\
        "confidence=--draft-policy confidence" "calibrated=--draft-policy calibrated" \\
        "calibrated2=--draft-policy calibrated" "confidence2=--draft-policy confidence"
    nix develop -c python3 tools/gemma4/draft_policy_bench.py table ab.json

Workloads (d0 unless --depth adds a shared earlier turn of that many tokens,
reused from the prompt cache): `prose` (12 greedy summary-and-story prompts
over the model-bench synthetic passages, 256 tokens each), `story`
(temperature 1, top-k 64, top-p 0.95, repeat penalty 1.05, 12 seeds x 128),
`code-write` (a new Python module from a spec, greedy, 384), `code-edit`
(refactor a repository file given in context, greedy, 384), `repetitive`
(copy the passage verbatim, greedy, 256) and `code-sampled` (the code-write
spec at temperature 0.6, top-p 0.95, 12 seeds x 192).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gufo.model_bench.llm import TASKS, synthetic_text  # noqa: E402

WORDS = 1800  # about 2K tokens of the model-bench synthetic passage
SEEDS = 12
CODE_SPEC = ("Write a complete, production-quality Python module implementing a thread-safe LRU cache "
             "with per-entry TTL expiry. Include type hints, docstrings, a small CLI demo, and a pytest "
             "test suite covering eviction order, expiry and concurrent access. Output only the code.")
STORY = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "repeat_penalty": 1.05}


def workloads() -> list[tuple[str, int, list[dict]]]:
    """(name, max_tokens, requests); each request is {prompt, sampling...}."""
    code_file = (ROOT / "tools/gemma4/image_bench.py").read_text()
    edit = ("Here is a Python script:\n\n```python\n" + code_file + "```\n\nRefactor it: add type hints to "
            "every function, rename the function `main` to `run`, and replace every f-string that formats "
            "a float with an explicit `format()` call. Output the complete updated file and nothing else.")
    passage = synthetic_text(100_000, WORDS)
    return [
        ("prose", 256, [{"prompt": synthetic_text(200_000 + i, WORDS) + TASKS["prose"], "temperature": 0}
                        for i in range(SEEDS)]),
        ("story", 128, [{"prompt": passage + TASKS["story"], **STORY, "seed": s} for s in range(1, SEEDS + 1)]),
        ("code-write", 384, [{"prompt": CODE_SPEC, "temperature": 0}]),
        ("code-edit", 384, [{"prompt": edit, "temperature": 0}]),
        ("repetitive", 256, [{"prompt": passage + TASKS["repetition"], "temperature": 0}]),
        ("code-sampled", 192, [{"prompt": CODE_SPEC, "temperature": 0.6, "top_p": 0.95, "seed": s}
                               for s in range(1, SEEDS + 1)]),
    ]


def post(port: int, body: dict) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def run_policy(args: argparse.Namespace, server_args: list[str]) -> dict:
    prefix = []
    if args.depth:
        prefix = [{"role": "user", "content": synthetic_text(31_337, int(args.depth / 1.137))},
                  {"role": "assistant", "content": "Understood."}]
    log_dir = ROOT / "artifacts/gemma4"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [str(args.gufo), "serve", "--port", str(args.port), "--sessions", "1", "llm",
               "--model", args.model, "--served-model-name", "bench", "--think", "off",
               "--speculative", "mtp", "--mtp-model", args.mtp,
               "--context", str(16384 + args.depth), *server_args]
    with open(log_dir / "draft-policy-bench-server.log", "w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(1200):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{args.port}/ready", timeout=2)
                    break
                except Exception:
                    if server.poll() is not None:
                        raise SystemExit(f"server exited; see {log_dir / 'draft-policy-bench-server.log'}")
                    time.sleep(0.5)
            post(args.port, {"model": "bench", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8})
            results = {}
            for name, max_tokens, requests in workloads():
                if args.workloads and name not in args.workloads:
                    continue
                tokens = ms = drafted = accepted = 0
                hashes = []
                for request in requests:
                    request = dict(request)
                    prompt = request.pop("prompt")
                    reply = post(args.port, {"model": "bench", "max_tokens": max_tokens, **request,
                                             "messages": prefix + [{"role": "user", "content": prompt}]})
                    timings = reply["timings"]
                    tokens += timings["predicted_n"]
                    ms += timings["predicted_ms"]
                    drafted += timings.get("draft_n", 0)
                    accepted += timings.get("draft_n_accepted", 0)
                    text = reply["choices"][0]["message"]["content"]
                    hashes.append(hashlib.sha256(text.encode()).hexdigest()[:16])
                cycles = tokens - accepted
                results[name] = {"tok_s": 1000 * tokens / ms, "tokens": tokens,
                                 "drafts_per_cycle": drafted / cycles, "accepted_per_cycle": accepted / cycles,
                                 "hashes": hashes}
            return {"server_args": server_args, "workloads": results}
        finally:
            server.terminate()
            server.wait()


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    doc = json.loads(out.read_text()) if out.exists() else {}
    doc.setdefault("meta", {
        "tool": "tools/gemma4/draft_policy_bench.py",
        "model": Path(args.model).name, "mtp": Path(args.mtp).name, "depth": args.depth,
        "gufo": str(args.gufo), "revision": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "date": time.strftime("%Y-%m-%d")})
    doc.setdefault("runs", {})
    for spec in args.policies:
        name, _, server_args = spec.partition("=")
        doc["runs"][name] = run_policy(args, server_args.split())
        out.write_text(json.dumps(doc, indent=1) + "\n")
        r = doc["runs"][name]["workloads"]
        print(name, " ".join(f"{w}:{v['tok_s']:.2f}" for w, v in r.items()), flush=True)
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    """Mean of the runs whose names start with each policy name."""
    for path in args.files:
        doc = json.loads(Path(path).read_text())
        runs = doc["runs"]
        policies = sorted({name.rstrip("0123456789") for name in runs})
        base = policies[1] if len(policies) > 1 and policies[0] == "calibrated" else policies[0]
        print(f"### {Path(path).name}: {doc['meta']['model']}, depth {doc['meta']['depth']}")
        print("| Workload | " + " | ".join(f"{p} (tok/s)" for p in policies) + " | Same greedy text |")
        print("|---|" + "---:|" * len(policies) + "---|")
        names = next(iter(runs.values()))["workloads"]
        for workload in names:
            cells, hashes = [], []
            for policy in policies:
                members = [r["workloads"][workload] for n, r in runs.items() if n.rstrip("0123456789") == policy]
                mean = sum(m["tok_s"] for m in members) / len(members)
                base_members = [r["workloads"][workload] for n, r in runs.items()
                                if n.rstrip("0123456789") == base]
                base_mean = sum(m["tok_s"] for m in base_members) / len(base_members)
                gain = "" if policy == base else f" ({100 * (mean / base_mean - 1):+.1f}%)"
                cells.append(f"{mean:.2f}{gain}")
                hashes += [m["hashes"] for m in members]
            sampled = workload in ("story", "code-sampled")
            same = "—" if sampled else ("yes" if all(h == hashes[0] for h in hashes) else "NO")
            print(f"| {workload} | " + " | ".join(cells) + f" | {same} |")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--mtp", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--gufo", type=Path, default=ROOT / "result/bin/gufo")
    run.add_argument("--depth", type=int, default=0, help="tokens of shared earlier conversation")
    run.add_argument("--port", type=int, default=58123)
    run.add_argument("--workloads", nargs="*", default=[])
    run.add_argument("policies", nargs="+", help="NAME=SERVER_ARGS, e.g. calibrated='--draft-policy calibrated'")
    table = sub.add_parser("table")
    table.add_argument("files", nargs="+")
    args = parser.parse_args()
    return cmd_run(args) if args.command == "run" else cmd_table(args)


if __name__ == "__main__":
    raise SystemExit(main())
