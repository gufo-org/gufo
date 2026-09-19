#!/usr/bin/env python3
"""gufo-mp-experiment — run multiple SHQ-T16 mixed-precision recipes and
measure retained quality + size, side-by-side.

For each preset in tools/gufo/recipe.py: quantize, benchmark candidate-vs-
teacher (matched-token KL/ppl/top1), record artifact bytes + per-tier tensor
counts. Prints a comparison table.

Usage:
  gufo-mp-experiment --source DIR --suite FILE --teacher-artifact DIR
                       [--imatrix DIR] [--presets a,b,c] [--json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo import recipe as recipe_mod


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--teacher-artifact", required=True)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--presets", default="bulk_g64,embed_only,embed_attn,mirror_no_lin,unsloth_mirror,full_shq8")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    this_dir = Path(__file__).resolve().parent
    presets = [p.strip() for p in args.presets.split(",")]
    env = dict(os.environ)
    results = {}
    for preset in presets:
        qdir = f"artifacts/quant-{preset}"
        plan = f"artifacts/work/plan-{preset}.json"
        import shutil
        shutil.rmtree(qdir, ignore_errors=True)  # no stale shards in size tally
        qa = ["--source", args.source, "--out", qdir, "--plan", plan,
              "--recipe", preset]
        if args.imatrix:
            qa += ["--imatrix", args.imatrix]
        run([sys.executable, str(this_dir / "gufo-quantize.py")] + qa)

        # artifact bytes
        total = 0
        for p in Path(qdir).rglob("*.shq4"):
            total += p.stat().st_size
        mb = total / 1e6

        env["GUFO_PLAN"] = plan
        r = run([sys.executable, str(this_dir / "gufo-bench.py"),
                 "--source", args.source, "--quant", qdir,
                 "--suite", args.suite, "--teacher-artifact", args.teacher_artifact,
                 "--repeats", str(args.repeats), "--json"], env=env)
        # stdout is pure JSON under --json; grab first { .. last }
        js = r.stdout[r.stdout.find("{"):r.stdout.rfind("}") + 1]
        bench = json.loads(js)

        # tier counts from the plan
        pd = json.loads(Path(plan).read_text("utf-8"))
        tiers = {}
        for info in pd["tensors"].values():
            tiers[info["format"]] = tiers.get(info["format"], 0) + 1

        results[preset] = {
            "recipe": preset,
            "size_mb": round(mb, 1),
            "tiers": tiers,
            "kl_mean": bench["quality"]["kl_mean"],
            "kl_median": bench["quality"]["kl_median"],
            "kl_p95": bench["quality"]["kl_p95"],
            "kl_max": bench["quality"]["kl_max"],
            "top1": bench["quality"]["top1_agreement"],
            "ppl": bench["quality"]["perplexity_candidate"],
        }
        print(f"[{preset}] KL {results[preset]['kl_mean']:.4f} "
              f"top1 {results[preset]['top1']:.4f} "
              f"ppl {results[preset]['ppl']:.3f} size {mb:.1f}MB", flush=True)

    if args.json:
        print(json.dumps({"schema": "gufo.mp-experiment.v1", "results": results}, indent=2))
    else:
        print("\n== mixed-precision quality vs size ==")
        hdr = f"{'recipe':16s} {'MB':>6s} {'KL mean':>8s} {'KL p95':>8s} {'top1':>6s} {'ppl':>6s}"
        print(hdr)
        for p, r in results.items():
            print(f"{p:16s} {r['size_mb']:6.1f} {r['kl_mean']:8.4f} "
                  f"{r['kl_p95']:8.4f} {r['top1']:6.3f} {r['ppl']:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
