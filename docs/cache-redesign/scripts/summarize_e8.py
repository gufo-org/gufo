#!/usr/bin/env python3
"""Markdown tables for E8 (revision 2) from results/e2/*/*/e8*.json."""
import json

from paths import WORK as HERE

NAMES = {"fn": "Flash-Next", "q27": "27B"}
RUNS = ("w1", "w2", "w3", "w4", "w2-c4", "w3-c4")
E2 = HERE / "results" / "e2"


def load(model, run, tag=""):
    return json.loads((E2 / model / run / f"e8{tag}.json").read_text())


def main():
    e4 = json.loads((E2 / "e4.json").read_text())
    print("| Run | Measured prefill | Today | Phase 0 | Hybrid | Hybrid + dense "
          "| Reuse, simulated today vs actual | Requests within 64 tokens "
          "| Refusals: server / simulated |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in NAMES:
        for run in RUNS:
            d = load(model, run)
            v = d["variants"]
            n = len(json.loads((E2 / model / run / "requests.json").read_text()))
            cells = " | ".join(f"{v[k]['prefill_s']:.0f} s"
                               for k in ("today", "phase0", "hybrid", "dense"))
            dev = 100 * (v["today"]["cached"] - d["actual_cached"]) / d["actual_cached"]
            print(f"| {NAMES[model]} {run.upper()} | {d['actual_prefill_s']:.0f} s | {cells} "
                  f"| {dev:+.1f}% | {v['today']['within_64_of_actual']}/{n} "
                  f"| {d['actual_ram_refusals']} / {v['today']['ram_refusals']} |")
    print()
    print("| Model | Restart before request | Shutdown | Today | Phase 0 | Hybrid "
          "| Restored after restart: today / Phase 0 / hybrid |")
    print("| --- | ---: | --- | ---: | ---: | ---: | --- |")
    for model in NAMES:
        for r in (12, 20, 28):
            for abrupt in (False, True):
                d = load(model, "w1", f"-restart{r}" + ("-abrupt" if abrupt else ""))
                v = d["variants"]
                first = " / ".join(f"{v[k]['first_after_restart']['cached']:,}"
                                   for k in ("today", "phase0", "hybrid"))
                print(f"| {NAMES[model]} | {r} | {'abrupt' if abrupt else 'graceful'} "
                      f"| {v['today']['prefill_s']:.0f} s | {v['phase0']['prefill_s']:.0f} s "
                      f"| {v['hybrid']['prefill_s']:.0f} s | {first} |")
    print()
    print("| Run | Disk written: server | Today (sim) | Phase 0 | Hybrid | Hybrid + dense |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model in NAMES:
        for run in RUNS:
            v = load(model, run)["variants"]
            cells = " | ".join(f"{v[k]['disk_write_bytes'] / 1e9:.1f} GB"
                               for k in ("today", "phase0", "hybrid", "dense"))
            print(f"| {NAMES[model]} {run.upper()} | "
                  f"{e4[f'{model}/{run}']['disk_written_gb']:.1f} GB | {cells} |")


if __name__ == "__main__":
    main()
