#!/usr/bin/env python3
"""E7: today vs Phase 0 vs the hybrid design on the E2 traces, in seconds.

Variants (same traces, same budgets the server actually chose):
  today    full copies in RAM and on disk; disk only when RAM misses;
           automatic staging limit
  phase0   today's format plus the cheap fixes: disk consulted whenever it
           holds a longer prefix; writes streamed (no staging limit);
           Flash-Next RAM counted by unique bytes (its KV is already shared)
  hybrid   shared KV chunks in RAM and on disk, small checkpoints, one index
  dense    hybrid plus a checkpoint at every message boundary, persisted too

Time model, all measured on this machine:
  prefill   fit_prefill.py (t = a + bN + cN(D + N/2) + eD)
  capture   27B full copy 28 ms/GB (E4); fixed-state capture same rate
  disk      write+fsync 0.59 GB/s, cold read 1.1 GB/s (diskbench.py)
"""
import argparse
import json
import pathlib

import numpy as np

from simulate_e5 import (GRID, IM_START, LEARN_MIN, MIN_STEP, Entry, Seq,
                         Store, lcp, load)

from paths import WORK as HERE
SIZE = {"fn": (119328358, 27460), "q27": (243700000, 65536)}
CAPTURE_S_PER_BYTE = {"fn": None, "q27": 28e-3 / 1e9}
WRITE_BPS, READ_BPS = 0.59e9, 1.1e9


def prefill_seconds(coef, n, d):
    if n <= 0:
        return 0.0
    return coef["a_s"] + coef["b_s_per_token"] * n + \
        coef["c_s_per_token_ctx"] * n * (d + n / 2) + coef["e_s_per_ctx"] * d


def simulate(requests, variant, model, ram_budget, disk_budget, staging,
             sessions, coef, chunk=2048):
    fixed, per_token = SIZE[model]
    ram_policy = {"today": "full", "phase0": "chunked" if model == "fn" else "full",
                  "hybrid": "chunked", "dense": "chunked"}[variant]
    disk_policy = "full" if variant in ("today", "phase0") else "chunked"
    disk_fix = variant != "today"
    stage_limit = staging if variant == "today" else None
    dense = variant == "dense"
    ram = Store(ram_policy, ram_budget, fixed, per_token, chunk, max_entries=128)
    disk = Store(disk_policy, disk_budget, fixed, per_token, chunk)
    live, clock, lifetime = [], 0, None
    totals = {"cached": 0, "prefill_s": 0.0, "capture_s": 0.0,
              "disk_write_bytes": 0, "disk_restore_s": 0.0, "disk_restores": 0}
    rows = []
    for request in requests:
        clock += 1
        if request["server_lifetime"] != lifetime:
            lifetime = request["server_lifetime"]
            ram.entries.clear()
            ram.chunk_refs.clear()
            live.clear()
        prompt = request["prompt"]
        seq = Seq(prompt, chunk)
        starts = np.nonzero(prompt == IM_START)[0]
        stable = int(starts[-1]) if len(starts) else len(prompt)
        cached, source = 0, "none"
        for state, length in live:
            if lcp(state.tokens[:length], prompt) >= length and length > cached:
                cached, source = length, "live"
        hit = ram.lookup(prompt)
        if hit and hit.length > cached:
            cached, source = hit.length, "memory"
            hit.used = clock
        if source == "none" or disk_fix:
            hit = disk.lookup(prompt)
            if hit and hit.length > cached:
                cached, source = hit.length, "disk"
                hit.used = clock
                read = disk.full_bytes(hit.length) * (1.13 if disk_policy == "chunked" else 1.0)
                totals["disk_restore_s"] += read / READ_BPS
                totals["disk_restores"] += 1
        cached = min(cached, len(prompt))
        scale = request["prompt_tokens"] / max(1, request["tokenized_prompt"])
        n_new = request["prompt_tokens"] - round(cached * scale)
        totals["prefill_s"] += prefill_seconds(coef, n_new, round(cached * scale))
        totals["cached"] += round(cached * scale)
        captures = []
        grid = [g for g in range(GRID, len(prompt) - 128 + 1, GRID)
                if g >= cached + GRID]
        if len(grid) > 4:
            step = len(grid) / 4
            grid = [grid[min(len(grid) - 1, int(round((k + 1) * step)) - 1)]
                    for k in range(4)]
        captures += [(g, "grid") for g in grid]
        if dense:
            # Every message boundary in RAM; the first one (end of the system
            # prompt, where conversations usually diverge) is also persisted.
            captures += [(int(s), "boundary" if i > 1 else "system")
                         for i, s in enumerate(starts) if cached < s < stable]
        others = [e.seq.tokens[:e.length] for e in ram.entries] + \
                 [s.tokens[:n] for s, n in live]
        learned = max((lcp(prompt, o) for o in others), default=0)
        if learned >= cached + LEARN_MIN and learned < stable - 64:
            captures.append((learned, "branch"))
        captures.append((stable, "stable"))
        captures.append((len(prompt), "prompt"))
        for position, purpose in captures:
            if position > cached or purpose in ("stable", "prompt"):
                if ram.admit(Entry(seq, position, purpose, clock)) and \
                        CAPTURE_S_PER_BYTE[model]:
                    copied = ram.full_bytes(position) if ram_policy == "full" else fixed
                    totals["capture_s"] += copied * CAPTURE_S_PER_BYTE[model]
        persist = ("stable", "prompt", "branch", "system")
        for position, purpose in captures:
            if purpose not in persist:
                continue
            if purpose not in ("branch", "system"):
                near = disk.lookup(prompt, limit=position)
                if near and position - near.length < MIN_STEP:
                    continue
            entry = Entry(seq, position, purpose, clock)
            if disk_policy == "chunked":
                new_chunks = sum(1 for key in disk.chunk_keys(entry)
                                 if key not in disk.chunk_refs)
                staged = disk.entry_cost(entry, new_chunks)
            else:
                staged = disk.full_bytes(position)
            if stage_limit is not None and staged > stage_limit:
                continue
            if disk.admit(entry, ranked=False):
                totals["disk_write_bytes"] += staged
        full = np.concatenate([prompt, request["output"]])
        live.append((Seq(full, chunk), len(full)))
        live = live[-sessions:]
        rows.append({"cached": round(cached * scale), "source": source})
    totals["disk_write_s"] = totals["disk_write_bytes"] / WRITE_BPS
    return totals, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("workload")
    parser.add_argument("--restart-at", type=int, default=None)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--disk-budget", type=int, default=16 << 30)
    args = parser.parse_args()
    base = HERE / "results" / "e2" / args.model / args.workload
    log = (base / "server.log").read_text(errors="replace")
    ram_budget = int(log.split("snapshot_cache_configured", 1)[1]
                     .split(" capacity_bytes=", 1)[1].split()[0])
    staging = int(log.split("staging_capacity_bytes=", 1)[1].split()[0])
    coef = json.loads((HERE / "results" / "prefill_fit.json").read_text())[args.model]
    requests = load(args.model, args.workload)
    if args.restart_at is not None:
        for index, request in enumerate(requests):
            request["server_lifetime"] = int(index >= args.restart_at)
    import re
    actual = sum(int(n) / float(t) for n, t in re.findall(
        r"event=completed .*?prefill_tokens=(\d+).*?prefill_tps=([\d.]+)", log)
        if int(n) > 0 and float(t) > 0)
    report = {"model": args.model, "workload": args.workload,
              "actual_prefill_s": round(actual, 2),
              "restart_at": args.restart_at, "ram_budget": ram_budget,
              "staging": staging, "variants": {}}
    for variant in ("today", "phase0", "hybrid", "dense"):
        totals, rows = simulate(requests, variant, args.model, ram_budget,
                                args.disk_budget, staging, args.sessions, coef)
        report["variants"][variant] = {k: round(v, 2) if isinstance(v, float) else v
                                       for k, v in totals.items()}
        if args.restart_at is not None:
            report["variants"][variant]["first_after_restart"] = rows[args.restart_at]
    tag = f"-restart{args.restart_at}" if args.restart_at is not None else ""
    (base / f"e7{tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
