#!/usr/bin/env python3
"""E8: E7 with production admission rules and asynchronous disk publication.

Changes from simulate_e7.py:
- RAM admission follows src/cli/serve/continuation_cache.cpp: an incoming
  checkpoint may evict only entries whose removal rank is at most its own
  ceiling (MaxRemovalPriority: history 1, continuation/branch point 3); the
  lowest rank goes first, oldest within a rank; the entry the request
  restored from is protected; when nothing evictable frees enough bytes the
  checkpoint is refused (counted, to compare with the server's
  `event=snapshot action=skipped reason=byte_capacity` lines).
- Removal ranks follow RemovalPriority: branch points 3; entries that are a
  strict prefix of the incoming continuation 1 (history) or 2; history
  entries related to a retained continuation 1; continuations covered by a
  longer continuation 2; everything else 3.
- Disk writes go through one serial writer at the measured gufo rate
  (0.44 GB/s) and become restorable only when published; a restart drains
  the queue first, as the server does on shutdown.

Variants and time model are those of simulate_e7.py. Phase 0 here assumes
bounded, streamed transfers in both directions (writes and restores), which
today's code does not have: ReadImage loads a whole file and refuses files
larger than staging.
"""
import argparse
import datetime
import json
import pathlib
import re

import numpy as np

from simulate_e5 import GRID, IM_START, LEARN_MIN, MIN_STEP, Seq, lcp, load
from simulate_e7 import CAPTURE_S_PER_BYTE, READ_BPS, SIZE, prefill_seconds

from paths import WORK as HERE
WRITE_BPS = 0.44e9
CEILING = {"grid": 1, "boundary": 1, "system": 1, "stable": 3, "prompt": 3,
           "branch": 3}
CONTINUATION = {"stable", "prompt", "branch"}


class Entry:
    def __init__(self, seq, length, purpose, clock):
        self.seq, self.length, self.purpose, self.used = seq, length, purpose, clock

    def tokens(self):
        return self.seq.tokens[:self.length]

    def prefix_of(self, tokens):
        return lcp(self.tokens(), tokens) >= self.length


class Store:
    def __init__(self, policy, budget, fixed, per_token, chunk, max_entries=None):
        self.policy, self.budget = policy, budget
        self.fixed, self.per_token, self.chunk = fixed, per_token, chunk
        self.max_entries = max_entries
        self.entries, self.chunk_refs = [], {}
        self.refusals = 0

    def full_bytes(self, length):
        return self.fixed + self.per_token * length

    def keys(self, entry):
        return entry.seq.keys[:entry.length // self.chunk]

    def cost(self, entry, new_chunks):
        if self.policy == "full":
            return self.full_bytes(entry.length)
        tail = entry.length - (entry.length // self.chunk) * self.chunk
        return self.fixed + self.per_token * (tail + new_chunks * self.chunk)

    def used(self):
        if self.policy == "full":
            return sum(self.full_bytes(e.length) for e in self.entries)
        tails = sum(e.length - (e.length // self.chunk) * self.chunk for e in self.entries)
        return len(self.entries) * self.fixed + self.per_token * (
            tails + len(self.chunk_refs) * self.chunk)

    def lookup(self, tokens, limit=None):
        best = None
        for entry in self.entries:
            if limit is not None and entry.length > limit:
                continue
            if (best is None or entry.length > best.length) and entry.prefix_of(tokens):
                best = entry
        return best

    def rank(self, entry, incoming):
        peers = [p for p in self.entries if p is not entry and p.purpose in CONTINUATION]
        nexts = {int(p.seq.tokens[entry.length]) for p in peers
                 if p.length > entry.length and entry.prefix_of(p.tokens())}
        if entry.purpose in CONTINUATION and len(nexts) > 1:
            return 3
        if entry.purpose == "branch" and not any(
                p.purpose == "branch" and p.length > entry.length and
                entry.prefix_of(p.tokens()) for p in peers):
            return 3
        if incoming is not None and entry.length < len(incoming) and \
                entry.prefix_of(incoming):
            return 1 if entry.purpose not in CONTINUATION else 2
        for p in peers:
            if entry.purpose not in CONTINUATION and (
                    entry.prefix_of(p.tokens()) or p.prefix_of(entry.tokens())):
                return 1
            if entry.purpose in CONTINUATION and p.length > entry.length and \
                    entry.prefix_of(p.tokens()):
                return 2
        return 3

    def remove(self, entry):
        self.entries.remove(entry)
        if self.policy == "chunked":
            for key in self.keys(entry):
                self.chunk_refs[key] -= 1
                if not self.chunk_refs[key]:
                    del self.chunk_refs[key]

    def admit(self, entry, ranked=True, source=None):
        for other in self.entries:
            if other.length == entry.length and other.prefix_of(entry.seq.tokens):
                other.used = entry.used
                return False
        new = sum(1 for k in self.keys(entry) if k not in self.chunk_refs) \
            if self.policy == "chunked" else 0
        cost = self.cost(entry, new)
        if cost > self.budget:
            self.refusals += 1
            return False
        self.entries.append(entry)
        if self.policy == "chunked":
            for key in self.keys(entry):
                self.chunk_refs[key] = self.chunk_refs.get(key, 0) + 1
        ceiling = CEILING[entry.purpose] if ranked else 3
        incoming = entry.tokens() if entry.purpose in CONTINUATION else None
        while self.used() > self.budget or (
                self.max_entries and len(self.entries) > self.max_entries):
            candidates = []
            for e in self.entries:
                if e is entry or e is source:
                    continue
                r = self.rank(e, incoming) if ranked else 0
                if r <= ceiling:
                    candidates.append((r, e.used, e))
            if not candidates:
                self.remove(entry)
                self.refusals += 1
                return False
            self.remove(min(candidates, key=lambda c: (c[0], c[1]))[2])
        return cost


def parse_time(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").timestamp()


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
    queue, writer_free = [], 0.0
    live, clock, lifetime = [], 0, None
    totals = {"cached": 0, "prefill_s": 0.0, "capture_s": 0.0,
              "disk_write_bytes": 0, "disk_restore_s": 0.0, "disk_restores": 0}
    rows = []

    def publish(until):
        """Run the serial writer up to `until`: each job is checked for
        spacing and staging when the writer reaches it, as Save() does."""
        nonlocal queue, writer_free
        while queue and max(writer_free, queue[0][0]) <= until:
            enqueued, entry, prompt_tokens = queue.pop(0)
            start = max(writer_free, enqueued)
            if entry.purpose not in ("branch", "system"):
                near = disk.lookup(prompt_tokens, limit=entry.length)
                if near and entry.length - near.length < MIN_STEP:
                    continue
            if disk_policy == "chunked":
                new = sum(1 for k in disk.keys(entry) if k not in disk.chunk_refs)
                staged = disk.cost(entry, new)
            else:
                staged = disk.full_bytes(entry.length)
            if stage_limit is not None and staged > stage_limit:
                continue
            if disk.admit(entry, ranked=False):
                totals["disk_write_bytes"] += staged
                writer_free = start + staged / WRITE_BPS

    for request in requests:
        clock += 1
        now = parse_time(request["time"])
        if request["server_lifetime"] != lifetime:
            publish(float("inf"))
            writer_free = 0.0
            lifetime = request["server_lifetime"]
            ram.entries.clear()
            ram.chunk_refs.clear()
            live.clear()
        publish(now)
        prompt = request["prompt"]
        seq = Seq(prompt, chunk)
        starts = np.nonzero(prompt == IM_START)[0]
        stable = int(starts[-1]) if len(starts) else len(prompt)
        cached, source, source_entry = 0, "none", None
        for state, length in live:
            if lcp(state.tokens[:length], prompt) >= length and length > cached:
                cached, source = length, "live"
        hit = ram.lookup(prompt)
        if hit and hit.length > cached:
            cached, source, source_entry = hit.length, "memory", hit
            hit.used = clock
        if source == "none" or disk_fix:
            hit = disk.lookup(prompt)
            if hit and hit.length > cached and (
                    stage_limit is None or disk.full_bytes(hit.length) <= stage_limit):
                cached, source = hit.length, "disk"
                hit.used = clock
                read = disk.full_bytes(hit.length) * (1.13 if disk_policy == "chunked" else 1.0)
                totals["disk_restore_s"] += read / READ_BPS
                totals["disk_restores"] += 1
        cached = min(cached, len(prompt))
        scale = request["prompt_tokens"] / max(1, request["tokenized_prompt"])
        totals["prefill_s"] += prefill_seconds(
            coef, request["prompt_tokens"] - round(cached * scale), round(cached * scale))
        totals["cached"] += round(cached * scale)
        captures = []
        grid = [g for g in range(GRID, len(prompt) - 128 + 1, GRID) if g >= cached + GRID]
        if len(grid) > 4:
            step = len(grid) / 4
            grid = [grid[min(len(grid) - 1, int(round((k + 1) * step)) - 1)] for k in range(4)]
        captures += [(g, "grid") for g in grid]
        if dense:
            captures += [(int(s), "boundary" if i > 1 else "system")
                         for i, s in enumerate(starts) if cached < s < stable]
        others = [e.tokens() for e in ram.entries] + [s.tokens[:n] for s, n in live]
        learned = max((lcp(prompt, o) for o in others), default=0)
        if learned >= cached + LEARN_MIN and learned < stable - 64:
            captures.append((learned, "branch"))
        captures.append((stable, "stable"))
        captures.append((len(prompt), "prompt"))
        for position, purpose in captures:
            if position > cached or purpose in ("stable", "prompt"):
                admitted = ram.admit(Entry(seq, position, purpose, clock), source=source_entry)
                if admitted and CAPTURE_S_PER_BYTE[model]:
                    copied = ram.full_bytes(position) if ram_policy == "full" else fixed
                    totals["capture_s"] += copied * CAPTURE_S_PER_BYTE[model]
        for position, purpose in captures:
            if purpose in ("stable", "prompt", "branch", "system"):
                queue.append((now, Entry(seq, position, purpose, clock), prompt))
        full = np.concatenate([prompt, request["output"]])
        # The server also retains the finished state (prompt + output) as a
        # continuation snapshot, which competes for the same RAM budget.
        ram.admit(Entry(Seq(full, chunk), len(full), "prompt", clock),
                  source=source_entry)
        live.append((Seq(full, chunk), len(full)))
        live = live[-sessions:]
        rows.append({"cached": round(cached * scale), "source": source})
    totals["ram_refusals"] = ram.refusals
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
    actual_prefill = sum(int(n) / float(t) for n, t in re.findall(
        r"event=completed .*?prefill_tokens=(\d+).*?prefill_tps=([\d.]+)", log)
        if int(n) > 0 and float(t) > 0)
    actual_refusals = len(re.findall(
        r"event=snapshot action=skipped reason=byte_capacity", log))
    requests = load(args.model, args.workload)
    if args.restart_at is not None:
        for index, request in enumerate(requests):
            request["server_lifetime"] = int(index >= args.restart_at)
    actual = [r["cached_tokens"] or 0 for r in requests]
    report = {"model": args.model, "workload": args.workload,
              "restart_at": args.restart_at, "ram_budget": ram_budget,
              "staging": staging, "actual_prefill_s": round(actual_prefill, 2),
              "actual_cached": sum(actual), "actual_ram_refusals": actual_refusals,
              "variants": {}}
    for variant in ("today", "phase0", "hybrid", "dense"):
        totals, rows = simulate(requests, variant, args.model, ram_budget,
                                args.disk_budget, staging, args.sessions, coef)
        result = {k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()}
        if args.restart_at is None:
            result["within_64_of_actual"] = sum(
                abs(a - r["cached"]) <= 64 for a, r in zip(actual, rows))
        else:
            result["first_after_restart"] = rows[args.restart_at]
        report["variants"][variant] = result
    tag = f"-restart{args.restart_at}" if args.restart_at is not None else ""
    (base / f"e8{tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
