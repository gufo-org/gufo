#!/usr/bin/env python3
"""E5: replay E2 token traces through a model of the cache.

Two accounting policies share every other rule (capture points, lookup,
eviction ranks, disk spacing), so their difference isolates the effect of
storing shared KV once:

  full     today: every checkpoint costs fixed + per_token * position
  chunked  KV in CHUNK-token pieces keyed by their whole prefix, stored once;
           a checkpoint costs fixed + its unshared tail, plus any new chunks

The `full` policy is validated against what the server actually reused
(cached_tokens in the trace) before `chunked` is trusted.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from paths import WORK as HERE
IM_START = 248045
GRID = 2048
MIN_STEP = 2048
LEARN_MIN = 512


def lcp(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.nonzero(a[:n] != b[:n])[0]
    return int(diff[0]) if len(diff) else n


class Seq:
    """A token sequence with chunk keys computed once."""

    def __init__(self, tokens, chunk):
        self.tokens = tokens
        self.keys = []
        digest = hashlib.blake2b(digest_size=16)
        for end in range(chunk, len(tokens) + 1, chunk):
            digest.update(tokens[end - chunk:end].tobytes())
            self.keys.append(digest.copy().digest())


class Entry:
    def __init__(self, seq, length, purpose, clock):
        self.seq = seq
        self.length = length
        self.purpose = purpose  # grid, branch, stable, prompt, continuation
        self.used = clock

    def is_prefix_of(self, tokens):
        return lcp(self.seq.tokens[:self.length], tokens) >= self.length


class Store:
    """One tier: entries plus byte accounting under a budget."""

    def __init__(self, policy, budget, fixed, per_token, chunk, max_entries=None):
        self.policy = policy
        self.budget = budget
        self.fixed = fixed
        self.per_token = per_token
        self.chunk = chunk
        self.max_entries = max_entries
        self.entries = []
        self.chunk_refs = {}
        self.written = 0

    def full_bytes(self, length):
        return self.fixed + self.per_token * length

    def chunk_keys(self, entry):
        return entry.seq.keys[:entry.length // self.chunk]

    def entry_cost(self, entry, new_chunks):
        if self.policy == "full":
            return self.full_bytes(entry.length)
        tail = entry.length - (entry.length // self.chunk) * self.chunk
        return self.fixed + self.per_token * (tail + new_chunks * self.chunk)

    def used_bytes(self):
        if self.policy == "full":
            return sum(self.full_bytes(e.length) for e in self.entries)
        tails = sum(e.length - (e.length // self.chunk) * self.chunk
                    for e in self.entries)
        return (len(self.entries) * self.fixed
                + self.per_token * (tails + len(self.chunk_refs) * self.chunk))

    def lookup(self, tokens, limit=None):
        best = None
        for entry in self.entries:
            if limit is not None and entry.length > limit:
                continue
            if (best is None or entry.length > best.length) and entry.is_prefix_of(tokens):
                best = entry
        return best

    def rank(self, entry):
        """Approximates RemovalPriority: copies covered by a longer entry of the
        same sequence go first, branch points and last copies last."""
        if entry.purpose == "branch":
            return 3
        for other in self.entries:
            if other is not entry and other.length > entry.length and \
                    entry.is_prefix_of(other.seq.tokens[:other.length]):
                return 1 if entry.purpose == "grid" else 2
        return 3

    def remove(self, entry):
        self.entries.remove(entry)
        if self.policy == "chunked":
            for key in self.chunk_keys(entry):
                self.chunk_refs[key] -= 1
                if self.chunk_refs[key] == 0:
                    del self.chunk_refs[key]

    def admit(self, entry, ranked=True):
        for other in self.entries:
            if other.length == entry.length and other.is_prefix_of(entry.seq.tokens):
                other.used = entry.used
                return False
        new_chunks = sum(1 for key in self.chunk_keys(entry)
                         if key not in self.chunk_refs) if self.policy == "chunked" else 0
        cost = self.entry_cost(entry, new_chunks)
        if cost > self.budget:
            return False
        self.entries.append(entry)
        if self.policy == "chunked":
            for key in self.chunk_keys(entry):
                self.chunk_refs[key] = self.chunk_refs.get(key, 0) + 1
        while self.used_bytes() > self.budget or (
                self.max_entries and len(self.entries) > self.max_entries):
            candidates = [e for e in self.entries if e is not entry]
            if not candidates:
                self.remove(entry)
                return False
            key = (lambda e: (self.rank(e), e.used)) if ranked else (lambda e: e.used)
            self.remove(min(candidates, key=key))
        self.written += cost
        return True


def simulate(requests, policy, ram_budget, disk_budget, staging, fixed,
             per_token, chunk, sessions, disk_fix=False):
    ram = Store(policy, ram_budget, fixed, per_token, chunk, max_entries=128)
    disk = Store(policy, disk_budget, fixed, per_token, chunk) if disk_budget else None
    live = []  # (Seq, length) per session, most recent last
    clock = 0
    results = []
    lifetime = None
    for request in requests:
        clock += 1
        if request["server_lifetime"] != lifetime:
            lifetime = request["server_lifetime"]
            ram.entries.clear()
            ram.chunk_refs.clear()
            live.clear()
        prompt = request["prompt"]
        seq = Seq(prompt, chunk)
        last = np.nonzero(prompt == IM_START)[0]
        stable = int(last[-1]) if len(last) else len(prompt)
        # Live sessions: continuing the latest state needs no snapshot.
        cached, source = 0, "none"
        for state, length in live:
            if lcp(state.tokens[:length], prompt) >= length and length > cached:
                cached, source = length, "live"
        hit = ram.lookup(prompt)
        if hit and hit.length > cached:
            cached, source = hit.length, "memory"
            hit.used = clock
        # Today the disk is consulted only when RAM has no hit at all;
        # disk_fix also uses a disk entry longer than the RAM or live hit.
        if disk and (source == "none" or disk_fix):
            hit = disk.lookup(prompt)
            if hit and hit.length > cached:
                cached, source = hit.length, "disk"
                hit.used = clock
        cached = min(cached, len(prompt))
        # Captures: up to four grid points spread over the prefilled range,
        # a learned divergence point, the stable boundary and the prompt.
        captures = []
        grid = [g for g in range(GRID, len(prompt) - 128 + 1, GRID)
                if g >= cached + GRID]
        if len(grid) > 4:
            step = len(grid) / 4
            grid = [grid[min(len(grid) - 1, int(round((k + 1) * step)) - 1)]
                    for k in range(4)]
        captures += [(g, "grid") for g in grid]
        others = [e.seq.tokens[:e.length] for e in ram.entries] + \
                 [s.tokens[:n] for s, n in live]
        learned = max((lcp(prompt, o) for o in others), default=0)
        if learned >= cached + LEARN_MIN and learned < stable - 64:
            captures.append((learned, "branch"))
        captures.append((stable, "stable"))
        captures.append((len(prompt), "prompt"))
        for position, purpose in captures:
            if position > cached or purpose in ("stable", "prompt"):
                ram.admit(Entry(seq, position, purpose, clock))
        if disk:
            for position, purpose in captures:
                if purpose not in ("stable", "prompt", "branch"):
                    continue
                if purpose != "branch":
                    near = disk.lookup(prompt, limit=position)
                    if near and position - near.length < MIN_STEP:
                        continue
                entry = Entry(seq, position, purpose, clock)
                # Staging holds what one write must copy: the whole snapshot
                # today, only the fixed state and unshared KV when chunked.
                if policy == "chunked":
                    new_chunks = sum(1 for key in disk.chunk_keys(entry)
                                     if key not in disk.chunk_refs)
                    staged = disk.entry_cost(entry, new_chunks)
                else:
                    staged = disk.full_bytes(position)
                if staged > staging:
                    continue
                disk.admit(entry, ranked=False)
        full = np.concatenate([prompt, request["output"]])
        live.append((Seq(full, chunk), len(full)))
        live = live[-sessions:]
        results.append({"cached": cached, "source": source,
                        "ram_entries": len(ram.entries),
                        "ram_bytes": ram.used_bytes()})
    return results, {"ram_written": ram.written,
                     "disk_written": disk.written if disk else 0}


def load(model, workload):
    base = HERE / "results" / "e2" / model / workload
    meta = json.loads((base / "requests.json").read_text())
    data = np.load(base / "requests.npz")
    return [m | {"prompt": data[f"p{i}"], "output": data[f"o{i}"]}
            for i, m in enumerate(meta)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("workload")
    parser.add_argument("--fixed", type=float, required=True, help="bytes")
    parser.add_argument("--per-token", type=float, required=True, help="bytes")
    parser.add_argument("--ram-budget", type=int, required=True)
    parser.add_argument("--disk-budget", type=int, default=16 << 30)
    parser.add_argument("--staging", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--restart-at", type=int, default=None,
                        help="simulate a server restart before request N")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    requests = load(args.model, args.workload)
    if args.restart_at is not None:
        for index, request in enumerate(requests):
            request["server_lifetime"] = int(index >= args.restart_at)
    summary = {}
    variants = {"full": ("full", False), "full_diskfix": ("full", True),
                "chunked": ("chunked", False), "chunked_diskfix": ("chunked", True)}
    for name, (policy, disk_fix) in variants.items():
        rows, written = simulate(requests, policy, args.ram_budget,
                                 args.disk_budget, args.staging, args.fixed,
                                 args.per_token, args.chunk, args.sessions,
                                 disk_fix)
        summary[name] = {"rows": rows, **written}
    actual = [r["cached_tokens"] or 0 for r in requests]
    scales = [r["prompt_tokens"] / max(1, r["tokenized_prompt"]) for r in requests]
    for policy in variants:
        for row, scale in zip(summary[policy]["rows"], scales):
            row["cached"] = round(row["cached"] * scale)
    full = [r["cached"] for r in summary["full"]["rows"]]
    chunked = [r["cached"] for r in summary["chunked"]["rows"]]
    agree = sum(abs(a - b) <= 64 for a, b in zip(actual, full))
    prompt_total = sum(r["prompt_tokens"] for r in requests)
    report = {
        "requests": len(requests), "prompt_tokens": prompt_total,
        "actual_cached": sum(actual), "sim_full_cached": sum(full),
        "sim_chunked_cached": sum(chunked),
        "sim_full_diskfix_cached": sum(r["cached"] for r in summary["full_diskfix"]["rows"]),
        "sim_chunked_diskfix_cached": sum(r["cached"] for r in summary["chunked_diskfix"]["rows"]),
        "full_agrees_with_actual_within_64": agree,
        "ram_written_full": summary["full"]["ram_written"],
        "ram_written_chunked": summary["chunked"]["ram_written"],
        "disk_written_full": summary["full"]["disk_written"],
        "disk_written_chunked": summary["chunked"]["disk_written"],
        "per_request": [{"i": i, "prompt": r["prompt_tokens"], "actual": a,
                         "full": f["cached"], "full_src": f["source"],
                         "chunked": c["cached"], "chunked_src": c["source"]}
                        for i, (r, a, f, c) in enumerate(zip(
                            requests, actual, summary["full"]["rows"],
                            summary["chunked"]["rows"]))],
    }
    out = HERE / "results" / "e2" / args.model / args.workload / (f"e5-{args.tag}.json" if args.tag else "e5.json")
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_request"}))


if __name__ == "__main__":
    main()
