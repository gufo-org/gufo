#!/usr/bin/env python3
"""E8 (revision 2): today vs Phase 0 vs the hybrid, replayed as concurrent events.

Modelled after the production paths in src/cli/serve/:

Requests (text_model_runner.cpp)
- Each request starts when its generation started (the trace timestamp),
  captures its checkpoints when its prefill ends (received + time to first
  token) and finishes at received + duration, from the completion log line.
  Requests therefore overlap as they did on the server.
- Prompt-path captures, in order: on a cache hit, a frozen copy of the reused
  frontier (continuation); the stable boundary (continuation); the complete
  prompt as a retry copy (purpose retry). Grid points are history, learned
  divergence points are branch points.
- At the end of a request only the prompt-path snapshots are committed; the
  prompt + output state stays as the session's live frontier and is lost
  when another conversation takes that session.

RAM admission (continuation_cache.cpp)
- An incoming checkpoint may evict only entries whose RemovalPriority rank is
  at most MaxRemovalPriority of its purpose (retry 0, history 1, continuation
  and branch point 3); lowest rank first, oldest within a rank; the entry the
  request restored from is protected; otherwise the checkpoint is refused.

Disk (continuation_disk_store.cpp)
- One serial writer at the measured 0.44 GB/s. When the writer reaches a job
  it checks the 2,048-token spacing and staging, writes it, and only then
  publishes the entry and evicts to make room. In-flight writes are never
  restorable.
- A restart is graceful by default (the queue drains, as on SIGTERM);
  --abrupt drops queued and in-flight writes.

Chunks (hybrid variants)
- Chunk keys carry provenance: a request inherits its restore source's chunks
  for the rows it restored and gets new chunks (its own lineage) for every row
  it computed. Independent computations of equal tokens are not shared.

Phase 0 assumes bounded, streamed transfers in both directions; today both
writes and restores are limited by staging (ReadImage loads whole files).
"""
import argparse
import datetime
import json
import re

import numpy as np

from paths import WORK as HERE
from simulate_e5 import GRID, IM_START, LEARN_MIN, MIN_STEP, lcp, load
from simulate_e7 import CAPTURE_S_PER_BYTE, READ_BPS, SIZE, prefill_seconds

WRITE_BPS = 0.44e9
LIVE_TOLERANCE = 64
CEILING = {"retry": 0, "history": 1, "continuation": 3, "branch": 3}
CONTINUATION = {"continuation", "branch"}


class Seq:
    """Tokens plus provenance-aware chunk keys."""

    def __init__(self, tokens, chunk, lineage, inherited=()):
        self.tokens = tokens
        n = len(tokens) // chunk
        inherited = list(inherited[:n])
        self.keys = inherited + [(lineage, i) for i in range(len(inherited), n)]


class Entry:
    def __init__(self, seq, length, purpose, clock, stable=0):
        self.seq, self.length, self.purpose, self.used = seq, length, purpose, clock
        self.stable = stable

    def tokens(self):
        return self.seq.tokens[:self.length]

    def prefix_of(self, tokens):
        return lcp(self.tokens(), tokens) >= self.length


class Store:
    def __init__(self, policy, budget, fixed, per_token, chunk, max_entries=None):
        self.policy, self.budget = policy, budget
        self.fixed, self.per_token, self.chunk = fixed, per_token, chunk
        self.max_entries = max_entries
        self.entries, self.refs = [], {}
        self.refusals = 0

    def full_bytes(self, length):
        return self.fixed + self.per_token * length

    def keys(self, entry):
        return entry.seq.keys[:entry.length // self.chunk]

    def new_chunks(self, entry):
        return sum(1 for k in self.keys(entry) if k not in self.refs)

    def cost(self, entry):
        if self.policy == "full":
            return self.full_bytes(entry.length)
        tail = entry.length - (entry.length // self.chunk) * self.chunk
        return self.fixed + self.per_token * (tail + self.new_chunks(entry) * self.chunk)

    def used(self):
        if self.policy == "full":
            return sum(self.full_bytes(e.length) for e in self.entries)
        tails = sum(e.length - (e.length // self.chunk) * self.chunk for e in self.entries)
        return len(self.entries) * self.fixed + self.per_token * (
            tails + len(self.refs) * self.chunk)

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
        if entry.purpose in CONTINUATION:
            nexts = {int(p.seq.tokens[entry.length]) for p in peers
                     if p.length > entry.length and entry.prefix_of(p.tokens())}
            if len(nexts) > 1:
                return 3
            if entry.purpose == "branch" and not any(
                    p.purpose == "branch" and p.length > entry.length and
                    entry.prefix_of(p.tokens()) for p in peers):
                return 3
        if incoming is not None and entry.length < len(incoming) and entry.prefix_of(incoming):
            return {"retry": 0, "history": 1}.get(entry.purpose, 2)
        for p in peers:
            if entry.purpose == "retry" and p.length <= entry.stable and \
                    p.prefix_of(entry.tokens()):
                return 0
            if entry.purpose == "history" and (
                    entry.prefix_of(p.tokens()) or p.prefix_of(entry.tokens())):
                return 1
            if entry.purpose in CONTINUATION and p.length > entry.length and \
                    entry.prefix_of(p.tokens()):
                return 2
        return 3

    def _add(self, entry):
        self.entries.append(entry)
        if self.policy == "chunked":
            for key in self.keys(entry):
                self.refs[key] = self.refs.get(key, 0) + 1

    def remove(self, entry):
        self.entries.remove(entry)
        if self.policy == "chunked":
            for key in self.keys(entry):
                self.refs[key] -= 1
                if not self.refs[key]:
                    del self.refs[key]

    def admit(self, entry, ranked=True, protected=()):
        for other in self.entries:
            if other.length == entry.length and other.prefix_of(entry.seq.tokens):
                other.used = entry.used
                return 0
        cost = self.cost(entry)
        if cost > self.budget:
            self.refusals += 1
            return 0
        self._add(entry)
        ceiling = CEILING[entry.purpose] if ranked else 3
        incoming = entry.tokens() if entry.purpose in CONTINUATION else None
        while self.used() > self.budget or (
                self.max_entries and len(self.entries) > self.max_entries):
            candidates = []
            for e in self.entries:
                if e is entry or any(e is p for p in protected):
                    continue
                r = self.rank(e, incoming) if ranked else 0
                if r <= ceiling:
                    candidates.append((r, e.used, e))
            if not candidates:
                self.remove(entry)
                self.refusals += 1
                return 0
            self.remove(min(candidates, key=lambda c: (c[0], c[1]))[2])
        return cost


def stamp(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def timings(log):
    """request id -> (duration_s, queue_s, ttft_s) from completion lines."""
    out = {}
    for rid, rest in re.findall(r"request=(\S+) event=completed (.*)", log):
        fields = dict(f.split("=", 1) for f in rest.split() if "=" in f)
        if "duration_ms" in fields:
            out[rid] = (float(fields["duration_ms"]) / 1e3,
                        float(fields.get("queue_ms", 0)) / 1e3,
                        float(fields.get("ttft_ms", fields["duration_ms"])) / 1e3)
    return out


class Writer:
    """Serial disk writer: jobs become restorable only when their write ends."""

    def __init__(self, disk, stage_limit, totals):
        self.disk, self.stage_limit, self.totals = disk, stage_limit, totals
        self.queue, self.inflight, self.free = [], None, 0.0

    def enqueue(self, at, entry, prompt):
        self.queue.append((at, entry, prompt))

    def run(self, until):
        while True:
            if self.inflight and self.inflight[0] <= until:
                done, entry, staged = self.inflight
                if self.disk.admit(entry, ranked=False):
                    self.totals["disk_write_bytes"] += staged
                self.free, self.inflight = done, None
                continue
            if self.inflight is None and self.queue and \
                    max(self.free, self.queue[0][0]) <= until:
                at, entry, prompt = self.queue.pop(0)
                start = max(self.free, at)
                if entry.purpose != "branch":
                    near = self.disk.lookup(prompt, limit=entry.length)
                    if near and entry.length - near.length < MIN_STEP:
                        continue
                staged = self.disk.cost(entry)
                if self.stage_limit is not None and staged > self.stage_limit:
                    continue
                self.inflight = (start + staged / WRITE_BPS, entry, staged)
                continue
            return

    def stop(self, graceful):
        if graceful:
            self.run(float("inf"))
        self.queue, self.inflight = [], None


def simulate(requests, log, variant, model, ram_budget, disk_budget, staging,
             sessions, coef, abrupt=False, chunk=2048):
    fixed, per_token = SIZE[model]
    ram_policy = {"today": "full", "phase0": "chunked" if model == "fn" else "full",
                  "hybrid": "chunked", "dense": "chunked"}[variant]
    disk_policy = "full" if variant in ("today", "phase0") else "chunked"
    disk_fix = variant != "today"
    stage_limit = staging if variant == "today" else None
    dense = variant == "dense"
    totals = {"cached": 0, "prefill_s": 0.0, "capture_s": 0.0,
              "disk_write_bytes": 0, "disk_restore_s": 0.0, "disk_restores": 0}
    ram = Store(ram_policy, ram_budget, fixed, per_token, chunk, max_entries=128)
    disk = Store(disk_policy, disk_budget, fixed, per_token, chunk)
    writer = Writer(disk, stage_limit, totals)
    times = timings(log)
    events = []
    for i, r in enumerate(requests):
        # The trace stamps a generation record when generation starts, after
        # queueing; the completion line gives duration and time to first token
        # measured from when the request was received.
        start = stamp(r["time"])
        duration, queue, ttft = times.get(r["request"], (0.0, 0.0, 0.0))
        received = start - queue
        end = max(start, received + duration)
        capture = min(end, max(start, received + ttft))
        events += [(start, 0, i, "start"), (capture, 1, i, "capture"), (end, 2, i, "end")]
    events.sort()
    live = [None] * sessions      # (Seq, length) live frontier per session
    busy = [None] * sessions
    last_used = [0.0] * sessions
    ctx, rows, lifetime, clock = {}, [None] * len(requests), None, 0
    for t, _, i, kind in events:
        request = requests[i]
        if kind == "start" and request["server_lifetime"] != lifetime:
            if lifetime is not None:
                writer.stop(graceful=not abrupt)
            lifetime = request["server_lifetime"]
            ram.entries.clear()
            ram.refs.clear()
            live, busy = [None] * sessions, [None] * sessions
        writer.run(t)
        clock += 1
        prompt = request["prompt"]
        if kind == "start":
            cached, source, source_entry, source_keys, slot = 0, "none", None, (), None
            for s in range(sessions):
                state = live[s]
                if busy[s] is None and state:
                    # The output was re-tokenized from text separately, so it
                    # can differ by a few tokens from the next prompt's
                    # rendering; the server compares its own tokens exactly.
                    common = lcp(state[0].tokens[:state[1]], prompt)
                    if common >= state[1] - LIVE_TOLERANCE and common > cached:
                        cached, source, slot, source_keys = common, "live", s, state[0].keys
            hit = ram.lookup(prompt)
            if hit and hit.length > cached:
                cached, source, source_entry, source_keys = hit.length, "memory", hit, hit.seq.keys
                slot = None
                hit.used = clock
            if source == "none" or disk_fix:
                hit = disk.lookup(prompt)
                if hit and hit.length > cached and (
                        stage_limit is None or disk.full_bytes(hit.length) <= stage_limit):
                    cached, source, source_entry, source_keys = hit.length, "disk", hit, hit.seq.keys
                    slot = None
                    hit.used = clock
                    totals["disk_restore_s"] += disk.full_bytes(hit.length) * (
                        1.13 if disk_policy == "chunked" else 1.0) / READ_BPS
                    totals["disk_restores"] += 1
            cached = min(cached, len(prompt))
            if slot is None:
                free = [s for s in range(sessions) if busy[s] is None] or list(range(sessions))
                slot = min(free, key=lambda s: last_used[s])
                live[slot] = None   # another conversation's frontier is overwritten
            busy[slot], last_used[slot] = i, t
            seq = Seq(prompt, chunk, (lifetime, i), source_keys[:cached // chunk])
            ctx[i] = (seq, cached, source_entry, slot)
            scale = request["prompt_tokens"] / max(1, request["tokenized_prompt"])
            totals["prefill_s"] += prefill_seconds(
                coef, request["prompt_tokens"] - round(cached * scale), round(cached * scale))
            totals["cached"] += round(cached * scale)
            rows[i] = {"cached": round(cached * scale), "source": source}
        elif kind == "capture":
            seq, cached, source_entry, slot = ctx[i]
            starts = np.nonzero(prompt == IM_START)[0]
            stable = int(starts[-1]) if len(starts) else len(prompt)
            captures = []
            if cached and cached < stable:
                captures.append((cached, "continuation", False))   # frozen frontier
            grid = [g for g in range(GRID, len(prompt) - 128 + 1, GRID) if g >= cached + GRID]
            if len(grid) > 4:
                step = len(grid) / 4
                grid = [grid[min(len(grid) - 1, int(round((k + 1) * step)) - 1)] for k in range(4)]
            captures += [(g, "history", False) for g in grid]
            if dense:
                captures += [(int(s), "history", n == 1)
                             for n, s in enumerate(starts) if cached < s < stable]
            others = [e.tokens() for e in ram.entries] + [s[0].tokens[:s[1]] for s in live if s]
            learned = max((lcp(prompt, o) for o in others), default=0)
            if learned >= cached + LEARN_MIN and learned < stable - 64:
                captures.append((learned, "branch", True))
            if stable < len(prompt):
                captures.append((stable, "continuation", True))
            captures.append((len(prompt), "retry", True))
            for position, purpose, persist in captures:
                entry = Entry(seq, position, purpose, clock, stable=stable)
                if ram.admit(entry, protected=(source_entry,)) and CAPTURE_S_PER_BYTE[model]:
                    copied = ram.full_bytes(position) if ram_policy == "full" else fixed
                    totals["capture_s"] += copied * CAPTURE_S_PER_BYTE[model]
                if persist:
                    disk_purpose = "branch" if purpose in ("branch", "history") else purpose
                    writer.enqueue(t, Entry(seq, position, disk_purpose, clock, stable=stable),
                                   prompt)
        else:
            seq, cached, source_entry, slot = ctx.pop(i)
            full = np.concatenate([prompt, request["output"]])
            live[slot] = (Seq(full, chunk, (lifetime, i), seq.keys), len(full))
            busy[slot] = None
    writer.stop(graceful=True)
    totals["ram_refusals"] = ram.refusals
    totals["disk_write_s"] = totals["disk_write_bytes"] / WRITE_BPS
    return totals, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("workload")
    parser.add_argument("--restart-at", type=int, default=None)
    parser.add_argument("--abrupt", action="store_true",
                        help="the simulated restart drops queued and in-flight writes")
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
    actual_refusals = len(re.findall(r"event=snapshot action=skipped reason=byte_capacity", log))
    requests = load(args.model, args.workload)
    if args.restart_at is not None:
        for index, request in enumerate(requests):
            request["server_lifetime"] = int(index >= args.restart_at)
    actual = [r["cached_tokens"] or 0 for r in requests]
    report = {"model": args.model, "workload": args.workload, "revision": 2,
              "restart_at": args.restart_at, "abrupt": args.abrupt,
              "ram_budget": ram_budget, "staging": staging,
              "actual_prefill_s": round(actual_prefill, 2), "actual_cached": sum(actual),
              "actual_ram_refusals": actual_refusals, "variants": {}}
    for variant in ("today", "phase0", "hybrid", "dense"):
        totals, rows = simulate(requests, log, variant, args.model, ram_budget,
                                args.disk_budget, staging, args.sessions, coef, args.abrupt)
        result = {k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()}
        if args.restart_at is None:
            result["within_64_of_actual"] = sum(
                abs(a - r["cached"]) <= 64 for a, r in zip(actual, rows))
        else:
            result["first_after_restart"] = rows[args.restart_at]
        report["variants"][variant] = result
    tag = (f"-restart{args.restart_at}" if args.restart_at is not None else "") + \
          ("-abrupt" if args.abrupt else "")
    (base / f"e8{tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
