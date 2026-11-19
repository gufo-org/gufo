#!/usr/bin/env python3
"""Replay an expert-store routing trace to compare prefetch policies offline.

The store (expert_stream::Policy::trace_path) writes the seeded cache, then
every prepare (routes, per-route hits, timing); callers may add router
predictions for later layers (Store::TracePrediction; the deepseek41
extension records the top-12 for the next three layers from each layer's FFN
input). This tool mirrors the store's cache policy before frequency classes
(decayed scores, size classes, scan-resistant admission, lowest-score
victim), checks how far the replay reproduces the recorded hits, and
estimates each prefetch policy's read stalls.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_trace import NO, Prepare, Snapshot, read_trace  # noqa: E402,F401

RECENT_ROWS = 16
DECAY = 1600.0


@dataclass
class CachePolicy:
    """The store's rule by default; the fields vary it for comparison."""
    name: str = "current"
    decay: float = DECAY        # ticks (prepares) to halve a score
    count_cap: int = 4          # most routes counted per call
    recent_bonus: float = 4.0   # routed in the batch's last 16 rows
    admit_div: int = 32         # prefill admits counts >= max(admit_min, n / admit_div)
    admit_min: int = 4
    admit_all: bool = False     # prefill admits every expert it read
    floor: float = 0.0          # weight of the seeded (popularity) score kept as a floor


class Cache:
    """The expert store's slot cache (runtime/expert_store.cpp), host-side."""

    def __init__(self, snap: Snapshot, limit=None, policy: CachePolicy = CachePolicy()):
        self.policy = policy
        self.routed = None
        self.base = snap.score.copy() * policy.floor
        self.snap = snap
        self.tick = snap.tick
        self.score = snap.score.copy()
        self.last = snap.last.copy()
        self.limit = snap.cache_limit if limit is None else limit
        self.cache_bytes = snap.cache_bytes
        self.class_bytes = snap.class_bytes
        self.layer_class = snap.layer_class
        n = len(snap.slots)
        self.s_class = np.array([s[0] for s in snap.slots], np.int64)
        self.s_layer = np.array([s[1] if s[1] != NO else -1 for s in snap.slots], np.int64)
        self.s_expert = np.array([s[2] if s[2] != NO else -1 for s in snap.slots], np.int64)
        self.slot_of = -np.ones((snap.n_layer, snap.n_expert), np.int64)
        self.free = [[] for _ in snap.class_bytes]
        for i in range(n):
            if self.s_layer[i] >= 0:
                self.slot_of[self.s_layer[i], self.s_expert[i]] = i
            else:
                self.free[self.s_class[i]].append(i)
        self.grow_failure = [None] * len(snap.class_bytes)

    def decayed(self, layer, expert):
        return (self.score[layer, expert] / (1.0 + (self.tick - self.last[layer, expert]) / self.policy.decay)
                + self.base[layer, expert])

    def grow(self, c):
        if self.cache_bytes >= self.limit:
            return False
        slot_bytes = self.class_bytes[c]
        count = min(self.limit - self.cache_bytes, self.snap.chunk_bytes) // slot_bytes
        if not count:
            return False
        self.cache_bytes += count * slot_bytes
        start = len(self.s_class)
        self.s_class = np.concatenate([self.s_class, np.full(count, c)])
        self.s_layer = np.concatenate([self.s_layer, -np.ones(count, np.int64)])
        self.s_expert = np.concatenate([self.s_expert, -np.ones(count, np.int64)])
        self.free[c].extend(range(start, start + count))
        return True

    def acquire(self, layer, expert, allow_evict):
        own = self.layer_class[layer]
        if not self.free[own] and self.grow_failure[own] != self.tick and not self.grow(own):
            self.grow_failure[own] = self.tick
        for c in range(own, len(self.free)):
            if self.free[c]:
                return self.free[c].pop()
        if not allow_evict:
            return None
        live = (self.s_layer >= 0) & (self.s_class >= own)
        # The batch's own hits are read from their slots: never victims.
        if self.routed is not None and self.routed[0] == layer:
            live &= ~((self.s_layer == layer) & self.routed[1][np.maximum(self.s_expert, 0)])
        if not live.any():
            return None
        idx = np.nonzero(live)[0]
        sl, se = self.s_layer[idx], self.s_expert[idx]
        scores = (self.score[sl, se] / (1.0 + (self.tick - self.last[sl, se]) / self.policy.decay)
                  + self.base[sl, se])
        j = int(np.argmin(scores))
        if self.decayed(layer, expert) < scores[j]:
            return None
        victim = int(idx[j])
        self.slot_of[self.s_layer[victim], self.s_expert[victim]] = -1
        return victim

    def place(self, slot, layer, expert):
        self.s_layer[slot], self.s_expert[slot] = layer, expert
        self.slot_of[layer, expert] = slot

    def force(self, layer, expert):
        """Prefetch landing in a slot (mako): a free slot, else the lowest score's."""
        if self.slot_of[layer, expert] >= 0:
            return
        own = self.layer_class[layer]
        if not self.free[own] and self.grow_failure[own] != self.tick and not self.grow(own):
            self.grow_failure[own] = self.tick
        for c in range(own, len(self.free)):
            if self.free[c]:
                self.place(self.free[c].pop(), layer, expert)
                return
        live = (self.s_layer >= 0) & (self.s_class >= own)
        idx = np.nonzero(live)[0]
        if not len(idx):
            return
        sl, se = self.s_layer[idx], self.s_expert[idx]
        scores = (self.score[sl, se] / (1.0 + (self.tick - self.last[sl, se]) / self.policy.decay)
                  + self.base[sl, se])
        victim = int(idx[int(np.argmin(scores))])
        self.slot_of[self.s_layer[victim], self.s_expert[victim]] = -1
        self.place(victim, layer, expert)

    def resident(self, layer, expert):
        return self.slot_of[layer, expert] >= 0

    def prepare(self, layer, ids):
        """One prepare: scores and admission as the store does; returns the
        missing experts (ascending) and the hit mask per route."""
        n = ids.shape[0]
        flat = ids.reshape(-1)
        valid = flat >= 0
        counts = np.bincount(flat[valid], minlength=self.snap.n_expert)
        self.routed = (layer, (counts > 0) & (self.slot_of[layer] >= 0))
        recent_rows = ids[n - min(n, RECENT_ROWS):].reshape(-1)
        recent = np.bincount(recent_rows[recent_rows >= 0], minlength=self.snap.n_expert)
        self.tick += 1
        hits = np.zeros(flat.shape, bool)
        hits[valid] = self.slot_of[layer, flat[valid]] >= 0
        missing, admit = [], {}
        for e in np.nonzero(counts)[0]:
            e = int(e)
            pol = self.policy
            self.score[layer, e] = (self.decayed(layer, e) - self.base[layer, e]
                                    + min(int(counts[e]), pol.count_cap)
                                    + (pol.recent_bonus if recent[e] else 0.0))
            self.last[layer, e] = self.tick
            if self.slot_of[layer, e] < 0:
                missing.append(e)
                admit[e] = (pol.admit_all or n <= RECENT_ROWS or recent[e] > 0
                            or counts[e] >= max(pol.admit_min, n // pol.admit_div))
        return missing, admit, hits.reshape(ids.shape)

    def admit(self, layer, missing, admit):
        for e in missing:
            slot = self.acquire(layer, e, admit[e])
            if slot is not None:
                self.place(slot, layer, e)


def validate(snap, prepares, limit=None):
    cache = Cache(snap, limit)
    agree = total = 0
    first_bad = None
    for i, p in enumerate(prepares):
        missing, admit, hits = cache.prepare(p.layer, p.ids)
        mask = p.ids >= 0
        same = (hits == p.hits) | ~mask
        agree += int(same[mask].sum())
        total += int(mask.sum())
        if first_bad is None and not same.all():
            first_bad = i
        cache.admit(p.layer, missing, admit)
    return agree, total, first_bad


def summary(snap, prepares):
    decode = [p for p in prepares if p.n <= 16]
    prefill = [p for p in prepares if p.n > 16]
    routes = sum(int((p.ids >= 0).sum()) for p in prepares)
    hits = sum(int(p.hits.sum()) for p in prepares)
    tokens = sum(p.n for p in prepares if p.layer == 0)
    demand = sum(p.demand for p in prepares)
    wait = sum(p.read_wait for p in prepares)
    print(f"{len(prepares)} prepares ({len(decode)} decode-sized, {len(prefill)} prefill), {tokens} tokens; "
          f"{100 * hits / max(routes, 1):.2f}% route hits; {demand / max(tokens, 1) / 2**20:.2f} MiB read/token; "
          f"read wait {wait / max(tokens, 1) / 1e6:.2f} ms/token")
    with_pred = [p for p in decode if p.pred]
    print(f"{len(with_pred)} decode-sized prepares with predictions")


def recall(prepares):
    """Share of decode misses each lead's top-k prediction contains."""
    out = {}
    for lead in (1, 2, 3):
        for k in (4, 6, 8, 12):
            caught = misses = 0
            for p in prepares:
                pred = p.pred.get(lead)
                if pred is None:
                    continue
                predicted = set(int(x) for x in pred[:, :k].reshape(-1))
                for e, hit in zip(p.ids.reshape(-1), p.hits.reshape(-1)):
                    if e >= 0 and not hit:
                        misses += 1
                        caught += int(e) in predicted
            out[lead, k] = (caught, misses)
    return out


def fit_read_cost(prepares):
    """Read time = a + b * bytes, fitted on decode-sized prepares that waited
    on their reads (their read phase ended the prepare)."""
    xs, ys = [], []
    for p in prepares:
        if p.n <= 16 and p.read_wait > 0 and p.demand > 0:
            xs.append(p.demand)
            ys.append(p.t_end - p.t_routes)
    if len(xs) < 10:
        raise SystemExit("too few read-bound prepares to fit the read cost")
    b, a = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
    return max(a, 0.0), b


@dataclass
class Policy:
    name: str
    lead: int = 0          # 0: no prefetch
    k: int = 6             # top-k per row
    only_after_miss: bool = False
    slots: bool = False    # land in cache slots (else a stage adopted only by its target prepare)
    gate: bool = False     # mako's usefulness gate
    max_experts: int = 96  # cap per prepare, best ranked first (rank sum across rows)
    single_row: bool = False  # predict only for one-row batches
    switch_on: float = 0.0    # > 0: predict every layer while the recent misses per decode
                              # prepare exceed this, else only after a stalled layer
    switch_off: float = 0.0   # hysteresis: back to after-a-stall below this
    per_layer: bool = False   # the switch tracks each layer's own miss rate


class Drive:
    """One read queue: demand preempts queued prefetch, not the task in flight."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.busy_until = 0.0
        self.queue = []      # [layer, expert, bytes, eval]
        self.done = {}       # (eval, layer, expert) -> completion time
        self.issued_bytes = 0

    def advance(self, now):
        """Runs queued prefetch tasks that can start before `now`."""
        while self.queue and self.busy_until < now:
            layer, expert, size, ev = self.queue.pop(0)
            start = self.busy_until
            self.busy_until = start + self.b * size
            self.done[ev, layer, expert] = self.busy_until

    def demand(self, now, sizes):
        self.advance(now)
        start = max(now, self.busy_until)
        if not sizes:
            return now
        end = start + self.a + self.b * sum(sizes)
        self.busy_until = end
        return end


def simulate(snap, prepares, policy, a, b, pred_cost=0.0, cache_policy=CachePolicy(), limit=None,
             byte_cost=0.0):
    """byte_cost: critical-path ns per prefetched byte (staging copies and DMA
    competing with the decode kernels for memory bandwidth, worker contention)."""
    """Returns (seconds, demand bytes, prefetch bytes, useful prefetch bytes) over decode-sized prepares."""
    cache = Cache(snap, limit, cache_policy)
    drive = Drive(a, b)
    size = snap.expert_bytes
    t = 0.0
    prev_end = None
    ev = 0
    total = demand_total = pf_total = pf_useful = 0
    staged = {}            # (eval, layer) -> set of experts prefetched for that prepare
    recent_pf = []         # usefulness history for the gate: 1 used, 0 wasted
    probe = 0
    rate = np.zeros(snap.n_layer if policy.per_layer else 1)   # recent misses per decode prepare
    on = np.zeros(len(rate), bool)
    alpha = 1 / 64 if policy.per_layer else 1 / 2560          # ~64 evaluations either way
    switched = []
    for i, p in enumerate(prepares):
        if p.layer == 0:
            ev += 1
        gap = 0 if prev_end is None else max(0, p.t_routes - prev_end)
        # The trace run computed the next two layers' predictions before this
        # prepare; remove their cost, then charge the policy's own.
        traced = sum(1 for lead in (1, 2) for q in prepares[i + 1:i + 1 + lead]
                     if q.layer == p.layer + lead and lead in q.pred)
        gap = max(0.0, gap - traced * pred_cost)
        t_routes = t + gap
        base = p.t_end - p.t_routes
        overlap = base - p.read_wait
        missing, admit, _ = cache.prepare(p.layer, p.ids)
        decode = p.n <= 16
        need = []
        wait_until = t_routes
        if decode and policy.lead:
            got = staged.pop((ev, p.layer), set())
            for e in missing:
                key = (ev, p.layer, e)
                if e in got:
                    drive.advance(t_routes)
                    if key in drive.done:
                        wait_until = max(wait_until, drive.done.pop(key))
                    else:   # still queued: becomes a demand read
                        drive.queue = [q for q in drive.queue if (q[3], q[0], q[1]) != key]
                        need.append(size[p.layer])
                    pf_useful += size[p.layer]
                else:
                    need.append(size[p.layer])
            used = set(missing) & got
            for e in got:
                recent_pf.append(1 if e in used else 0)
            if policy.slots:
                # Completed reads that this prepare did not route stay cached.
                drive.advance(t_routes)
                for e in got - used:
                    if drive.done.pop((ev, p.layer, e), None) is not None:
                        cache.force(p.layer, e)
            recent_pf = recent_pf[-100:]
            # queued, never adopted: dropped (stage landing)
            drive.queue = [q for q in drive.queue if not (q[3] == ev and q[0] == p.layer)]
        else:
            need = [size[p.layer]] * len(missing)
        done = drive.demand(t_routes, need)
        end_reads = max(done, wait_until)
        stall = max(0.0, end_reads - t_routes - overlap)
        if not decode:
            stall = p.read_wait     # prefill is not simulated
        cache.admit(p.layer, missing, admit)
        demand_total += sum(need) if decode else p.demand
        end = t_routes + overlap + stall
        if decode:
            total += end - t_routes + gap
        # Predictions for later layers, issued at the end of this prepare.
        if decode and policy.lead:
            target = p.layer + policy.lead
            pred = None
            nxt = next((q for q in prepares[i + 1:i + 1 + policy.lead]
                        if q.layer == target), None)
            if nxt is not None:
                pred = nxt.pred.get(policy.lead)
            allowed = (not policy.only_after_miss or bool(missing)) and (not policy.single_row or p.n == 1)
            if policy.switch_on and target < snap.n_layer:
                j = target if policy.per_layer else 0
                if on[j] and rate[j] < policy.switch_off:
                    on[j] = False
                elif not on[j] and rate[j] > policy.switch_on:
                    on[j] = True
                allowed = on[j] or bool(missing)
                switched.append(on[j])
            if policy.gate and len(recent_pf) >= 100 and sum(recent_pf) < 0.35 * len(recent_pf):
                probe += 1
                allowed = allowed and probe % 16 == 0
            if pred is not None and allowed:
                end += pred_cost
                total += pred_cost
                weight = {}
                for row in pred[:, :policy.k]:
                    for rank, e in enumerate(row):
                        e = int(e)
                        if e >= 0 and not cache.resident(target, e):
                            weight[e] = weight.get(e, 0) + policy.k - rank
                want = sorted(weight, key=lambda e: -weight[e])[:policy.max_experts]
                drive.advance(end)
                for e in want:
                    drive.queue.append([target, e, size[target], ev])
                    pf_total += size[target]
                end += byte_cost * size[target] * len(want)
                total += byte_cost * size[target] * len(want)
                staged[ev, target] = set(want)
        if decode:
            j = p.layer if policy.per_layer else 0
            rate[j] += alpha * (len(missing) - rate[j])
        t = end
        prev_end = p.t_end
    return total / 1e9, demand_total, pf_total, pf_useful


def cache_eval(snap, prepares, policy, a, b, limit=None):
    """Misses and estimated read stalls under a cache policy, without prefetch."""
    cache = Cache(snap, limit, policy)
    size = snap.expert_bytes
    out = dict(dec_miss=0, dec_routes=0, dec_bytes=0, pre_bytes=0, dec_stall=0.0, pre_stall=0.0,
               dec_base=0.0, pre_base=0.0)
    for p in prepares:
        missing, admit, hits = cache.prepare(p.layer, p.ids)
        mask = p.ids >= 0
        bytes_ = len(missing) * size[p.layer]
        overlap = (p.t_end - p.t_routes) - p.read_wait
        stall = max(0.0, (a + b * bytes_ if bytes_ else 0.0) - overlap)
        if p.n <= 16:
            out["dec_miss"] += int((~hits & mask).sum())
            out["dec_routes"] += int(mask.sum())
            out["dec_bytes"] += bytes_
            out["dec_stall"] += stall
            out["dec_base"] += p.read_wait
        else:
            out["pre_bytes"] += bytes_
            out["pre_stall"] += stall
            out["pre_base"] += p.read_wait
        cache.admit(p.layer, missing, admit)
    return out


CACHE_POLICIES = [
    CachePolicy(),
    CachePolicy("decay x2", decay=DECAY * 2),
    CachePolicy("decay x4", decay=DECAY * 4),
    CachePolicy("decay x10", decay=DECAY * 10),
    CachePolicy("decay /2", decay=DECAY / 2),
    CachePolicy("no recency bonus", recent_bonus=0.0),
    CachePolicy("count cap 16", count_cap=16),
    CachePolicy("popularity floor 0.25", floor=0.25),
    CachePolicy("popularity floor 1", floor=1.0),
    CachePolicy("prompt admits n/64", admit_div=64),
    CachePolicy("prompt admits n/128", admit_div=128),
    CachePolicy("prompt admits >=2", admit_div=10**9, admit_min=2),
    CachePolicy("prompt admits all", admit_all=True),
]


PREFETCH_POLICIES = [
    Policy("no prefetch"),
    Policy("production", lead=1, k=6, only_after_miss=True),
    Policy("lead 1 top-6", lead=1, k=6),
    Policy("lead 1 top-6 cap 2", lead=1, k=6, max_experts=2),
    Policy("lead 1 top-6 cap 4", lead=1, k=6, max_experts=4),
    Policy("lead 1 top-6 cap 6", lead=1, k=6, max_experts=6),
    Policy("lead 1 top-6 cap 8", lead=1, k=6, max_experts=8),
    Policy("lead 1 top-8 cap 6", lead=1, k=8, max_experts=6),
    Policy("lead 1 top-6 single row", lead=1, k=6, single_row=True),
    Policy("lead 1 top-6 cap 4 after miss", lead=1, k=6, max_experts=4, only_after_miss=True),
    Policy("lead 2 top-6 cap 4", lead=2, k=6, max_experts=4),
    *[Policy(f"lead 1 top-6 cap 4 switch {on}", lead=1, k=6, max_experts=4, switch_on=on, switch_off=on * 0.7)
      for on in (0.05, 0.1, 0.2, 0.3)],
    *[Policy(f"lead 1 top-6 cap 4 per-layer switch {on}", lead=1, k=6, max_experts=4, switch_on=on,
             switch_off=on * 0.7, per_layer=True) for on in (0.05, 0.1, 0.2, 0.3)],
]

CONFIDENT_POLICIES = [
    Policy("no prefetch"),
    Policy("current", lead=1, k=6, only_after_miss=True),
    Policy("current cap 4", lead=1, k=6, only_after_miss=True, max_experts=4),
    Policy("after miss top-2 cap 2", lead=1, k=2, only_after_miss=True, max_experts=2),
    Policy("after miss top-1 cap 1", lead=1, k=1, only_after_miss=True, max_experts=1),
    Policy("always top-1 cap 1", lead=1, k=1, max_experts=1),
    Policy("always top-2 cap 2", lead=1, k=2, max_experts=2),
    Policy("switch 0.3 top-2 cap 2", lead=1, k=2, max_experts=2, switch_on=0.3, switch_off=0.2, per_layer=True),
    Policy("switch 0.5 top-6 cap 4", lead=1, k=6, max_experts=4, switch_on=0.5, switch_off=0.35, per_layer=True),
]

POLICIES = [
    Policy("no prefetch"),
    Policy("production: lead 1 top-6 after a miss", lead=1, k=6, only_after_miss=True),
    Policy("lead 1 top-4", lead=1, k=4),
    Policy("lead 1 top-6", lead=1, k=6),
    Policy("lead 1 top-8", lead=1, k=8),
    Policy("lead 1 top-12", lead=1, k=12),
    Policy("lead 2 top-4", lead=2, k=4),
    Policy("lead 2 top-6", lead=2, k=6),
    Policy("lead 2 top-8", lead=2, k=8),
    Policy("lead 2 top-6 gated", lead=2, k=6, gate=True),
    Policy("lead 1 top-6 gated", lead=1, k=6, gate=True),
    Policy("lead 1 top-6 into slots", lead=1, k=6, slots=True),
    Policy("lead 2 top-6 into slots", lead=2, k=6, slots=True),
    Policy("lead 2 top-6 into slots gated", lead=2, k=6, slots=True, gate=True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--pred-cost-us", type=float, default=0.0,
                        help="critical-path cost of one router prediction")
    parser.add_argument("--limit-gib", type=float,
                        help="cache size the run reached, when free memory stopped it below the cap")
    parser.add_argument("--read-cost", help="override the fitted read cost: 'ms,ms_per_MiB'")
    parser.add_argument("--cache", action="store_true", help="compare cache policies instead of prefetch")
    parser.add_argument("--byte-cost-ms-mib", type=float, default=0.0,
                        help="critical-path cost per prefetched MiB")
    parser.add_argument("--confident", action="store_true", help="compare high-confidence prefetch rules")
    parser.add_argument("--capped", action="store_true", help="compare capped prefetch under two cache policies")
    args = parser.parse_args()
    limit = int(args.limit_gib * 2**30) if args.limit_gib else None
    snap, prepares = read_trace(args.trace)
    summary(snap, prepares)
    agree, total, first_bad = validate(snap, prepares, limit)
    print(f"cache replay: {100 * agree / max(total, 1):.3f}% of route hits reproduced"
          + (f" (first difference at prepare {first_bad})" if first_bad is not None else ""))
    for (lead, k), (caught, misses) in recall(prepares).items():
        print(f"lead {lead} top-{k:2d}: {100 * caught / max(misses, 1):5.1f}% of {misses} decode misses predicted")
    a, b = fit_read_cost(prepares)
    if args.read_cost:
        a, per_mib = (float(x) for x in args.read_cost.split(","))
        a, b = a * 1e6, per_mib * 1e6 / 2**20
    print(f"read cost: {a / 1e6:.3f} ms + {b * 2**20 / 1e6:.4f} ms/MiB ({1 / b:.2f} GB/s)")
    if args.cache:
        ref = None
        for policy in CACHE_POLICIES:
            r = cache_eval(snap, prepares, policy, a, b, limit)
            ref = ref or r
            print(f"{policy.name:24s} decode misses {100 * r['dec_miss'] / max(r['dec_routes'], 1):5.2f}% "
                  f"({100 * (r['dec_miss'] / max(ref['dec_miss'], 1) - 1):+5.1f}%), reads decode "
                  f"{r['dec_bytes'] / 2**30:6.1f} GiB prefill {r['pre_bytes'] / 2**30:6.1f} GiB; stall decode "
                  f"{r['dec_stall'] / 1e9:5.2f} s ({(r['dec_stall'] - ref['dec_stall']) / 1e9:+5.2f}) prefill "
                  f"{r['pre_stall'] / 1e9:5.2f} s ({(r['pre_stall'] - ref['pre_stall']) / 1e9:+5.2f})")
        return 0
    if args.confident:
        base = None
        for policy in CONFIDENT_POLICIES:
            seconds, demand, pf, useful = simulate(snap, prepares, policy, a, b, args.pred_cost_us * 1e3,
                                                   CachePolicy(), limit, args.byte_cost_ms_mib * 1e6 / 2**20)
            base = base or seconds
            print(f"{policy.name:28s} decode {seconds:7.2f} s ({100 * (base / seconds - 1):+5.1f}%)  "
                  f"prefetch {pf / 2**30:6.1f} GiB ({100 * useful / max(pf, 1):4.1f}% used)")
        return 0
    if args.capped:
        for cpol in (CachePolicy(), CachePolicy("prompt admits >=2", admit_div=10**9, admit_min=2)):
            base = None
            for policy in PREFETCH_POLICIES:
                seconds, demand, pf, useful = simulate(snap, prepares, policy, a, b, args.pred_cost_us * 1e3,
                                                       cpol, limit)
                base = base or seconds
                print(f"[{cpol.name}] {policy.name:32s} decode {seconds:7.2f} s ({100 * (base / seconds - 1):+5.1f}%)"
                      f"  demand {demand / 2**30:6.1f} GiB  prefetch {pf / 2**30:6.1f} GiB "
                      f"({100 * useful / max(pf, 1):4.1f}% used)")
        return 0
    base = None
    for policy in POLICIES:
        seconds, demand, pf, useful = simulate(snap, prepares, policy, a, b, args.pred_cost_us * 1e3)
        base = base or seconds
        print(f"{policy.name:40s} decode {seconds:7.2f} s ({100 * (base / seconds - 1):+5.1f}%)  "
              f"demand {demand / 2**30:6.2f} GiB  prefetch {pf / 2**30:6.2f} GiB "
              f"({100 * useful / max(pf, 1):4.1f}% used)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
