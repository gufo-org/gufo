#!/usr/bin/env python3
"""Eviction rules on an expert-store routing trace, on
prefetch_sim.py's cache model without prefetch: the store's score and
decay/bonus variants, LRU, per-expert reuse-interval prediction, and
Belady's offline optimum (evict the expert whose next use is farthest), with
and without its admission bypass. `current` and prefetch_sim.py's cache
model are the store's rule before frequency classes (decayed score); the store now evicts
by frequency class, `classlru:2048/1/16/0/4` (its seeded uses at half the
ranking weight are not modeled); traces record those uses where the score was.

  evict_sim.py TRACE                      # the default rule set
  evict_sim.py TRACE current,lru,belady   # named rules
  evict_sim.py TRACE 'grid:1/4;2/8'       # multiple of prefetch_sim.DECAY / recency bonus
  evict_sim.py TRACE 'reuse:reuse-b/4000/0.25'  # kind / first gap / gap rate
  evict_sim.py TRACE 'classlru:2048/1/16/0/4/0'  # llama.cpp #31233: half-life tokens /
      # prefill counts as use / clock rows per prepare / store-score prefill gate /
      # CRF per use min(count, n) / recent-rows bonus
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefetch_sim as ps

class Rule(ps.Cache):
    def __init__(self, snap, kind, prepares=None, **kw):
        pol = ps.CachePolicy(kind, **{k: v for k, v in kw.items() if k in ("decay", "recent_bonus", "count_cap")})
        super().__init__(snap, None, pol)
        self.kind = kind
        self.i = -1
        self.g0 = kw.get("g0", 4000.0)
        self.grate = kw.get("grate", 0.25)
        self.gap = np.full((snap.n_layer, snap.n_expert), np.nan)
        # ClassLRU (llama.cpp #31233): uses decayed by a per-layer token clock,
        # class floor(log2 CRF) set at each use, victim max age / 2^class.
        self.half = kw.get("half", 128.0)
        self.prefill_uses = kw.get("prefill_uses", True)
        self.crf = np.zeros((snap.n_layer, snap.n_expert))
        self.cls = np.zeros((snap.n_layer, snap.n_expert))
        self.luse = np.zeros((snap.n_layer, snap.n_expert))
        self.clock = np.zeros(snap.n_layer)
        if kw.get("seed_crf"):  # history from the store's seed score
            self.crf = snap.score.copy()
            ok = self.crf >= 1
            self.cls[ok] = np.minimum(7, np.floor(np.log2(self.crf[ok])))
        self.clock_cap = kw.get("clock_cap", 1 << 30)
        self.gate = kw.get("gate", False)  # prefill admission by the store's score
        self.cur_n = 0
        self.rbonus = kw.get("rbonus", 0.0)  # CRF added when routed in the last 16 rows
        self.inc_cap = kw.get("inc_cap", 1)  # CRF added per use: min(count, inc_cap)
        if kind in ("belady", "belady-admit"):
            L, E = snap.n_layer, snap.n_expert
            occ = {}
            for i, p in enumerate(prepares):
                for e in np.unique(p.ids[p.ids >= 0]):
                    occ.setdefault((p.layer, int(e)), []).append(i)
            self.occ = {k: np.array(v) for k, v in occ.items()}
            self.nxt = np.full((L, E), 1 << 40, np.int64)
            for (l, e), v in self.occ.items():
                self.nxt[l, e] = v[0]

    def value(self, sl, se):
        if self.kind == "lru":
            return self.last[sl, se].astype(float)
        if self.kind.startswith("reuse"):
            g = np.where(np.isnan(self.gap[sl, se]), self.g0, self.gap[sl, se])
            age = self.tick - self.last[sl, se]
            if self.kind == "reuse-a":
                nxt = self.last[sl, se] + g
            elif self.kind == "reuse-d":
                nxt = self.last[sl, se] + g + age
            else:  # reuse-b: overdue experts are due one gap from now
                nxt = np.where(age < g, self.last[sl, se] + g, self.tick + g)
            return -nxt.astype(float)
        if self.kind in ("belady", "belady-admit"):
            return -self.nxt[sl, se].astype(float)
        if self.kind == "classlru":
            return -(self.clock[sl] - self.luse[sl, se]) / 2.0 ** self.cls[sl, se]
        return self.score[sl, se] / (1.0 + (self.tick - self.last[sl, se]) / self.policy.decay)

    def prepare(self, layer, ids):
        self.i += 1
        if self.kind.startswith("reuse"):
            for e in np.unique(ids[ids >= 0]):
                e = int(e)
                if self.last[layer, e] > 0:
                    d = self.tick + 1 - self.last[layer, e]
                    g = self.gap[layer, e]
                    self.gap[layer, e] = d if np.isnan(g) else g + self.grate * (d - g)
        self.cur_n = ids.shape[0]
        if self.kind == "classlru":
            n = ids.shape[0]
            if n <= 16 or self.prefill_uses:
                es, cnt = np.unique(ids[ids >= 0], return_counts=True)
                c = (self.crf[layer, es] * np.exp2(-(self.clock[layer] - self.luse[layer, es]) / self.half)
                     + np.minimum(cnt, self.inc_cap))
                if self.rbonus:
                    tail = ids[max(0, n - 16):]
                    c = c + self.rbonus * np.isin(es, tail[tail >= 0])
                self.crf[layer, es] = c
                self.cls[layer, es] = np.minimum(7, np.floor(np.log2(c)))
                self.luse[layer, es] = self.clock[layer]
            self.clock[layer] += min(n, self.clock_cap)
        out = super().prepare(layer, ids)
        if self.kind in ("belady", "belady-admit"):
            for e in np.unique(ids[ids >= 0]):
                v = self.occ[(layer, int(e))]
                j = np.searchsorted(v, self.i, side="right")
                self.nxt[layer, int(e)] = v[j] if j < len(v) else 1 << 40
        return out

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
        if self.routed is not None and self.routed[0] == layer:
            live &= ~((self.s_layer == layer) & self.routed[1][np.maximum(self.s_expert, 0)])
        if not live.any():
            return None
        idx = np.nonzero(live)[0]
        vals = self.value(self.s_layer[idx], self.s_expert[idx])
        j = int(np.argmin(vals))
        if self.kind == "classlru" and self.gate and self.cur_n > 16:
            if self.decayed(layer, expert) < self.decayed(self.s_layer[idx[j]], self.s_expert[idx[j]]):
                return None
        elif self.kind != "belady-admit" and self.value(np.array([layer]), np.array([expert]))[0] < vals[j]:
            return None
        victim = int(idx[j])
        self.slot_of[self.s_layer[victim], self.s_expert[victim]] = -1
        return victim

def evaluate(snap, prepares, cache, a, b):
    size = snap.expert_bytes
    miss = routes = 0; dbytes = pbytes = 0; dstall = pstall = 0.0
    for p in prepares:
        missing, admit, hits = cache.prepare(p.layer, p.ids)
        mask = p.ids >= 0
        by = len(missing) * size[p.layer]
        overlap = (p.t_end - p.t_routes) - p.read_wait
        st = max(0.0, (a + b * by if by else 0.0) - overlap)
        if p.n <= 16:
            miss += int((~hits & mask).sum()); routes += int(mask.sum()); dbytes += by; dstall += st
        else:
            pbytes += by; pstall += st
        cache.admit(p.layer, missing, admit)
    return miss / routes, dbytes, pbytes, dstall, pstall

def main():
    snap, prepares = ps.read_trace(Path(sys.argv[1]))
    a, b = ps.fit_read_cost(prepares)
    D = ps.DECAY
    rules = [("current", {}), ("decay /2", dict(decay=D / 2)), ("decay /4", dict(decay=D / 4)),
             ("decay /8", dict(decay=D / 8)), ("bonus 8", dict(recent_bonus=8.0)),
             ("bonus 16", dict(recent_bonus=16.0)), ("decay /2 bonus 8", dict(decay=D / 2, recent_bonus=8.0)),
             ("decay /4 bonus 8", dict(decay=D / 4, recent_bonus=8.0)), ("lru", {}), ("belady", {}), ("belady-admit", {})]
    if len(sys.argv) > 2 and sys.argv[2].startswith("classlru:"):
        rules = [("current", {})]
        for spec in sys.argv[2][9:].split(";"):
            f = spec.split("/")
            h, pf, cap, gate, inc, rb, sd = f + ["1073741824", "0", "1", "0", "0"][len(f) - 2:]
            rules.append((f"classlru {h} {'pf' if pf == '1' else 'nopf'} cap {cap}{' gate' if gate == '1' else ''} inc {inc} rb {rb}{' seed' if sd == '1' else ''}",
                          dict(kind="classlru", half=float(h), prefill_uses=pf == "1", clock_cap=int(cap),
                               gate=gate == "1", inc_cap=int(inc), rbonus=float(rb), seed_crf=sd == "1")))
    elif len(sys.argv) > 2 and sys.argv[2].startswith("reuse:"):
        rules = [("current", {})]
        for spec in sys.argv[2][6:].split(";"):
            k, g0, gr = spec.split("/")
            rules.append((f"{k} g0 {g0} rate {gr}", dict(kind=k, g0=float(g0), grate=float(gr))))
    elif len(sys.argv) > 2 and sys.argv[2].startswith("grid:"):
        rules = []
        for spec in sys.argv[2][5:].split(";"):
            d, bo = (float(x) for x in spec.split("/"))
            rules.append((f"decay x{d:g} bonus {bo:g}", dict(decay=D * d, recent_bonus=bo)))
    elif len(sys.argv) > 2:
        rules = [r for r in rules if r[0] in sys.argv[2].split(",")]
    ref = None
    for name, kw in rules:
        kind = kw.pop("kind", None) or (name if name in ("lru", "belady", "belady-admit") else "score")
        c = Rule(snap, kind, prepares, **kw)
        m, db, pb, ds, pst = evaluate(snap, prepares, c, a, b)
        ref = ref or (m, ds, pst)
        print(f"{name:20s} decode misses {100*m:5.2f}% ({100*(m/ref[0]-1):+6.1f}%)  reads decode {db/2**30:6.1f} GiB "
              f"prefill {pb/2**30:6.1f} GiB  stall decode {ds/1e9:6.2f} s ({(ds-ref[1])/1e9:+5.2f}) "
              f"prefill {pst/1e9:6.2f} s ({(pst-ref[2])/1e9:+5.2f})", flush=True)

main()
