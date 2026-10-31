#!/usr/bin/env python3
"""Gemma 4 sampled MTP proposals replayed: expected tokens per ms under q variants.

Reads `--draft-trace` records of sampled cycles run with `--draft-policy fixed
--draft-tokens 15` (every position verified), which carry each drafter
proposal's top-64 candidates, the proposal q and the target distribution p of
its row. For each way of building q from the drafter's candidates (the logged
q, no top-p, top-p 0.99, other temperatures) it computes per position the
chain acceptance sum(min(p, q)) and the probability that a sibling drawn from
q without the draft is accepted once the draft is rejected (recursive
rejection sampling, as `VerifyWithSibling`), then scores the best fixed
(drafts, siblings) choice and a per-cycle oracle at draft_policy.cpp's prices
(`--experts` for the 26B-A4B; `--scale` multiplies the depth-0 verification
rows, 1.75 for 31B UD-Q8_K_XL). Assumes each position's p and drafter
candidates do not depend on which earlier drafts were drawn.

    python3 tools/gemma4/proposal_replay.py --scale 1.75 TRACE.jsonl
"""
import json, math, sys, argparse

DEPTHS = [0, 4096, 16384, 32768, 65536]
STEP = [1.93, 2.39, 2.68, 3.13, 4.04]
VERIFY = [
    [99.5, 99.8, 100.6, 101.4, 102.0, 102.8, 103.3, 103.9, 109.0, 109.6, 110.1, 111.1, 111.7, 112.3, 112.7, 113.1],
    [104.1, 105.1, 106.3, 107.9, 108.5, 110.3, 111.5, 112.8, 120.5, 121.8, 122.6, 123.7, 125.2, 126.7, 126.8, 129.9],
    [108.2, 109.6, 111.2, 113.3, 116.4, 118.8, 121.4, 123.8, 134.6, 136.8, 139.5, 141.7, 144.0, 146.1, 146.9, 149.4],
    [112.5, 114.9, 117.2, 119.6, 124.6, 128.5, 132.7, 136.6, 149.7, 153.8, 157.9, 161.1, 164.9, 167.6, 169.2, 173.8],
    [121.0, 124.3, 128.3, 132.4, 140.2, 145.8, 153.8, 160.0, 180.0, 187.3, 194.8, 202.0, 208.2, 212.0, 216.0, 222.7],
]


EXPERT_STEP = [1.25, 1.43, 1.63, 1.83, 2.17]
EXPERT_VERIFY = [
    [18.3, 21.1, 23.1, 25.4, 27.3, 29.1, 30.8, 31.8, 33.2, 34.4, 37.1, 40.6, 42.3, 44.8, 44.6, 46.0],
    [19.8, 22.8, 25.7, 28.0, 30.0, 32.7, 34.7, 36.2, 37.1, 38.9, 42.0, 45.5, 48.6, 50.9, 50.8, 52.4],
    [20.8, 24.0, 26.6, 29.3, 31.9, 34.8, 37.5, 39.2, 40.7, 42.8, 46.2, 49.9, 51.8, 54.5, 54.7, 56.8],
    [21.7, 25.2, 28.0, 31.0, 34.3, 36.8, 39.4, 41.3, 43.6, 46.0, 49.6, 54.0, 56.6, 59.7, 59.9, 61.8],
    [23.6, 25.7, 29.1, 32.4, 36.8, 40.5, 43.9, 46.2, 49.2, 52.2, 57.0, 61.4, 66.3, 69.5, 69.9, 72.8],
]
EXPERTS = False


def costs(context, scale):
    global STEP, VERIFY
    if EXPERTS:
        STEP, VERIFY = EXPERT_STEP, EXPERT_VERIFY
    hi = 1
    while hi + 1 < len(DEPTHS) and context > DEPTHS[hi]:
        hi += 1
    lo = hi - 1
    t = (context - DEPTHS[lo]) / (DEPTHS[hi] - DEPTHS[lo])
    lerp = lambda a, b: max(a, a + t * (b - a))
    step = lerp(STEP[lo], STEP[hi])
    verify = [0.0] * 33
    for r in range(1, 17):
        verify[r] = lerp(VERIFY[lo][r - 1], VERIFY[hi][r - 1]) + (scale - 1) * VERIFY[0][r - 1]
    for r in range(17, 33):
        verify[r] = verify[r - 1] + (verify[16] - verify[15])
    return step, verify


def q_from_logits(ids, logits, temperature, top_k, top_p):
    """The request sampler's filters on the drafter's candidates (no penalties)."""
    order = sorted(range(len(ids)), key=lambda i: -logits[i])
    if top_k > 0:
        order = order[:top_k]
    t = max(temperature, 1e-6)
    m = max(logits[i] for i in order)
    w = [math.exp((logits[i] - m) / t) for i in order]
    z = sum(w)
    probs = [x / z for x in w]
    keep = len(order)
    if top_p < 1.0:
        c = 0.0
        for n, pr in enumerate(probs):
            c += pr
            if c >= top_p:
                keep = n + 1
                break
    z = sum(probs[:keep])
    return {ids[order[n]]: probs[n] / z for n in range(keep)}


def accept_and_sibling(p, q):
    """alpha = sum min(p, q); hit = P(draft rejected and its sibling, drawn from
    q without the draft, accepted) / P(draft rejected), as VerifyWithSibling."""
    alpha = sum(min(pv, q.get(t, 0.0)) for t, pv in p.items())
    reject = 1.0 - alpha
    if reject <= 1e-12:
        return alpha, 0.0
    res = {t: max(pv - q.get(t, 0.0), 0.0) for t, pv in p.items()}
    tot = sum(res.values())
    if tot <= 0:
        return alpha, 0.0
    r = {t: v / tot for t, v in res.items() if v > 0}
    joint = 0.0  # P(reject x and accept sibling)
    for x, qx in q.items():
        px = p.get(x, 0.0)
        rej_x = qx * (1.0 - min(1.0, px / qx)) if qx > 0 else 0.0
        if rej_x <= 0:
            continue
        rest = 1.0 - qx
        if rest <= 1e-12:
            continue
        acc = 0.0
        for y, qy in q.items():
            if y == x or qy <= 0:
                continue
            own = qy / rest
            acc += own * min(1.0, r.get(y, 0.0) / own)
        joint += rej_x * acc
    return alpha, joint / reject


def variants(rec):
    T, K, P = rec['temperature'], rec['top_k'], rec['top_p']
    return {
        'production': lambda j: {i: v for i, v in zip(rec['proposal_dist'][j]['ids'], rec['proposal_dist'][j]['v'])},
        'no_top_p': lambda j: q_from_logits(rec['draft_candidates'][j]['ids'], rec['draft_candidates'][j]['v'], T, K, 1.0),
        'top_p_0.99': lambda j: q_from_logits(rec['draft_candidates'][j]['ids'], rec['draft_candidates'][j]['v'], T, K, 0.99),
        'refiltered_same': lambda j: q_from_logits(rec['draft_candidates'][j]['ids'], rec['draft_candidates'][j]['v'], T, K, P),
        'temp_x0.8': lambda j: q_from_logits(rec['draft_candidates'][j]['ids'], rec['draft_candidates'][j]['v'], T * 0.8, K, P),
        'temp_x1.2': lambda j: q_from_logits(rec['draft_candidates'][j]['ids'], rec['draft_candidates'][j]['v'], T * 1.2, K, P),
        'temp_x1.2_no_top_p': lambda j: q_from_logits(rec['draft_candidates'][j]['ids'], rec['draft_candidates'][j]['v'], T * 1.2, K, 1.0),
    }


def cycle_value(alpha, hit, K, s, step, verify):
    tokens, reach = 1.0, 1.0
    for j in range(K):
        tokens += reach * (alpha[j] + ((1 - alpha[j]) * hit[j] if j < s else 0.0))
        reach *= alpha[j]
    steps = K + 1 if K < 15 else K
    return tokens, step * steps + verify[K + 1 + s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--scale', type=float, default=1.75)
    ap.add_argument('--min-positions', type=int, default=4)
    ap.add_argument('--experts', action='store_true')
    a = ap.parse_args()
    global EXPERTS
    EXPERTS = a.experts
    cycles = []
    for f in a.traces:
        for line in open(f):
            r = json.loads(line)
            if not r.get('sampled') or r.get('sessions') != 1:
                continue
            n = min(len(r.get('target_dist', [])), len(r.get('proposal_dist', [])), len(r.get('draft_candidates', [])))
            if n >= a.min_positions:
                cycles.append((r, n))
    print(f"{len(cycles)} sampled cycles, {sum(n for _, n in cycles)} positions")
    results = {}
    for name in variants(cycles[0][0]):
        per_cycle = []
        a_sum = h_sum = n_pos = with_sib = 0
        for r, n in cycles:
            v = variants(r)[name]
            al, hi = [], []
            for j in range(n):
                p = {i: x for i, x in zip(r['target_dist'][j]['ids'], r['target_dist'][j]['v'])}
                q = v(j)
                aa, hh = accept_and_sibling(p, q)
                al.append(aa); hi.append(hh)
                a_sum += aa; h_sum += hh; n_pos += 1; with_sib += len(q) > 1
            per_cycle.append((al, hi, costs(r['position'], a.scale)))
        # best fixed (K, s) over all cycles, and the per-cycle oracle
        best = None
        for K in range(0, 16):
            for s in range(0, K + 1):
                tok = ms = 0.0
                for al, hi, (step, verify) in per_cycle:
                    k = min(K, len(al)); ss = min(s, k)
                    t, m = cycle_value(al, hi, k, ss, step, verify)
                    tok += t; ms += m
                if best is None or tok / ms > best[0]:
                    best = (tok / ms, K, s)
        tok = ms = 0.0
        tok0 = ms0 = 0.0
        for al, hi, (step, verify) in per_cycle:
            opt = max((cycle_value(al, hi, k, s, step, verify) for k in range(len(al) + 1) for s in range(k + 1)), key=lambda x: x[0] / x[1])
            opt0 = max((cycle_value(al, hi, k, 0, step, verify) for k in range(len(al) + 1)), key=lambda x: x[0] / x[1])
            tok += opt[0]; ms += opt[1]; tok0 += opt0[0]; ms0 += opt0[1]
        results[name] = dict(alpha=a_sum / n_pos, hit=h_sum / n_pos, multi_token_q=with_sib / n_pos,
                             best_fixed=best[0] * 1000, best_K=best[1], best_s=best[2],
                             oracle=tok / ms * 1000, oracle_no_siblings=tok0 / ms0 * 1000)
    base = results['production']
    for name, x in results.items():
        print(f"{name:20s} alpha {x['alpha']:.3f} hit {x['hit']:.3f} q>1 token {x['multi_token_q']:.0%}  "
              f"best fixed {x['best_fixed']:.2f} tok/s (K={x['best_K']}, s={x['best_s']}, {x['best_fixed']/base['best_fixed']-1:+.1%})  "
              f"oracle {x['oracle']:.2f} ({x['oracle']/base['oracle']-1:+.1%}), oracle no siblings {x['oracle_no_siblings']:.2f}")


main()
