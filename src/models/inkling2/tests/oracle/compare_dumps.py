#!/usr/bin/env python3
"""Compare two I2OR dumps (reference first). Reports KL over the reference top-64 + tail bucket,
top-1 agreement and mean |logit diff| at the reference's top-64, overall and per position range.
usage: compare_dumps.py ref.bin ours.bin [boundaries...]   (boundaries default 1024 2048 8192)"""
import math, struct, sys

def load(path):
    raw = open(path, 'rb').read()
    assert raw[:4] == b'I2OR'
    _, n, top = struct.unpack_from('<III', raw, 4)
    rec = 4 + top * 12
    out = []
    for p in range(n):
        o = 16 + p * rec
        lse = struct.unpack_from('<f', raw, o)[0]
        ids = struct.unpack_from('<%di' % top, raw, o + 4)
        lg = struct.unpack_from('<%df' % top, raw, o + 4 + top * 4)
        at = struct.unpack_from('<%df' % top, raw, o + 4 + top * 8)
        out.append((lse, ids, lg, at))
    return out

def kl(ref, our):
    lse_r, ids_r, lg_r, _ = ref
    lse_o, _, _, at_o = our
    pr = [math.exp(l - lse_r) for l in lg_r]
    po = [math.exp(l - lse_o) for l in at_o]
    tr, to = max(1.0 - sum(pr), 1e-12), max(1.0 - sum(po), 1e-12)
    s = sum(p * (math.log(p) - math.log(q)) for p, q in zip(pr, po) if p > 0)
    s += tr * (math.log(tr) - math.log(to))
    return s

ref, our = load(sys.argv[1]), load(sys.argv[2])
n = min(len(ref), len(our))
bounds = [int(x) for x in sys.argv[3:]] or [1024, 2048, 8192]
edges = [0] + [b for b in bounds if b < n] + [n]
def report(name, a, b):
    ks = [kl(ref[i], our[i]) for i in range(a, b)]
    t1 = sum(ref[i][1][0] == our[i][1][0] for i in range(a, b))
    md = sum(sum(abs(x - y) for x, y in zip(ref[i][2], our[i][3])) / 64 for i in range(a, b)) / (b - a)
    w = max(range(len(ks)), key=lambda i: ks[i])
    print(f"{name:>14}: n={b-a:5d} meanKL={sum(ks)/len(ks):.3e} maxKL={max(ks):.3e}@{a+w} "
          f"top1={t1/(b-a)*100:.2f}% mean|dlogit|={md:.3e}")
report('all', 0, n)
for a, b in zip(edges, edges[1:]):
    report(f'[{a},{b})', a, b)
