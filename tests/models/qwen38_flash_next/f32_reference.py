#!/usr/bin/env python3
"""Compact F32 reference fixtures for Qwen3.8 Flash-Next prefill logits.

The CPU reference (`qwen38_flash_next_reference_probe --dump`) writes the
F32 logits of every prompt position. A fixture keeps, for the positions a
`qwen38_flash_next_gpu_probe --logit-rows N` dump covers (the last N rows of
every prefill pass), the reference's log-sum-exp over the full vocabulary and
its top-K token ids and logits. That is enough to score a GPU dump without
running the reference again:

  * KL(reference || gpu), exact over the reference top-K plus the remaining
    probability mass as one grouped term (a lower bound of the full KL);
  * agreement of the top-1 token and the reference top-1's rank on the GPU;
  * the negative log likelihood of the prompt's actual next token, which the
    reference measures over its full vocabulary.

  f32_reference.py write --reference ref.bin --tokens N --batch B --rows R
                         [--top-k K] --out NAME.npz
  f32_reference.py compare --manifest manifest.json --dumps DIR [--label L]
                           [--against DIR2 --against-label L2]

`compare` expects DIR/NAME.bin for every prompt in the manifest, dumped with
the manifest's batch and row count. With --against it also reports the paired
per-position KL and next-token NLL differences between the two builds with
bootstrap 95% intervals (NLL also per prompt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VOCAB = 248320


def dumped_positions(tokens: int, batch: int, rows: int) -> np.ndarray:
    """Positions of the rows a --logit-rows dump holds, in file order."""
    out = []
    for start in range(0, tokens, batch):
        end = min(start + batch, tokens)
        kept = min(rows, end - start)
        out.extend(range(end - kept, end))
    return np.asarray(out, dtype=np.int64)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def write(args: argparse.Namespace) -> int:
    positions = dumped_positions(args.tokens, args.batch, args.rows)
    ref = np.memmap(args.reference, dtype=np.float32, mode="r")
    if ref.size % VOCAB != 0 or ref.size // VOCAB < args.tokens:
        print(f"{args.reference}: expected {args.tokens} rows", file=sys.stderr)
        return 1
    ref = ref.reshape(-1, VOCAB)
    k = args.top_k
    lse = np.empty(len(positions), dtype=np.float32)
    ids = np.empty((len(positions), k), dtype=np.uint32)
    top = np.empty((len(positions), k), dtype=np.float32)
    for i, p in enumerate(positions):
        row = np.asarray(ref[p], dtype=np.float64)
        m = row.max()
        lse[i] = m + np.log(np.exp(row - m).sum())
        order = np.argpartition(-row, k)[:k]
        order = order[np.argsort(-row[order], kind="stable")]
        ids[i] = order
        top[i] = row[order]
    np.savez_compressed(args.out, positions=positions.astype(np.uint32),
                        lse=lse, top_ids=ids, top_logits=top)
    print(f"{args.out}: {len(positions)} positions, top-{k}")
    return 0


def score(fixture: dict, gpu: np.ndarray, next_tokens: np.ndarray) -> dict:
    """Per-position metrics of one GPU dump against a fixture."""
    lg = log_softmax(gpu)
    ids = fixture["top_ids"].astype(np.int64)
    lp_ref = fixture["top_logits"].astype(np.float64) - fixture["lse"][:, None]
    p_ref = np.exp(lp_ref)
    lp_gpu = np.take_along_axis(lg, ids, axis=1)
    kl_top = (p_ref * (lp_ref - lp_gpu)).sum(axis=1)
    tail_ref = np.clip(1.0 - p_ref.sum(axis=1), 1e-300, 1.0)
    tail_gpu = np.clip(1.0 - np.exp(lp_gpu).sum(axis=1), 1e-300, 1.0)
    kl = kl_top + tail_ref * (np.log(tail_ref) - np.log(tail_gpu))
    ref_top1 = ids[:, 0]
    gpu_top1 = lg.argmax(axis=1)
    rank = (lg > np.take_along_axis(lg, ref_top1[:, None], axis=1)).sum(axis=1)
    has_next = next_tokens >= 0
    nll = np.full(len(gpu), np.nan)
    nll[has_next] = -lg[np.nonzero(has_next)[0], next_tokens[has_next]]
    return {"kl": kl, "top1": gpu_top1 == ref_top1, "rank": rank, "nll": nll}


def reference_nll(fixture: dict, next_tokens: np.ndarray) -> np.ndarray:
    """The reference's NLL of each next token, when it is in its top-K."""
    ids = fixture["top_ids"].astype(np.int64)
    lp = fixture["top_logits"].astype(np.float64) - fixture["lse"][:, None]
    out = np.full(len(ids), np.nan)
    for i, t in enumerate(next_tokens):
        hit = np.nonzero(ids[i] == t)[0]
        if t >= 0 and len(hit):
            out[i] = -lp[i, hit[0]]
    return out


def ids_fnv1a(tokens: np.ndarray) -> str:
    """gpu_probe's FNV-1a over the token ids as little-endian int32 bytes."""
    h = 0xCBF29CE484222325
    for byte in tokens.astype("<i4").tobytes():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:x}"


def load(manifest_path: Path, dumps: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    results = {}
    for prompt in manifest["prompts"]:
        fixture = dict(np.load(base / prompt["fixture"]))
        tokens = np.asarray(json.loads((base / prompt["tokens"]).read_text()),
                            dtype=np.int64)
        positions = fixture["positions"].astype(np.int64)
        nxt = np.where(positions + 1 < len(tokens),
                       tokens[np.minimum(positions + 1, len(tokens) - 1)], -1)
        if len(tokens) != prompt["token_count"]:
            raise SystemExit(f"{prompt['name']}: {len(tokens)} token ids, "
                             f"manifest {prompt['token_count']}")
        # gpu_probe --logit-rows records the batch and row count beside the
        # dump: another batch can give the same row count at other positions.
        meta_path = dumps / f"{prompt['name']}.bin.json"
        if not meta_path.exists():
            raise SystemExit(f"{meta_path}: missing; dump with gpu_probe "
                             f"--logit-rows")
        meta = json.loads(meta_path.read_text())
        expected = {"tokens": len(tokens), "ids_fnv1a": ids_fnv1a(tokens),
                    "batch": prompt["batch"],
                    "rows_per_pass": manifest["rows_per_pass"]}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise SystemExit(f"{meta_path}: {key} {meta.get(key)}, "
                                 f"fixture {value}")
        gpu = np.fromfile(dumps / f"{prompt['name']}.bin", dtype=np.float32)
        gpu = gpu.reshape(-1, VOCAB)
        if len(gpu) != len(positions):
            raise SystemExit(f"{prompt['name']}: dump has {len(gpu)} rows, "
                             f"fixture {len(positions)}")
        results[prompt["name"]] = (score(fixture, gpu, nxt),
                                   reference_nll(fixture, nxt))
    return results


def summary(label: str, results: dict) -> None:
    # NLL: the GPU's next-token NLL over every scored position. NLL@ref and
    # ref NLL: GPU and reference over the positions whose next token is in
    # the reference's stored top-k (the only ones the reference can score).
    print(f"\n{label}")
    print(f"{'prompt':<8} {'rows':>5} {'KL mean':>9} {'median':>9} {'p99':>8} "
          f"{'max':>7} {'top-1':>7} {'NLL':>7} {'NLL@ref':>8} {'ref NLL':>8}")
    every = {"kl": [], "top1": [], "nll": [], "nll_ref": [], "ref": []}
    for name, (m, ref_nll) in results.items():
        scored = ~np.isnan(m["nll"])
        both = scored & ~np.isnan(ref_nll)
        print(f"{name:<8} {len(m['kl']):>5} {m['kl'].mean():>9.5f} "
              f"{np.median(m['kl']):>9.5f} {np.quantile(m['kl'], 0.99):>8.4f} "
              f"{m['kl'].max():>7.4f} {m['top1'].mean():>7.3f} "
              f"{m['nll'][scored].mean():>7.4f} {m['nll'][both].mean():>8.4f} "
              f"{ref_nll[both].mean():>8.4f}")
        every["kl"].append(m["kl"])
        every["top1"].append(m["top1"])
        every["nll"].append(m["nll"][scored])
        every["nll_ref"].append(m["nll"][both])
        every["ref"].append(ref_nll[both])
    kl = np.concatenate(every["kl"])
    print(f"{'all':<8} {len(kl):>5} {kl.mean():>9.5f} {np.median(kl):>9.5f} "
          f"{np.quantile(kl, 0.99):>8.4f} {kl.max():>7.4f} "
          f"{np.concatenate(every['top1']).mean():>7.3f} "
          f"{np.concatenate(every['nll']).mean():>7.4f} "
          f"{np.concatenate(every['nll_ref']).mean():>8.4f} "
          f"{np.concatenate(every['ref']).mean():>8.4f}")


def bootstrap(diff: np.ndarray, rng: np.random.Generator) -> tuple:
    boot = [rng.choice(diff, size=len(diff)).mean() for _ in range(2000)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return diff.mean(), lo, hi


def paired(a: dict, b: dict, label_a: str, label_b: str) -> None:
    rng = np.random.default_rng(0)
    da = np.concatenate([a[n][0]["kl"] for n in a])
    db = np.concatenate([b[n][0]["kl"] for n in a])
    mean, lo, hi = bootstrap(da - db, rng)
    print(f"\nKL({label_a}) - KL({label_b}): mean {mean:+.5f}, "
          f"95% interval [{lo:+.5f}, {hi:+.5f}] over {len(da)} positions")
    # Next-token NLL over the positions both builds score, overall and per
    # prompt.
    every = []
    for name in a:
        na, nb = a[name][0]["nll"], b[name][0]["nll"]
        both = ~np.isnan(na) & ~np.isnan(nb)
        diff = na[both] - nb[both]
        every.append(diff)
        mean, lo, hi = bootstrap(diff, rng)
        print(f"NLL({label_a}) - NLL({label_b}) {name:<6}: mean {mean:+.4f}, "
              f"95% interval [{lo:+.4f}, {hi:+.4f}] over {len(diff)} positions")
    diff = np.concatenate(every)
    mean, lo, hi = bootstrap(diff, rng)
    print(f"NLL({label_a}) - NLL({label_b}) all   : mean {mean:+.4f}, "
          f"95% interval [{lo:+.4f}, {hi:+.4f}] over {len(diff)} positions")


def compare(args: argparse.Namespace) -> int:
    a = load(Path(args.manifest), Path(args.dumps))
    summary(args.label, a)
    if args.against:
        b = load(Path(args.manifest), Path(args.against))
        summary(args.against_label, b)
        paired(a, b, args.label, args.against_label)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("--reference", required=True)
    w.add_argument("--tokens", type=int, required=True)
    w.add_argument("--batch", type=int, required=True)
    w.add_argument("--rows", type=int, required=True)
    w.add_argument("--top-k", type=int, default=256)
    w.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--manifest", required=True)
    c.add_argument("--dumps", required=True)
    c.add_argument("--label", default="candidate")
    c.add_argument("--against")
    c.add_argument("--against-label", default="baseline")
    args = parser.parse_args()
    return write(args) if args.command == "write" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
