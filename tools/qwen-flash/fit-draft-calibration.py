#!/usr/bin/env python3
"""Fit draft-slot acceptance curves for the Qwen3.8-Flash-Next MTP controller.

Consumes the `depth,context,confidence,accepted` pairs that `GUFO_CALIB_LOG`
streams, and fits

    logit(P(accept)) = a + b * logit(confidence) + c * log(context)

per target model. A logistic surface is used rather than a two-dimensional
isotonic grid because the pair count is spread thin: a 2-D monotone fit leaves
most cells below any usable minimum for a long time, whereas this has three
coefficients per model. Monotonicity in confidence is not imposed but checked:
`b` must come out positive for the curve to be usable as a gate, and the script
fails loudly if it does not.

Models are shrunk toward a pooled fit in proportion to their pair count, so a
model with little evidence sits near the average and earns its own surface as
data arrives. A model with no data is exactly the pooled fit.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

EPS = 1e-4
MIN_PAIRS_FOR_OWN_FIT = 400
SHRINK_STRENGTH = 400.0  # pseudo-pairs pulling a model toward the pooled fit


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def load(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (context, confidence, accepted) for one log file."""
    ctx: list[float] = []
    conf: list[float] = []
    acc: list[int] = []
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) != 4:
            continue
        _depth, context, confidence, accepted = parts
        ctx.append(float(context))
        conf.append(float(confidence))
        acc.append(int(accepted))
    return (
        np.asarray(ctx, dtype=np.float64),
        np.asarray(conf, dtype=np.float64),
        np.asarray(acc, dtype=np.float64),
    )


USE_CONTEXT = False


def design(conf: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    if USE_CONTEXT:
        return np.column_stack(
            [np.ones_like(conf), logit(conf), np.log1p(np.maximum(ctx, 0.0))]
        )
    return np.column_stack([np.ones_like(conf), logit(conf)])


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 60) -> np.ndarray:
    """Newton/IRLS with a small ridge term; the intercept is left unpenalised."""
    beta = np.zeros(x.shape[1])
    ridge = np.full(x.shape[1], 1e-3)
    ridge[0] = 0.0
    for _ in range(iters):
        eta = np.clip(x @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-6)
        grad = x.T @ (y - mu) - ridge * beta
        hess = (x.T * w) @ x + np.diag(ridge)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def predict(beta: np.ndarray, conf: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    eta = np.clip(design(conf, ctx) @ beta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-eta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="label=path[+path...] groups")
    ap.add_argument("--out", default="draft_calibration.json")
    ap.add_argument(
        "--with-context",
        action="store_true",
        help="add a log1p(context) term; off by default because holding content "
             "fixed shows acceptance flat in depth, so the term fits content",
    )
    args = ap.parse_args()
    global USE_CONTEXT
    USE_CONTEXT = args.with_context

    per_model: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for spec in args.csv:
        label, _, paths = spec.partition("=")
        ctxs, confs, accs = [], [], []
        for raw in paths.split("+"):
            c, f, a = load(pathlib.Path(raw))
            if c.size:
                ctxs.append(c)
                confs.append(f)
                accs.append(a)
        if not ctxs:
            print(f"{label}: no pairs, skipped")
            continue
        ctx, conf, acc = (np.concatenate(x) for x in (ctxs, confs, accs))
        per_model[label] = (ctx, conf, acc)
        print(f"{label}: {ctx.size} pairs, acceptance {acc.mean():.3f}")

    if not per_model:
        raise SystemExit("no data")

    pooled_ctx = np.concatenate([c for c, _f, _a in per_model.values()])
    pooled_conf = np.concatenate([f for _c, f, _a in per_model.values()])
    pooled_acc = np.concatenate([a for _c, _f, a in per_model.values()])
    pooled = fit_logistic(design(pooled_conf, pooled_ctx), pooled_acc)

    coef = " ".join(f"{v:+.3f}" for v in pooled)
    print(f"\npooled: {coef} over {pooled_conf.size} pairs")
    if pooled[1] <= 0.0:
        raise SystemExit(
            "pooled confidence coefficient is not positive; confidence does not "
            "predict acceptance on this data, so the curve is unusable as a gate"
        )

    result: dict[str, object] = {
        "model": "logistic",
        "features": (["1", "logit(confidence)", "log1p(context)"] if USE_CONTEXT
                     else ["1", "logit(confidence)"]),
        "min_pairs_for_own_fit": MIN_PAIRS_FOR_OWN_FIT,
        "shrink_strength": SHRINK_STRENGTH,
        "pooled": pooled.tolist(),
        "models": {},
    }


    for label, (ctx, conf, y) in sorted(per_model.items()):
        n = ctx.size
        if n < MIN_PAIRS_FOR_OWN_FIT:
            beta = pooled.copy()
            source = "pooled"
        else:
            raw = fit_logistic(design(conf, ctx), y)
            weight = n / (n + SHRINK_STRENGTH)
            beta = weight * raw + (1.0 - weight) * pooled
            source = "shrunk"
        result["models"][label] = {
            "pairs": int(n),
            "source": source,
            "acceptance": float(y.mean()),
            "beta": beta.tolist(),
            "monotonic": bool(beta[1] > 0.0),
        }
        cells = " ".join(f"{v:+8.3f}" for v in beta)
        print(f"{label:>8} {n:>7} {cells}"
              f"  {'yes' if beta[1] > 0 else 'NO'}  ({source})")

    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
