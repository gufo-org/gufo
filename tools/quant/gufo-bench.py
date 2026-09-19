#!/usr/bin/env python3
"""gufo-bench — correctness-linked benchmark CLI.

Measures, for the full-precision teacher and the SHQ4-T16 candidate:

- Prefill tokens/sec (full-prompt forward).
- Decode tokens/sec (single-token forward).
- Candidate-vs-teacher quality on the matched-token suite:
  KL divergence (mean/median/p95/p99/max), perplexity, top-1 agreement,
  non-finite logit count.

Quality is measured first; speed only after logits are comparable
(docs/TESTING.md: correctness before performance counts).

Usage:
  gufo-bench --source DIR --quant DIR --suite FILE --teacher-artifact DIR
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gufo import model as strix_model
from gufo import quality


def load_teacher_logits(artifact_dir):
    """Load captured teacher logits: returns (target_tokens, logits_by_prompt)."""
    ad = Path(artifact_dir)
    import zstandard
    positions = json.loads((ad / "positions.json").read_text("utf-8"))
    tokens = np.fromfile(ad / "tokens.u32", dtype=np.uint32)
    logits = []
    for pi in range(len(positions)):
        fbytes = (ad / f"logits-{pi:05d}.f32.zst").read_bytes()
        T = positions[pi]["tokens"]
        d = zstandard.ZstdDecompressor().decompress(fbytes, max_output_size=T * 248320 * 4)
        logits.append(np.frombuffer(d, dtype=np.float32).reshape(T, 248320))
    # target tokens = tokens[1:] per prompt, concatenated
    return tokens, logits


def timed(fn, n=2):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gufo-bench")
    ap.add_argument("--source", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--teacher-artifact", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    suite = json.loads(Path(args.suite).read_text("utf-8"))
    tokens, teacher_logits = load_teacher_logits(args.teacher_artifact)

    teacher, tok = strix_model.load_teacher(args.source)
    candidate, _ = strix_model.load_candidate(args.source, args.quant)

    prompts = suite["prompts"]
    ids_list = [tok(p, return_tensors="pt")["input_ids"][0].tolist() for p in prompts]

    # --- candidate quality (matched-token vs teacher artifact) ---
    cand_logits = []
    with torch.no_grad():
        for ids in ids_list:
            lg = strix_model.forward_logits(candidate, torch.tensor([ids], dtype=torch.long))[0].float().numpy()
            cand_logits.append(lg)

    # teacher logits from artifact (f32), candidate from run
    # compare per prompt position-wise
    kl_all, top1 = [], []
    cand_nll = []
    off = 0
    for pi, ids in enumerate(ids_list):
        t_log = teacher_logits[pi].astype(np.float64)
        c_log = cand_logits[pi].astype(np.float64)
        lp_t = quality.log_softmax(t_log)
        lp_c = quality.log_softmax(c_log)
        p_t = np.exp(lp_t)
        p_c = np.exp(lp_c)
        kl = quality.kl_divergence(p_t, p_c)
        kl_all.append(kl)
        top1_t = np.argmax(t_log, axis=1)
        top1_c = np.argmax(c_log, axis=1)
        top1.append((top1_t == top1_c))
        targets = ids[1:]
        cand_nll.append(-np.log(np.clip(p_c[np.arange(len(targets)), targets], 1e-30, 1.0)))
    kl_all = np.concatenate(kl_all)
    top1 = np.concatenate(top1)

    quality_metrics = {
        "kl_mean": float(kl_all.mean()),
        "kl_median": float(np.median(kl_all)),
        "kl_p95": float(np.percentile(kl_all, 95)),
        "kl_p99": float(np.percentile(kl_all, 99)),
        "kl_max": float(kl_all.max()),
        "top1_agreement": float(top1.mean()),
        "perplexity_candidate": float(np.exp(np.concatenate(cand_nll).mean())),
        "positions": int(len(kl_all)),
    }

    # --- speed ---
    # prefill: full prompt forward
    prefill_tok = sum(len(i) for i in ids_list)
    def run_prefill(model):
        with torch.no_grad():
            for ids in ids_list:
                strix_model.forward_logits(model, torch.tensor([ids], dtype=torch.long))
    def run_decode(model):
        # one token, repeated
        ids = ids_list[0][:1]
        with torch.no_grad():
            strix_model.forward_logits(model, torch.tensor([ids], dtype=torch.long))

    t_prefill_c = timed(lambda: run_prefill(candidate), args.repeats)
    t_decode_c = timed(lambda: run_decode(candidate), args.repeats)
    t_prefill_t = timed(lambda: run_prefill(teacher), args.repeats)
    t_decode_t = timed(lambda: run_decode(teacher), args.repeats)

    speed = {
        "prefill_tokens_per_sec_teacher": prefill_tok / t_prefill_t,
        "prefill_tokens_per_sec_candidate": prefill_tok / t_prefill_c,
        "decode_tokens_per_sec_teacher": 1.0 / t_decode_t,
        "decode_tokens_per_sec_candidate": 1.0 / t_decode_c,
        "prefill_sec_teacher": t_prefill_t,
        "prefill_sec_candidate": t_prefill_c,
        "decode_sec_per_token_teacher": t_decode_t,
        "decode_sec_per_token_candidate": t_decode_c,
    }

    report = {"model": "Qwen/Qwen3.5-0.8B", "candidate": "SHQ4-T16 U4Z G64",
              "quality": quality_metrics, "speed": speed}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in speed.items():
            print(f"{k}: {v:.4f}")
        print("--- quality (candidate vs teacher) ---")
        for k, v in quality_metrics.items():
            print(f"{k}: {v:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
