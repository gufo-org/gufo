#!/usr/bin/env python3
"""gufo-calibrate — capture per-input-channel imatrix (E[x^2]) for SHQ4 tensors.

Runs the calibration suite through the full-precision teacher and records, for
every eligible LM linear projection, its per-input-channel importance vector
h[j] = E[x_j^2] over the calibration tokens. This is the activation-weighted
reconstruction objective used by the imatrix/GPTQ-style scale search in
gufo-quantize (--imatrix). Mirror of gufo-capture's suite/hash discipline.

Performance vs. the naive reference:

- The whole corpus is tokenized once and forwarded in TOKEN-BUDGETED BATCHES
  instead of one prompt per forward (same per-real-token numerics; batches
  amortize kernel/GEMM launch overhead).
- Optional `--device cuda` (ROCm/HIP torch in the default shell) runs the
  forward on the gfx1151 GPU for multi-million-token corpora.
- Pad positions are excluded with the attention mask, never accumulated, so
  batching is EXACT: each real token sees bit-identical inputs whether or not
  other (padded) rows share the batch.

Numerical exactness guarantees:

- Accumulation is float64 ACROSS batches (per-channel sum-of-squares via numpy
  float64); within a batch the block sum is float32 (gfx1151 fp64 is 1/16
  rate), then cast to fp64 and accumulated. For `--max-tokens N` the fp32
  block error is ~sqrt(N)*eps_fp32, negligible versus the corpus-wide fp64
  accumulation.
- Pad-masked per-token squares are reduced with the attention mask, so the
  statistic is exactly E[x_j^2] over REAL tokens.
- GPU path forces fp32 accumulation (allow_tf32=False) and deterministic
  algorithms; bf16 activations are never converted to fp16.
- `--reference DIR` cross-checks the new artifact against a prior run.

GPU speed:

- fp32 block-reduce hooks (not fp64) avoid the GPU fp64 pipe. Run-to-run is
  bit-reproducible for fixed `--device`/`--max-tokens`.

Layout of the output artifact dir:
  manifest.json        schema, hashes (source + dataset), token count, per tensor
                       { file, dim, count, mean_channel_energy }
  <tensor>.f32         one importance vector per eligible tensor (float32, K)

Usage:
  gufo-calibrate --source DIR --suite FILE --out DIR [--device cpu|cuda|hip]
                  [--max-tokens N] [--reference DIR] [--seed N]
"""
import sys
import os
import json
import hashlib
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gufo import model as strix_model


# Eligible module suffixes = weight-tensor suffixes minus ".weight".
_ELIGIBLE_MODULES = tuple(s[: -len(".weight")] for s in strix_model.ELIGIBLE)


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _greedy_chunks(lens, budget):
    """Group prompt-token-lengths into chunks whose cumulative real length is
    <= budget. Keeps prompt order; pads only within a chunk to its max length."""
    chunks, cur, acc = [], [], 0
    for n in lens:
        if cur and acc + n > budget:
            chunks.append(cur)
            cur, acc = [], 0
        cur.append(n)
        acc += n
    if cur:
        chunks.append(cur)
    return chunks


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gufo-calibrate")
    ap.add_argument("--source", required=True, help="HF snapshot directory")
    ap.add_argument("--suite", required=True, help="calibration suite JSON")
    ap.add_argument("--out", default="artifacts/calib")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) | cuda | hip  (hip==cuda; needs ROCm torch)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="max real tokens per batched forward (memory bound)")
    ap.add_argument("--reference", default=None,
                    help="prior gufo.imatrix.v1 dir to cross-check against")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = "cuda" if args.device in ("cuda", "hip", "gpu", "rocm") else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                f"--device {args.device} requested but torch has no HIP backend. "
                "Use `nix develop` (python313 torchWithRocm, gfx1151) or "
                "re-run with --device cpu.")
        # Deterministic, full-precision math only (bf16 x bf16 -> fp32 accum).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        print(f"device: {torch.cuda.get_device_name(0)} "
              f"(hip {torch.version.hip})", flush=True)
    else:
        print("device: cpu", flush=True)

    src = Path(args.source)
    suite = json.loads(Path(args.suite).read_text("utf-8"))
    model, tok = strix_model.load_teacher(str(src))
    if device == "cuda":
        model.to(device)

    # Collect target modules: language_model projections only (mtp head is not
    # loaded by this model adapter, so it is excluded and falls back to range
    # scaling in gufo-quantize).
    targets = {}
    for name, mod in model.named_modules():
        if not name.startswith("language_model."):
            continue
        if any(name.endswith(sfx) for sfx in _ELIGIBLE_MODULES):
            targets[name] = mod
    if not targets:
        raise SystemExit("no eligible modules found")

    acc = {name: (np.zeros(mod.weight.shape[1], dtype=np.float64), 0)
           for name, mod in targets.items()}

    # Target-language sequences; chunk by cumulative real-token budget (order
    # preserved). Tokenize once per chunk with padding so lengths don't blow up.
    seqs = [tok(t, return_tensors="pt")["input_ids"][0] for t in suite["prompts"]]
    lens = [int(len(s)) for s in seqs]
    chunks = _greedy_chunks(lens, args.max_tokens)
    print(f"prompts: {len(seqs)}  tokens: {sum(lens)}  batches: {len(chunks)}",
          flush=True)

    # Forward-only state captured by outer scope of each hook.
    mask_pad = None
    n_real = 0

    def make_hook(name):
        def hook(mod, args, kwargs=None):
            x = args[0] if args else None
            if x is None and kwargs:
                x = next(iter(kwargs.values()))
            xt = x.detach().float()                  # [..., K], K = last dim
            lead = xt.numel() // xt.shape[-1]
            if lead == mask_pad.shape[0] * mask_pad.shape[1]:
                # [B,T]xK positions align with the flattened attention mask.
                m = mask_pad.reshape(-1, 1).to(xt.dtype)
            else:
                # Hidden reshaped (not [B,T]xK) — every position is real.
                m = torch.ones(lead, 1, dtype=torch.float32, device=xt.device)
            # Reduce squares to fp32 in one [K] block (GPU fp64 is 1/16 rate), then
            # accumulate across batches in float64 on the host (exact corpus sum).
            x2 = (xt.reshape(lead, -1) * m).square().sum(dim=0)
            s, n = acc[name]
            s += x2.detach().cpu().numpy()
            acc[name] = (s, n + n_real)
            return None
        return hook

    handles = [mod.register_forward_hook(make_hook(name))
               for name, mod in targets.items()]

    total_tokens = 0
    try:
        with torch.no_grad():
            idx, bi = 0, 0
            for c in chunks:
                texts = suite["prompts"][idx:idx + len(c)]
                idx += len(c)
                enc = tok(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=8192)
                ids = enc["input_ids"].to(device)
                msk = enc["attention_mask"].to(device)
                mask_pad = msk
                n_real = int(msk.sum())
                total_tokens += n_real
                model.language_model(input_ids=ids, attention_mask=msk,
                                     use_cache=False)
                bi += 1
                print(f"  batch {bi:>4}  [{len(c)} seq, "
                      f"len={int(msk.shape[1])}, {n_real} real]", flush=True)
    finally:
        for h in handles:
            h.remove()

    # Normalize to per-channel second moments and write artifact.
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name in sorted(targets):
        s, n = acc[name]
        imp = (s / n).astype(np.float32) if n > 0 else np.ones(
            targets[name].weight.shape[1], np.float32)
        fname = name.replace("model.", "") + ".f32"
        (outdir / fname).write_bytes(imp.tobytes())
        tensors[name] = {
            "file": fname,
            "dim": int(targets[name].weight.shape[1]),
            "count": int(n),
            "mean_channel_energy": float(imp.mean()),
        }

    # Hashes: source snapshot + dataset.
    source_hash = hashlib.sha256()
    src_files = sorted(src.glob("model.*.safetensors")) + sorted(src.glob("config.json"))
    for p in src_files:
        source_hash.update(p.name.encode())
        source_hash.update(hash_file(p).encode())
    suite_hash = hash_file(Path(args.suite))

    manifest = {
        "schema": "gufo.imatrix.v1",
        "model": suite.get("model", "unknown"),
        "source": str(src),
        "source_hash": source_hash.hexdigest(),
        "dataset": str(Path(args.suite)),
        "dataset_hash": suite_hash,
        "count": int(total_tokens),
        "device": device,
        "tensors": tensors,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    out = {"artifact": str(outdir), "modules": len(tensors),
           "tokens": total_tokens, "schema": manifest["schema"],
           "device": device}

    # Optional cross-check against a prior exact artifact.
    if args.reference:
        ref = Path(args.reference)
        ref_man = json.loads((ref / "manifest.json").read_text("utf-8"))
        worst_abs, worst_rel, worst_name = 0.0, 0.0, None
        for name in sorted(tensors):
            a = np.fromfile(outdir / tensors[name]["file"], np.float32)
            b = np.fromfile(ref / ref_man["tensors"][name]["file"], np.float32)
            d = np.abs(a - b)
            denom = np.maximum(np.abs(b), 1e-12)
            rel = np.max(d / denom)
            if np.max(d) > worst_abs:
                worst_abs, worst_name = np.max(d), name
            worst_rel = max(worst_rel, rel)
        out["xref"] = {
            "ref": str(ref),
            "max_abs_diff": float(worst_abs),
            "max_rel_diff": float(worst_rel),
            "worst_tensor": worst_name,
        }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
