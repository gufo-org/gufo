#!/usr/bin/env python3
"""gufo-quantize — deterministic SHQ-T16 conversion of a source snapshot.

Validates the source snapshot and converts each tensor using a
mixed-precision recipe (tools/gufo/recipe.py):
SHQ4-G64-U4Z (bulk), SHQ4-G32-U4Z (attention), SHQ8-G64 (high-precision tier),
or BF16 (kept). The recipe is per-tensor, so precision varies across layers
and components the way the unsloth Q4_K_M recipe does.

Deterministic: identical inputs -> identical bytes.
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from gufo import safetensors, shq
from gufo import recipe as recipe_mod
from gufo.manifest import write_json


def read_bf16_tensor(path, info, dtype, shape, data_off, offset=0) -> np.ndarray:
    data = safetensors.read_tensor(path, info, dtype, shape, data_off, offset)
    # bf16 is stored as uint16 bits in the TOP 16 bits of an f32; shift left 16.
    return (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


def quantize_one(name, shape, data, fmt, importance=None) -> dict:
    """Quantize W to the recipe encoding. Returns planes dict + stats."""
    W = data.reshape(shape)
    if fmt.startswith("SHQ4"):
        G = 32 if "G32" in fmt else 64
        planes = shq.quantize_shq4(W, group_size=G, importance=importance)
        Wd = shq.dequant_shq4(planes)[: shape[0], : shape[1]]
    elif fmt.startswith("SHQ6"):
        planes = shq.quantize_shq6(W, group_size=64)
        Wd = shq.dequant_shq6(planes)[: shape[0], : shape[1]]
    elif fmt.startswith("SHQ8"):
        planes = shq.quantize_shq8(W, group_size=64)
        Wd = shq.dequant_shq8(planes)[: shape[0], : shape[1]]
    else:
        raise ValueError(f"{name}: cannot quantize to {fmt}")
    err = np.abs(Wd - W)
    stats = {
        "format": planes["format"],
        "max_abs_err": float(err.max()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean_abs_err": float(err.mean()),
    }
    return planes, stats


def load_imatrix(imatrix_dir: str) -> dict:
    """Load a gufo-calibrate artifact dir -> {tensor name: importance[K]}."""
    d = Path(imatrix_dir)
    manifest = json.loads((d / "manifest.json").read_text("utf-8"))
    if manifest.get("schema") != "gufo.imatrix.v1":
        raise SystemExit(f"{imatrix_dir}: not a gufo.imatrix.v1 artifact")
    out = {}
    for name, info in manifest["tensors"].items():
        raw = (d / info["file"]).read_bytes()
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.shape[0] != info["dim"]:
            raise SystemExit(f"{name}: imatrix dim mismatch {arr.shape[0]} != {info['dim']}")
        # calibrate keys are module names (language_model...); quantize keys
        # are safetensors names (model.<module>.weight)
        out["model." + name + ".weight"] = arr
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gufo-quantize")
    ap.add_argument("--source", required=True, help="HF snapshot directory")
    ap.add_argument("--out", default="artifacts/quant")
    ap.add_argument("--plan", default="artifacts/work/quantization-plan.json")
    ap.add_argument("--recipe", default="embed_ffn",
                    help="preset recipe name (recipe.py PRESETS) or path to a "
                         "JSON rule list [[substr, fmt], ...]  (default embed_ffn: "
                         "embed+ffn_down SHQ8, rest SHQ4 G64, per MIXED_PRECISION.md)")
    ap.add_argument("--imatrix", default=None,
                    help="gufo-calibrate artifact dir; enables imatrix/GPTQ-style "
                         "importance-weighted scale search")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    # resolve recipe rules
    rp = Path(args.recipe)
    if rp.exists():
        rules = [(str(k), str(v)) for k, v in json.loads(rp.read_text("utf-8")).items()]
    elif args.recipe in recipe_mod.PRESETS:
        rules = recipe_mod.PRESETS[args.recipe]
    else:
        raise SystemExit(f"unknown recipe preset '{args.recipe}'")
    rid = recipe_mod.recipe_id(rules)

    imatrix = load_imatrix(args.imatrix) if args.imatrix else None
    if imatrix:
        print(f"imatrix: {len(imatrix)} tensors from {args.imatrix}")

    src = Path(args.source)
    index = src / "model.safetensors.index.json"
    tensors, files = safetensors.inspect_snapshot(src, index if index.exists() else None)
    p = files[0]
    infos = safetensors.validate_file(p)
    hdr, data_off = safetensors.read_header(p)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recipe = {"family": "SHQ-T16", "recipe_id": rid, "rules": rules,
              "scale_search": "imatrix-weighted-ls" if imatrix else "range"}
    plan = {"schema": "gufo.quant-plan.v1", "recipe": recipe,
            "source_manifest": str(Path(args.source) / ".."),
            "tensors": {}}
    plan_file = Path(args.plan)

    n_q = {"SHQ4-G64-U4Z": 0, "SHQ4-G32-U4Z": 0, "SHQ6-G64": 0, "SHQ8-G64": 0}
    for name in sorted(infos):
        dtype, shape, (b0, b1) = infos[name]
        if name == "__metadata__" or not name.endswith(".weight"):
            continue
        fmt = recipe_mod.resolve(name, rules)
        # Quantization is for linear projections only: 1D norms/bias stay BF16.
        if len(shape) != 2:
            fmt = "BF16"
        if fmt == "BF16":
            plan["tensors"][name] = {"format": "BF16", "shape": list(shape), "shard": "source-copy"}
            continue
        if dtype != "BF16":
            plan["tensors"][name] = {"format": "BF16", "shape": list(shape), "note": "source dtype not BF16"}
            continue
        data = read_bf16_tensor(p, infos[name], dtype, shape, data_off)
        imp = imatrix.get(name) if imatrix else None
        if imp is not None and imp.shape[0] != shape[1]:
            raise SystemExit(f"{name}: imatrix dim {imp.shape[0]} != K {shape[1]}")
        # SHQ6/SHQ8 have no imatrix search; SHQ4-G32/G64 use it when available.
        if fmt.startswith("SHQ6") or fmt.startswith("SHQ8"):
            imp = None
        planes, stats = quantize_one(name, shape, data, fmt, importance=imp)
        if imp is not None:
            stats["scale_search"] = "imatrix-weighted-ls"
            stats["imatrix_mean"] = float(imp.mean())
        # write shard
        shard = out / (name.replace(".", "/") + ".shq4")
        shard.parent.mkdir(parents=True, exist_ok=True)
        meta = {"name": name, "shape": shape, "format": planes["format"],
                 "Np": planes["Np"], "Kp": planes["Kp"],
                 "group_size": planes["group_size"],
                 "w_len": len(planes["weight"]),
                 "s_len": len(planes["scale"]),
                 "z_len": len(planes["zero"])}
        with shard.open("wb") as f:
            f.write(json.dumps(meta).encode("utf-8"))
            f.write(b"\n")
            f.write(planes["weight"])
            f.write(planes["scale"])
            f.write(planes["zero"])
        plan["tensors"][name] = {
            "format": planes["format"], "recipe_fmt": fmt,
            "shard": str(shard), "shape": list(shape), **stats,
        }
        n_q[fmt] += 1
        del data

    write_json(plan_file, plan)
    if args.json:
        print(json.dumps({"plan": str(plan_file), "recipe": rid,
                          "quantized": dict(n_q)}, indent=2))
    else:
        print(f"quantization plan: {plan_file}  (recipe '{rid}')")
        print(f"quantized per tier: " +
              ", ".join(f"{k}:{v}" for k, v in n_q.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
