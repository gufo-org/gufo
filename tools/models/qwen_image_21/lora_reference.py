#!/usr/bin/env python3
"""FP64 reference merge of a Qwen-Image-2.1 LoRA file, compared with a Gufo dump.

numpy only, offline. Reads the adapter and the base transformer safetensors
itself and applies the ComfyUI LoRA conventions (master
54a4e56ad1cabd17871068f160d0e75a1557db34, cited by path and line; no ComfyUI
code is copied) in float64, independently of Gufo's loader. DIR is the output
of `qwen_image_21_lora_dump MODEL_DIR DIR --lora NAME=FILE --scale S`. Prints
a JSON report and exits 1 when a gate fails:
  patched tensors:   max_ulp <= 1 vs bf16(R) except cancellation elements
                     (|R| << |W| or |BA| << sum |B_ik A_kj|), which must be
                     within 2**-8 BF16 ULP of the operand scale
                     |W| + sum |strength*alpha*B_ik*A_kj| (DoRA: times the row
                     factor; .diff: |R-W|); ulp_mismatch_frac <= 1e-3,
                     rel_l2 <= 1e-2, delta_energy > 0
  unpatched tensors: bit-identical to the base checkpoint
`--derive-strip OUT` instead writes a copy of FILE without its full
replacement `.weight` tensors (e.g. Fun-Acc PDD projections).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Suffix -> role. Layouts accepted by ComfyUI comfy/weight_adapter/lora.py
# L159-196 (lora_up/lora_down, lora_B/lora_A, lora.up/lora.down, bare
# lora_B/lora_A, lora_B.default) plus the bare lora_up/lora_down pair.
ROLES = (
    (".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"),
    (".lora_A.weight", "down"), (".lora_B.weight", "up"),
    (".lora_down.weight", "down"), (".lora_up.weight", "up"),
    (".lora.down.weight", "down"), (".lora.up.weight", "up"),
    (".lora_A", "down"), (".lora_B", "up"),
    (".lora_down", "down"), (".lora_up", "up"),
    (".alpha", "alpha"),          # comfy/lora.py L45-49
    (".dora_scale", "dora"),      # comfy/lora.py L51-55
    (".diff", "diff"),            # comfy/lora.py L76-80
)
PREFIXES = ("diffusion_model.transformer.", "base_model.model.transformer.",
            "base_model.model.", "diffusion_model.", "transformer.")
FUSED = ".img_mlp.gate_up"
EPS32 = float(np.finfo(np.float32).eps)  # comfy/weight_adapter/base.py L299
GATES = {"max_ulp": 1, "ulp_mismatch_frac": 1e-3, "rel_l2": 1e-2}
CANCEL_ULP = 2.0**-8  # 256 FP32 ULPs, in BF16 ULPs at the operand scale


class SafeTensors:
    """Minimal safetensors reader: BF16, F16 and F32 as float64."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(length))
        self.start = 8 + length
        self.metadata = header.pop("__metadata__", {}) or {}
        self.header = header
        self.data = np.memmap(self.path, dtype=np.uint8, mode="r")

    def raw(self, name):
        begin, end = self.header[name]["data_offsets"]
        return self.data[self.start + begin:self.start + end]

    def get(self, name):
        info = self.header[name]
        raw, dtype = self.raw(name), info["dtype"]
        if dtype == "BF16":
            bits = raw.view("<u2").astype(np.uint32) << 16
            values = bits.view(np.float32)
        elif dtype == "F16":
            values = raw.view("<f2")
        elif dtype == "F32":
            values = raw.view("<f4")
        else:
            raise ValueError(f"{self.path}: {name}: unsupported dtype {dtype}")
        return values.astype(np.float64).reshape(info["shape"])


class Base:
    """Base transformer tensors by Gufo name ("transformer." + checkpoint key)."""

    def __init__(self, model):
        root = Path(model) / "transformer"
        index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())
        self.files = {"transformer." + k: root / v for k, v in index["weight_map"].items()}
        self.shards = {}

    def __contains__(self, name):
        return name in self.files

    def get(self, name):
        path = self.files[name]
        if path not in self.shards:
            self.shards[path] = SafeTensors(path)
        return self.shards[path].get(name[len("transformer."):])


def module_targets(module, base):
    """Base tensors a module updates; gate_up is the fused [gate_layer; proj]."""
    # Row order: comfy/ldm/qwen_image21/model.py L59 and L67 (gate_up feeds
    # swiglu), comfy/ops.py L1020-1022 (gate, up = chunk(2)), unfused L68
    # silu(gate_layer) * proj, and comfy/lora.py L340-342 (halves 0 and H).
    if module.endswith(FUSED):
        stem = module[:-len(FUSED)] + ".img_mlp."
        names = ["transformer." + stem + "gate_layer.weight", "transformer." + stem + "proj.weight"]
    else:
        names = ["transformer." + module + ".weight"]
    return names if all(name in base for name in names) else None


def parse(lora, base):
    """Group adapter tensors by transformer module."""
    kohya = {}
    for name in base.files:
        if name.endswith(".weight"):
            module = name[len("transformer."):-len(".weight")]
            kohya[module.replace(".", "_")] = module
            if module.endswith(".img_mlp.gate_layer"):
                fused = module[:-len(".gate_layer")] + ".gate_up"
                kohya[fused.replace(".", "_")] = fused
    modules = {}
    for key in lora.header:
        role = next((r for s, r in ROLES if key.endswith(s)), None)
        if role is None:
            raise ValueError(f"{key}: not a LoRA/DoRA/diff tensor")
        module = key[:-len(next(s for s, r in ROLES if key.endswith(s)))]
        for prefix in PREFIXES:
            if module.startswith(prefix):
                module = module[len(prefix):]
                break
        if module.startswith("lora_unet_"):  # comfy/lora.py L196-205
            module = kohya.get(module[len("lora_unet_"):], module)
        if module_targets(module, base) is None:
            raise ValueError(f"{key}: no base tensor for module {module}")
        parts = modules.setdefault(module, {})
        if role in parts:
            raise ValueError(f"{key}: duplicate {role} for {module}")
        parts[role] = key
    return modules


def metadata_alpha(lora):
    """diffusers lora_adapter_metadata (LORA_ADAPTER_METADATA_KEY): (alpha, rslora)."""
    text = lora.metadata.get("lora_adapter_metadata")
    if not text:
        return None, False
    config = json.loads(text)
    pick = lambda key: config.get("transformer." + key, config.get(key))
    for key in ("alpha_pattern", "rank_pattern"):
        if pick(key):
            raise ValueError(f"metadata {key} is unsupported")
    if pick("use_dora"):
        raise ValueError("metadata use_dora is unsupported")
    return pick("lora_alpha"), bool(pick("use_rslora"))


def reference(module, parts, lora, base, strength, meta):
    """(targets, base W, reference R, term magnitudes) for one module, float64.

    The last item holds, per target, a function (rows, cols) -> sum of the
    magnitudes of the terms added to W at those elements (the BA dot product
    terms, times the DoRA row factor), or None for `.diff`.
    """
    terms = None
    targets = module_targets(module, base)
    blocks = [base.get(t) for t in targets]
    blocks = [b.reshape(b.shape[0], -1) for b in blocks]
    w = np.concatenate(blocks, axis=0)
    if "diff" in parts:
        # comfy/lora.py L529-534: weight += strength * diff (alpha unused).
        if {"down", "up", "alpha", "dora"} & parts.keys():
            raise ValueError(f"{module}: diff mixed with a low-rank pair")
        diff = lora.get(parts["diff"])
        if diff.size != w.size:
            raise ValueError(f"{module}: diff shape {diff.shape}")
        r = w + strength * diff.reshape(w.shape)
    else:
        if "down" not in parts or "up" not in parts:
            raise ValueError(f"{module}: incomplete pair {sorted(parts)}")
        down, up = lora.get(parts["down"]), lora.get(parts["up"])
        rank = down.shape[0]
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != rank or \
                up.shape[0] != w.shape[0] or down.shape[1] != w.shape[1]:
            raise ValueError(f"{module}: pair {up.shape} x {down.shape} vs {w.shape}")
        # comfy/weight_adapter/lora.py L248-251: alpha / rank, else 1.
        # Without a .alpha tensor, diffusers metadata alpha (rslora: sqrt).
        if "alpha" in parts:
            scale = float(lora.get(parts["alpha"]).reshape(-1)[0]) / rank
        elif meta[0] is not None:
            scale = meta[0] / (np.sqrt(rank) if meta[1] else rank)
        else:
            scale = 1.0
        delta = up @ down
        # Per element sum_k |B_ik A_kj|, only where compare() needs it.
        products = lambda i, j: np.einsum("nk,kn->n", np.abs(up[i]), np.abs(down[:, j]))
        if "dora" in parts:
            # comfy/weight_adapter/base.py L275-307, output axis: norms of the
            # weight before this patch, plus FP32 eps; strength blends.
            magnitude = lora.get(parts["dora"]).reshape(-1, 1)
            if magnitude.shape[0] != w.shape[0]:
                raise ValueError(f"{module}: input-axis dora_scale")
            calc = w + scale * delta
            norm = np.sqrt(np.sum(w * w, axis=1, keepdims=True)) + EPS32
            factor = magnitude / norm
            calc *= factor
            r = calc if strength == 1.0 else w + strength * (calc - w)
            blend = 0.0 if strength == 1.0 else 1.0
            terms = lambda i, j: abs(strength) * (np.abs(factor[i, 0]) * (
                np.abs(w[i, j]) + abs(scale) * products(i, j)) + blend * np.abs(w[i, j]))
        else:
            # comfy/weight_adapter/lora.py L283: weight += strength*alpha*BA.
            r = w + (strength * scale) * delta
            terms = lambda i, j: abs(strength * scale) * products(i, j)
    rows = np.cumsum([b.shape[0] for b in blocks])[:-1]
    offsets = [0, *rows]
    split = [None if terms is None else
             (lambda i, j, o=o: terms(i + o, j)) for o in offsets]
    return targets, np.split(w, rows), np.split(r, rows), split


def bf16_spacing(x):
    """BF16 ULP at |x| (8 significant bits, subnormal spacing 2**-133)."""
    _, exponent = np.frexp(x)
    return np.ldexp(1.0, np.maximum(exponent, -125) - 8)


def bf16_round(x):
    """Round float64 to the nearest BF16 value, ties to even (one rounding)."""
    spacing = bf16_spacing(x)
    return np.rint(x / spacing) * spacing


def bf16_order(x):
    """Integer order of BF16 values, so differences count ULPs."""
    bits = (x.astype(np.float32).view(np.uint32) >> 16).astype(np.int64)
    return np.where(bits & 0x8000, -(bits & 0x7FFF), bits)


def compare(g, w, r, patched, terms=None):
    rb = bf16_round(r)
    ulps = np.abs(bf16_order(g) - bf16_order(rb))
    norm = lambda x: float(np.sqrt(np.sum(x * x)))
    change = norm(r - w)
    # When W + delta cancels (|R| << |W|), or the BA dot product itself does
    # (|delta| << sum |B_ik A_kj|), any FP32 merge is many BF16 ULPs of R away
    # from the FP64 result. Such elements pass only if the error is within
    # CANCEL_ULP of a BF16 ULP at the operand scale: |W| plus the summed
    # magnitudes of the added terms (|R-W| for `.diff`).
    far = ulps > GATES["max_ulp"]
    rows, cols = np.nonzero(far)
    added = np.abs(r[far] - w[far]) if terms is None else terms(rows, cols)
    operand = np.abs(w[far]) + added
    excess = np.abs(g[far] - r[far]) / bf16_spacing(operand)
    stats = {
        "patched": patched,
        "ulp_mismatch_frac": float(np.mean(ulps != 0)),
        "max_ulp": int(ulps.max()),
        "cancellation_elements": int(far.sum()),
        "max_cancellation_error": float(excess.max(initial=0)),
        "rel_l2": norm(g - r) / max(norm(r), 1e-300),
        "delta_rel_l2": norm(g - r) / change if change else None,
        "delta_energy": change / max(norm(w), 1e-300),
    }
    if patched:
        ok = (stats["max_cancellation_error"] <= CANCEL_ULP and
              stats["ulp_mismatch_frac"] <= GATES["ulp_mismatch_frac"] and
              stats["rel_l2"] <= GATES["rel_l2"] and stats["delta_energy"] > 0)
    else:
        ok = bool(np.array_equal(g, w))
    stats["pass"] = ok
    return stats


def derive_strip(path, output):
    """Copy a safetensors file without plain `.weight` tensors."""
    lora = SafeTensors(path)
    keep, dropped = {}, []
    offset = 0
    for name, info in sorted(lora.header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        if name.endswith(".weight") and not any(name.endswith(s) for s, _ in ROLES):
            dropped.append(name)
            continue
        size = info["data_offsets"][1] - info["data_offsets"][0]
        keep[name] = dict(info, data_offsets=[offset, offset + size])
        offset += size
    header = dict(keep)
    if lora.metadata:
        header["__metadata__"] = lora.metadata
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(output, "wb") as stream:
        stream.write(len(text).to_bytes(8, "little"))
        stream.write(text)
        for name in keep:
            stream.write(lora.raw(name).tobytes())
    json.dump({"output": str(output), "dropped": dropped, "kept": len(keep)}, sys.stdout, indent=1)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dump", type=Path, help="qwen_image_21_lora_dump output directory")
    parser.add_argument("--derive-strip", type=Path, metavar="OUT")
    args = parser.parse_args()
    if args.derive_strip:
        derive_strip(args.lora, args.derive_strip)
        return 0
    if not args.model or not args.dump:
        parser.error("--model and --dump are required unless --derive-strip is given")
    base = Base(args.model)
    lora = SafeTensors(args.lora)
    modules = parse(lora, base)
    meta = metadata_alpha(lora)
    index = json.loads((args.dump / "index.json").read_text())
    digest = hashlib.sha256()
    with open(args.lora, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    report = {"lora": str(args.lora), "sha256": digest.hexdigest(), "scale": args.scale,
              "modules": len(modules), "tensors": {},
              "gates": dict(GATES, cancellation_error=CANCEL_ULP)}

    def check(target, w, r, patched, terms=None):
        rows, cols = index[target]
        g = np.fromfile(args.dump / f"{target}.f32", dtype="<f4").astype(np.float64)
        stats = compare(g.reshape(rows, cols), w.reshape(rows, cols), r.reshape(rows, cols),
                        patched, terms)
        report["tensors"][target] = stats
        print(f"{'PASS' if stats['pass'] else 'FAIL'} {target} max_ulp={stats['max_ulp']} "
              f"cancel={stats['cancellation_elements']} "
              f"mismatch={stats['ulp_mismatch_frac']:.2e} rel_l2={stats['rel_l2']:.2e}",
              file=sys.stderr)

    referenced = set()
    for module, parts in sorted(modules.items()):
        names = module_targets(module, base)
        referenced.update(names)
        if not any(name in index for name in names):
            continue
        targets, ws, rs, terms = reference(module, parts, lora, base, args.scale, meta)
        for target, w, r, t in zip(targets, ws, rs, terms):
            if target in index:
                check(target, w, r, True, t)
    for target in sorted(set(index) - referenced):
        if target not in base:
            raise ValueError(f"dump has unknown tensor {target}")
        w = base.get(target)
        check(target, w, w, False)
    tensors = report["tensors"].values()
    patched = [s for s in tensors if s["patched"]]
    report["not_dumped"] = sorted(referenced - set(index))
    report["worst"] = {
        "max_ulp": max((s["max_ulp"] for s in patched), default=0),
        "max_cancellation_error": max((s["max_cancellation_error"] for s in patched), default=0),
        "ulp_mismatch_frac": max((s["ulp_mismatch_frac"] for s in patched), default=0),
        "rel_l2": max((s["rel_l2"] for s in patched), default=0),
        "min_delta_energy": min((s["delta_energy"] for s in patched), default=0),
    }
    report["compared"] = {"patched": len(patched), "unpatched": len(tensors) - len(patched)}
    report["pass"] = bool(tensors) and all(s["pass"] for s in tensors)
    json.dump(report, sys.stdout, indent=1)
    print()
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
