#!/usr/bin/env python3
"""gufo-gguf — inspect and dequant-q4 GGUF model files.

Offline comparison tool for the Qwen3.5 SHQ4 benchmark (docs/BENCHMARKS.md,
benchmarks/qwen3.5-0.8b/README.md). It reads a llama.cpp GGUF file and lets
you answer, for an unsloth quantization of the same model:

  * What ggml quant type does each tensor use, at what bits/weight (bpw)?
  * Does the file carry the MTP head (mtp tensors)?
  * Per-component model card (embed / linear-attn / full-attn / mlp / mtp / norm).
  * Dequant of Q4-family tensors back to f32/bf16, so their reconstruction
    error vs the bf16 source can be scored the same way gufo-quantize.py
    scores our SHQ4 tensors.

Minimal by design: header + KV metadata + tensor info + a small dequant set
(the Q4 family present in unsloth Qwen Q4_K_M / Q4_0 files: F16/BF16/F32,
Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, and the K-family Q4_K/Q5_K/Q6_K). Other ggml types
(IQ/UD dynamic, legacy Q2_K/Q3_K) are reported but not dequantized — add a
codec when a specific comparison needs it.

Usage:
  gufo-gguf --gguf FILE [--card] [--types] [--tensors]
  gufo-gguf --gguf FILE --dequant NAME                # print one tensor stats
  gufo-gguf --gguf FILE --dequant-dir DIR --max N     # dump first N tensors to .npy
  gufo-gguf --gguf FILE --recon --bf16-source SRC_MANIFEST_DIR \
             [--plan artifacts/work/quantization-plan.json]   # per-tensor error
"""
from __future__ import annotations

import argparse
import json
import mmap
import numpy as np
import os
from pathlib import Path
import re
import struct
import sys
import zlib

# --------------------------------------------------------------------------
# GGML value types (GGUF metadata)
# --------------------------------------------------------------------------
GGUF_TYPE = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
    6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64",
    12: "float64",
}

# --------------------------------------------------------------------------
# GGML tensor types (ggml.h enum). bpw + block codec per type.
# --------------------------------------------------------------------------
# bytes per name -> packing bytes
GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4",
    1000: "GUFO_SHQ4_T16", 1001: "GUFO_SHQ6_T16", 1002: "GUFO_SHQ8_T16",
}

# type -> (block_size, type_size_bytes)
GGML_TYPE_BLOCK = {
    0: (1, 4),     # f32
    1: (1, 2),     # f16
    2: (32, 18),   # q4_0: d + qs
    3: (32, 20),   # q4_1: d+m + qs
    6: (32, 22),   # q5_0
    7: (32, 24),   # q5_1
    8: (32, 34),   # q8_0
    10: (256, 2 + 2 + 16 + 64),  # q2_K
    11: (256, 2 + 64 + 32 + 12),  # q3_K
    12: (256, 4 + 12 + 128),   # q4_K  (d:2,dmin:2,scales:12,qs:128)
    13: (256, 4 + 12 + 32 + 128),  # q5_K
    14: (256, 2 + 128 + 64 + 16),  # q6_K
    15: (256, 4 + 256 + 32),    # q8_K
    16: (256, 2 + 64),   # iq2_xxs: d + qs[QK_K/8] as uint16
    20: (32, 18),   # iq4_nl
    21: (256, 2 + 64 + 8 + 32 + 4),  # iq3_s: d,qs,qh,signs,scales
    23: (256, 2 + 2 + 4 + 128),  # iq4_xs
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    30: (1, 2),    # bf16
    34: (256, 2 + 4 + (256 - 4 * 4) // 5),  # tq1_0
    35: (256, 2 + 64),   # tq2_0
}

# --------------------------------------------------------------------------
# Q4-family dequant codecs (llama.cpp ggml-quants.c, reference ports).
# QK_K = 256 super-block.
# --------------------------------------------------------------------------
QK_K = 256


def _fp16_to_fp32(u):
    # GGUF is written with native (host) endianness; x86 = little-endian.
    return np.frombuffer(u, dtype="<u2").astype(np.uint16).view("float16").astype(np.float32)


def _bf16_to_fp32(u):
    a = np.frombuffer(u, dtype="<u2").astype(np.uint32) << 16
    return a.view("float32")


def _dequant_f16(f):
    return _fp16_to_fp32(f)


def _dequant_bf16(f):
    return _bf16_to_fp32(f)


def _dequant_f32(f):
    return np.frombuffer(f, dtype="<f4").copy()


def _dequant_q4_0(f):
    nblk = len(f) // 18
    b = f.reshape(nblk, 18)
    d = _fp16_to_fp32(b[:, 0:2].reshape(-1))  # (nblk,)
    qs = b[:, 2:].reshape(nblk, 16)
    lo = (qs & 0x0F).astype(np.float32) - 8
    hi = (qs >> 4).astype(np.float32) - 8
    out = np.empty((nblk, 32), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None]
    out[:, 16:32] = hi * d[:, None]
    return out.reshape(-1)


def _dequant_q8_0(f):
    nblk = len(f) // 34
    b = f.reshape(nblk, 34)
    d = _fp16_to_fp32(b[:, 0:2].reshape(-1))
    qs = np.frombuffer(b[:, 2:].reshape(-1), dtype=np.int8).reshape(nblk, 32).astype(np.float32)
    return (qs * d[:, None]).reshape(-1)


def _dequant_q4_K(f):
    # block_q4_K: d(2) dmin(2) scales(12) qs(128)
    nblk = len(f) // 144
    b = f.reshape(nblk, 144)
    d = _fp16_to_fp32(b[:, 0:2].reshape(-1))
    dmin = _fp16_to_fp32(b[:, 2:4].reshape(-1))
    scales = b[:, 4:16]              # (nblk,12)
    qs = b[:, 16:144]                # (nblk,128)
    # 8 per-32-block (d,m) 6-bit pairs
    d8 = np.empty((nblk, 8), dtype=np.int32)
    m8 = np.empty((nblk, 8), dtype=np.int32)
    for j in range(4):
        d8[:, j] = scales[:, j] & 63
        m8[:, j] = scales[:, j + 4] & 63
    for j in range(4, 8):
        d8[:, j] = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        m8[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    out = np.empty((nblk, QK_K), dtype=np.float32)
    for k in range(4):
        d1 = d * d8[:, 2 * k].astype(np.float32); m1 = dmin * m8[:, 2 * k].astype(np.float32)
        d2 = d * d8[:, 2 * k + 1].astype(np.float32); m2 = dmin * m8[:, 2 * k + 1].astype(np.float32)
        q = qs[:, k * 32:(k + 1) * 32]
        lo = (q & 0x0F).astype(np.float32)
        hi = (q >> 4).astype(np.float32)
        out[:, k * 64:k * 64 + 32] = lo * d1[:, None] - m1[:, None]
        out[:, k * 64 + 32:k * 64 + 64] = hi * d2[:, None] - m2[:, None]
    return out.reshape(-1)


def _dequant_q5_K(f):
    # block_q5_K: d(2) dmin(2) scales(12) qh(32) qs(128)
    nblk = len(f) // 176
    b = f.reshape(nblk, 176)
    d = _fp16_to_fp32(b[:, 0:2].reshape(-1))
    dmin = _fp16_to_fp32(b[:, 2:4].reshape(-1))
    scales = b[:, 4:16]
    qh = b[:, 16:48]
    qs = b[:, 48:176]
    d8 = np.empty((nblk, 8), dtype=np.int32)
    m8 = np.empty((nblk, 8), dtype=np.int32)
    for j in range(4):
        d8[:, j] = scales[:, j] & 63
        m8[:, j] = scales[:, j + 4] & 63
    for j in range(4, 8):
        d8[:, j] = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        m8[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    out = np.empty((nblk, QK_K), dtype=np.float32)
    for k in range(4):
        d1 = d * d8[:, 2 * k].astype(np.float32); m1 = dmin * m8[:, 2 * k].astype(np.float32)
        d2 = d * d8[:, 2 * k + 1].astype(np.float32); m2 = dmin * m8[:, 2 * k + 1].astype(np.float32)
        q = qs[:, k * 32:(k + 1) * 32]
        h = qh[:, 0:32]
        b1 = ((h & (1 << (2 * k))) != 0).astype(np.float32) * 16
        b2 = ((h & (2 << (2 * k))) != 0).astype(np.float32) * 16
        lo = (q & 0x0F).astype(np.float32) + b1
        hi = (q >> 4).astype(np.float32) + b2
        out[:, k * 64:k * 64 + 32] = lo * d1[:, None] - m1[:, None]
        out[:, k * 64 + 32:k * 64 + 64] = hi * d2[:, None] - m2[:, None]
    return out.reshape(-1)


def _dequant_q6_K(f):
    # block_q6_K: ql(128) qh(64) scales(16) d(2)
    nblk = len(f) // 210
    b = f.reshape(nblk, 210)
    d = _fp16_to_fp32(b[:, 208:210].reshape(-1))
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    sc = np.frombuffer(b[:, 192:208].reshape(-1), dtype=np.int8).reshape(nblk, 16).astype(np.int32)
    out = np.empty((nblk, QK_K), dtype=np.float32)
    for g in range(2):  # two 128-lane halves
        qlg = ql[:, g * 64:(g + 1) * 64]
        qhg = qh[:, g * 32:(g + 1) * 32]
        scg = sc[:, g * 8:(g + 1) * 8]
        sel = (np.arange(32) // 16)
        q1 = ((qlg[:, 0:32] & 0x0F) | ((qhg & 0x03) << 4)).astype(np.int32) - 32
        q2 = ((qlg[:, 32:64] & 0x0F) | ((qhg >> 2 & 0x03) << 4)).astype(np.int32) - 32
        q3 = ((qlg[:, 0:32] >> 4) | ((qhg >> 4 & 0x03) << 4)).astype(np.int32) - 32
        q4 = ((qlg[:, 32:64] >> 4) | ((qhg >> 6 & 0x03) << 4)).astype(np.int32) - 32
        s1 = scg[:, sel + 0].astype(np.float32)
        s2 = scg[:, sel + 2].astype(np.float32)
        s3 = scg[:, sel + 4].astype(np.float32)
        s4 = scg[:, sel + 6].astype(np.float32)
        base = g * 128
        out[:, base + 0:base + 32] = d[:, None] * s1 * q1
        out[:, base + 32:base + 64] = d[:, None] * s2 * q2
        out[:, base + 64:base + 96] = d[:, None] * s3 * q3
        out[:, base + 96:base + 128] = d[:, None] * s4 * q4
    return out.reshape(-1)


DEQUANT = {
    0: _dequant_f32,
    1: _dequant_f16,
    2: _dequant_q4_0,
    8: _dequant_q8_0,
    12: _dequant_q4_K,
    13: _dequant_q5_K,
    14: _dequant_q6_K,
    30: _dequant_bf16,
}


def bpw(t):
    bs, ts = GGML_TYPE_BLOCK[t]
    return ts * 8 / bs


# --------------------------------------------------------------------------
# GGUF binary reader
# --------------------------------------------------------------------------
class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def u8(self): v = self.data[self.pos]; self.pos += 1; return v
    def i8(self): return struct.unpack_from("<b", self.data, self.pos)[0] if self._take(1) else 0
    def u16(self): v = struct.unpack_from("<H", self.data, self.pos)[0]; self.pos += 2; return v
    def i16(self): v = struct.unpack_from("<h", self.data, self.pos)[0]; self.pos += 2; return v
    def u32(self): v = struct.unpack_from("<I", self.data, self.pos)[0]; self.pos += 4; return v
    def i32(self): v = struct.unpack_from("<i", self.data, self.pos)[0]; self.pos += 4; return v
    def u64(self): v = struct.unpack_from("<Q", self.data, self.pos)[0]; self.pos += 8; return v
    def i64(self): v = struct.unpack_from("<q", self.data, self.pos)[0]; self.pos += 8; return v
    def f32(self): v = struct.unpack_from("<f", self.data, self.pos)[0]; self.pos += 4; return v
    def f64(self): v = struct.unpack_from("<d", self.data, self.pos)[0]; self.pos += 8; return v

    def _take(self, n):
        assert self.pos + n <= len(self.data), f"overrun @{self.pos}+{n}"
        return self.pos + n <= len(self.data)

    def string(self):
        ln = self.u64()
        s = self.data[self.pos:self.pos + ln]
        self.pos += ln
        return s.decode("utf-8", "replace")

    def read_value(self, tag):
        if tag == 0: return self.u8()
        if tag == 1: return self.i8()
        if tag == 2: return self.u16()
        if tag == 3: return self.i16()
        if tag == 4: return self.u32()
        if tag == 5: return self.i32()
        if tag == 6: return self.f32()
        if tag == 7: return bool(self.u8())
        if tag == 8: return self.string()
        if tag == 9:
            atype = self.u32()
            nelem = self.u64()
            return [self.read_value(atype) for _ in range(nelem)]
        if tag == 10: return self.u64()
        if tag == 11: return self.i64()
        if tag == 12: return self.f64()
        raise ValueError(f"unknown gguf value tag {tag}")

    def skip_value(self, tag):
        if tag == 9:
            atype = self.u32()
            nelem = self.u64()
            for _ in range(nelem):
                self.skip_value(atype)
            return
        if tag == 8:
            ln = self.u64()
            self.pos += ln
            return
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        self.pos += sizes[tag]
        if self.pos > len(self.data):
            raise ValueError("value overruns buffer")


def parse_gguf(path):
    with Path(path).open("rb") as source:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return _parse_gguf_data(data)


def _parse_gguf_data(data):
    assert data[0:4] == b"GGUF", "not a GGUF file"
    r = Reader(data)
    r.pos = 4
    ver = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()

    metadata = {}
    user_kv = {}
    for _ in range(n_kv):
        key = r.string()
        tag = r.u32()
        if key.startswith("tokenizer."):
            # vocab arrays are huge and unneeded for inspection; skip without storing.
            r.skip_value(tag)
            continue
        val = r.read_value(tag)
        if r.pos > 262144:
            pass  # ok, bigger metadata is fine if the file is fully read
        if key.startswith("general."):
            metadata[key] = val
        else:
            user_kv[key] = val

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        # dims are packed as u64 (ggml ne[] are int64); file order = innermost first.
        ne = tuple(r.u64() for _ in range(n_dims))
        ttype = r.u32()
        offset = r.u64()
        tensors.append({
            "name": name, "shape": list(reversed(ne)), "type": ttype,
            "type_name": GGML_TYPE_NAME.get(ttype, f"T{ttype}"),
            "offset": offset,
        })

    # data section is aligned per general.alignment (writer pads after TI data)
    alignment = metadata.get("general.alignment", 32)
    data_offset = (r.pos + alignment - 1) // alignment * alignment
    return {
        "version": ver, "n_tensors": n_tensors, "metadata": metadata,
        "user_kv": user_kv, "tensors": tensors, "data_offset": data_offset,
        "file_size": len(data),
    }


def tensor_bytes(state, t):
    bs, ts = GGML_TYPE_BLOCK.get(t["type"], (None, None))
    if bs is None:
        return None
    ne = 1
    for d in t["shape"]:
        ne *= d
    nblk = ne // bs
    return nblk * ts


# --------------------------------------------------------------------------
# Qwen3.5 component classifier (matches safetensors layout)
# --------------------------------------------------------------------------
def component_of(name):
    n = name
    if "token_embd" in n or "embed_tokens" in n:
        return "embed"
    if "mtp" in n:
        return "mtp"
    if "vision" in n:
        return "vision"
    if "attn_norm" in n or "post_attention_norm" in n or "q_norm" in n or "k_norm" in n \
       or "ssm_norm" in n or "input_layer" in n or "output_norm" in n:
        return "norm"
    if "ffn" in n or ".mlp." in n:
        return "mlp"
    if "linear_attn" in n or "ssm" in n or "in_proj" in n or "out_proj" in n:
        return "linear_attn"
    if "attn_q" in n or "attn_k" in n or "attn_v" in n or "attn_output" in n \
       or ".self_attn." in n or ".q_proj" in n:
        return "full_attn"
    return "other"


def build_card(state):
    """Per-component model card: tenrsor count, params, quantized bytes, bpw avg."""
    comps = {}
    for t in state["tensors"]:
        c = component_of(t["name"])
        e = comps.setdefault(c, {"tensors": 0, "params": 0, "qbytes": 0, "bf16bytes": 0, "types": {}, "bpwm": []})
        e["tensors"] += 1
        ne = 1
        for d in t["shape"]:
            ne *= d
        e["params"] += ne
        tb = tensor_bytes(state, t)
        if tb is None:
            e["types"].setdefault(t["type_name"], {"count": 0, "bpw": None})
            e["types"][t["type_name"]]["count"] += 1
            continue
        e["qbytes"] += tb
        e["bf16bytes"] += ne * 2
        e["types"].setdefault(t["type_name"], {"count": 0, "bpw": bpw(t["type"])})
        e["types"][t["type_name"]]["count"] += 1
        e["bpwm"].append(bpw(t["type"]))

    for c, e in comps.items():
        e["avg_bpw"] = round(float(np.mean(e["bpwm"])), 3) if e["bpwm"] else None

    has_mtp = "mtp" in comps
    return comps, has_mtp


def gguf_name_to_st(name):
    """Map a Qwen3.5 GGUF tensor name onto the corresponding safetensors key.

    llama.cpp drops the vision encoder and (for qwen35) the MTP head; only the
    language_model body is shared. Returns None for tensors with no 1:1 twin
    (e.g. packed attn_qkv, ssm_a, ssm_dt.bias, attn_gate).
    """
    if name == "token_embd.weight":
        return "model.language_model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.language_model.norm.weight"
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return None
    n, sub = m.group(1), m.group(2)
    base = f"model.language_model.layers.{n}"
    tbl = {
        "attn_norm": "input_layernorm",
        "post_attention_norm": "post_attention_layernorm",
        "attn_q_norm": "self_attn.q_norm",
        "attn_k_norm": "self_attn.k_norm",
        "ssm_norm": "linear_attn.norm",
        "ssm_out": "linear_attn.out_proj",
        "ssm_beta": "linear_attn.in_proj_b",
        "ssm_alpha": "linear_attn.in_proj_a",
        "ssm_conv1d": "linear_attn.conv1d",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    stem = sub[:-7] if sub.endswith(".weight") else sub
    if stem not in tbl:
        return None
    return f"{base}.{tbl[stem]}.weight"


# --------------------------------------------------------------------------
# Reconstruction error vs bf16 source (safetensors): per tensor + per component.
# --------------------------------------------------------------------------
def recon_compare(state, gguf_path, src_dir, plan=None, max_tensors=None):
    """Dequant GGUF tensors, compare to bf16 safetensors, and cross-reference our
    SHQ4 per-tensor stats from the quantization plan.

    Reports every tensor with a 1:1 safetensors twin (all layers, both quantized
    and f32-kept), so each component shows how much information the unsloth GGUF
    retained vs bf16, side-by-side with our own SHQ4 retention. Returns rows.
    """
    from safetensors import safe_open
    st_files = sorted(Path(src_dir).glob("*.safetensors"))
    handles = [safe_open(str(fp), framework="pt", device="cpu") for fp in st_files]

    def st_get(key):
        for h in handles:
            if key in h.keys():
                return h.get_tensor(key).float().numpy()
        return None

    fdata = np.memmap(gguf_path, mode="r", dtype=np.uint8)
    base = state["data_offset"]
    rows = []
    checked = 0
    QUANT = {2, 8, 12, 13, 14}  # Q4_0, Q8_0, Q4_K, Q5_K, Q6_K
    for t in state["tensors"]:
        key = gguf_name_to_st(t["name"])
        layer = "final" if "output_norm" in t["name"] \
                else ("embd" if "token_embd" in t["name"]
                      else int(re.match(r"blk\.(\d+)", t["name"]).group(1)))
        rows.append({"name": t["name"], "st_key": key, "gguf_type": t["type_name"],
                     "type": t["type"], "bpw": bpw(t["type"]), "shape": t["shape"],
                     "layer": layer})
        if key is None:
            rows[-1].update({"note": "no 1:1 safetensors twin (packed/ssm-small), skipped"})
            continue
        ref = st_get(key)
        if ref is None:
            rows[-1].update({"note": "safetensors key absent"})
            continue
        c = component_of(key)
        rows[-1]["component"] = c
        refm = np.abs(ref).mean()
        # our SHQ4 retention for this weighted key (mean_abs_err / mean|w|)
        pinfo = (plan or {}).get(key, {})
        if pinfo.get("format") == "SHQ4_T16_V1_U4Z_G64":
            rows[-1]["our_rel"] = float(pinfo["mean_abs_err"] / max(refm, 1e-12))
            rows[-1]["our_mae"] = float(pinfo["mean_abs_err"])
        if t["type"] not in DEQUANT:
            rows[-1].update({"quantized": False, "note": f"no codec ({t['type_name']}); unmapped"})
            continue
        tb = tensor_bytes(state, t)
        blob = np.frombuffer(fdata[base + t["offset"]:base + t["offset"] + tb], dtype=np.uint8)
        deq = DEQUANT[t["type"]](blob).reshape(t["shape"]).astype(np.float32)
        reff = ref.reshape(t["shape"])
        if deq.shape != reff.shape:
            rows[-1].update({"note": f"shape mismatch {tuple(deq.shape)}"})
            continue
        err = (reff - deq).reshape(-1)
        rows[-1]["quantized"] = t["type"] in QUANT
        rows[-1]["mae"] = float(np.abs(err).mean())
        rows[-1]["rmse"] = float(np.sqrt((err ** 2).mean()))
        rows[-1]["max_abs_err"] = float(np.abs(err).max())
        if t["type"] in QUANT:
            rows[-1]["rel_mae"] = float(rows[-1]["mae"] / max(refm, 1e-12))
        else:
            # unquantized (f16/bf16/f32): subtract any constant parameterization
            # offset (Qwen3.5 norms are stored as w+1 in gguf) before scoring.
            resid = (err - err.mean()).reshape(-1)
            rows[-1]["rel_mae"] = float(np.abs(resid).mean() / max(refm, 1e-12))
            rows[-1]["retained"] = "exact" if float(np.abs(resid).mean()) < 1e-6 else "close"
        rows[-1]["n_params"] = len(err)
        checked += 1
        if max_tensors and checked >= max_tensors:
            break
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="gufo-gguf")
    ap.add_argument("--gguf", required=True, help="path to .gguf file")
    ap.add_argument("--card", action="store_true", help="print per-component model card")
    ap.add_argument("--types", action="store_true", help="list ggml types in file")
    ap.add_argument("--tensors", action="store_true", help="list all tensors")
    ap.add_argument("--dequant", metavar="NAME", help="dequant one tensor, print stats")
    ap.add_argument("--dequant-dir", metavar="DIR", help="dequant N tensors to .npy")
    ap.add_argument("--max", type=int, default=8, help="max tensors for --dequant-dir")
    ap.add_argument("--recon", action="store_true", help="reconstruction vs bf16 source")
    ap.add_argument("--bf16-source", metavar="DIR", help="bf16 safetensors source dir")
    ap.add_argument("--plan", metavar="JSON", help="our quantization-plan.json for SHQ4 stats")
    ap.add_argument("--out", metavar="JSON", help="write JSON result")
    args = ap.parse_args(argv)

    state = parse_gguf(args.gguf)

    print(f"GGUF v{state['version']}  {state['file_size']/1e6:.1f} MB")
    print(f"tensors: {state['n_tensors']}")

    has_mtp = any("mtp" in t["name"] for t in state["tensors"])
    print(f"MTP tensors present: {has_mtp}")

    if args.types:
        from collections import Counter
        c = Counter((t["type_name"], t["type"]) for t in state["tensors"])
        print("\n== ggml types ==")
        total_b = 0
        for (name, tid), cnt in sorted(c.items(), key=lambda x: -x[1]):
            bp = bpw(tid)
            nb = 0
            for t in state["tensors"]:
                if t["type"] == tid:
                    tb = tensor_bytes(state, t)
                    nb += tb if tb else 0
            total_b += nb
            print(f"  {name:8s} n={cnt:5d}  bpw={bp:.3f}  bytes={nb/1e6:8.1f} MB")
        print(f"  total quantized bytes {total_b/1e6:.1f} MB")

    if args.tensors:
        print("\n== tensors ==")
        for t in state["tensors"]:
            tb = tensor_bytes(state, t)
            sz = f"{tb/1e6:.2f}MB" if tb else "-"
            print(f"  {t['type_name']:8s} {str(t['shape']):24s} {sz:8s}  {t['name']}")

    if args.card:
        comps, has_mtp2 = build_card(state)
        print("\n== model card (per component) ==")
        hdr = f"{'component':14s} {'tensors':>8s} {'params(M)':>9s} {'qbytes(MB)':>9s} {'avg_bpw':>8s}"
        print(hdr)
        for c, e in sorted(comps.items()):
            print(f"{c:14s} {e['tensors']:8d} {e['params']/1e6:9.2f} {e['qbytes']/1e6:9.1f} "
                  f"{str(e['avg_bpw']):>8s}")

    if args.dequant:
        t = next((x for x in state["tensors"] if x["name"] == args.dequant), None)
        if t is None:
            print(f"tensor '{args.dequant}' not found"); return 1
        if t["type"] not in DEQUANT:
            print(f"no codec for {t['type_name']}"); return 1
        tb = tensor_bytes(state, t)
        fdata = np.memmap(args.gguf, mode="r", dtype=np.uint8)
        base = state["data_offset"]
        blob = np.frombuffer(fdata[base + t["offset"]:base + t["offset"] + tb], dtype=np.uint8)
        deq = DEQUANT[t["type"]](blob)
        print(f"\n{args.dequant}: {t['type_name']} shape={t['shape']} -> dequant {deq.shape}")
        print(f"  min={deq.min():.5f} max={deq.max():.5f} rms={np.sqrt((deq**2).mean()):.5f}")

    if args.recon:
        if not args.bf16_source:
            print("--recon requires --bf16-source"); return 1
        plan = json.loads(Path(args.plan).read_text("utf-8")).get("tensors", {}) if args.plan else {}
        rows = recon_compare(state, args.gguf, args.bf16_source, plan=plan)
        scored = [r for r in rows if "rel_mae" in r]  # has a 1:1 safetensors twin

        print("\n== per-component retention (mean rel_mae vs bf16 source) ==")
        print(f"{'component':12s} {'n':>3s} {'gguf bpw':>9s} {'unsloth rel':>11s} {'our SHQ4 rel':>12s}")
        comps = {}
        for r in scored:
            comps.setdefault(r["component"], []).append(r)
        for c, lst in sorted(comps.items()):
            n = len(lst)
            bp = sum(x["bpw"] for x in lst) / n
            rel = sum(x["rel_mae"] for x in lst) / n
            our = [x["our_rel"] for x in lst if "our_rel" in x]
            ours = f"{sum(our)/len(our):.4f}" if our else "(kept bf16)"
            print(f"{c:12s} {n:3d} {bp:9.2f} {rel:11.4f} {ours:>12s}")

        print("\n== per-layer, per-tensor retention (unsloth GGUF vs our SHQ4) ==")
        hdr = f"{'layer':>6s} {'tensor':52s} {'gguf':>6s} {'bpw':>5s} {'unsloth%':>8s} {'ours%':>7s}"
        print(hdr)
        for r in scored:
            ours = f"{r['our_rel']*100:.2f}" if "our_rel" in r else "kept"
            print(f"{str(r['layer']):>6s} {r['name'][:52]:52s} {r['gguf_type']:>6s} "
                  f"{r['bpw']:5.2f} {r['rel_mae']*100:8.2f} {ours:>7s}")
        print("\nnote: 'kept' = our SHQ4 keeps that tensor bf16; unquantized f32 gguf rows"
              " score after removing the constant Qwen3.5 norm offset.")

    if args.out:
        comps, has_mtp2 = build_card(state)
        out = {
            "schema": "gufo.gguf.v1",
            "file": str(args.gguf),
            "version": state["version"],
            "n_tensors": state["n_tensors"],
            "file_size": state["file_size"],
            "metadata": state["metadata"],
            "general_arch": state["metadata"].get("general.architecture"),
            "has_mtp": has_mtp,
            "components": {k: {kk: v if kk != "types" else {x: yy for x, yy in v.items()}
                               for kk, v in e.items() if kk != "bpwm"}
                           for k, e in comps.items()},
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
