"""SHQ-T16 mixed-precision recipes.

A recipe is an ordered list of rules mapping safetensors weight names to an
SHQ-T16 encoding. First match wins; a default governs the bulk eligible set
and BF16 the rest. Encodings:

  SHQ4-G64-U4Z   4.5  bpw  bulk linear tensors
  SHQ4-G32-U4Z   4.625bpw  sensitive attention / output tensors
  SHQ6-G64       6.56 bpw  Q6_K-class tier (embed, ffn_down, linear-attn)
  SHQ8-G64       8.25 bpw  high-precision tier (embed, ffn_down, linear-attn)
  BF16           16   bpw  norms, conv1d, small projections, vision

Per-tensor precision (never per-block) keeps hot kernels branch-free; SHQ6 and
SHQ8 share the SHQ4 T16 tile layout, so the decode GEMV kernel structure is
identical (only per-weight read width differs), which is what makes the tiers
hardware-safe on gfx1151 / XDNA2. SHQ6 is the Q6_K-class tier: 6-bit codes
packed 4-per-3-bytes, expanded to INT8 in the kernel (storage != compute).

Presets mirror the unsloth Q4_K_M recipe (Q6_K embed, Q6_K ffn_down, Q5_K/Q8_0
linear-attn, Q4_K attention). SHQ6 replaces the Q6_K choices; SHQ8 is the
higher-precision fallback tier.
"""

from __future__ import annotations

# Ordered rules: (substring, encoding). First match wins.
# Default bulk: eligible linear projections stay SHQ4-G64-U4Z unless a rule
# upcasts them.

EMBED = "embed_tokens.weight"
FFN_DOWN = "mlp.down_proj.weight"
MLP_UP = "mlp.up_proj.weight"
MLP_GATE = "mlp.gate_proj.weight"
LIN_QKV = "linear_attn.in_proj_qkv.weight"
LIN_Z = "linear_attn.in_proj_z.weight"
LIN_OUT = "linear_attn.out_proj.weight"
ATTN = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight")

PRESETS = {
    # baseline: uniform SHQ4 G64 on all eligible, embed/norms bf16
    "bulk_g64": [],
    # unsloth-mirror: upcast embed + ffn_down + linear-attn proj to SHQ8,
    # attention to G32; mlp gate/up stay SHQ4 G64
    "unsloth_mirror": [
        (EMBED, "SHQ8-G64"),
        (FFN_DOWN, "SHQ8-G64"),
        (LIN_QKV, "SHQ8-G64"),
        (LIN_Z, "SHQ8-G64"),
        (LIN_OUT, "SHQ8-G64"),
        (ATTN[0], "SHQ4-G32-U4Z"),
        (ATTN[1], "SHQ4-G32-U4Z"),
        (ATTN[2], "SHQ4-G32-U4Z"),
        (ATTN[3], "SHQ4-G32-U4Z"),
    ],
    # upcast embed + ffn_down + attention-G32; linear-attn stays SHQ4 G64
    "mirror_no_lin": [
        (EMBED, "SHQ8-G64"),
        (FFN_DOWN, "SHQ8-G64"),
        (ATTN[0], "SHQ4-G32-U4Z"),
        (ATTN[1], "SHQ4-G32-U4Z"),
        (ATTN[2], "SHQ4-G32-U4Z"),
        (ATTN[3], "SHQ4-G32-U4Z"),
    ],
    # embed-only upcast: cheapest quality win, everything else SHQ4 G64
    "embed_only": [
        (EMBED, "SHQ8-G64"),
    ],
    # embed + attention-G32 only (no ffn_down upcast)
    "embed_attn": [
        (EMBED, "SHQ8-G64"),
        (ATTN[0], "SHQ4-G32-U4Z"),
        (ATTN[1], "SHQ4-G32-U4Z"),
        (ATTN[2], "SHQ4-G32-U4Z"),
        (ATTN[3], "SHQ4-G32-U4Z"),
    ],
    # ffn_down-only upcast, embed stays bf16 (isolate ffn_down lever)
    "ffn_only": [
        (FFN_DOWN, "SHQ8-G64"),
    ],
    # embed + ffn_down SHQ8, no G32 attention (isolate ffn_down w/o embed noise)
    "embed_ffn": [
        (EMBED, "SHQ8-G64"),
        (FFN_DOWN, "SHQ8-G64"),
    ],
    # unsloth-mirror but with SHQ6 (Q6_K-class) instead of SHQ8 on the
    # upcast set: embed + ffn_down + linear-attn -> SHQ6, attention G32, rest Q4
    "shq6_mirror": [
        (EMBED, "SHQ6-G64"),
        (FFN_DOWN, "SHQ6-G64"),
        (LIN_QKV, "SHQ6-G64"),
        (LIN_Z, "SHQ6-G64"),
        (LIN_OUT, "SHQ6-G64"),
        (ATTN[0], "SHQ4-G32-U4Z"),
        (ATTN[1], "SHQ4-G32-U4Z"),
        (ATTN[2], "SHQ4-G32-U4Z"),
        (ATTN[3], "SHQ4-G32-U4Z"),
    ],
    # embed + ffn_down -> SHQ6, linear-attn stays SHQ4 G64 (smaller Q6 tier)
    "shq6_ffn": [
        (EMBED, "SHQ6-G64"),
        (FFN_DOWN, "SHQ6-G64"),
    ],
    # full SHQ8: every eligible tensor to 8.25bpw (max quality ceiling)
    "full_shq8": [
        (EMBED, "SHQ8-G64"),
        (FFN_DOWN, "SHQ8-G64"),
        (MLP_UP, "SHQ8-G64"),
        (MLP_GATE, "SHQ8-G64"),
        (LIN_QKV, "SHQ8-G64"),
        (LIN_Z, "SHQ8-G64"),
        (LIN_OUT, "SHQ8-G64"),
        (ATTN[0], "SHQ8-G64"),
        (ATTN[1], "SHQ8-G64"),
        (ATTN[2], "SHQ8-G64"),
        (ATTN[3], "SHQ8-G64"),
        ("mtp.", "SHQ8-G64"),
    ],
}

# Bulk default applied to the ELIGIBLE linear set when no rule matches.
DEFAULT_BULK = "SHQ4-G64-U4Z"
DEFAULT_KEEP = "BF16"

# Substrings that are eligible for quantization (bulk set).
ELIGIBLE = (
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.out_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    "mtp.fc.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
)



def is_eligible(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in ELIGIBLE)


def resolve(name: str, rules: list) -> str:
    """Pick an encoding for a tensor name under the given recipe rules."""
    for pat, fmt in rules:
        if pat in name:
            return fmt
    if is_eligible(name):
        return DEFAULT_BULK
    return DEFAULT_KEEP


def bpw(fmt: str) -> float:
    return {"SHQ4-G64-U4Z": 4.5, "SHQ4-G32-U4Z": 4.625, "SHQ6-G64": 6.5625,
            "SHQ8-G64": 8.25, "BF16": 16.0}.get(fmt, 0.0)


def recipe_id(rules: list) -> str:
    """Stable id for a rule list (preset name if it matches)."""
    for name, r in PRESETS.items():
        if r == rules:
            return name
    return "custom"
