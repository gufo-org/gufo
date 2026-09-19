"""Qwen3.5-0.8B model adapter: teacher load, candidate load (SHQ4 dequant
swap), forward, logit extraction.

Teacher: transformers Qwen3_5ForConditionalGeneration, bf16 source, CPU.
Logits are computed as last_hidden_state @ embed_tokens.T (tied LM head).
Candidate: same architecture with eligible SHQ4 shards dequantized back to
bf16 before load. This measures OUR quantization quality, not torch kernels.
"""

from __future__ import annotations

import json
import numpy as np
import torch

from . import shq

# Eligible SHQ4 tensors in the current recipe (see gufo-quantize.py).
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


def dequant_shard(shard_path) -> np.ndarray:
    """Read an SHQ4/SHQ8 shard file and dequantize to float32 W[Np,Kp]."""
    with open(shard_path, "rb") as f:
        first = f.readline()
        meta = json.loads(first.decode("utf-8"))
        planes = {
            "weight": f.read(meta["w_len"]),
            "scale": f.read(meta["s_len"]),
            "zero": f.read(meta["z_len"]),
            "Np": meta["Np"], "Kp": meta["Kp"],
            "group_size": meta["group_size"],
            "symmetric": False,
        }
    if meta["format"].startswith("SHQ6"):
        return shq.dequant_shq6(planes)
    if meta["format"].startswith("SHQ8"):
        return shq.dequant_shq8(planes)
    return shq.dequant_shq4(planes)


def load_teacher(source_dir: str):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(source_dir, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(source_dir)
    return model, tok


def load_candidate(source_dir: str, quant_dir: str):
    """Load the source model with eligible SHQ4 shards dequantized to bf16."""
    import safetensors.torch
    from transformers import AutoModel, AutoTokenizer

    # Build state dict from source safetensors, then replace eligible tensors.
    import glob
    st = {}
    for f in glob.glob(f"{source_dir}/*.safetensors"):
        st.update(safetensors.torch.load_file(f))
    # safetensors keys carry a leading 'model.' that torch module params do not;
    # the mtp (speculative) head is not part of Qwen3_5Model and is dropped.
    st = {k.removeprefix("model."): v for k, v in st.items()
          if not k.startswith("mtp.")}

    # Read quantization plan for shard paths (GUFO_PLAN override for experiments).
    import os
    default_plan = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "artifacts", "work", "quantization-plan.json")
    plan_path = os.environ.get("GUFO_PLAN", default_plan)
    with open(plan_path) as f:
        plan = json.load(f)

    for name, info in plan["tensors"].items():
        if info["format"].startswith("SHQ4") or info["format"].startswith("SHQ6") \
           or info["format"].startswith("SHQ8"):
            shard = info["shard"]
            W = dequant_shard(shard)[: info["shape"][0], : info["shape"][1]]
            key = name.removeprefix("model.")
            st[key] = torch.from_numpy(W).to(torch.bfloat16)

    model = AutoModel.from_pretrained(source_dir, dtype=torch.bfloat16)
    # Keep only keys the model actually holds, then swap in candidate weights.
    allowed = set(model.state_dict().keys())
    st = {k: v for k, v in st.items() if k in allowed}
    model.load_state_dict(st, strict=True)
    tok = AutoTokenizer.from_pretrained(source_dir)
    return model, tok


def forward_logits(model, input_ids: torch.Tensor):
    """Teacher/candidate logits for a batch: last_hidden_state @ embed.T."""
    out = model.language_model(input_ids=input_ids, use_cache=False)
    h = out.last_hidden_state
    emb = model.language_model.embed_tokens.weight
    return torch.matmul(h, emb.T)  # [B, T, vocab]
