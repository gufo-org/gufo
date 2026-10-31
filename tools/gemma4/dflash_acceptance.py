#!/usr/bin/env python3
"""Teacher-forced DFlash acceptance for Gemma 4 31B along Gufo transcripts.

Usage (drafter: z-lab/gemma-4-31B-it-DFlash; transcripts and features from
gemma4_gpu_probe --chat ... --generate N --tokens-out --taps 1,12,23,35,46,57
--taps-out, plus --logits-out --first PROMPT-1 for sampled mode):
  nix develop -c python3 tools/gemma4/dflash_acceptance.py --upstream model.py \
    --draft DIR --gguf TARGET.gguf --tokens T.i32 --taps F.f32 --prompt-len N \
    --mode greedy|sampled [--logits L.g4lg] --embed-scale 73.3212

Inputs come from gemma4_gpu_probe: the token sequence (prompt + reply), the
residual features after the drafter's target layers, and (sampled mode) the
target logits of the reply. The drafter is the unmodified upstream
DFlashDraftModel (z-lab/dflash dflash/model.py); the target's tied embedding
and head come from the same GGUF the engine runs (Q5_K, dequantized).

greedy: simulates greedy DFlash decoding exactly along the target's greedy
        transcript (acceptance = leading drafts equal to the transcript).
sampled: at every `stride`-th reply position, expected accepted drafts under
        p/q rejection, sum_k prod_{i<=k} sum_x min(p_i, q_i), with the
        target distributions teacher-forced on the transcript.
Both report tokens per cycle for verification widths W (W - 1 drafts).
"""
import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
# The z-lab/dflash revision the Qwen DFlash2 reference also pins.
UPSTREAM_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"
UPSTREAM_SHA256 = "f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_logits(path):
    with open(path, "rb") as f:
        magic, version, rows, vocab = struct.unpack("<4I", f.read(16))
        assert magic == 0x474C3447 and version == 1
        positions = np.frombuffer(f.read(4 * rows), dtype="<u4")
        logits = np.frombuffer(f.read(4 * rows * vocab), dtype="<f4").reshape(rows, vocab)
    return positions, logits


class TargetShim(nn.Module):
    """The target pieces DFlash reads: raw input embeddings and the tied head."""

    def __init__(self, embedding):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.lm_head = nn.Linear(embedding.shape[1], embedding.shape[0], bias=False)
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--draft", required=True, help="dir with config.json + model.safetensors")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--taps", required=True)
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--mode", choices=["greedy", "sampled"], required=True)
    ap.add_argument("--logits", help="target logits of the reply (sampled mode)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--label", default="")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--causal", choices=["config", "false", "true"], default="config")
    ap.add_argument("--embed-scale", type=float, default=None,
                    help="override input_embedding_scale (Gemma scales embeddings by sqrt(hidden))")
    args = ap.parse_args()

    dev = "cuda"
    import hashlib
    if hashlib.sha256(Path(args.upstream).read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise SystemExit(f"--upstream is not dflash/model.py at {UPSTREAM_REVISION}")
    up = load_module("dflash_upstream", args.upstream)
    from safetensors.torch import load_file
    from transformers import Qwen3Config

    cfg_json = json.loads((Path(args.draft) / "config.json").read_text())
    config = Qwen3Config(**cfg_json)
    config.dflash_config = cfg_json.get("dflash_config", {})
    if args.causal != "config":
        config.is_causal = args.causal == "true"
    model = up.DFlashDraftModel(config)
    state = load_file(str(Path(args.draft) / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    model = model.to(dev, torch.bfloat16).eval()
    layers = model.target_layer_ids
    block = model.block_size

    codec = load_module("gufo_gguf", ROOT / "tools/gufo/gguf.py")
    st = codec.parse_gguf(args.gguf)
    info = next(t for t in st["tensors"] if t["name"] == "token_embd.weight")
    raw = np.memmap(args.gguf, mode="r", dtype=np.uint8, offset=st["data_offset"] + info["offset"],
                    shape=(codec.tensor_bytes(st, info),))
    emb = torch.from_numpy(np.asarray(codec.DEQUANT[info["type"]](raw), dtype=np.float32)
                           .reshape(info["shape"])).to(dev, torch.bfloat16)
    target = TargetShim(emb).to(dev)

    tokens = np.fromfile(args.tokens, dtype="<i4")
    n = len(tokens)
    hidden = config.hidden_size
    taps = np.fromfile(args.taps, dtype="<f4").reshape(n, len(layers), hidden)
    feats = torch.from_numpy(taps.reshape(n, len(layers) * hidden)).to(dev, torch.bfloat16)
    ids = torch.from_numpy(tokens.astype(np.int64)).to(dev)
    s0 = args.prompt_len
    mask = int(model.mask_token_id)
    widths = [w for w in (2, 4, 6, 8, 12, 16) if w <= block]

    # A full cache: the drafter's own mask applies the sliding window, and a
    # sliding cache cannot be cropped back once it held more than the window.
    from transformers import DynamicCache
    cache = DynamicCache()
    fed = 0  # context rows already in the drafter cache

    @torch.no_grad()
    def draft_logits(s):
        nonlocal fed
        block_ids = torch.full((1, block), mask, dtype=torch.long, device=dev)
        block_ids[0, 0] = ids[s]
        scale = (args.embed_scale if args.embed_scale is not None
                 else float(up._draft_value(model.config, "input_embedding_scale", 1.0)))
        noise = up._raw_input_embeddings(target, block_ids, scale)
        ctx = feats[fed:s][None]
        pos = torch.arange(fed, s + block, device=dev)[None]
        h = model(target_hidden=ctx, noise_embedding=noise, position_ids=pos,
                  past_key_values=cache, use_cache=True)[:, 1 - block:, :]
        up._crop_to(cache, s)
        fed = s
        return model.compute_logits(h, target.lm_head)[0].float()  # [block-1][vocab]

    if args.mode == "greedy":
        per_width = {w: [] for w in widths}
        s = s0
        cycles = 0
        while s + 1 < n:
            drafts = torch.argmax(draft_logits(s), dim=-1)
            if args.show and cycles < args.show:
                vocab = json.load(open(str(Path(args.tokens).parent / "vocab.json"))) if cycles == 0 else vocab
                txt = lambda xs: "".join(vocab[int(x)] for x in xs).replace("▁", " ")
                print("anchor", repr(txt([ids[s]])), "| truth", repr(txt(ids[s + 1:s + 16])), "| draft", repr(txt(drafts)))
            remain = n - 1 - s
            truth = ids[s + 1: s + 1 + min(block - 1, remain)]
            match = (drafts[: len(truth)] == truth).int()
            k = int(match.cumprod(0).sum().item())
            for w in widths:
                per_width[w].append(min(k, w - 1) + 1)
            s += k + 1  # the full block's greedy path (accepted + bonus)
            cycles += 1
        out = {w: float(np.mean(v)) for w, v in per_width.items()}
        print(json.dumps({"label": args.label, "mode": "greedy", "reply_tokens": n - s0,
                          "cycles_w16": cycles, "tokens_per_cycle": out}))
        return

    positions, logits = read_logits(args.logits)
    row_of = {int(p): i for i, p in enumerate(positions)}
    lg = torch.from_numpy(logits)
    exp = {w: [] for w in widths}
    for s in range(s0, n - 1, args.stride):
        dl = draft_logits(s)
        q = up._sampling_probs(dl, args.temperature, args.top_p, args.top_k)  # [block-1][vocab]
        alphas = []
        for i in range(1, block):
            p_row = row_of.get(s + i - 1)
            if p_row is None or s + i >= n:
                break
            p = up._sampling_probs(lg[p_row][None].to(dev), args.temperature, args.top_p, args.top_k)[0]
            alphas.append(float(torch.minimum(p, q[i - 1]).sum().item()))
        for w in widths:
            e, run = 0.0, 1.0
            for a in alphas[: w - 1]:
                run *= a
                e += run
            exp[w].append(1.0 + e)
    out = {w: float(np.mean(v)) for w, v in exp.items()}
    print(json.dumps({"label": args.label, "mode": "sampled", "reply_tokens": n - s0,
                      "samples": len(exp[widths[0]]), "tokens_per_cycle": out}))


if __name__ == "__main__":
    main()
