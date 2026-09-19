# Mixed-Precision Experiment: Qwen3.5-0.8B (SHQ-T16 tiers)

Status: 2026-08-13. Compares SHQ-T16 mixed-precision recipes against the
bf16 teacher, same matched-token suite as the base benchmark. Same source
revision, tokenizer, imatrix (`artifacts/calib`, 666-token disjoint set).

Teacher suite: 78 scored positions, SHA-256
`b33862883e78e7500cc8553cffe68ec8c44ca5ac1ceca8cbccb3070f6d42b131`.
Calibration suite SHA-256:
`f1314e80685482cd80c99a6093054ce44ec2b326fa2edc7c87065dc249a1b37e`.

Recipes are per-tensor tiers (never per-block). Tiers used:

| encoding | bpw | role |
|---|---|---|
| SHQ4-G64-U4Z | 4.50 | bulk linear tensors |
| SHQ4-G32-U4Z | 4.625 | sensitive attention (evaluated, no gain) |
| SHQ6-G64 | 6.56 | Q6_K-class tier (embed, ffn_down, linear-attn) |
| SHQ8-G64 | 8.25 | high-precision fallback tier |
| BF16 | 16 | norms, conv1d, small projections, vision |

**True model size** = quantized shard bytes + BF16-kept tensor bytes (source
bf16). Shard-only totals undercount BF16-kept tensors (embed is 508 MB bf16).

## Two independent size gaps vs unsloth Q4_K_M (533 MB)

1. **Vision tower (~200 MB)** — we keep `model.visual.*` bf16. llama.cpp /
   unsloth **drop the vision encoder entirely** (their GGUF has zero vision
   tensors). For a text-only serving runtime this is dead weight: the single
   biggest lever. Product decision: drop vision for the text artifact.
2. **Tier gap** — we upcast with SHQ8 (8.25 bpw) where unsloth uses Q6_K
   (6.56 bpw) / Q5_K (5.5 bpw). Added **SHQ6** (6.56 bpw, signed 6-bit, 4-per-3
   bytes, INT8 kernel expansion) to close it.

## Results (KL = matched-token mean; suite 78 positions)

| recipe | true+vision MB | true-vision MB | KL mean | KL p95 | top1 | ppl |
|---|---|---|---|---|---|---|
| bulk_g64 (baseline) | 990.7 | 789.8 | 0.1154 | 0.434 | 0.833 | 3.882 |
| embed_ffn (SHQ8) | 789.5 | 588.6 | 0.0866 | 0.308 | 0.872 | 3.957 |
| shq6_ffn | 703.0 | **502.0** | 0.0886 | 0.296 | 0.872 | 3.915 |
| unsloth_mirror (SHQ8) | 884.4 | 683.5 | 0.0383 | 0.156 | 0.910 | 3.831 |
| shq6_mirror | 750.7 | **549.8** | 0.0498 | 0.172 | 0.885 | 4.003 |
| full_shq8 | 999.1 | 798.2 | 0.0013 | 0.003 | 0.974 | 3.453 |

unsloth Q4_K_M = 533 MB (text-only, vision+MTP dropped).

## Levers (isolated)

1. **ffn_down upcast is the quality lever.** KL 0.1154 -> 0.0862 (SHQ8) /
   0.0886 (SHQ6) for 25 tensors. Mirrors unsloth Q6_K ffn_down.
2. **embed upcast is a free size cut** (quality-neutral): SHQ6 6.56bpw saves
   246 MB vs bf16. The bf16-embed policy over-spends the largest tensor.
3. **SHQ6 vs SHQ8 on the upcast set** saves 86 MB (shq6_ffn) to 134 MB
   (shq6_mirror) at a small quality cost (0.0886 vs 0.0866; 0.0498 vs 0.0383).
   SHQ6 is the Q6_K-class tier that closes the unsloth size gap.
4. **G32 attention adds nothing** (mirror_no_lin 0.0887 vs embed_ffn 0.0866;
   embed_attn worse than embed_only). Drop G32 — no quality, extra kernel path.
5. **linear_attn upcast is the biggest further lever** (0.0886 -> 0.0498 via
   shq6_mirror, +48 MB). Mirrors unsloth Q8_0/Q5_K linear-attn.
6. **Vision drop (~200 MB)** is the dominant size lever, independent of
   quantization.

## Recommended (text-only deployment, vision dropped)

- **shq6_ffn**: 502 MB, KL 0.0886 — size-competitive with unsloth's 533 MB at
  comparable quality. Minimum-size variant.
- **shq6_mirror**: 550 MB, KL 0.0498 — best quality-per-MB; linear_attn SHQ6
  adds 48 MB for a large KL gain. Preferred when the quality floor is tight.

Keep MTP (speculative) as SHQ4 — unsloth drops it, we keep it for speculative
decoding (~22 MB).

## Strix Halo hardware compatibility (format-level)

Measured kernel speed is **not** available yet — `src/main.cpp` is only a
device probe; SHQ4/SHQ6/SHQ8 decode GEMV kernels are the next milestone. What
is verified at the format level:

- SHQ6 and SHQ8 share the **identical T16 tile layout and packing order** as
  SHQ4 (SHQ6 byte-identical to reference loop; conformance suite passes). The
  decode GEMV kernel is one family — only the per-weight read width differs
  (SHQ6 is 3 bytes per 4 weights). SHQ6 expands to INT8 in the kernel
  (storage != compute, per docs/QUANTIZATION.md).
- Precision is **per-tensor**, never per-block, so hot kernels do not branch.
- SHQ6 (6.56 bpw) costs ~1.46x SHQ4 decode bandwidth (vs SHQ8's 2x), so it is
  the right tier for upcast tensors on the bandwidth-bound decode path.
- Chosen recipe keeps the bulk hot path on a single SHQ4-G64 group size (G32
  dropped).

## Tooling

`tools/gufo/recipe.py` defines presets; `gufo-quantize.py --recipe <name>`
converts; `gufo-mp-experiment.py` quantizes + benches each preset and prints
the comparison table. Plans in `artifacts/work/plan-<preset>.json`,
shards in `artifacts/quant-<preset>/`.
