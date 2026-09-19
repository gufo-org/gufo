# Benchmark: Qwen3.5-0.8B, SHQ4-T16 U4Z G64 candidate vs bf16 teacher

Status: first vertical slice, 2026-08-11. CPU-only reference path (torch
fallback for linear attention; no flash-linear-attention/causal-conv1d fast
path). Measures OUR SHQ4 quantization quality; speed here is the reference
runtime, not Gufo kernels yet.

Methodology: `docs/BENCHMARKS.md` (matched-token, per position; utilities
`gufo-capture.py` / `gufo-bench.py`). Raw artifacts gitignored under
`artifacts/`.

Source: `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`, bf16.

Teacher suite: `tools/quant/suites/teacher.json`, 78 scored positions,
SHA-256 `b33862883e78e7500cc8553cffe68ec8c44ca5ac1ceca8cbccb3070f6d42b131`.
Calibration suite: `tools/quant/suites/calib.json`, SHA-256
`f1314e80685482cd80c99a6093054ce44ec2b326fa2edc7c87065dc249a1b37e`.

Candidate: all LM linear projections (qkv/z/out/mlp/full-attn/mtp) quantized
SHQ4-T16 U4Z G64; norms, embeddings, conv1d, vision kept bf16. 158 tensors
quantized, 267 MB artifact.

## How the models are pulled from Hugging Face

Everything runs inside `nix develop` (flake adds `huggingface-hub`).

```bash
nix develop

# 1. bf16 safetensors source (tokenizer + weights + config)
mkdir -p artifacts/source
hf download Qwen/Qwen3.5-0.8B \
  --local-dir artifacts/source \
  --revision 2fc06364715b967f1860aea9cf38778875588b17

# 2. unsloth GGUF, closest 4-bit resolution (Q4_K_M, 533 MB)
mkdir -p artifacts/gguf
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf \
  --local-dir artifacts/gguf
```

`hf download` is the `huggingface-hub` CLI (same lib that
`gufo-inspect.py` uses). Pin the source revision — the benchmark is only
comparable against the same revision, suite, tokenizer, and reference
runtime. Other 4-bit resolutions available on the unsloth repo: `Q4_K_S`,
`IQ4_XS`, `IQ4_NL`, `Q4_0`, `UD-Q4_K_XL` (dynamic). We compare against
`Q4_K_M` because it is the standard closest-resolution 4-bit reference and
ships an imatrix file (`imatrix_unsloth.gguf_file`).

## Model components and MTP

Component map (24 blocks; layers 0-2 linear, 3 full, repeating per
`full_attention_interval: 4`):

| component | tensors | params (M) | our candidate | unsloth Q4_K_M gguf |
| --- | --- | --- | --- | --- |
| embed | 1 | 254.3 | bf16 (kept) | Q6_K 6.56 bpw |
| full_attn | 42 | 157.3 | SHQ4 4.5 bpw | Q4_K 4.5 bpw |
| linear_attn (ssm) | 108 | 38.8 | SHQ4 (in_proj_qkv/z/out), bf16 rest | Q8_0/Q5_K/F32 |
| mlp | 72 | 264.2 | SHQ4 4.5 bpw | Q4_K gate/up, Q6_K down |
| norm | 79 | 0.1 | bf16 | F32 |
| MTP head | 1 block + fc | ~50 | SHQ4 (quantized) | **absent** |

**MTP answer: the model has it, the GGUF does not.** The HF model config sets
`mtp_num_hidden_layers: 1` (model card: "MTP: trained with multi-steps"), and
the safetensors carry `mtp.layers.0.*` + `mtp.fc`. Our candidate quantizes the
MTP head to SHQ4 (`artifacts/quant/mtp/`). The unsloth GGUF has zero MTP
tensors (`gufo-gguf.py` reports `MTP tensors present: False`) — the llama.cpp
`qwen35` conversion drops the speculative head. MTP is a multi-token draft
head, not part of the single-token NLL/KL benchmark above, so the two files
are comparable on the shared LM body; only our build carries the MTP head
(4-bit) for speculative decoding later.

## Comparison vs unsloth Q4_K_M (closest 4-bit resolution)

Both quants are scored against the same bf16 source, tensor by tensor:

```bash
nix develop
tools/quant/gufo-gguf.py --gguf artifacts/gguf/Qwen3.5-0.8B-Q4_K_M.gguf \
  --recon --bf16-source artifacts/source \
  --plan artifacts/work/quantization-plan.json
```

`gufo-gguf.py` parses the GGUF header/tensor-info, dequants the Q4 family
(Q4_0/Q4_K/Q5_K/Q6_K/Q8_0), and reports per-tensor `rel_mae` (mean abs error
/ mean |w|) — the same scale `gufo-quantize.py` records for our SHQ4, merged
from `quantization-plan.json`. Weight reconstruction is a per-component proxy;
the logit-level matched-token KL above stays the authoritative whole-model
metric (per-layer logit attribution is still a planned slice).

### Per-component retention (mean rel_mae vs bf16)

| component | n | gguf bpw | unsloth Q4_K_M | our SHQ4 |
| --- | --- | --- | --- | --- |
| embed | 1 | 6.56 | 1.9% | kept bf16 |
| full_attn | 24 | 4.84 | 7.1% | 10.7% |
| linear_attn | 90 | 8.5-17.3 | 1.1% | 10.6% |
| mlp | 72 | 4.84 | 6.7% | 10.1% |
| norm | 36 | 32 | ~0% | kept bf16 |

Per-layer/per-tensor rows print below the aggregate (e.g. layer 0:
`ffn_gate Q4_K 4.50bpw unsloth 7.53% ours 9.92%`, `ffn_down Q6_K 6.56bpw
unsloth 1.85% ours 9.97%`, `ssm_out Q5_K 5.50bpw unsloth 4.05% ours 10.32%`).

### Reading

- Unsloth retains more per weight on the Q4_K attention/mlp tensors (7% vs
  our 10%) because Q4_K_M uses **imatrix / mixed-precision recipes** (Q6_K
  for embed and ffn_down, Q5_K/Q8_0 for linear-attn projections) plus
  importance-weighted quantization. Our SHQ4 slice is uniform U4Z G64 on
  everything with imatrix-weighted scale search. With a proper disjoint
  calibration set, imatrix closes most of the logit-level gap (see quality
  table below); the remaining per-weight rel_mae gap is still dominated by
  the **mixed-precision recipe** (upcast embed + ffn_down + linear-attn
  projections), not by scale selection.
- We retain more where we stay bf16 (embed, norms) — 0% loss.
- linear_attn compares unevenly because the recipes pick different tensors to
  quantize (their ssm small projections vs our qkv/z/out). Per-tensor rows
  are the apples-to-apples view; the aggregate bpw column shows why.

## Quality (candidate vs teacher, matched-token suite, 78 positions)

| metric | range | imatrix |
| --- | --- | --- |
| KL mean | 0.172 | 0.117 |
| KL median | 0.044 | 0.040 |
| KL p95 | 0.794 | 0.424 |
| KL p99 | 1.734 | 1.212 |
| KL max | 2.419 | 1.759 |
| top-1 agreement | 0.821 | 0.872 |
| teacher perplexity | 3.481 | 3.481 |
| candidate perplexity | 4.259 | 3.942 |

`range` = min/max scale search; `imatrix` = importance-weighted scale search
calibrated on `tools/quant/suites/calib.json` (666 tokens, disjoint from this eval
suite). Calibrating on the eval suite itself changed nothing (KL 0.172) —
the disjoint calibration set is what unlocks imatrix.

## Speed (CPU reference runtime)

| metric | teacher (bf16) | candidate (SHQ4 dequant bf16) |
| --- | --- | --- |
| prefill tokens/s | 99.8 | 49.1 |
| decode tokens/s | 4.09 | 4.55 |
| decode ms/token | 245 | 220 |

Note: candidate dequant-reloads to bf16 and runs through the same torch path,
so speed is the reference runtime, not our kernels. Kernel speed is a later
milestone.

## Mixed-precision revision (see MIXED_PRECISION.md)

The uniform-SHQ4 slice is superseded by a per-tensor tier recipe. Experiments
across presets show:

- ffn_down upcast is the quality lever (KL 0.1154 -> 0.0862 SHQ8 / 0.0886 SHQ6).
- embed upcast is a free 246 MB size cut (quality-neutral); bf16-embed
  over-spent the largest tensor.
- SHQ6 (6.56 bpw, Q6_K-class) replaces SHQ8 on the upcast set: saves 86-134 MB
  at small quality cost. This is the tier that closes the unsloth size gap.
- G32 attention buys no quality; dropped.
- linear_attn upcast is the biggest further lever (KL 0.0886 -> 0.0498 shq6).
- Vision tower (~200 MB) should be dropped for text-only serving (llama.cpp /
  unsloth drop it); dominant size lever.
- Chosen text-only deployment: `shq6_ffn` (embed+ffn_down SHQ6, rest SHQ4 G64;
  502 MB, KL 0.0886) size-competitive with unsloth's 533 MB; quality variant
  `shq6_mirror` (550 MB, KL 0.0498).

## Next steps

- Port SHQ4/SHQ8 decode GEMV to HIP (gfx1151) and XDNA2 (AIE2P); consume the
  packed planes directly (no dequant-to-bf16). SHQ8 shares the SHQ4 T16 tile
  layout, so it needs no separate kernel family.
- Imatrix scale search is DONE and validated: `tools/quant/gufo-calibrate.py` +
  `--imatrix` with the disjoint 666-token `tools/quant/suites/calib.json` drops
  logit KL mean 0.172 -> 0.117, top-1 0.821 -> 0.872, ppl 4.259 -> 3.942
  (measured on this eval suite; calibration set disjoint). Self-calibrating
  on the eval suite is a documented anti-pattern — it changes nothing.
- The remaining per-weight gap vs Q4_K_M is dominated by mixed precision, not
  scale search: add recipe support to promote embed + ffn_down (+ linear-attn
  projections) to SHQ8/Q6_K-class bpw, mirroring unsloth's Q6_K/Q5_K/Q8_0
  choices; then re-run the retention table.
- G32 quality groups for sensitive attention tensors.
- Per-layer quality breakdown (see `docs/BENCHMARKS.md`).
- GGUF cross-quant matrix: run `gufo-gguf.py --recon` on `Q4_K_S`, `Q4_0`,
  `IQ4_XS`, `Q6_K` and tabulate retention per layer across resolutions.
