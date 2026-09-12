# Offline conversion toolchain (tools/)

This page covers the offline weight-conversion and quantization pipeline only.
The GPU performance tooling that also lives under `tools/` -- `tools/bench/`,
`tools/prof/prof.py`, and `tools/prof/isa_mix.py` -- is documented in
[docs/PERFORMANCE.md](../docs/PERFORMANCE.md) under "Commands".

Python-only, torch-free serving. Never a transitive dependency of the server.
Run inside `nix develop` (flake adds torch, transformers, safetensors,
huggingface-hub, numpy, scipy, zstandard).

All model weights, logit dumps, and quantized artifacts live under `artifacts/`
which is gitignored — nothing committed.

## Layout & Domains

The `tools/` directory is organized into domain-specific subdirectories:

- `tools/ds4/` — DeepSeek V4 quality and performance workflow
- `tools/quant/` — Offline model inspection, quantization, calibration, evaluation, and test suites
- `tools/bench/` — GPU & kernel standalone microbenchmarks (HIP / C++)
- `tools/prof/` — rocprofv3 profiling wrappers and RDNA 3.5 ISA mix analyzer
- `tools/serving/` — HTTP serving benchmark harness
- `tools/h3/` — Multimodal & diffusion (MiniMax H3) manifest and quality verification
- `tools/audio/` — TTS reference runners and audio quality evaluation
- `tools/ci/` — Repository verification, dependency license audit, and docs checking
- `tools/gufo/` — Shared Python library (`shq`, `safetensors`, `recipe`, `model`, `quality`, `manifest`, `h3_*`)

## Commands

```bash
# 1. download safetensors + tokenizer to artifacts/source
# 2. validate safetensors, write source manifest
tools/quant/gufo-inspect.py --source artifacts/source --revision <sha> \
  --out artifacts/work/source-manifest.json

# 3. capture full-precision teacher logits + perplexity (matched-token)
tools/quant/gufo-capture.py --source artifacts/source \
  --suite tools/quant/suites/teacher.json --out artifacts/teacher

# 3b. capture per-input-channel imatrix (E[x^2]) for imatrix-weighted scale search
#   (calibration suite MUST be disjoint from the eval suite; see tools/quant/suites/calib.json)
#   CPU default; batched -> fast, exact fp64 reduction, bit-reproducible:
#   tools/quant/gufo-calibrate.py --source artifacts/source \
#     --suite tools/quant/suites/calib.json --out artifacts/calib
#   GPU (gfx1151 ROCm torch, default shell) for big corpora:
#   tools/quant/gufo-calibrate.py --source artifacts/source \
#     --suite tools/quant/suites/calib.json --out artifacts/calib --device cuda
#   --max-tokens N bounds real tokens per batched forward; --max-tokens 1
#   reproduces the legacy one-prompt-at-a-time result. --reference DIR
#   cross-checks against a prior artifact.
tools/quant/gufo-calibrate.py --source artifacts/source \
  --suite tools/quant/suites/calib.json --out artifacts/calib

# 4. quantize to a mixed-precision research recipe (tool default embed_ffn:
#   embed+ffn_down SHQ8, rest SHQ4 G64). This is not the Qwen3.8 production
#   default; that recipe remains unset until model-specific calibration and
#   native kernel gates pass. See benchmarks/qwen3.5-0.8b/MIXED_PRECISION.md.
#   --imatrix DIR enables importance-weighted SHQ4 scale search
#   --recipe NAME selects a preset (bulk_g64/embed_only/embed_ffn/ffn_only/
#   embed_attn/mirror_no_lin/unsloth_mirror/shq6_ffn/shq6_mirror/full_shq8)
#   or a JSON rule file; tiers SHQ4-G64/G32, SHQ6-G64, SHQ8-G64, BF16
tools/quant/gufo-quantize.py --source artifacts/source \
  --out artifacts/quant --plan artifacts/work/quantization-plan.json --imatrix artifacts/calib
# 4b. sweep mixed-precision presets, benchmark each, print comparison table
tools/quant/gufo-mp-experiment.py --source artifacts/source \
  --suite tools/quant/suites/teacher.json --teacher-artifact artifacts/teacher \
  --imatrix artifacts/calib --json
# 5. benchmark: candidate-vs-teacher quality + prefill/decode speed
tools/quant/gufo-bench.py --source artifacts/source --quant artifacts/quant \
  --suite tools/quant/suites/teacher.json --teacher-artifact artifacts/teacher

# 6. inspect an external GGUF (e.g. unsloth) and score it vs bf16, cross-quant
tools/quant/gufo-gguf.py --gguf artifacts/gguf/Qwen3.5-0.8B-Q4_K_M.gguf --card
# per-tensor retention vs bf16, side-by-side with our SHQ4 (from the plan)
tools/quant/gufo-gguf.py --gguf artifacts/gguf/Qwen3.5-0.8B-Q4_K_M.gguf --recon \
  --bf16-source artifacts/source --plan artifacts/work/quantization-plan.json
```

## Modules

- `gufo/safetensors.py` — safetensors validation + lazy read (Source Contract)
- `gufo/shq.py` — SHQ4-T16 / SHQ6-T16 / SHQ8-T16 quantize/pack/dequant (candidate contract; portable v1 not yet frozen)
- `gufo/conformance.py` — byte-exact conformance vectors (T1)
- `gufo/manifest.py` — source-manifest / quantization-plan writers
- `gufo/model.py` — Qwen3.5 teacher/candidate load + logit extraction
- `gufo/quality.py` — KL, perplexity, top-k agreement
- `tools/quant/gufo-capture.py` — teacher logit artifact (chunked zstd)
- `tools/quant/gufo-calibrate.py` — per-input-channel E[x^2] imatrix artifact
- `gufo/recipe.py` — SHQ-T16 mixed-precision recipe presets (per-tensor tiers)
- `tools/quant/gufo-quantize.py` — deterministic conversion (recipe + range/imatrix search)
- `tools/quant/gufo-mp-experiment.py` — quantize+bench sweep across presets (comparison table)
- `tools/quant/gufo-bench.py` — correctness-linked benchmark
- `tools/serving/gufo-serving-bench.py` — canonical C=1/C=2/C=4 HTTP serving benchmark
- `tools/quant/gufo-gguf.py` — GGUF header/tensor-info inspection + Q4-family dequant
  (`--card` model card, `--recon` per-tensor retention vs bf16 with our SHQ4
  stats merged from the quantization plan; see `benchmarks/qwen3.5-0.8b/`)

## Conformance tests

```bash
python3 -c "import sys; sys.path.insert(0,'tools'); from gufo.conformance import run_all; run_all()"
```

## Layout contract (SHQ4-T16)

See docs/QUANTIZATION.md. Byte layout: `qweight[n_tile][k_group][k16][lane=16][k_pair=8]`,
scales/zeros BF16 + packed UINT4 per (tile,group,lane). U4Z dequant:
`s * (q - z)`, scale rounded to BF16 RNE-ties-even before code selection.

Scale selection is deterministic, two modes:

- `range` (default): per-channel-per-group min/max range (v1).
- `imatrix-weighted-ls` (`--imatrix DIR`): per-input-channel E[x^2]
  importance; per (tile,group) all 16 lanes solved at once. For each lane,
  enumerate zero candidates {round(-min/s_r)+d, d in -1..1}, refine scale by
  weighted least squares (`s = sum(h*w*d)/sum(h*d^2)`, d = q-z) for 2 rounds,
  fix s to BF16 RNE, keep best (s,z,q) by weight sensitivity `S = sum_h(w-ŵ)^2`.
  Result is never worse than the range baseline per block.

Planes are byte-identical in layout for both modes; only s/z/q values differ.

## Performance

`gufo-shq` is fully batch-vectorized and chunk-parallel. Scale/zero/code
selection and packing run as numpy ops on one `[B,16,G]` block array;
chunks fan out across threads (numpy releases the GIL in its C loops, so
elementwise/bandwidth-bound work parallelizes). Threads default to
`min(4, cores)` (memory-bandwidth bound past 4); override with
`GUFO_QUANT_THREADS`, fixed chunk size with `GUFO_QUANT_CHUNK`.
`dequant_shq4` is batch-vectorized too.

Whole-model Qwen3.5-0.8B (158 tensors, G64) rough timings on this box:

| path | time (1 thread) | time (4 threads) |
| --- | --- | --- |
| range | ~30s | ~6s |
| imatrix | ~30s | ~16s |

(The 4-thread row is the default. The prior per-block Python-loop version took
minutes and choked imatrix; this is ~20x faster and scales memory-bounded.)
# Qwen27B

`qwen27b/check.py` owns the focused checks (`fast`, `kernels`, `model`, `serving`).
Its optional `reference` suite compares target logits with an explicitly
provided BF16 artifact. `qwen27b/drafts.py` compares Q4/Q8/BF16 DFlash2 companions and refuses
incomplete or mismatching results. `--baseline-binary` interleaves two releases
and also requires identical autoregressive token traces between them.
`qwen27b/dflash_gemm_bench.hip` measures exact BF16, Q4/Q5/Q6/Q8 and IQ4_XS
matrix geometries using the production templates; build it with `tools/bench/build.sh` inside Nix.
Its optional final batch argument selects 3–8 projection rows or sixteen
BF16 injection rows, for example `q4 17408 5120 24 7`,
`q8 5120 17408 24 7` or `bf16 5120 25600 24 16`.
Injection repeats one weight matrix and uses the production cache hint
(`bf16 1024 5120 24 16` covers K/V); decoding rotates at least 128 MiB of weights.
`qwen27b/deltanet_bench.hip` checks exact recurrence and state-only replay
while rotating the 144 MiB target state; `qwen27b/prefill_deltanet_bench.hip`
contains the separate prefill ablations. Both use the same fast Nix builder.
`qwen27b/attention_bench.hip` compares scalar and batched attention. It checks exact outputs
beside component timings, including the 4K split-K boundary; its `rope` mode
compares separate and fused draft normalization/RoPE.
`qwen27b/prefill_gemm_bench.hip` compares production Q8_0 gate/up GEMM
with separate or fused SwiGLU, including every activation byte and scale.
It replaces the old standalone W8A8 kernel copies.
`qwen27b/dflash_reference.py` checks a GPU trace against
pinned upstream PyTorch operators using the same GGUF weights. See
`benchmarks/qwen3.8-27b/README.md` and its quality report for commands and evidence.
