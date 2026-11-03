# verify_sweep: tune verification launch shapes for a Qwen GGUF

During speculative decoding the target model verifies 2-8 draft rows at a
time, and the DFlash2 draft runs its block at the same widths. Each K-quant
projection then runs `SmallBatchKQuantExactFp32GEMMKernel`, and how its work
is split over waves (waves per block, rows per wave, 16- or 32-lane groups,
staging tiles) decides how close the GPU gets to its memory bandwidth.
`src/families/qwen/kernels/verify_gemm_27b.hip` holds the shapes measured
faster on gfx1151 for Qwen3.8-27B UD-Q4_K_XL and its Q4_K_M DFlash2 draft; any
projection it does not list keeps the generic `LaunchBatchedQuantGEMMFp32`
dispatch.

A different GGUF, quantization mix, draft setting, GPU or toolchain changes
which projections and row counts matter, so the shapes have to be measured
again. This tool does that in a few minutes.

## Use

```sh
# production optimization, as the timings should be
nix develop -c cmake --preset release -DGUFO_BUILD_TOOLS=ON
nix develop -c cmake --build --preset release --target qwen27b_verify_sweep
# stop every gufo or llama.cpp server: the GPU has to be idle
nix develop -c python3 tools/qwen27b/verify_sweep/sweep.py \
  --binary build/release/qwen27b_verify_sweep /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

`sweep.py` lists the model's projections (one example tensor per type and
shape, with the joined gate/up matrix where both halves are stored back to
back), sweeps every projection used by at least `--min-layers` layers at
`--batches` rows, and prints the best exact candidate per projection plus the
winning shapes as `Launch<...>` lines. Options: `--batches 3,4,5`,
`--rounds 9`, `--min-gain 5` (percent), `--out sweep.log` for the raw lines.
For a DFlash2 draft GGUF pass `--min-layers 1`: the draft has only five
layers, so some projections occur once. Its shapes go into `LaunchQ4K`, the
draft block of `verify_gemm_27b.hip`.

The binary also runs on its own:
`qwen27b_verify_sweep <gguf> <offset> <type> <m> <k> <batch> [rounds]`, with
the absolute file offset of the tensor and the GGUF type id (12 Q4_K, 13 Q5_K,
14 Q6_K, 23 IQ4_XS).

## Method

- Weights are the real tensors from the GGUF, read at their file offset.
- Production is what `LaunchProjection` runs: `TryLaunchVerifyGemm27`, then
  `LaunchBatchedQuantGEMMFp32`. The `generic` candidate is
  `LaunchBatchedQuantGEMMFp32` alone, the dispatch without any listed shape;
  where it beats production, the listed shape is a regression and is printed
  for removal. Every candidate's output is compared bit for bit with
  production; inexact candidates are reported and never timed.
- Each launch uses another copy of the weights, enough copies for at least
  512 MiB, so no run reads from the 32 MiB MALL cache. Without that, matrices
  under ~35 MB look faster than the memory allows.
- Production and candidates alternate within each round; the result is the
  median over the rounds, 20 launches per sample after 3 warm-up launches.

## Taking a result over

Add the shape to `verify_gemm_27b.hip` under its format and projection,
keyed by type, `m`, `k` and row count, only where it beats `generic` by the
`--min-gain` margin. The target and the DFlash2 draft share some projection
shapes (the Q4_K FFN and attention matrices), and one entry serves both, so
sweep both GGUFs and judge a shared shape on both results. Then rebuild,
rerun the sweep to confirm no listed shape loses to `generic`, confirm greedy
output is unchanged end to end and measure a real workload: a per-launch gain
only matters in proportion to the projection's share of the cycle.

## Limits

- Only the Qwen GPU path (`src/models/qwen/hip`: Qwen3.8-27B and the DFlash2
  drafts) and the formats Q4_K, Q5_K, Q6_K and IQ4_XS.
- The candidate set is the neighbourhood of shapes picked so far, not an
  exhaustive search.
- The current shapes were measured on gfx1151 under Linux with ROCm 10.1.
  Shapes picked under Windows with TheRock ROCm 10.0.0 were up to 29% slower
  than `generic` there (16-lane groups at six to eight rows), so a new
  toolchain needs a new sweep.
