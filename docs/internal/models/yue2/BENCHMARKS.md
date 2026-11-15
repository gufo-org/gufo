# YuE2 benchmarks

Windows gfx1151 (Strix Halo, 96 GB carve-out), TheRock 10.0.0, `build/rock10`
release binary. One `gufo music` request, model load excluded:
`examples/song.json` (style and lyrics, seed 831001).

## 2026-11-15: BF16 projection outputs, matrix value product, sampler

The same fixed-score request (1059-token prefix), one change at a time:

| Change | Semantic | Acoustic | Total |
| --- | --- | --- | --- |
| Before (hipBLASLt plan, FP32 outputs) | 36.8 s | 34.8 s | 74.4 s |
| Prefill-sized projections write BF16 | 36.7 s | 29.8 s | 69.1 s |
| NAR attention value product on WMMA | 36.7 s | 26.9 s | 66.2 s |
| Top-k by a k-entry heap | 36.3 s | 27.0 s | 65.9 s |

The last row is a 79.8 s song in 65.9 s (0.83x real time). Every consumer of a
projection already rounded its FP32 input to BF16, so hipBLASLt now writes BF16
directly: at 1997 rows its BF16-output kernels run at 37-47 TFLOP/s against
23-37 for FP32 output (`gemm_bench`), and the shape rule's rank 1 is within
15% of the best candidate on every YuE2 shape. The NAR attention rounds the
probabilities to BF16 before the value product, as PyTorch's fused BF16
attention does, which lets that product use WMMA as well; one acoustic
velocity evaluation went from about 535 ms to 420 ms. The sampler found the
top 30 of 32769 codec rows with `nth_element` over every finite row (305 us
per token); one pass with a 30-entry min-heap takes 71 us and selects the same
rows (the seeded song is byte-identical).

Oracles (`yue2_hip_oracle_test`, now printed to seven digits):

| Stage | Before | After |
| --- | --- | --- |
| Teacher-forced logits, worst cosine | 0.9999607 (26/26 argmax) | 0.9999769 (26/26) |
| Velocity at t = 1, relative error | 0.01105 | 0.01118 |
| 32-step latents, relative error | 0.01105 | 0.01328 |
| VAE audio, relative error | 6.4e-7 | 6.4e-7 |

The latent difference is accumulation order compounding over 64 evaluations;
the single-step velocity error is unchanged.

Decode stays at 55 tok/s. A captured HIP graph of one decode step replays in
16.6 ms, the same as eager launches, so launch overhead is not what limits
it. The shared GEMV streams the decode shapes at about 240 GB/s in isolation;
the remaining time per token is the small per-layer kernels (two norms, RoPE,
SwiGLU, attention merge) between the projections and the 134 MB codec head.
`hipEventElapsedTime` reported 1.6 ms for that same 16.6 ms step on this
Windows stack, so per-op event timings there are not trustworthy.

## 2026-11-15: hipBLASLt for prefill-sized GEMMs

Same request with the 929-token score from the first run passed as
`--score` (1059-token prefix), so both binaries play the same score:

| Stage | hipBLAS `GemmEx` | hipBLASLt plan |
| --- | --- | --- |
| Semantic | 2019 tokens, 37.2 s | 1995 tokens, 36.8 s |
| Acoustic | 2021 rows, 117.5 s | 1997 rows, 34.8 s |
| Decode | 2.2 s | 2.1 s |
| Total | 80.8 s song in 158.0 s | 79.8 s song in 74.4 s (0.93x real time) |

`hipblasGemmEx` (BF16 in, FP32 out) runs at about 4 TFLOP/s on every NAR
projection; hipBLASLt's heuristic candidate 4 runs at 21-37 TFLOP/s from 200
to 4500 rows (`gemm_bench`, rows x {2048, 1024, 6144} x {2048, 6144}). That is
the FP32-output rank Qwen-Image-2.1 already uses, so both models share the
plan cache in `src/families/qwen/eager/blas.cpp`. The NAR GEMMs now take about
0.22 s per velocity evaluation; the rest of the acoustic stage is attention
over the AR prefix plus the latent rows.

Sampling diverges between the two runs (2019 against 1995 codec tokens)
because the prefill logits change in the last bits; the stage oracles in
`yue2_hip_oracle_test` still pass.

## 2026-11-14: first cut

Planning `full`:

| Stage | Work | Time |
| --- | --- | --- |
| Plan | 929 score tokens | 19.9 s (47 tok/s) |
| Semantic | 1996 codec tokens | 36.3 s (55 tok/s) |
| Acoustic | 32 midpoint steps over 1998 rows | 115.3 s |
| Decode | 2 tiles, 79.8 s of audio | 2.1 s |
| Total | 79.8 s song | 174.0 s (2.2x real time) |

Decode is weight-bandwidth bound (about 2.9 GB of BF16 weights per token).
