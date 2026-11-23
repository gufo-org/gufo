# YuE2 benchmarks

Windows gfx1151 (Strix Halo, 96 GB carve-out), TheRock 10.0.0, `build/rock10`
release binary. One `gufo music` request, model load excluded:
`examples/song.json` (style and lyrics, seed 831001).

## 2026-11-20: 12-bit decode rows

Windows, the fixed-score request (929-token score, 1059-token prefix, 1995
codec tokens), on `main` at 8327275c:

| Change | Semantic | Acoustic | Decode | Total | Load |
| --- | --- | --- | --- | --- | --- |
| Before | 29.9 s (15.0 ms per token) | 20.3 s | 1.6 s | 52.2 s | 4.3 s |
| Decode projections and head on 12-bit rows | 26.3 s | 20.3 s | 1.6 s | 48.5 s | 4.6 s |
| Three 2048-column rows per workgroup | 24.8 s (12.4 ms per token) | 20.3 s | 1.6 s | 47.0 s | 4.6 s |

A 79.8 s song takes 47.0 s (0.59x real time), and every row writes the same
audio bytes.

Decode was at the memory rate: about 2.9 GB of BF16 weights per token at 240
GB/s. The only way left to speed it up was to read fewer bytes. The decode
copies (`hip/kernels.hpp`) keep a weight's sign and 7 mantissa bits and
replace its 8-bit exponent by a 4-bit code relative to the row's largest
exponent. The 0.04% of weights below that range sit in a per-row escape
list, so every weight decodes to its exact BF16 value. The GEMV adds the
products in the order of the shared BF16 kernel and gives the same bits; a
verify pass at load decodes every packed weight and compares it with the
tensor.

- The copies take 2.5 GiB of device memory (the AR projections of 28 layers
  and the head). Prefill and the acoustic path keep the BF16 tensors for
  hipBLASLt.
- Packing runs on the GPU from the uploaded tensors: 0.3 s of load for the
  scan, write and verify passes. Packing on 16 CPU threads from the mapped
  file took 1.9 s.
- In isolation the packed GEMV streams 213-232 GB/s of packed bytes: 1.29x
  the BF16 kernel on q/k/v, o and down, and 1.15x on gate/up and the head.
  Those two have many 3 KiB rows, and one workgroup per row cannot keep the
  stream at the memory rate. In situ the semantic stage took 26.3 s, 25.3 s,
  24.9 s and 27.9 s with one to four 2048-column rows per workgroup; one
  row stays best for the 6144-column rows, which are as long as three.
- What is left of a 12.4 ms token: about 9.5 ms of packed weights, 1.2 ms of
  attention over the BF16 KV cache (also at the memory rate) and the small
  per-layer kernels.

The acoustic stage per velocity evaluation (318 ms at 1997 rows over a
3055-token prefix), from host-clock timings with a synchronize after each
operator: GEMMs 189 ms, attention 97 ms, norms, SwiGLU, RoPE and the value
transpose 21 ms. Each synchronize costs about 0.16 ms here, which the
figures exclude. hipBLASLt runs these shapes at 32-39 TFLOP/s in isolation
on random weights and at 26-32 in situ. Removing parts of the attention kernel
shows its time: 41% QK matrix products, 30% value products, 14% `expf` and
15% staging and the rest, so it is bound by the matrix instructions and not
by memory. Sixteen or 32 waves per workgroup, which read each key tile for
two or four times as many queries, took 20.1 s and 21.4 s against 20.3 s.

## 2026-11-15: Linux kernel pass

Linux gfx1151 (Strix Halo, 128 GB), ROCm 10.1 from `nix develop`,
`build/release` binary. The same fixed-score request: `examples/song.json`
style and lyrics with the 871-token `city_lights` score passed as `--score`
(1001-token prefix), seed 831001, one change at a time:

| Change | Semantic | Acoustic | Decode | Total |
| --- | --- | --- | --- | --- |
| Before (`b4be1f0b`) | 1803 tokens, 32.39 s | 23.97 s | 1.91 s | 58.54 s |
| Acoustic attention: Q in registers, K and transposed V tiles in LDS | 32.37 s | 21.10 s | 1.91 s | 55.64 s |
| Decode attention: 64-key splits, coalesced loads | 1813 tokens, 29.81 s | 20.81 s | 1.92 s | 52.81 s |
| Gate/up projections on hipBLASLt's first candidate | 29.76 s | 18.65 s | 1.93 s | 50.60 s |
| RoPE table; register-resident RMSNorm (bit-identical) | 29.56 s | 18.56 s | 1.92 s | 50.31 s |
| One wave per head for QK norm/RoPE (bit-identical) | 29.53 s | 18.15 s | 1.92 s | 49.86 s |
| Decode merge over four thread groups | 1816 tokens, 29.36 s | 18.21 s | 1.94 s | 49.76 s |
| Weights 256-byte aligned on the device | 1801 tokens, 26.20 s | 17.70 s | 1.91 s | 46.06 s |
| VAE GEMMs on hipBLASLt FP32 | 26.21 s | 17.63 s | 1.40 s | 45.49 s |

A 72.0 s song now takes 45.5 s (0.63x real time); decode went from 17.96 to
14.55 ms per token. Token counts change where the logits change in the last
bits, so sampling diverges; the bit-identical rows replay the same song.

- **Weight alignment.** Only the safetensors payload is copied to the device,
  but the copy was shifted by the header's alignment, leaving every
  projection at 8 bytes mod 16. The decode GEMV then fell back to 8-byte
  loads. The shift now puts the large matrices on 256-byte boundaries; the
  GEMV reads at 239 GB/s in situ (the measured DRAM read ceiling is 235-245
  GB/s), so decode projections are at the roofline.
- **Acoustic attention.** The segmented kernel reloaded Q per key tile and
  staged probabilities through LDS: 4.45 ms per layer and evaluation, about
  15 TFLOP/s. The new kernel keeps each wave's 16 queries in registers,
  computes scores transposed so each lane owns one query's softmax, and reads
  values from a transposed copy (the prefix is transposed once per solve):
  2.59 ms, about 26 TFLOP/s. It runs at 238 VGPRs and close to the LDS read
  rate WMMA needs; more queries per wave do not fit in registers.
- **Decode attention.** 256-key splits gave 40 workgroups at 2000 keys and a
  serial 256-step value loop: 97 us per layer. 64-key splits with four lanes
  per key and 16-byte value loads take 36 us (about 216 GB/s of KV).
- **hipBLASLt candidates.** At 1800 rows the shared policy's second candidate
  (a 64-row tile) re-reads each weight tile 29 times; the first (128-row)
  candidate runs the 6144-wide gate/up projections in 1528 us instead of
  2161 us in situ. Other shapes and ranks measured within noise in situ, and
  a fused QKV projection saves about 6% of one GEMM in isolation, not enough
  for 470 MB of duplicated weights.
- **VAE.** hipBLAS runs the FP32 convolution GEMMs at about 3 TFLOP/s;
  hipBLASLt's 29th candidate (112x64 tile) is 2.6-3.2x faster on the 128- to
  2048-channel stages and as fast on the memory-bound 64-channel ones, and is
  correct for the unaligned tap offsets.

Rejected: several rows per GEMV block (same per-row order, bit-identical)
was slower in situ, 32.4 s and 34.4 s for two and four rows; one wave per row
reaches only 90-160 GB/s in isolation. 64-key acoustic tiles with register
prefetch gained 0.1 s over 32-key tiles.

Oracles (`yue2_hip_oracle_test`); the decode-path check is new and runs the
teacher-forced rows through one-token appends:

| Stage | Before | After |
| --- | --- | --- |
| Teacher-forced logits, worst cosine | 0.9996836 (26/26 argmax) | 0.9998944 (26/26) |
| Decode-path logits, worst cosine | 0.9998679 (25/26) | 0.9999079 (26/26) |
| Velocity at t = 1, relative error | 0.011344 | 0.011550 |
| 32-step latents, relative error | 0.013287 | 0.010156 |
| VAE audio, relative error | 6.48e-7 | 6.11e-7 |

The argmax misses are one-ulp BF16 ties in the upstream logits (gap 0.125 at
logits near 18). The functional music suite passes.

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
