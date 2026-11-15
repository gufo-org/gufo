# YuE2 benchmarks

Windows gfx1151 (Strix Halo, 96 GB carve-out), TheRock 10.0.0, `build/rock10`
release binary. One `gufo music` request, model load excluded:
`examples/song.json` (style and lyrics, seed 831001).

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
