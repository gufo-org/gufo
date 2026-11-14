# YuE2 benchmarks

2026-11-14, Windows gfx1151 (Strix Halo, 96 GB carve-out), TheRock 10.0.0,
`build/rock10` release binary. One `gufo music` request, model load excluded:
`examples/song.json` (style and lyrics, planning `full`, seed 831001).

| Stage | Work | Time |
| --- | --- | --- |
| Plan | 929 score tokens | 19.9 s (47 tok/s) |
| Semantic | 1996 codec tokens | 36.3 s (55 tok/s) |
| Acoustic | 32 midpoint steps over 1998 rows | 115.3 s |
| Decode | 2 tiles, 79.8 s of audio | 2.1 s |
| Total | 79.8 s song | 174.0 s (2.2x real time) |

The acoustic stage dominates: 64 non-causal passes whose attention uses the
fused WMMA kernel with an FP32 value product. Decode is weight-bandwidth bound
(about 2.9 GB of BF16 weights per token).
