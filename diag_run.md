# Diagnostic: GPU inference hang isolation (issue #162 scouting, Q8_K_L file)

## Setup confirmed
- `result/bin/` exists. `result/bin/strix-server` present. No `strix-bench` (bench is `bench` subcommand of `strix-server`; used as such).
- Model present at M (Q8_K_L gguf). `[Model Load]: 1.15 s` — loader accepted it.
- NOTE: bench table labels model "Qwen3.8-27B BF16" but size 26.12 GiB (= ~28 GB), which matches Q8_K
  quantization, NOT BF16 (BF16 27B would be ~54.6 GiB). So the Q8_K_L file IS loading as quantized;
  "BF16" is only a display-name string.

## Result: NO HANG REPRODUCED
- `RUN1_EXIT=0` (complete, NOT 124). `RUN2_EXIT=0` (complete).
- Neither run hung; both finished well under the 100s timeout.

## STRIX_PROFILE does NOT print per-layer timing
Searched log: 0 lines contain "layer", "attn", "ssm", "ffn", "embed" as per-layer markers.
`STRIX_PROFILE` emits per-batch section timings, NOT per-layer:
`Norms / Input Proj / SSM Recur / SSM Out / FFN (3 GEMM)`. Per-layer hang localization with this
binary's output format is not possible.

## RUN1 — LAST 20 lines of diag_run1.log (verbatim)
```
[STRIX_PROFILE B=4] Total: 1185.09 ms (3.38 tok/s)
  - Norms:      1.37 ms
  - Input Proj: 318.59 ms
  - SSM Recur:  3.00 ms
  - SSM Out:    103.37 ms
  - FFN (3 GEMM): 758.74 ms

[STRIX_PROFILE B=4] Total: 1180.23 ms (3.39 tok/s)
  - Norms:      1.38 ms
  - Input Proj: 315.05 ms
  - SSM Recur:  3.17 ms
  - SSM Out:    102.66 ms
  - FFN (3 GEMM): 757.95 ms
| Qwen3.8-27B BF16               |  26.12 GiB |    27.32 B | ROCm (HIP) |  99 |             pp4 |          3.33 ± 0.00 |

[STRIX_PROFILE B=16] Total: 1260.86 ms (12.69 tok/s)
  - Norms:      1.48 ms
  - Input Proj: 336.63 ms
  - SSM Recur:  6.54 ms
  - SSM Out:    110.47 ms
  - FFN (3 GEMM): 805.71 ms
| Qwen3.8-27B BF16               |  26.12 GiB |    27.32 B | ROCm (HIP) |  99 |             tg2 |          0.08 ± 0.00 |
```
No error / hang / abort lines. Completed all batches (B=32, B=4, B=4, B=16).

## RUN2 — LAST 20 lines of diag_run2.log (verbatim, file is 8 lines)
```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 126976 MiB):
  Device 0: AMD Radeon 8060S Graphics, gfx1151, Wave Size: 32, VRAM: 126976 MiB
[Model Load]: 1.15535 s
| model                          |       size |     params | backend    | ngl |            test |                   t/s |
| ------------------------------ | ---------- | ---------- | ---------- | --- | --------------- | --------------------- |
| Qwen3.8-27B BF16               |  26.12 GiB |    27.32 B | ROCm (HIP) |  99 |             pp1 |          0.83 ± 0.00 |
| Qwen3.8-27B BF16               |  26.12 GiB |    27.32 B | ROCm (HIP) |  99 |             tg1 |          0.08 ± 0.00 |
```

## GPU util% during RUN1 (rocm-smi, sampled every 15s)
```
t=15s   GPU%  100%  (temp 61°C, pwr 81.4W, VRAM 39%)
t=30s   GPU%  100%  (temp 63°C, pwr 82.8W, VRAM 39%)
t=45s   GPU%    8%  (ramping down; VRAM 29%)
t=60s   GPU%    0%  (idle)
t=75s   GPU%    0%  (idle)
```
GPU saturates for ~30-35s (the benchmark), then idles. Process finished ~35-40s total. No stuck-at-100% spin.

## HYPOTHESIS
No hang occurs in the basic `bench` path (`-p 4 -n 2`, `-p 1 -n 1`) on this machine/session today; both
complete in seconds. The prior "chronic >5min hang" is NOT triggered by these commands. The hang trigger
likely lives in a *different* config not exercised here: long-generation `-n` decode loops, `--validate-prefill`,
a specific batch size, or a larger working set/sequence length — NOT in model load, embed prefill, or an
early layer. Need a longer/different command (e.g. `-p 128 -n 128` or `--validate-prefill 128`) to reproduce.

## Residual risks
- GPU util sampled only on RUN1, and only every 15s; may miss short spikes.
- RUN1 log has no per-layer timing, so exact hang layer cannot be pinned with this binary's profile format.
- "BF16" label vs Q8_K size mismatch should be reconciled (is the file being dequantized to BF16 on load,
  i.e. 26.12 GiB is the on-disk Q8_K but run happens BF16 after dequant?). Worth confirming; if so, the
  Q8 path is NOT actually being executed end-to-end yet.
