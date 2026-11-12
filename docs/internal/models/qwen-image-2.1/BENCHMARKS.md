# Qwen-Image-2.1 benchmarks

AMD Strix Halo gfx1151, BF16 checkpoint, production `nix build` binary.
Report wall time per complete image plus prompt, denoising and VAE time.
Warm runs exclude model loading. HTTP timings use no profiler or correctness
observer; GPU times come from a separate request-only profile.

Cold-file-cache control, 2026-09-21: metadata readiness **0.27 s**;
first 256×256/two-step generation **8.83 s**, subsequent identical request
**0.42 s**. First-request time includes loading active components.
Vision and VAE-encoder weights load only for editing. These short controls
measure loading and replay, not normal 40-step generation quality.

| Mode | Output | Steps | Prompt (s) | Denoising (s) | VAE (s) | Total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Generation, C1 | 1024×1024 | 40 | 0.086 | 89.03 | 0.77 | 89.89 |
| Edit, one reference, C1 | 1024×1024 | 40 | 3.11 | 126.30 | 0.81 | 130.23 |

Median of three warm runs, ROCm 10.1, seed 42, 2026-11-01.

## Qwen-Image-2.1-Turbo

Same host, build and method; separate Turbo weights with eight steps.

| Mode | Output | Steps | Prompt (s) | Denoising (s) | VAE (s) | Total (s) | vs base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generation, C1 | 1024×1024 | 8 | 0.088 | 17.78 | 0.76 | 18.63 | 4.8× |
| Edit, one reference, C1 | 1024×1024 | 8 | 2.99 | 27.16 | 0.77 | 30.93 | 4.2× |
| Edit, original plus mask, C1 | 1024×1024 | 8 | 5.01 | 35.37 | 0.77 | 41.16 | — |
| Generation, C1 | 1920×1088 | 8 | 0.091 | 47.97 | 2.57 | 50.63 | — |

Resident weights match the base model (28.5 GiB; 29.8 GiB with the editing
vision encoder).

Concurrent requests are isolated and queued; concurrent GPU batching is not
implemented. Do not report a two-step execution smoke check as model quality
or normal generation performance.

Short development control, warm C1, 1024×1024, two steps, seed 42:

| Prompt (s) | Denoising (s) | VAE (s) | HTTP total (s) |
| ---: | ---: | ---: | ---: |
| 0.089 | 4.865 | 0.826 | 5.838 |

A separate profile records **5.76 s of GPU work**: native projections 1.72 s,
fused feed-forward 1.41 s, fused attention 1.49 s and native convolution 0.55 s.
GPU idle time is 0.5% of the request span. The 5 s target has not been reached.
This control is for kernel iteration; the default 40-step performance remains
TODO and quality is measured separately in [QUALITY.md](QUALITY.md).

PNG uses low compression with adaptive filters and preserves every RGBA pixel.
The control's response PNG is approximately 1.00 MB; compression trades response
size for lower CPU latency.

## LoRA adapters

Adapters are merged into the weights, so denoising runs the same kernels.
Switching the selected set rebuilds the affected weights before the request
(`lora_ms` in the request log, also counted in Total); repeated requests with
the same set pay nothing.

| 1024×1024, 40 steps, seed 7 | Denoising (s) | `lora_ms` |
| --- | ---: | ---: |
| No adapters | 93.1-93.3 | 0 |
| One adapter, unchanged set | 93.1-93.3 | 0 |

About 2.33 s per step either way.

| Switch | `lora_ms` |
| --- | ---: |
| First selection after startup | ≈1795 |
| One adapter → another | ≈1790 |
| Adapter → none | 931 |

Measured on gfx1151 with the final merge implementation (host-widened factors,
pooled scratch). Stacked-adapter and DoRA switches were not re-measured.
