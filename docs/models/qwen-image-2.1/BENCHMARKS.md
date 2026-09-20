# Qwen-Image-2.1 benchmarks

AMD Strix Halo gfx1151, BF16 checkpoint, production `nix build` binary.
Report wall time per complete image plus prompt, denoising and VAE time.
Warm runs exclude model loading. Profiles and correctness observers are untimed.

| Mode | Output | Steps | Prompt (s) | Denoising (s) | VAE (s) | Total (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Generation, C1 | 1024×1024 | 40 | TODO | TODO | TODO | TODO |
| Edit, one reference, C1 | 1024×1024 | 40 | TODO | TODO | TODO | TODO |

Concurrent requests are isolated and queued; concurrent GPU batching is not
implemented. Do not report a two-step execution smoke check as model quality
or normal generation performance.

Short development control, warm C1, 1024×1024, two steps, seed 42:

| Prompt (s) | Denoising (s) | VAE (s) | HTTP total (s) |
| ---: | ---: | ---: | ---: |
| 0.091 | 6.407 | 1.518 | 8.311 |

A separate request-only profile records 7.90 s of GPU work: native projections
3.89 s, fused attention 1.85 s, other BLAS 0.98 s. GPU idle time is 1.4% of the
request span. This control is for kernel iteration, not the default 40-step
performance or image-quality measurement.
