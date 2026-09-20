# Qwen-Image-2.1 benchmarks

AMD Strix Halo gfx1151, BF16 checkpoint, production `nix build` binary.
Report wall time per complete image plus prompt, denoising and VAE time.
Warm runs exclude model loading. HTTP timings use no profiler or correctness
observer; GPU times come from a separate request-only profile.

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
| 0.089 | 4.865 | 0.826 | 5.838 |

A separate profile records **5.76 s of GPU work**: native projections 1.72 s,
fused feed-forward 1.41 s, fused attention 1.49 s and native convolution 0.55 s.
GPU idle time is 0.5% of the request span. The 5 s target has not been reached.
This control is for kernel iteration; the default 40-step performance remains
TODO and quality is measured separately in [EVALUATION.md](EVALUATION.md).

PNG uses low compression with adaptive filters and preserves every RGBA pixel.
The control's response PNG is approximately 1.00 MB; compression trades response
size for lower CPU latency.
