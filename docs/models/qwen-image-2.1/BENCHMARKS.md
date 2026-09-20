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
