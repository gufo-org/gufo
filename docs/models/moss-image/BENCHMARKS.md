# MOSS-Image benchmarks

AMD Strix Halo gfx1151 on **Windows 11** (TheRock 10.0, `windows-release`
preset), BF16 transformer/VAE safetensors and the BF16 GGUF text encoder,
2026-10-30. One server, requests in the order listed, seeds as given. Linux
`nix build` numbers are TODO.

| Output | Steps | Prompt (s) | Denoising (s) | VAE (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512×512, first request | 20 | 3.58 | 17.67 | 0.47 | 21.73 |
| 512×512 | 30 | 0.61 | 26.13 | 0.38 | 27.12 |
| 1024×1024 | 50 | 0.61 | 169.17 | 1.80 | 171.60 |
| 1024×1024, second prompt | 50 | 0.77 | 176.77 | 1.65 | 179.21 |

Prompt time includes both prompts (classifier-free guidance); the first request
also uploads the transformer. Every step runs the transformer twice (positive
and negative branch). The server answered `/health` 4.1 s after start; weights
upload on first use.

Peak GPU memory of the server process (Windows per-process counters,
1 s samples): 28,161 MiB dedicated after the 512² requests, 29,391 MiB during
1024² requests, plus about 0.5 GiB shared. About 15.3 GiB is the BF16 text
encoder and 9.5 GiB the transformer; the rest is activations and the pool.
