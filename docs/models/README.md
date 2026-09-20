# Models

Linux x86-64 / AMD Strix Halo gfx1151 is the production target. Build with
`nix build`; Python/reference toolchains are development-only. Model weights
are downloaded separately and are not part of the runtime package.

| Model | Weights / supported inputs | Guide |
| --- | --- | --- |
| DeepSeek V4 Flash | Flash 0731 mixed IQ2/Q2/Q8 GGUF; text | [Usage and modes](deepseek-v4-flash/README.md) |
| Qwen3.8 27B | Q4/Q8 GGUF; text/images | [Usage and modes](qwen3.8-27b/README.md) |
| Qwen3.8 Flash-Next | Sharded Q4 GGUF; text/images | [Usage and modes](qwen3.8-flash-next/README.md) |
| Qwen3-ASR 1.7B | BF16 safetensors; audio to text | [Usage](qwen3-asr/README.md) |
| Qwen3-TTS 12Hz 1.7B | BF16 safetensors; text/reference audio to speech | [Voice modes](qwen3-tts/README.md) |
| Qwen-Image-2.1 | BF16 safetensors; generation and image editing | [Usage](qwen-image-2.1/README.md) |
| MiniMax H3 FL2VA | Pinned safetensors; text to audiovisual output | [Presets and usage](minimax-h3/README.md) |

Each model folder contains:

- `README.md`: model card, acquisition, usage and modes.
- `BENCHMARKS.md`: current retained measurements, scope/date and TODO cells.
- `EVALUATION.md`: maintained tests, independent references and unresolved gaps.
- `EXPERIMENTS.md`: short retained/rejected decisions.
- `artifacts/`: only useful machine-readable results/calibration, when present.

Keep executable fixtures with their tests. New logit/trace dumps and local
profiles belong in ignored top-level `artifacts/`; commit only independent
reference evidence needed by a maintained check. Old experiment records remain
in Git history. See [benchmark methodology](../BENCHMARKS.md).
