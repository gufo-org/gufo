# Qwen3-ASR 1.7B implementation notes

Developer detail behind the [public model page](../../../models/qwen3-asr.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm](ROCM.md)

Native BF16 safetensors speech recognition on gfx1151. The 1.7B checkpoint is
supported; 0.6B, forced alignment and audio GGUFs are unsupported. Python runs
only offline reference checks. Generation is deterministic greedy decoding.

isolated by FIFO admission between audio chunks; cancellation removes queued
work. `--context` controls capacity per chunk; `--served-model-name` sets
the public model ID (repeatable; extra names are accepted aliases). TTS runs
separately with `gufo serve tts`.
