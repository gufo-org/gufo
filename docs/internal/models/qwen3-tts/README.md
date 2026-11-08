# Qwen3-TTS 12Hz 1.7B implementation notes

Developer detail behind the [public model page](../../../models/qwen3-tts.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm](ROCM.md)

Native gfx1151 speech synthesis from BF16 safetensors, with an F32 waveform
decoder. Checkpoint configuration selects the variant; 0.6B, 25Hz and GGUF
checkpoints are unsupported. Official Python is an offline oracle only.

voices replaces it. This preserves generated audio and reduces warm latency.

## Sampling and streaming
