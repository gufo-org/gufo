# Changelog

Notable user-facing changes are recorded here. Gufo follows
[Semantic Versioning](https://semver.org/) under the compatibility policy in
[the release guide](docs/RELEASING.md).

## [0.1.0] - 2026-09-28

Initial public development release for AMD Strix Halo (`gfx1151`).

### Features

- Native text inference and OpenAI-compatible serving for Qwen3.8 27B,
  Qwen3.8 Flash-Next and DeepSeek V4 Flash.
- Speech recognition with Qwen3-ASR and speech synthesis and voice cloning
  with Qwen3-TTS.
- Image generation with Qwen-Image-2.1 and experimental MiniMax H3
  video/audio generation.
- Continuous batching, request cancellation and conversation caching as
  first-class serving workloads.
- Reproducible Nix and CMake production builds specialized for Strix Halo.

### Performance

- Model-owned HIP kernels and speculative decoding paths for DFlash2, MTP and
  DSpark, with published matched quality and performance reports.

[0.1.0]: https://github.com/gufo-org/gufo/releases/tag/v0.1.0
