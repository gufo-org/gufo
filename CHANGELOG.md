# Changelog

Notable user-facing changes are recorded here. Gufo follows
[Semantic Versioning](https://semver.org/) under the compatibility policy in
[the release guide](docs/RELEASING.md).

## [0.2.0](https://github.com/gufo-org/gufo/compare/v0.1.1...v0.2.0) (2026-09-29)


### Features

* **sampling:** use official text-model defaults ([#282](https://github.com/gufo-org/gufo/issues/282)) ([eb91584](https://github.com/gufo-org/gufo/commit/eb915840ffb62a8ec4b5c1adb41b04b5c1c75892))


### Bug Fixes

* **serve:** bound hardware compute queues per server ([#317](https://github.com/gufo-org/gufo/issues/317)) ([9da89d6](https://github.com/gufo-org/gufo/commit/9da89d64b03c13f76085f6221074e927f9d91472))
* **server:** accept dotted and namespaced tool names ([#314](https://github.com/gufo-org/gufo/issues/314)) ([fee9d2a](https://github.com/gufo-org/gufo/commit/fee9d2a4c17ea2ff43d36672d9e693029bb810d4))

## [0.1.1](https://github.com/gufo-org/gufo/compare/v0.1.0...v0.1.1) (2026-09-28)


### Bug Fixes

* **server:** handle repeated tool-call parameters ([30392d5](https://github.com/gufo-org/gufo/commit/30392d5bbe96dc925f81e955d9e0285f4351ff34))

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
