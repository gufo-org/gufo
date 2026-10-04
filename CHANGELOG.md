# Changelog

Notable user-facing changes are recorded here. Gufo follows
[Semantic Versioning](https://semver.org/) under the compatibility policy in
[the release guide](docs/RELEASING.md).

## [0.6.0](https://github.com/gufo-org/gufo/compare/v0.5.0...v0.6.0) (2026-10-03)


### Features

* **serve:** share in-flight prefixes between concurrent requests ([#382](https://github.com/gufo-org/gufo/issues/382)) ([c1eba3d](https://github.com/gufo-org/gufo/commit/c1eba3de7ffc1f71c9ad6e395138f9bf89ddcc7b))


### Bug Fixes

* **serve:** exit and report device_lost when the GPU context is lost ([#390](https://github.com/gufo-org/gufo/issues/390)) ([ee2bff3](https://github.com/gufo-org/gufo/commit/ee2bff34d8cbb95a29f8abe85e046dd49382b5c9))
* **server:** improve error messaging on streaming generation failure ([#385](https://github.com/gufo-org/gufo/issues/385)) ([bf60539](https://github.com/gufo-org/gufo/commit/bf605399477b694c19adb6313952ecbf0986d8be))
* **server:** preserve JSON string ownership during tool recovery ([#396](https://github.com/gufo-org/gufo/issues/396)) ([2c6a106](https://github.com/gufo-org/gufo/commit/2c6a1064f39a4d3ea0b8d92efea0beedf18150f1))
* **serve:** separate Messages thinking blocks and accept the thinking field ([#380](https://github.com/gufo-org/gufo/issues/380)) ([b27f1ec](https://github.com/gufo-org/gufo/commit/b27f1ec0189e5410028029f249915dfa8813b749))

## [0.5.0](https://github.com/gufo-org/gufo/compare/v0.4.0...v0.5.0) (2026-10-02)


### Features

* **server:** accept WebP and other spellings of image data URLs ([#352](https://github.com/gufo-org/gufo/issues/352)) ([d707143](https://github.com/gufo-org/gufo/commit/d707143223c52b91da3f0fb231ba85ac3eab247c))


### Bug Fixes

* **cache:** keep cache reuse advancing as conversations grow ([#358](https://github.com/gufo-org/gufo/issues/358)) ([9afdd48](https://github.com/gufo-org/gufo/commit/9afdd4802fbe4f4afeb5607a8cc4da905f8e4d70))
* **cache:** keep cached conversations independent of execution sessions ([#369](https://github.com/gufo-org/gufo/issues/369)) ([d7cf7ee](https://github.com/gufo-org/gufo/commit/d7cf7ee4b8e3a506ee1ec9c301e8a3efd10c6c94))
* **cache:** retain prefixes across conversation history edits ([#362](https://github.com/gufo-org/gufo/issues/362)) ([68e8475](https://github.com/gufo-org/gufo/commit/68e8475dbdf9dcddbe0a3ea645646b453b0bb89c))
* **qwen-image:** prevent source-noise reuse in image edits ([#377](https://github.com/gufo-org/gufo/issues/377)) ([1071b36](https://github.com/gufo-org/gufo/commit/1071b361eb69e561125543382ebbf3f212c2cc5a))
* **server:** advertise loaded model input modalities ([#367](https://github.com/gufo-org/gufo/issues/367)) ([5451525](https://github.com/gufo-org/gufo/commit/54515255300de99a8cbbae192d25b42959b5c973))
* **server:** keep literal tool markers inside constrained reasoning ([#361](https://github.com/gufo-org/gufo/issues/361)) ([2ab0c4b](https://github.com/gufo-org/gufo/commit/2ab0c4b70ffec6be8c306a6ed299acef76c903e7))
* **server:** preserve native tool schemas and historical calls ([#373](https://github.com/gufo-org/gufo/issues/373)) ([594a623](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e))


### Performance

* **serve:** skip checkpoints that barely advance a prefix ([#348](https://github.com/gufo-org/gufo/issues/348)) ([93af45d](https://github.com/gufo-org/gufo/commit/93af45d48a712d2c14105ef5a9347942e2260f14))

## [0.4.0](https://github.com/gufo-org/gufo/compare/v0.3.0...v0.4.0) (2026-10-01)


### Features

* **serve:** expose live token and request metrics ([#351](https://github.com/gufo-org/gufo/issues/351)) ([03d7c72](https://github.com/gufo-org/gufo/commit/03d7c727c48c62960549ef5f27cd2ea9897c873e))
* **serve:** report cache eviction and retained snapshot capacity ([#353](https://github.com/gufo-org/gufo/issues/353)) ([bda078c](https://github.com/gufo-org/gufo/commit/bda078c3c3b087bd4bf5a9b09f4fd38653eb2e94))
* **server:** make log verbosity configurable with --log-level ([#319](https://github.com/gufo-org/gufo/issues/319)) ([6a32726](https://github.com/gufo-org/gufo/commit/6a32726068f21f94db0761edba1da00eefe99c4b))
* **server:** support llama-server return_progress on streaming completions ([#344](https://github.com/gufo-org/gufo/issues/344)) ([741722b](https://github.com/gufo-org/gufo/commit/741722b04eb474b3d75c7e2c0a4d727ae21f6577))


### Bug Fixes

* **qwen-flash:** preserve seeded MTP replay across cache reuse ([#330](https://github.com/gufo-org/gufo/issues/330)) ([a917b79](https://github.com/gufo-org/gufo/commit/a917b790df8d5fd98205abddd3b7c1afd0ca9458))
* **serve:** preserve native tool calls during constrained decoding ([#324](https://github.com/gufo-org/gufo/issues/324)) ([b26de0d](https://github.com/gufo-org/gufo/commit/b26de0d30caa363cc6694bddd094af9bd88ec62b))
* **server:** keep SSE streams alive during generation ([#334](https://github.com/gufo-org/gufo/issues/334)) ([c6e1069](https://github.com/gufo-org/gufo/commit/c6e10690c7532a99257b58319ca619a52351ddca))


### Performance

* **qwen-flash:** accelerate greedy penalties and fix sampling ranges ([#332](https://github.com/gufo-org/gufo/issues/332)) ([7e450e8](https://github.com/gufo-org/gufo/commit/7e450e8c0bc5458d056b34675e6d7f844d0ef23f))


### Documentation

* explain how the KV cache works in gufo ([#360](https://github.com/gufo-org/gufo/issues/360)) ([446cb14](https://github.com/gufo-org/gufo/commit/446cb141f8b312befdbfb58945c01db3897ba09a))

## [0.3.0](https://github.com/gufo-org/gufo/compare/v0.2.0...v0.3.0) (2026-09-30)


### Features

* **cli:** print startup banner on interactive commands ([#323](https://github.com/gufo-org/gufo/issues/323)) ([f783fed](https://github.com/gufo-org/gufo/commit/f783fedb9bea2ec7de941f6da4e02f4a4596b29e))


### Documentation

* update readme with link to gufo forks ([#327](https://github.com/gufo-org/gufo/issues/327)) ([8eedee6](https://github.com/gufo-org/gufo/commit/8eedee6fd904b8e6812f740f777fe84940f341c5))

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
