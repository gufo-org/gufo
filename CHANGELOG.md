# Changelog

Notable user-facing changes are recorded here. Gufo follows
[Semantic Versioning](https://semver.org/) under the compatibility policy in
[the release guide](docs/RELEASING.md).

## [0.10.0](https://github.com/gufo-org/gufo/compare/v0.9.1...v0.10.0) (2026-10-09)


### Features

* **serve:** support tools and streaming on /v1/messages ([#469](https://github.com/gufo-org/gufo/issues/469)) ([50900eb](https://github.com/gufo-org/gufo/commit/50900eb717925cc8e85ed952e1a75e522958a7f6))


### Performance

* **ds4:** keep the 8-row IQ2 gate/up tiles out of scratch ([#482](https://github.com/gufo-org/gufo/issues/482)) ([e9b0f0b](https://github.com/gufo-org/gufo/commit/e9b0f0b34b41b1968dcf6eb2f215cbfd53ae4ebe))
* **qwen-flash:** accelerate prefill and MTP catch-up ([#485](https://github.com/gufo-org/gufo/issues/485)) ([fd747a5](https://github.com/gufo-org/gufo/commit/fd747a51951ccd09eda20de5c34313670dc9b9d3))
* **qwen-image:** unroll the fused attention value halves ([#481](https://github.com/gufo-org/gufo/issues/481)) ([b2438a4](https://github.com/gufo-org/gufo/commit/b2438a4a12717cd0970b46bb293875eaf31d9839))


### Documentation

* **cache:** add cache redesign implementation cards ([#489](https://github.com/gufo-org/gufo/issues/489)) ([d221a01](https://github.com/gufo-org/gufo/commit/d221a01c74052737a04148ef86a47e1db3e270b7))
* **cache:** propose continuation cache redesign RFC ([#488](https://github.com/gufo-org/gufo/issues/488)) ([b3facce](https://github.com/gufo-org/gufo/commit/b3facce1180e4bb44ac25c8f8a0fee348ae36da9))

## [0.9.1](https://github.com/gufo-org/gufo/compare/v0.9.0...v0.9.1) (2026-10-08)


### Bug Fixes

* **cache:** drop the fixed 1 GiB cap on automatic disk staging ([#473](https://github.com/gufo-org/gufo/issues/473)) ([b39c530](https://github.com/gufo-org/gufo/commit/b39c530e70e87f4340e2230a155fd16d066d19f3))


### Performance

* **hip:** keep prefill GEMMs spill-free on clang 23 ([#459](https://github.com/gufo-org/gufo/issues/459)) ([f17e37b](https://github.com/gufo-org/gufo/commit/f17e37b8bb7df5fb83ea7ce6d4dc4ef6d6677253))
* **qwen-flash:** 4096-token prefill chunks with an overlapped n-gram gather ([#470](https://github.com/gufo-org/gufo/issues/470)) ([47b6391](https://github.com/gufo-org/gufo/commit/47b639159315fcdba17e6a144d67e273de9ead6e))


### Code Refactoring

* **qwen:** centralize control-token literals in one header ([#464](https://github.com/gufo-org/gufo/issues/464)) ([7701ba7](https://github.com/gufo-org/gufo/commit/7701ba769f454a580ffcbaa29538d65e7f33b38c))

## [0.9.0](https://github.com/gufo-org/gufo/compare/v0.8.1...v0.9.0) (2026-10-07)


### Features

* **server:** add opt-in --trace for text request content ([#458](https://github.com/gufo-org/gufo/issues/458)) ([ba97a2b](https://github.com/gufo-org/gufo/commit/ba97a2b33f3ab0ca3864b9b7b51c62dff199268f))


### Bug Fixes

* **cache:** exclude free CMA pages from the automatic RAM budget ([88d2139](https://github.com/gufo-org/gufo/commit/88d213943e4f97c292f1561ccae32bb42e2b35c3))
* **cache:** keep learned branch points under RAM pressure ([#466](https://github.com/gufo-org/gufo/issues/466)) ([7da04cd](https://github.com/gufo-org/gufo/commit/7da04cd37f70eb1c455fae7ceb2306f0cbbf4173))
* **serve:** default the per-client queue cap to --max-pending ([#467](https://github.com/gufo-org/gufo/issues/467)) ([04205d4](https://github.com/gufo-org/gufo/commit/04205d44611fdccca2be0ae70e685894fbd0d19a))
* **server:** hoist mid-conversation system messages to the leading block ([#449](https://github.com/gufo-org/gufo/issues/449)) ([05688f9](https://github.com/gufo-org/gufo/commit/05688f94aec4ecae21ea9be0d9eb5c6516e18829))


### Performance

* **qwen-flash:** reduce prefill work at long context ([#463](https://github.com/gufo-org/gufo/issues/463)) ([33d1b20](https://github.com/gufo-org/gufo/commit/33d1b208a72e9664fdc1267ae1c0d465cfb9b8f9))
* **qwen:** avoid flash-next prompt checkpoint copies ([#445](https://github.com/gufo-org/gufo/issues/445)) ([82711d8](https://github.com/gufo-org/gufo/commit/82711d8c8ceb8bda9024914b6e7ac8495ba41742))

## [0.8.1](https://github.com/gufo-org/gufo/compare/v0.8.0...v0.8.1) (2026-10-06)


### Bug Fixes

* **server:** keep tool calls in native model syntax ([#441](https://github.com/gufo-org/gufo/issues/441)) ([21d6e64](https://github.com/gufo-org/gufo/commit/21d6e64f137f6bcbc5a8bf63f900cab648188df7))
* **server:** remove leading answer blank lines after reasoning ([#446](https://github.com/gufo-org/gufo/issues/446)) ([4ec92f2](https://github.com/gufo-org/gufo/commit/4ec92f2d48bf594503f06917a5d96e9da2978aa7))
* **vision:** accept image histories within model context ([#447](https://github.com/gufo-org/gufo/issues/447)) ([167bad6](https://github.com/gufo-org/gufo/commit/167bad69e8ac45f0fdf4bb376959a10caca2844c))

## [0.8.0](https://github.com/gufo-org/gufo/compare/v0.7.1...v0.8.0) (2026-10-05)


### Features

* OpenAI Responses compatibility ([#434](https://github.com/gufo-org/gufo/issues/434)) ([d921a4b](https://github.com/gufo-org/gufo/commit/d921a4bd956424241e3e050cf981023b5b81e475))

## [0.7.1](https://github.com/gufo-org/gufo/compare/v0.7.0...v0.7.1) (2026-10-05)


### Bug Fixes

* **cache:** keep the stable checkpoint before replaced trailing user context ([#407](https://github.com/gufo-org/gufo/issues/407)) ([3d73237](https://github.com/gufo-org/gufo/commit/3d732377e5ac257fa3f1a59f57a17163e65763b5))
* **serve:** accept the reasoning fields Claude Code sends to /v1/messages ([#405](https://github.com/gufo-org/gufo/issues/405)) ([56383be](https://github.com/gufo-org/gufo/commit/56383be0718ea0ce203fc42a55581304f0bbc253))
* **serve:** preserve active disk-cache writes during startup ([#392](https://github.com/gufo-org/gufo/issues/392)) ([207bb4e](https://github.com/gufo-org/gufo/commit/207bb4e0bf6cd0c666674794ec1f0dcf0eb49cb9))
* **serve:** reclaim output capacity from abandoned streams ([#394](https://github.com/gufo-org/gufo/issues/394)) ([d85fb0f](https://github.com/gufo-org/gufo/commit/d85fb0fcd3fb906ea845f6e1404222d686b74e07))
* **server:** end DeepSeek tool output after the call block ([#397](https://github.com/gufo-org/gufo/issues/397)) ([0df6ba3](https://github.com/gufo-org/gufo/commit/0df6ba3e1b5eade79782aacac559a02ea5468c2b))
* **server:** keep DeepSeek client markup as content, as llama.cpp does ([#420](https://github.com/gufo-org/gufo/issues/420)) ([a72fc1e](https://github.com/gufo-org/gufo/commit/a72fc1e58c8d4784e7b824b4aad93a76b5a197b1))
* **server:** keep tool-call framing out of assistant content ([#400](https://github.com/gufo-org/gufo/issues/400)) ([d910b92](https://github.com/gufo-org/gufo/commit/d910b92f9d722ee4a0ed8770c16749dea472afe2))
* **server:** preserve literal reasoning tags when thinking is disabled ([#391](https://github.com/gufo-org/gufo/issues/391)) ([6c0d493](https://github.com/gufo-org/gufo/commit/6c0d493eb3cbcbc8571648260d952a3198fab784))


### Performance

* **qwen-flash:** faster prefill projections, attention and indexer ([#421](https://github.com/gufo-org/gufo/issues/421)) ([653a318](https://github.com/gufo-org/gufo/commit/653a31830447be5068a448a0284e43669d52f436))
* **sampling:** skip vocabulary blocks that cannot change the selected tokens ([#415](https://github.com/gufo-org/gufo/issues/415)) ([b945d0a](https://github.com/gufo-org/gufo/commit/b945d0afdb26e4b790b8cb260762105e167ee1a3))


### Code Refactoring

* **serve:** keep Messages reasoning parsers together ([#428](https://github.com/gufo-org/gufo/issues/428)) ([c3f4a9c](https://github.com/gufo-org/gufo/commit/c3f4a9c26031b04f34b84a96f4d8cfc8f4337ad2))

## [0.7.0](https://github.com/gufo-org/gufo/compare/v0.6.0...v0.7.0) (2026-10-04)


### Features

* **serve:** learn shared-prefix boundaries in the RAM cache ([#386](https://github.com/gufo-org/gufo/issues/386)) ([2ba3be2](https://github.com/gufo-org/gufo/commit/2ba3be24039186e00d2cecdb969f99ad695e1161))
* **serve:** let an explicit RAM cache limit exceed the automatic budget ([#384](https://github.com/gufo-org/gufo/issues/384)) ([53c5906](https://github.com/gufo-org/gufo/commit/53c590649295edf63abcc06117231cdc890b69c7))
* **serve:** report live sessions and llama.cpp-compatible metrics ([#389](https://github.com/gufo-org/gufo/issues/389)) ([1b4e682](https://github.com/gufo-org/gufo/commit/1b4e6825f2cf6fa2203af3b4b3596bf25e4bfa4e))
* **server:** export speculative verification round metrics ([#403](https://github.com/gufo-org/gufo/issues/403)) ([8bdde80](https://github.com/gufo-org/gufo/commit/8bdde807e57fadfe57f4a1005707559ae6afc82f))


### Bug Fixes

* **server:** detect idle GPU loss and defer streaming success ([#406](https://github.com/gufo-org/gufo/issues/406)) ([6a9ea9f](https://github.com/gufo-org/gufo/commit/6a9ea9f263de4598a4c49954f10c5d285bcf7635))
* **server:** parse tool output using the admitted request format ([#393](https://github.com/gufo-org/gufo/issues/393)) ([c33e050](https://github.com/gufo-org/gufo/commit/c33e050eced6389852617994fe7349367df4c900))
* **server:** reuse replayed tool turns with union and typed arguments ([#404](https://github.com/gufo-org/gufo/issues/404)) ([ea06418](https://github.com/gufo-org/gufo/commit/ea064189976242f33f54bac92be6e0cdbd33fc48))

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
