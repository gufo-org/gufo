# Capability conformance for source extensions

Conformance is retained evidence for a particular extension, source/API revision,
platform and advertised capability set. There is no universal certification
command. Shared API fixtures, model numerical checks and production HTTP
workloads establish different parts of the contract.

Start with the [extension author guide](../extensions.md) for implementation and
the real LFM3 example.

## Evidence required from an extension author

- Pin Gufo, extension and reference revisions, compiler/dependencies, model
  artifacts, settings and selected suites. Retain actual commands, output,
  failures, skipped cases and artifact hashes.
- Check registration, public-header isolation, HIP-free probes, advertised IDs,
  ambiguous selection, excluded keys, option types and unsupported inputs.
- Run actual API inference, advertised streaming/batching/cancellation contracts,
  error recovery and capability-specific scheduler boundaries. A disconnected
  client alone does not prove cancellation reached the execution request.
- Keep independent numerical references and model-quality tests. Stub runners
  and metadata-only template builds do not establish numerical quality.
- Publish limitations and performance measurements. Core extensions retain the
  repository's matched per-request/per-phase timing gates and release policy.
  Community extensions supply quality/performance evidence without blocking
  Gufo releases on their speed. Missing weights or a skipped required mode is
  incomplete evidence, never a quality pass.

Use the same installed executable that will be distributed. Repeat install,
provenance and applicable conformance checks for each promised path: native
Linux/CMake, the Nix package, the extracted native Windows ZIP and the published
container digest. Record platform-specific gaps explicitly.

## Existing contracts and production workloads

| Capability | Weight-free contract coverage | Actual execution evidence |
| --- | --- | --- |
| Text generation | `text_model_runner_test`, `text_generation_scheduler_test`, HTTP envelope tests, extension runtime/options fixtures | Explicit suites in `tests/functional/run.py`: discovery, responses, stops, conversation and batching; tool/forced-choice suites for tool support; long-context/cache for continuation; declared draft modes with actual drafts; cancellation/device-loss execution and independent logits/quality checks |
| Embeddings | `embedding_gemma2_embeddings_api_test`, extension embedding contracts | `tests/functional/model_services.py embedding`; select `--embedding-suite full` for qualification, preserving independent text/image/audio/video references for every advertised modality |
| Speech recognition | `qwen3_asr_audio_api_test`, audio/WebSocket contracts | `tests/functional/model_services.py asr`; transcript quality, formats/SSE, concurrency, rejected inputs, cancellation/recovery; additional live/WebSocket checks when advertised |
| Speech synthesis | `qwen3_tts_audio_api_test`, voice/stream contracts | `tests/functional/model_services.py tts`; use the checkpoint's variant, pinned independent ASR evaluator, audio validity/intelligibility, deterministic replay, streaming, concurrency and cancellation/recovery |
| Image generation | `qwen_image_21_test`, image HTTP API contracts | `tools/models/qwen_image_21/check_server.py` exercises the existing provider's HTTP generation/edit/replay/isolation; retain independent image quality and the standard speed benchmark separately |
| Video generation | `video_api_test`, job scheduling/cancellation contracts | Actual job submission/status/result, invalid input, cancellation/recovery and independent video/audio quality; the existing H3 tools live under `tools/h3/` and are model-specific, not an oracle for another model |
| Decisions | Public decision registry/runner contracts and the provider's host framing/calibration fixtures | Actual `gufo decide` and `/v1/systemone` requests, advertised multimodal input, invalid-input recovery and independent score/calibration references; LFM3's external `tests/functional/d1.py` is a concrete provider-owned example |

These named model suites cover the current in-tree implementations. An external
provider needs equivalent assertions for its advertised contracts and its own
independent numerical fixtures; a different model cannot inherit another model's
quality verdict. Keep model-specific checks in its source repository and HTTP
regression workloads in `tests/functional/` or its external functional fragment.

For a focused text run, set the required preset and modality expectation to match
the provider, then select suites explicitly:

```sh
python3 tests/functional/run.py --output artifacts/text-conformance \
  --sampling-preset "$SAMPLING_PRESET" --expected-input-modalities text \
  --suite discovery --suite responses --suite stops --suite conversation --suite batch \
  -- /absolute/path/to/installed/gufo serve llm --model /absolute/path/to/model.gguf
```

The existing harness supports its declared Qwen/DeepSeek presets; a new model
must provide an appropriate harness adapter instead of choosing an incompatible
preset. Add the relevant tool suites when advertised and `long-context`/`cache`
when continuation is implemented. Do not use an all-model sweep for a packaging
or documentation edit. See [functional runner guidance](../../tests/functional/README.md)
for the executable audio/embedding recipes, oracle requirements and timing rules.

## Repository contract checks and the external template

The bounded extension suites exercise the public source API and independent
negative fixtures without loading model weights:

```sh
nix develop -c python3 tests/extensions/metadata_test.py
nix develop -c python3 tests/extensions/metadata_test.py --schema
nix develop -c python3 tests/extensions/discovery_test.py
nix develop -c python3 tests/tools/check_extension_api_test.py
nix develop -c python3 tests/tools/extension_distribution_test.py
python3 tools/ci/generate-infrastructure.py --check
python3 tools/ci/check-extension-api.py
python3 tools/ci/check-dependencies.py
```

The external template fixture checks actual compilation through public headers,
registration, diagnostic provenance and installation of its license. It has no
`EmbeddingRunner` implementation; `available` metadata must not be presented as
successful embedding inference. Use a genuine external provider for its model's
actual capability evidence.

For every declared generated output, run its drift check. Generated model CI,
documentation and functional registries are checked by `generate-infrastructure`;
the public header closure by `check-extension-api`; target/include isolation by
the repository checks described in [TESTING.md](../TESTING.md). An author-owned
generator needs its own deterministic drift gate. Do not regenerate independent
reference fixtures together with a candidate API change and call them an oracle.

See [distribution and support](DISTRIBUTION.md) for promotion, abandoned
maintainers, native-code trust and installed artifact checks.
