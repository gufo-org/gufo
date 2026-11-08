# Extension conformance

What an extension must prove before it is published, reviewed for the
registry or promoted to core. Conformance is retained evidence for one
extension, source and API revision, platform and set of advertised
capabilities; there is no single certification command. The author workflow is
in [writing an extension](../writing-extensions.md).

## Evidence required from an author

- Pin Gufo, extension and reference revisions, compiler and dependencies,
  model artifacts, settings and selected suites. Retain the actual commands,
  output, failures, skipped cases and artifact hashes.
- Check registration, public-header isolation, GPU-free probes, advertised IDs,
  ambiguous selection, excluded keys, option types and unsupported inputs.
- Run real API inference and every advertised streaming, batching,
  cancellation, error-recovery and scheduler contract. A disconnected client
  alone doesn't prove cancellation reached the execution.
- Keep independent numerical references and model-quality tests. Stub runners
  and metadata-only template builds establish no numerical quality.
- Publish limitations and performance. Core extensions keep the repository's
  per-request, per-phase timing gates and release policy; community extensions
  supply their own evidence and never block a Gufo release. Missing weights or
  a skipped required mode is incomplete evidence, never a pass.

Test the artifact you distribute. Repeat install, provenance and the applicable
checks for every promised path: native Linux CMake, the Nix package, a module
installed with `gufo extension install`, the extracted Windows ZIP and the
published container digest. Record platform gaps explicitly.

A registry review is a source review against the registry's `REVIEW.md`, not
a conformance run; the extension's own documentation carries this evidence.

## Existing contracts and workloads

| Capability | Weight-free contract coverage | Real execution evidence |
| --- | --- | --- |
| Text generation | `text_model_runner_test`, `text_generation_scheduler_test`, HTTP envelope tests, extension runtime and option fixtures | `tests/functional/run.py` suites: discovery, responses, stops, conversation and batching; tool and forced-choice suites for tools; `long-context` and `cache` for continuation; declared draft modes with real drafts; cancellation, device loss and independent logit checks |
| Embeddings | `embedding_gemma2_embeddings_api_test`, embedding contracts | `tests/functional/model_services.py embedding`, with `--embedding-suite full` and independent references for every advertised modality |
| Speech recognition | `qwen3_asr_audio_api_test`, audio and WebSocket contracts | `tests/functional/model_services.py asr`: transcript quality, formats, SSE, concurrency, rejected inputs, cancellation and recovery, live checks when advertised |
| Speech synthesis | `qwen3_tts_audio_api_test`, voice and stream contracts | `tests/functional/model_services.py tts`: pinned independent ASR evaluator, audio validity, replay, streaming, concurrency, cancellation and recovery |
| Image generation | `qwen_image_21_test`, image API contracts | `tools/models/qwen_image_21/check_server.py` for generation, editing, replay and isolation, plus independent image quality and the standard speed benchmark |
| Video generation | `video_api_test`, job scheduling and cancellation contracts | Job submission, status, result, invalid input, cancellation and independent video and audio quality; `tools/h3/` is H3-specific, not an oracle for other models |
| Decisions | Decision registry and runner contracts, the provider's framing and calibration fixtures | `gufo decide` and `/v1/systemone` requests, advertised multimodal input, invalid-input recovery and independent score references; LFM3's `tests/functional/d1.py` is an example |

These suites cover the in-tree models. An external provider needs equivalent
assertions for its own contracts and its own independent fixtures; one model
can't inherit another's quality verdict. Keep model checks in the extension's
repository and HTTP workloads in `tests/functional/` or its functional
fragment.

A focused text run selects suites explicitly:

```sh
python3 tests/functional/run.py --output artifacts/text-conformance \
  --sampling-preset "$SAMPLING_PRESET" --expected-input-modalities text \
  --suite discovery --suite responses --suite stops --suite conversation --suite batch \
  -- /absolute/path/to/installed/gufo serve llm --model /absolute/path/to/model.gguf
```

The harness supports its declared Qwen and DeepSeek presets; a new model needs
its own harness adapter. Add tool suites when tools are advertised and
`long-context` and `cache` when continuation is. See the
[functional runner guide](../../tests/functional/README.md) for audio and
embedding recipes, oracle requirements and timing rules.

## Repository checks and the template

Weight-free suites exercise the public API and negative fixtures:

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

The template fixture checks compilation through public headers, registration,
diagnostic provenance and license installation. It has no runner, so its
`available` metadata must never be presented as inference. Generated CI,
documentation and functional registries are drift-checked by
`generate-infrastructure`, the public header closure by `check-extension-api`,
and target and include isolation by the checks in [testing](testing.md). Never
regenerate independent reference fixtures together with a candidate API change.

See [extension distribution](extension-distribution.md) for support states,
the registry, trust and install audits.
