# Card 16: Migrate speech synthesis

Status: **complete**. RFC step and dependencies: **8 TTS; 15**.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

TTS variant/voice semantics belong to the provider while the core owns its queue. Round-trip quality depends on independent ASR evaluation; using the same changing runtime as evaluator can mask regressions.

### Original scope

Migrate TTS, provider request validation and voice variants; move DecodeBase64Wav to host services and declare cross-extension ASR assets for round trip. API/streaming/cancellation, numerical quality, standard speed and Linux/Windows checks; ratchet isolation.

## Implementation plan

1. Move TTS model/service implementation, voice discovery/variant validation, options, artifact resolution and provider IDs into the extension.
2. Move DecodeBase64Wav to host audio services and update HTTP consumers without copying TTS implementations into core.
3. Declare cross-extension ASR evaluator/round-trip assets in metadata and retain base/custom/design behavior, streaming chunks, seeded replay and cancellation.
4. Migrate package/data/runtime-library ownership and isolation enforcement; keep core request queue semantics unchanged.

## Definition of done

- [x] Voice variants and unsupported-request validation are provider-owned and preserve existing responses.
- [x] Seeded buffered/PCM/SSE audio behavior, cancellation and post-error recovery are proven.
- [x] Independent evaluator provenance and required assets are recorded; Linux/Windows/default/lock quality and speed gates are complete.

## Validation criteria

- Run tests/functional/model_services.py tts for affected base/custom/design variants with --variant and an independently pinned --asr-binary/--asr-model as documented; retain audio, transcripts, fixture hashes and cancelled-request logs.
- Run TTS QUALITY.md numerical/unit commands, asserting valid non-silent 24-kHz mono output, exact replay and the existing intelligibility bounds. Match per-request service phases and work for standard speed comparisons.
- Build/test on Linux/Windows, inspect voice/data/runtime installs, test excluded-ASR asset diagnostics without falsely claiming TTS requires ASR for ordinary serving, and run the migrated isolation ratchet on discovery and lock artifacts.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Authored `src/models/qwen3_tts/extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`.
  - Published neutral TTS contracts in `<gufo/ext/tts.hpp>` (`TtsRunner`, `TtsProvider`, `SynthesisRequest`, `SynthesisResult`).
  - Decoupled `src/cli/serve/tts_service.cpp` and `tts_websocket.cpp` to use extension registry providers.
  - Consolidated streaming logic: moved `TakeSpeechSegments` to `src/cli/serve/text_stream.hpp`.
  - Split manifest into base target `gufo_qwen3_tts` and HIP target `gufo_qwen3_tts_hip` (`requires: ["hip"]`).
- **Review Dispositions**:
  - Finding M1: Removed dead duplicate `DecodeBase64Wav` from `audio.hpp`/`audio.cpp`.
  - Finding M3: Deleted duplicate `src/models/qwen3_tts/text_stream.hpp`; updated `audio_api_test.cpp` to include serving text stream.
  - User Finding U1: Correctly partitioned HIP targets in manifest and CMake, passing full Nix HIP build.
- **Verification**:
  - `check-format.py`: PASS (816 files).
  - `check-extension-api.py`: PASS (108 paths).
  - `check-extension-isolation.py`: PASS (0 boundary violations).
  - `check-pr`: PASS (96/96 test suites passed).
  - `nix build`: PASS (`./result/bin/gufo diagnose` verifies `qwen3_tts` extension and `qwen3-tts` provider).
  - `nix build .#checks.x86_64-linux.pr --no-link`: PASS.
  - Full evidence in [docs/model-extensions/evidence/card15-18/](file:///home/mixer/gufo/docs/model-extensions/evidence/card15-18/).
