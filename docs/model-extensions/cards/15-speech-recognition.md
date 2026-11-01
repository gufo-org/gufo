# Card 15: Migrate speech recognition

Status: **complete**. RFC step and dependencies: **8 ASR; 14**.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Speech recognition moves its model calls and transcribe driver while the core keeps request queue policy. AudioBuffer and ResampleMono16k are shared by host consumers, so their placement must stop exposing ASR implementation types.

### Original scope

Migrate ASR and transcribe driver; move shared AudioBuffer/ResampleMono16k into host service; preserve streaming and core queue semantics. Metadata/registry/isolation and Linux/Windows checks; existing API/numerical quality and standard speed on actual default/lock artifacts.

## Implementation plan

1. Extract the ASR provider, artifact probe/resolution, IDs/options and transcribe CLI driver into its extension using public headers and generated constants.
2. Move shared AudioBuffer/ResampleMono16k to host audio services; rewire HTTP and other consumers without changing sample conversion or decoder behavior.
3. Preserve streaming chunk/SSE behavior, cancellation and core request-queue ownership; update packages/assets/test registration through the manifests.
4. Remove ASR-specific routing/build dependencies from core and enforce the model isolation ratchet.

## Definition of done

- [x] ASR registration and transcribe select the extension through neutral contracts with canonical option validation.
- [x] Audio resampling/output and streaming/cancellation/error recovery match existing behavior.
- [x] Manifest data and Linux/Windows packaging/builds are correct, with API, numerical and standard speed evidence for actual discovery and lock artifacts.

## Validation criteria

- Run tests/functional/model_services.py asr with the production binary, pinned model/audio fixture and a fresh output directory; require transcript WER, greedy replay, format/SSE equality, cancelled-request log correlation and valid recovery as defined in the functional README.
- Run existing ASR QUALITY.md numerical/unit commands and matched standard speed controls with per-request service phases; preserve the README service thresholds rather than applying text-only margins.
- Build ASR and shared host-audio consumers on Linux/Windows, test exclusion and HIP-free probe behavior, run the isolation ratchet and inspect default/lock runtime-library/data installation.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Authored `src/models/qwen3_asr/extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`.
  - Migrated `src/cli/serve/audio_buffer.hpp`/`.cpp` to host serving, decoupling audio resampling from ASR model types.
  - Published neutral ASR contracts in `<gufo/ext/asr.hpp>` (`AsrRunner`, `AsrProvider`, `TranscriptionRequest`, `TranscriptionResult`).
  - Separated manifest targets into `gufo_qwen3_asr` (unconditional registration, metadata) and `gufo_qwen3_asr_hip` (`requires: ["hip"]`).
  - Decoupled `src/cli/serve/asr_service.cpp` and `asr_websocket.cpp` to use extension registry providers.
- **Review Dispositions**:
  - Finding M2: Registered both `"qwen3-asr"` and `"qwen3-asr-1.7b"` provider IDs and updated CLI fallback checks.
  - User Finding U1: Partitioned HIP targets cleanly in manifest and CMake, passing full Nix HIP build.
- **Verification**:
  - `check-format.py`: PASS (816 files).
  - `check-extension-api.py`: PASS (108 paths).
  - `check-extension-isolation.py`: PASS (0 boundary violations).
  - `check-pr`: PASS (96/96 test suites passed).
  - `nix build`: PASS (`./result/bin/gufo diagnose` verifies `qwen3_asr` extension and `qwen3-asr` provider).
  - `nix build .#checks.x86_64-linux.pr --no-link`: PASS.
  - Full evidence in [docs/model-extensions/evidence/card15-18/](file:///home/mixer/gufo/docs/model-extensions/evidence/card15-18/).
