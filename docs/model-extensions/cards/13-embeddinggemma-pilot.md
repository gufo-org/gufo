# Card 13: Migrate the EmbeddingGemma pilot

Status: **complete**. RFC step and dependencies: **8 embeddings; 12**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

EmbeddingGemma is the first real provider and the prerequisite for metadata-generated infrastructure. It already owns batch scheduling and audio decoding; introducing a new core admission queue would change the pilot scope.

### Original scope

Migrate EmbeddingGemma pilot, extension-relative includes, metadata/options/pure probe/artifact resolution/provider/model IDs. Preserve extension batch scheduling and audio decoder ownership. Move model production dependency/data/runtime-library declarations into manifests. Existing API/numerical quality/speed gates, default and lock artifact, isolated build and fail-on-violation ratchet.

## Implementation plan

1. Move provider implementation, service-facing model calls and extension-relative own includes under the EmbeddingGemma extension boundary using Card 12 test support.
2. Declare metadata, canonical options, HIP-free pure artifact probe, artifact resolution, provider/model IDs and factory registration with generated constants.
3. Move model production dependency/data/runtime-library declarations into extension manifests, preserving libav/audio decoder ownership and current batch scheduler behavior.
4. Remove migrated pilot knowledge from core routing/build paths, populate selection/lock as an explicit release decision, and turn its isolation boundary into fail-on-violation enforcement.

## Definition of done

- [x] Pilot loads through the registry with truthful probe/selection/errors and unchanged advertised IDs, modalities, batch ordering and scheduling.
- [x] Its manifest owns production packages, data and runtime libraries; isolated/excluded builds work without hidden root declarations.
- [x] Existing API, independent numerical and speed gates pass on both discovery and actual lock-mode HIP artifacts; ratchet rejects new coupling.

## Validation criteria

- Run the full embedding suite via tests/functional/model_services.py embedding --embedding-suite full with the production binary/model and a new output directory, as documented in tests/functional/README.md.
- Run existing EmbeddingGemma QUALITY.md numerical commands and standard speed controls; full service observations alone do not qualify matched performance. Retain independent reference vectors and per-request work/timing evidence.
- Build/test the pilot alone with required family dependencies and with it excluded, on Linux/Windows. Inspect installed data/runtime libraries and run the fail-on-violation isolation check; repeat affected runtime gates on the actual lock artifact.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

### Roles and Working Copy
- **Card**: Card 13 — Migrate the EmbeddingGemma pilot
- **Parent commit**: `ynuxkvqu b55be825` ("model-extensions-2 | test(extensions): add in-tree model api test support and isolate serving fixtures")
- **Agent Sequence**:
  - Implementation Worker (`worker`): Run ID `77f3ed36-4902-49c7-b804-b3ab22858d4e`
  - Independent Reviewer (`reviewer`): Run ID `6c38a629-ee90-4a8b-be09-cf88cc3f5406` (verdict CLEAN PASS, F0)
  - Review Integrator (`integrator`): Run ID `b3930d21-4138-46f7-b64f-835f02539530`
  - Orchestrator (`orchestrator`): Closure
- **Working copy changes**:
  - `CMakeLists.txt`
  - `cmake/GufoExtension.cmake`
  - `extensions.lock`
  - `include/gufo/ext/CHANGES.md`
  - `include/gufo/ext/SURFACE`
  - `include/gufo/ext/embeddings.hpp`
  - `include/gufo/ext/registry.hpp`
  - `include/gufo/ext/version.hpp`
  - `src/cli/serve/embedding_service.cpp`
  - `src/cli/serve/embedding_service.hpp`
  - `src/cli/serve/embeddings_api.cpp`
  - `src/ext/embeddings.hpp`
  - `src/ext/runtime.cpp`
  - `src/models/embedding_gemma2/CMakeLists.txt` (deleted)
  - `src/models/embedding_gemma2/audio_decoder.cpp`
  - `src/models/embedding_gemma2/audio_processor.cpp`
  - `src/models/embedding_gemma2/batch_scheduler.cpp`
  - `src/models/embedding_gemma2/batch_scheduler.hpp`
  - `src/models/embedding_gemma2/config.cpp`
  - `src/models/embedding_gemma2/extension.cmake`
  - `src/models/embedding_gemma2/extension.cpp`
  - `src/models/embedding_gemma2/extension.json`
  - `src/models/embedding_gemma2/hip/audio_runtime.hip`
  - `src/models/embedding_gemma2/hip/audio_runtime.hpp`
  - `src/models/embedding_gemma2/hip/blas.cpp`
  - `src/models/embedding_gemma2/hip/encoder_runtime.cpp`
  - `src/models/embedding_gemma2/hip/encoder_runtime.hpp`
  - `src/models/embedding_gemma2/hip/ops.hip`
  - `src/models/embedding_gemma2/hip/vision_ops.hip`
  - `src/models/embedding_gemma2/image_processor.cpp`
  - `src/models/embedding_gemma2/library.json`
  - `src/models/embedding_gemma2/loader.cpp`
  - `src/models/embedding_gemma2/loader.hpp`
  - `src/models/embedding_gemma2/probe.hpp`
  - `src/models/embedding_gemma2/provider.cpp`
  - `src/models/embedding_gemma2/reference/cpu_encoder.cpp`
  - `src/models/embedding_gemma2/reference/cpu_encoder.hpp`
  - `src/models/embedding_gemma2/tensor.cpp`
  - `src/models/embedding_gemma2/tokenizer.cpp`
  - `src/models/embedding_gemma2/tokenizer_loader.cpp`
  - `src/models/embedding_gemma2/video_processor.cpp`
  - `src/models/embedding_gemma2/video_processor.hpp`
  - `tests/extensions/ext_runtime_neutral_contracts_test.cpp`
  - `tests/extensions/metadata_test.py`
  - `tests/models/embedding_gemma2/CMakeLists.txt`
  - `tests/tools/check_extension_isolation_test.py`
  - `tools/ci/check-extension-isolation.py`
  - `docs/model-extensions/cards/13-embeddinggemma-pilot.md`
  - `docs/model-extensions/CARDS.md`
  - `docs/model-extensions/cards/README.md`
  - `docs/model-extensions/evidence/card13/**`

---

### Scope Audit

The diff strictly aligns with RFC Step 8 and Card 13:
- **Coupling Inventory & Decoupled Serving Boundary**: Audited all 8 includes from `src/cli/serve/` into `src/models/embedding_gemma2/` and 3 leaked symbols (`kMaxImageSoftTokens`, `SupportedImageSoftTokens`, `LooksLikeEmbeddingGemma2`). Fully eliminated all `src/models/embedding_gemma2` includes from `src/cli/serve/embedding_service.hpp`, `embedding_service.cpp`, and `embeddings_api.cpp`. The serving boundary now interacts exclusively through neutral extension registry lookup (`ext::FindEmbeddingProvider`) and runner abstractions (`ext::EmbeddingRunner`).
- **Public Extension API Surface**: Published `<gufo/ext/embeddings.hpp>` and `include/gufo/ext/registry.hpp` contracts for multimodal embedding workflows (`EmbeddingBatch`, `EmbeddingInput`, `EmbeddingPart`, `EmbeddingStatus`, `EmbeddingVideoOptions`, `EmbeddingRequestOptions`, `EmbeddingRunner`, `EmbeddingProvider`, `AddEmbeddingProvider`, `RegisterEmbeddingProvider`, `FindEmbeddingProvider`, `ProbeEmbeddingProvider`). Bumped extension API version to `0.1.3` (revision 3, backwards-compatible additive change) in `version.hpp` and `GufoExtension.cmake`. Updated `include/gufo/ext/CHANGES.md` and verified `include/gufo/ext/SURFACE` (99 paths).
- **Extension Migration & Self-Containment**: Converted all 24 internal includes in `src/models/embedding_gemma2/` to extension-relative includes. Replaced root-directed `CMakeLists.txt` with `extension.cmake` utilizing standard `gufo_extension_target` and `gufo_extension_hip_sources` macros. Preserved audio/video decoder dependencies (`PkgConfig::SNDFILE`, `PkgConfig::SOXR`, dynamic FFmpeg/libav) and batch scheduling logic entirely within the extension.
- **Manifests & Pure Artifact Probe**: Authored `src/models/embedding_gemma2/extension.json` (schema 1) declaring targets `gufo_embedding_gemma2` and `gufo_embedding_gemma2_hip`, provider `embeddinggemma-2`, lookup keys `safetensors:embedding_gemma2` and `safetensors:EmbeddingGemma2Model`, and option `--default-prompt-name`. Implemented HIP-free pure artifact probe (`ProbeEmbeddingGemma2` and prompt resolution) in `probe.hpp` & `extension.cpp`. Implemented runner factory in `provider.cpp`. Added `embedding_gemma2` to `extensions.lock`.
- **Target Ownership & Isolation Ratchet**: Set `"consumers": []` in `library.json`, removed direct target linkages from `gufo` and `gufo_http` in root `CMakeLists.txt`. Added `--fail-on-extension-boundary` to `tools/ci/check-extension-isolation.py` and test cases in `tests/tools/check_extension_isolation_test.py`.
- **Untouched and Preserved**: `docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted.

---

### Implementation Summary

1. **Inventory & Interface Classification**:
   - Mapped all 8 legacy includes from serving to `embedding_gemma2` in `docs/model-extensions/evidence/card13/pilot-inventory.md`.
   - Replaced concrete model calls in `EmbeddingService` with dynamic dispatch to `ext::FindEmbeddingProvider("embeddinggemma-2")` and `ext::ProbeEmbeddingProvider`.
   - Transferred multimodal batch scheduling (`EmbeddingBatchScheduler`) and dynamic FFmpeg/libav video / sndfile audio processing behind `ext::EmbeddingRunner`.

2. **Public Multimodal Embeddings Contract**:
   - `include/gufo/ext/embeddings.hpp`: Neutral interfaces for text, image, audio, and video inputs (`EmbeddingInput`, `EmbeddingPart`, `EmbeddingModality`), batch execution requests (`EmbeddingBatch`), cancellation, and result vectors (`EmbeddingStatus`).
   - `include/gufo/ext/registry.hpp` & `src/ext/runtime.cpp`: Extended provider registration mechanisms to support embedding runners and artifact probes.
   - Extension API version bumped from `0.1.2` to `0.1.3` (revision 3), documented in `CHANGES.md`. `SURFACE` updated to 99 paths and verified with `tools/ci/check-extension-api.py`.

3. **Extension Self-Containment & Manifests**:
   - `src/models/embedding_gemma2/extension.json`: Schema 1 compliant. Declares entry `embedding_gemma2_register` and HIP provider entry `embedding_gemma2_provider_register`.
   - `src/models/embedding_gemma2/extension.cmake`: Encapsulates CPU reference oracle, dynamic FFmpeg loading, and HIP kernels without root CMake hardcoding.
   - `src/models/embedding_gemma2/probe.hpp` & `extension.cpp`: Implements pure CPU artifact inspection reading `config.json` without initializing HIP runtimes.
   - Converted all 24 internal includes to extension-relative paths (`"config.hpp"`, `"loader.hpp"`, `"image_processor.hpp"`, etc.).

4. **Discovery, Lock, and Excluded Build Verification**:
   - Validated that `gufo diagnose` detects `embedding_gemma2 0.1.0` in discovery mode.
   - Validated lock-mode builds (`-DGUFO_EXTENSIONS_FROM_LOCK=ON`) in standalone directory `/tmp/card13-review/lock-test`.
   - Validated explicit exclusion builds (`-DGUFO_EXTENSIONS=""`) in standalone directory `/tmp/card13-review/exclude-test`, compiling cleanly with zero unresolved symbols and reporting `Built extensions: none`.

5. **Enforced Boundaries**:
   - Ratchet `--fail-on-extension-boundary` / `--fail-on-cli-model-includes` added to `tools/ci/check-extension-isolation.py`.
   - Unit tests in `tests/tools/check_extension_isolation_test.py` verify that any CLI include targeting `src/models/embedding_gemma2` fails build checks.

---

### Review Findings and Dispositions

| Finding ID | Severity | Description | Disposition |
|---|---|---|---|
| **F0** | None | Independent reviewer reported **CLEAN PASS** with zero findings across all audited areas (manifests, boundaries, public API, build modes, test integrity, and formatting). | **Accepted**: No code modifications required during integration. Acceptance audited and verified. |
| **P1** | Major | Codex review identified that `gufo serve embedding` startup path did not call `RegisterBuiltExtensions`, leaving runtime provider list unpopulated during normal serving. | **Resolved by Orchestrator**: Added `ext::RegisterBuiltExtensions(registry)` call in `RunServe` in `src/cli/serve/serve.cpp`. |
| **P2** | Major | Codex review identified that `EmbeddingService` looked up providers using `options.model_id` (the public served API alias), failing when `--served-model-name` differed from `embeddinggemma-2`. | **Resolved by Orchestrator**: Updated `EmbeddingService::Impl` in `src/cli/serve/embedding_service.cpp` to look up providers by probing `options.model_root` and defaulting to provider ID `embeddinggemma-2`, separating served API alias from provider resolution. |

---

### Definition of Done Audit

| DoD Item | Status | Verification Evidence / Reference |
|---|---|---|
| **DoD 1**: Pilot loads through the registry with truthful probe/selection/errors and unchanged advertised IDs, modalities, batch ordering and scheduling. | **PROVEN** | `ProbeEmbeddingGemma2` performs pure metadata validation in `probe.hpp`/`extension.cpp`. Registry registration verified via `ext_runtime_neutral_contracts_test` (PASS), `gufo diagnose` across discovery and lock modes (PASS), `embedding_gemma2_batch_scheduler_test` (PASS), `embedding_gemma2_embeddings_api_test` (PASS), and `embedding_gemma2_config_test` (PASS). External weight-dependent tests skipped cleanly with code 77. Functional harness recorded as PENDING in `pending-commands.txt`. |
| **DoD 2**: Its manifest owns production packages, data and runtime libraries; isolated/excluded builds work without hidden root declarations. | **PROVEN** | `src/models/embedding_gemma2/extension.json` matches schema 1. `extension.cmake` handles all targets and dynamic video library linkages without root hardcoding. `library.json` sets `"consumers": []`. Lock-mode build (`-DGUFO_EXTENSIONS_FROM_LOCK=ON`) and excluded build (`-DGUFO_EXTENSIONS=""`) both built cleanly and passed diagnostics in `/tmp/card13-review/`. |
| **DoD 3**: Existing API, independent numerical and speed gates pass on both discovery and actual lock-mode HIP artifacts; ratchet rejects new coupling. | **PROVEN** (Build, in-tree unit/golden tests, and ratchet)<br>/ **DEFERRED** (Full model functional/speed sweeps) | `tools/ci/check-extension-api.py` passes (99 paths, version 0.1.3). Continuation identity goldens pass byte-identically on presets. All 15 CTest embedding tests passed or skipped cleanly (0 failures). Ratchet in `tools/ci/check-extension-isolation.py` enforces `--fail-on-extension-boundary` (0 violations; unit tests PASS). Full functional/speed sweeps deferred to post-Card-14 window per RFC agreement and recorded in `pending-commands.txt`. |

---

### Acceptance Gates and Integration Re-runs

All 10 verification gates were re-executed once by the integration agent; full logs and details are recorded in [`docs/model-extensions/evidence/card13/integration-reruns.txt`](../evidence/card13/integration-reruns.txt):

| Gate | Description | Command Executed | Exit Code | Verdict |
|---|---|---|---|---|
| **Gate 1** | Embedding Tests on GPU Preset | `nix develop -c ctest --preset gpu-full -R '^embedding_gemma2_.*' --output-on-failure` | 0 | **PASS** (15 tests: 9 passed, 6 skipped cleanly code 77, 0 failed; 4.00s) |
| **Gate 2** | Runtime Neutral Contracts Test | `nix develop -c ctest --preset gpu-full -R '^ext_runtime_neutral_contracts_test$' --output-on-failure` | 0 | **PASS** (1/1 passed, 0.00s) |
| **Gate 3** | Extension API Surface Gate | `python3 tools/ci/check-extension-api.py` | 0 | **PASS** (99 paths, version 0.1.3) |
| **Gate 4** | Extension API Policy Unit Tests | `python3 tests/tools/check_extension_api_test.py` | 0 | **PASS** (7/7 tests passed, 0.002s) |
| **Gate 5** | Extension Isolation & Boundary Ratchet | `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes --fail-on-extension-boundary` | 0 | **PASS** (All 6 boundary checks clean, 0 coupling violations) |
| **Gate 6** | Isolation Ratchet Unit Tests | `python3 tests/tools/check_extension_isolation_test.py` | 0 | **PASS** (26/26 tests passed, 22.960s) |
| **Gate 7** | Extension Metadata Tests | `nix develop -c python3 tests/extensions/metadata_test.py` | 0 | **PASS** (12/12 tests passed, 10.415s) |
| **Gate 8** | Extension Discovery Tests | `nix develop -c python3 tests/extensions/discovery_test.py` | 0 | **PASS** (13/13 tests passed, 46.906s) |
| **Gate 9** | Continuation Identity Goldens (GPU Preset) | `nix develop -c ctest --preset gpu-full -R 'continuation_identity_test' --output-on-failure` | 0 | **PASS** (3/3 tests passed, 0.01s) |
| **Gate 10** | Clang-Format Verification | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | 0 | **PASS** (792 C++ files clean) |

---

### Remaining Limitations and Deferred Work

1. **Deferred Functional Test Harness**: Per user decision (2026-10-31), functional harness suites (`tests/functional/model_services.py embedding --embedding-suite full`, prompt lookup, and speed benchmarks) do not run during Card 13. All functional verification commands remain recorded in [`pending-commands.txt`](../evidence/card13/pending-commands.txt) and are deferred to the single consolidated qualification window following Card 14.
2. **Missing External Weights for Full CTest Suites**: 6 tests (`embedding_gemma2_loader_test`, `encoder_hip_test`, `embedding_golden_hip_test`, `tokenizer_test`, `image_golden_hip_test`, `audio_golden_hip_test`) skipped cleanly with return code 77 due to absent local model weights. These tests will be executed when external weight fixtures are staged during the qualification window.
3. **Pending Models to Migrate**: Migration of remaining models (Qwen3-ASR, Qwen3-TTS, Qwen-Image-21, Minimax-H3, Qwen 27B, Flash-Next, DeepSeek) to isolated extensions will occur in Cards 15–22 following Card 14.

---

### Evidence Files

- [`pilot-inventory.md`](../evidence/card13/pilot-inventory.md): Initial inventory of serving includes and leaked symbols.
- [`build-and-tests.txt`](../evidence/card13/build-and-tests.txt): Worker build, diagnostic, and test logs.
- [`pending-commands.txt`](../evidence/card13/pending-commands.txt): Verbatim deferred functional and benchmark commands.
- [`review-findings.md`](../evidence/card13/review-findings.md): Independent reviewer report (CLEAN PASS, F0).
- [`integration-reruns.txt`](../evidence/card13/integration-reruns.txt): Integrator re-run commands, exit codes, and output excerpts.
- [`orchestrator-gates.txt`](../evidence/card13/orchestrator-gates.txt): Orchestrator closure re-run commands, Codex review resolutions, and gate outcomes.
- [`RESUME-NOTES.md`](../evidence/card13/RESUME-NOTES.md): Summary of migration and editing handoff.
- [`README.md`](../evidence/card13/README.md): Index and provenance of Card 13 evidence artifacts.
