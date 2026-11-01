# Cards 15–18 Integration & Verification Evidence

This directory consolidates the execution records, review findings, resolution logs, and verification evidence for **Cards 15, 16, 17, and 18** of the Model Extension Migration roadmap.

## Migrated Models

| Card | Model ID | Domain | Base Target | HIP Target | Manifest |
|---|---|---|---|---|---|
| **Card 15** | `qwen3_asr` | Speech Recognition (ASR) | `gufo_qwen3_asr` | `gufo_qwen3_asr_hip` | `src/models/qwen3_asr/extension.json` |
| **Card 16** | `qwen3_tts` | Speech Synthesis (TTS) | `gufo_qwen3_tts` | `gufo_qwen3_tts_hip` | `src/models/qwen3_tts/extension.json` |
| **Card 17** | `qwen_image_21` | Image Generation (DiT) | `gufo_qwen_image_21` | `gufo_qwen_image_21_hip` | `src/models/qwen_image_21/extension.json` |
| **Card 18** | `minimax_h3` | Video Generation | `gufo_minimax_h3` | `gufo_minimax_h3_hip` | `src/models/minimax_h3/extension.json` |

---

## Review and User Findings Resolution

All findings identified during review and Nix validation were resolved prior to sign-off:

1. **U1 — Separate HIP Targets in Extension Manifests**:
   - `cmake/GufoExtension.cmake` enforces that `gufo_extension_hip_sources` can only attach to targets declaring `"requires": ["hip"]`.
   - Updated manifests (`extension.json`) and CMake files (`extension.cmake`) for all 4 models to split into base target (unconditional registration, metadata) and HIP target (kernels, runtime, requires HIP).
   - Added `roc::hipblaslt` and `roc::rocblas` to `gufo_minimax_h3_hip`.
   - Staged changes; `nix build` and `nix build .#checks.x86_64-linux.pr` completed cleanly with code 0.
2. **M1 — Duplicate Dead Code Elimination**:
   - Removed obsolete `DecodeBase64Wav` helper and constants from `src/models/qwen3_tts/audio.hpp` and `audio.cpp`.
3. **M2 — Qwen3-ASR Provider Alias**:
   - Registered both `"qwen3-asr"` and `"qwen3-asr-1.7b"` provider IDs in `src/models/qwen3_asr/provider.cpp` and `extension.cpp`.
   - Updated `src/cli/serve/asr_service.cpp` to recognize both `"qwen3-asr"` and `"qwen3-asr-1.7b"` aliases.
4. **M3 — Speech Segments Stream Consolidation**:
   - Deleted duplicate `src/models/qwen3_tts/text_stream.hpp`.
   - Updated `tests/models/qwen3_tts/audio_api_test.cpp` to include `src/cli/serve/text_stream.hpp`.
5. **N1 — Extension API Changes Documentation**:
   - Updated `include/gufo/ext/CHANGES.md` to reference `gufo::platform::RunCapture` and `RunMediaMux` (removing stale `RunSubprocess` reference).
6. **N2 — Public Platform Forwarding Header Alignment**:
   - Removed `namespace gufo::ext::platform` alias wrapper from `include/gufo/ext/platform/process.hpp`, aligning with `files.hpp` and `pages.hpp`.
   - Updated `src/models/minimax_h3/media.cpp` to invoke `gufo::platform::RunCapture` and `RunMediaMux` directly.

---

## Verification Gates Status

| Verification Gate | Command | Result | Details |
|---|---|---|---|
| Infrastructure sync | `python3 tools/ci/generate-infrastructure.py --check` | **PASS** | Extensions and provider registry tables up to date |
| Public API surface | `python3 tools/ci/check-extension-api.py` | **PASS** | 108 surface paths validated |
| Extension isolation | `python3 tools/ci/check-extension-isolation.py --fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes` | **PASS** | 0 strict boundary violations across 1371 files |
| Documentation links | `python3 tools/ci/check-docs.py` | **PASS** | 321 files, 1605 links validated |
| Code formatting | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | **PASS** | 816 C++ files verified clean |
| Metadata unit tests | `nix develop -c python3 -m unittest tests/extensions/metadata_test.py` | **PASS** | 13 unit tests passed |
| Isolation unit tests | `python3 -m unittest tests/tools/check_extension_isolation_test.py` | **PASS** | 27 unit tests passed |
| Hosted PR suite | `nix develop -c cmake --build --preset cpu-test --target check-pr` | **PASS** | 96/96 test suites passed (152s) |
| Nix production build | `nix build` | **PASS** | Produces `./result/bin/gufo`; all 5 extensions built & active |
| Nix PR check derivations | `nix build .#checks.x86_64-linux.pr --no-link` | **PASS** | All 4 check derivations built and passed |

> [!NOTE]
> End-to-end functional model harness sweeps (`tests/functional/run.py`) and live inference benchmarks are deferred to the unified post-Card-14 qualification window per RFC guidelines.

---

## Evidence Artifacts

- [integration-reruns.txt](file:///home/mixer/gufo/docs/model-extensions/evidence/card15-18/integration-reruns.txt): Complete gate execution transcripts and diagnostic output.
- [review-findings.md](file:///home/mixer/gufo/docs/model-extensions/evidence/card15-18/review-findings.md): Pre-integration reviewer findings and status ledger.
- [build-and-tests.txt](file:///home/mixer/gufo/docs/model-extensions/evidence/card15-18/build-and-tests.txt): Initial worker build logs.
- [input-brief-integration.md](file:///home/mixer/gufo/docs/model-extensions/evidence/card15-18/input-brief-integration.md): Integration mission brief.
