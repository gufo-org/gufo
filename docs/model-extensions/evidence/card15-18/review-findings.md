# Cards 15–18 Independent Review Findings: Migrate Speech, Audio, Image, and Video Models

Reviewer: Independent Read-Only Review Agent  
Review Date: 2026-11-01  
Repository: `/home/mixer/gufo`  
Parent Revision: `yuvlvuwx 0125a678 model-extensions-4* | fix(extensions): version metadata discovery contract`  
Working Copy: `zonoxvml 715be0d1` (uncommitted, audit of Cards 15–18 migration)  
Audit Verdict: **PASS with 3 Minor and 2 Nit findings** (0 Critical, 0 Major)

---

## 1. Executive Summary

Cards 15, 16, 17, and 18 represent the migration of all four multimodal / non-text model families into isolated extensions:
1. **Card 15 (`qwen3_asr`)**: Speech recognition contracts in `include/gufo/ext/asr.hpp`, host audio services (`AudioBuffer`, `ResampleMono16k`, `DecodeWav`) in `src/cli/serve/audio_buffer.{hpp,cpp}`, `qwen3_asr` extension structure, CLI and `gufo transcribe` decoupling.
2. **Card 16 (`qwen3_tts`)**: Speech synthesis contracts in `include/gufo/ext/tts.hpp`, `ext::Registry` TTS methods, `DecodeBase64Wav` in host audio services, `qwen3_tts` extension structure, CLI decoupling (`tts_service`, `tts_websocket`, `audio_tts_api`).
3. **Card 17 (`qwen_image_21`)**: Image generation contracts in `include/gufo/ext/image.hpp`, `ext::Registry` image methods, `qwen_image_21` extension structure, CLI decoupling (`image_api`).
4. **Card 18 (`minimax_h3`)**: Video generation contracts in `include/gufo/ext/video.hpp`, `ext::Registry` video methods, `minimax_h3` extension structure, OS-specific subprocess execution extracted into clean platform leaves (`src/core/platform/process.{hpp,posix.cpp,windows.cpp}`), CLI decoupling (`video`, `video_api`, `video_jobs`).

### API and Invariants Verification
- **Extension API Versioning**: Bumped to `0.1.5` (`kApiRevision = 5`) in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`. Documented in `include/gufo/ext/CHANGES.md`. `include/gufo/ext/SURFACE` accurately tracks 108 public paths.
- **Extensions Lock**: `extensions.lock` updated with all 5 migrated extensions (`embedding_gemma2`, `minimax_h3`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`).
- **Isolation Ratchet**: `tools/ci/check-extension-isolation.py` enforces all 5 migrated extensions. Zero boundary violations across all 3 strict gates (`--fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes`).
- **Platform Conventions**: Zero `<windows.h>`, `_WIN32`, or `_MSC_VER` macros introduced outside `src/core/platform/`. `minimax_h3/media_windows.cpp` was cleanly replaced by platform leaves.
- **Repository Integrity**: `docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted.
- **Verification Gates**: 100% passed (generator drift, schema/isolation unit tests, clang-format across 817 files, docs link check across 319 markdown files, 26 ctest CPU tests for migrated models and contracts, and sandboxed nix PR checks `checks.x86_64-linux.pr`). GPU speed and functional benchmarks are systematically recorded as `PENDING` in `docs/model-extensions/evidence/card15-18/pending-commands.txt` for the post-Card-14 qualification window.

---

## 2. Findings by Severity

### Critical Findings
*None.*

### Major Findings
*None.*

---

### Minor Findings

#### Finding M1 (Minor): Duplicate dead code for `DecodeBase64Wav` retained in `src/models/qwen3_tts/`

- **Severity**: Minor
- **Affected paths**:
  - [`src/models/qwen3_tts/audio.hpp:17-19`](file:///home/mixer/gufo/src/models/qwen3_tts/audio.hpp#L17-L19)
  - [`src/models/qwen3_tts/audio.cpp:217-236`](file:///home/mixer/gufo/src/models/qwen3_tts/audio.cpp#L217-L236)
- **Concrete problem and violation of requirements**:
  Card 16 specifies: *"Move DecodeBase64Wav to host audio services and update HTTP consumers without copying TTS implementations into core."*
  `DecodeBase64Wav` was properly extracted into `src/cli/serve/audio_buffer.hpp` and `src/cli/serve/audio_buffer.cpp`. However, the original implementation in `src/models/qwen3_tts/audio.cpp` and declaration in `src/models/qwen3_tts/audio.hpp` were left intact. No internal code in `src/models/qwen3_tts` or its tests invokes `DecodeBase64Wav`.
- **Practical impact**:
  Harmless to runtime behavior, but leaves dead duplicate code inside the `qwen3_tts` extension.
- **Suggested correction**:
  Remove `DecodeBase64Wav` from [`src/models/qwen3_tts/audio.hpp`](file:///home/mixer/gufo/src/models/qwen3_tts/audio.hpp) and [`src/models/qwen3_tts/audio.cpp`](file:///home/mixer/gufo/src/models/qwen3_tts/audio.cpp).

---

#### Finding M2 (Minor): Model ID alias asymmetry in `AsrService` fallback lookup

- **Severity**: Minor
- **Affected paths**:
  - [`src/cli/serve/asr_service.cpp:13-23`](file:///home/mixer/gufo/src/cli/serve/asr_service.cpp#L13-L23)
  - [`src/models/qwen3_asr/extension.cpp:37-42`](file:///home/mixer/gufo/src/models/qwen3_asr/extension.cpp#L37-L42)
  - [`src/models/qwen3_asr/provider.cpp:47-52`](file:///home/mixer/gufo/src/models/qwen3_asr/provider.cpp#L47-L52)
- **Concrete problem and violation of requirements**:
  In `src/cli/serve/asr_service.cpp`:
  ```cpp
  if (options.model_id.empty()) {
    options.model_id = "qwen3-asr-1.7b";
  }
  const ext::AsrProvider* provider = nullptr;
  if (!options.model_root.empty()) {
    provider = ext::ProbeAsrProvider(options.model_root.string());
  }
  if (!provider && !options.model_id.empty()) {
    provider = ext::FindAsrProvider(options.model_id);
  }
  ```
  `qwen3_asr` registers itself with `.id = "qwen3-asr"`. If `options.model_root` is empty (e.g. during certain programmatic service initializations without a disk directory), `ProbeAsrProvider` is bypassed and `FindAsrProvider("qwen3-asr-1.7b")` is called. Because `FindAsrProvider` performs exact string matching against registered provider IDs, lookup fails. In contrast, `src/cli/serve/image_api.cpp` handles both `"qwen-image-21"` and `"qwen-image-2.1"`.
- **Practical impact**:
  In the CLI server, `options.model_root` is passed from `--model`, so `ProbeAsrProvider` succeeds. However, fallback lookup by model ID fails if `"qwen3-asr-1.7b"` is supplied without probing.
- **Suggested correction**:
  In `src/cli/serve/asr_service.cpp`, fallback to `"qwen3-asr"` when `options.model_id == "qwen3-asr-1.7b"`, or register `"qwen3-asr-1.7b"` alongside `"qwen3-asr"` in the ASR provider registry.

---

#### Finding M3 (Minor): Duplication of `TakeSpeechSegments` text-streaming helper

- **Severity**: Minor
- **Affected paths**:
  - [`src/cli/serve/text_stream.hpp:13-54`](file:///home/mixer/gufo/src/cli/serve/text_stream.hpp#L13-L54)
  - [`src/models/qwen3_tts/text_stream.hpp:13-54`](file:///home/mixer/gufo/src/models/qwen3_tts/text_stream.hpp#L13-L54)
- **Concrete problem and violation of requirements**:
  `TakeSpeechSegments` is a host WebSocket serving helper for streaming TTS chunking. During migration, it was moved to `src/cli/serve/text_stream.hpp` to prevent CLI include boundary violations in `tts_websocket.cpp`. However, `src/models/qwen3_tts/text_stream.hpp` was retained so that `tests/models/qwen3_tts/audio_api_test.cpp` could compile. Nothing in the `qwen3_tts` model itself requires `TakeSpeechSegments`.
- **Practical impact**:
  Redundant duplicate header logic maintained across two separate subdirectories (`src/cli/serve/` and `src/models/qwen3_tts/`).
- **Suggested correction**:
  Move the unit tests for `TakeSpeechSegments` into a host/CLI test target (e.g., in `tests/cli/`), and remove `src/models/qwen3_tts/text_stream.hpp`.

---

### Nit Findings

#### Finding N1 (Nit): Discrepancy in `include/gufo/ext/CHANGES.md` referencing non-existent `RunSubprocess`

- **Severity**: Nit
- **Affected paths**:
  - [`include/gufo/ext/CHANGES.md:16-17`](file:///home/mixer/gufo/include/gufo/ext/CHANGES.md#L16-L17)
- **Concrete problem and violation of requirements**:
  `include/gufo/ext/CHANGES.md` states:
  `- Platform Process (include/gufo/ext/platform/process.hpp): Subprocess execution contracts RunSubprocess, RunMediaMux, RunCapture.`
  However, `src/core/platform/process.hpp` exposes `RunCapture`, `RunMediaMux`, `FfmpegProgram`, `FfprobeProgram`, and `PartialMediaPath`. There is no symbol named `RunSubprocess`.
- **Practical impact**:
  Documentation inaccuracy in public extension changelog.
- **Suggested correction**:
  Remove `RunSubprocess` from the list of declared process contracts in [`include/gufo/ext/CHANGES.md`](file:///home/mixer/gufo/include/gufo/ext/CHANGES.md).

---

#### Finding N2 (Nit): Namespace aliasing inconsistency in `include/gufo/ext/platform/process.hpp`

- **Severity**: Nit
- **Affected paths**:
  - [`include/gufo/ext/platform/process.hpp:5-7`](file:///home/mixer/gufo/include/gufo/ext/platform/process.hpp#L5-L7)
- **Concrete problem and violation of requirements**:
  `include/gufo/ext/platform/process.hpp` defines:
  ```cpp
  namespace gufo::ext::platform {
  using namespace gufo::platform;
  }
  ```
  In contrast, other platform headers under `include/gufo/ext/platform/` (`fs.hpp`, `library.hpp`, `mapped_file.hpp`, `os_info.hpp`, `pages.hpp`, `prefault.hpp`, `publish.hpp`) do not create `gufo::ext::platform`, instead directly exposing `gufo::platform`. In `src/models/minimax_h3/media.cpp`, calls use `ext::platform::*`.
- **Practical impact**:
  Cosmetic stylistic inconsistency across platform forwarding headers.
- **Suggested correction**:
  Either align all platform forwarding headers to declare `gufo::ext::platform` aliases or keep symbols uniformly under `gufo::platform`.

---

## 3. Validation Gate Commands for Integration Agent

The following validation commands were run and verified by the reviewer:

1. **Infrastructure Generator Drift Check**:
   ```sh
   python3 tools/ci/generate-infrastructure.py --check
   ```
2. **Extension API Surface & Tool Unit Tests**:
   ```sh
   python3 tools/ci/check-extension-api.py
   python3 tests/tools/check_extension_api_test.py
   ```
3. **Extension Isolation Boundary Ratchet**:
   ```sh
   python3 tools/ci/check-extension-isolation.py --fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes
   python3 tests/tools/check_extension_isolation_test.py
   ```
4. **Documentation and Formatting Checks**:
   ```sh
   python3 tools/ci/check-docs.py
   nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
   ```
5. **CTest Migration and Contract Suite**:
   ```sh
   nix develop -c ctest --test-dir build/cpu-test -R 'qwen3_asr|qwen3_tts|qwen_image_21|minimax_h3|video_|ext_runtime|audio_buffer' --output-on-failure
   ```
6. **Extension Metadata Configuration Tests**:
   ```sh
   nix develop -c python3 -m unittest tests/extensions/metadata_test.py
   ```
7. **Sandboxed Nix PR Checks**:
   ```sh
   nix build .#checks.x86_64-linux.pr --no-link
   ```
