# Cards 15–18 Resume Notes: Migrate Multimodal Models (ASR, TTS, Image, Video)

## Summary of Accomplished Work

Cards 15–18 represent the migration of all four non-text / multimodal model families into standalone extensions:
- **Card 15**: Speech Recognition (`src/models/qwen3_asr`)
- **Card 16**: Speech Synthesis (`src/models/qwen3_tts`)
- **Card 17**: Image Generation (`src/models/qwen_image_21`)
- **Card 18**: Video Generation (`src/models/minimax_h3`)

### 1. Platform & Shared Audio Decoupling
- Extracted OS-specific subprocess execution from `minimax_h3/media_posix.cpp` and `minimax_h3/media_windows.cpp` into clean platform leaves under `src/core/platform/process.hpp`, `src/core/platform/process_posix.cpp`, and `src/core/platform/process_windows.cpp` (in `namespace gufo::platform`). Exposed publicly via `include/gufo/ext/platform/process.hpp`.
- Extracted shared `AudioBuffer`, `ResampleMono16k`, and `DecodeBase64Wav` out of model directories into host audio services: `src/cli/serve/audio_buffer.hpp` and `src/cli/serve/audio_buffer.cpp`.
- Model extensions and CLI code now use neutral contracts without cross-model contamination or exposing internal ASR/TTS implementation types.

### 2. Neutral Extension Contracts & Extension API 0.1.5
- Defined neutral contracts:
  * ASR: `src/ext/asr.hpp` forwarded to `include/gufo/ext/asr.hpp` (`AsrRunner`, `AsrProvider`, `AsrProbe`, `AsrArtifactResolver`, `TranscriptionRequest`, `TranscriptionResult`).
  * TTS: `src/ext/tts.hpp` forwarded to `include/gufo/ext/tts.hpp` (`TtsRunner`, `TtsProvider`, `TtsProbe`, `TtsArtifactResolver`, `TtsSynthesisRequest`, `TtsSynthesisResult`).
  * Image: `src/ext/image.hpp` forwarded to `include/gufo/ext/image.hpp` (`ImageRunner`, `ImageProvider`, `ImageProbe`, `ImageArtifactResolver`, `ImageRequest`, `ImageResult`, `ImageStatus`).
  * Video: `src/ext/video.hpp` forwarded to `include/gufo/ext/video.hpp` (`VideoRunner`, `VideoProvider`, `VideoProbe`, `VideoArtifactResolver`, `VideoJobRequest`, `VideoJobStatus`).
- Updated `ext::Registry` in `include/gufo/ext/registry.hpp` and `src/ext/runtime.cpp` with provider registration, lookup, and probing for ASR, TTS, Image, and Video capabilities.
- Bumped extension API version to `0.1.5` (`kApiRevision = 5`) in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`. Updated `include/gufo/ext/CHANGES.md` and regenerated `include/gufo/ext/SURFACE` (108 public paths).

### 3. Model Extension Migrations
- **Card 15 (`qwen3_asr`)**:
  * Replaced `CMakeLists.txt` with `extension.cmake`. Added `extension.cpp`, `provider.cpp`, `probe.hpp`.
  * `transcription.hpp` now forwards to `ext::asr.hpp`.
  * `src/cli/serve/asr_service.hpp`, `asr_service.cpp`, `asr_websocket.cpp`, `audio_asr_api.cpp`, and `src/cli/transcribe/transcribe.cpp` migrated to use `ext::FindAsrProvider` and `ext::ProbeAsrProvider`.
- **Card 16 (`qwen3_tts`)**:
  * Replaced `CMakeLists.txt` with `extension.cmake`. Added `extension.cpp`, `provider.cpp`, `probe.hpp`.
  * `synthesis.hpp` now forwards to `ext::tts.hpp`.
  * `src/cli/serve/tts_service.hpp`, `tts_service.cpp`, `audio_tts_api.cpp`, and `tts_websocket.cpp` migrated to use `ext::FindTtsProvider` and `ext::ProbeTtsProvider`.
- **Card 17 (`qwen_image_21`)**:
  * Replaced `CMakeLists.txt` with `extension.cmake`. Added `extension.cpp`, `provider.cpp`, `probe.hpp`.
  * `model.hpp` updated to alias request/result types to `ext::ImageRequest`, `ext::ImageResult`.
  * `src/cli/serve/image_api.hpp` and `image_api.cpp` migrated to neutral contracts.
- **Card 18 (`minimax_h3`)**:
  * Replaced `CMakeLists.txt` with `extension.cmake`. Added `extension.cpp`, `provider.cpp`, `probe.hpp`, `media.cpp`.
  * Removed `media_posix.cpp` and `media_windows.cpp`; subprocess calls use `gufo::platform::RunProcessSync`.
  * `src/cli/video/video.hpp`, `video.cpp`, `src/cli/serve/video_api.cpp`, `video_jobs.hpp`, `video_jobs.cpp` migrated to neutral contracts.

### 4. Extensions Lock & Isolation Ratchet
- Added `"qwen3_asr"`, `"qwen3_tts"`, `"qwen_image_21"`, `"minimax_h3"` to `extensions.lock`.
- Updated `tools/ci/check-extension-isolation.py` so all 5 migrated extensions (`embedding_gemma2`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`, `minimax_h3`) are strictly enforced against CLI direct access.
- Isolation check passed with 0 violations across all 3 boundaries (`--fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes`).

### 5. Verification Results
- All 11 verification gates passed (100% check-pr tests, full nix derivation checks, 0 drift on infrastructure generator, clang-format clean, docs clean).

### 6. Editing Ownership Handoff
- Worker editing phase is COMPLETE.
- Sole editing ownership is handed off to the review and integration agents.
