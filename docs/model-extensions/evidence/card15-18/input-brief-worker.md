> Instructions given to the implementation agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Worker brief — Cards 15–18: Migrate Speech, Audio, Image, and Video Models

You are the **implementation agent** (`worker`) for Cards 15, 16, 17, and 18 of the model-extensions RFC in
`/home/mixer/gufo`. You have **sole editing ownership** of the shared worktree while you run.

## Absolute constraints

1. **Never run `jj commit`, `jj new`, `jj squash`, `git commit`, or anything that
   changes revision state.** The orchestrator commits. Read-only commands (`jj st`,
   `jj diff`, `jj log`) are permitted.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
   Do not read or write `HANDOFF.md` (deleted; its decisions now live in `CARDS.md` and `cards/README.md`).
3. **Scope discipline**: during this card run only unit/identity/CPU/hosted/static
   gates. Functional harness runs (`tests/functional/run.py` rows), main-control timings,
   and standard speed benchmarks are **deferred to the single post-Card-14 qualification window**
   per standing user decision. Record them as PENDING with their exact commands in
   `docs/model-extensions/evidence/card15-18/pending-commands.txt`, never as passes.
4. Scratch goes in `/tmp/card15-18-*` only (logs, caches, temporary build trees) and MUST
   be deleted when the test finishes, failures included.
   No evidence file over 1 MB under `docs/`.
   No raw full build logs, no per-row report trees under `docs/`.
5. Keep tool output small: redirect build and test logs to files in `/tmp/card15-18-logs/`
   and inspect using `tail -n 40` or `grep`. Never output multi-megabyte tool outputs.
6. Host RAM: Before running any GPU-heavy operation, check `free -g` and keep at least
   15 GiB headroom. The `llama-swap.service` must stay running; do not stop services.
7. Preserve unrelated work and user changes.

## Objectives & Context

Migrate the four multimodal/non-text models (RFC step 8):
- **Card 15**: Speech Recognition (`src/models/qwen3_asr`)
- **Card 16**: Speech Synthesis (`src/models/qwen3_tts`)
- **Card 17**: Image Generation (`src/models/qwen_image_21`)
- **Card 18**: Video Generation (`src/models/minimax_h3`)

### Detailed Plan for Each Card

#### 1. Card 15: Speech Recognition (`qwen3_asr`)
- Declare neutral ASR contracts in `include/gufo/ext/asr.hpp` (forwarding to `src/ext/asr.hpp`):
  `AsrRunner`, `AsrProvider`, `AsrProbe`, `AsrArtifactResolver`, `AsrRunnerFactory`, `TranscriptionRequest`, `TranscriptionResult`, `CancellationCheck`.
  Register in `ext::Registry` (`AddAsrProvider`, `FindAsrProvider`, `ProbeAsrProvider`).
- Move shared `AudioBuffer` and `ResampleMono16k` out of `qwen3_asr` into host audio services (`src/cli/serve/audio_buffer.hpp` / `audio_buffer.cpp`).
- Decouple `src/cli/serve/asr_service.hpp`, `asr_service.cpp`, `asr_websocket.cpp`, `audio_asr_api.cpp` and `src/cli/transcribe/transcribe.cpp` from `src/models/qwen3_asr/`.
- Replace `src/models/qwen3_asr/CMakeLists.txt` with `extension.cmake`. Add `extension.cpp`, `provider.cpp`, `probe.hpp`.
- Convert internal includes in `src/models/qwen3_asr/` to extension-relative.

#### 2. Card 16: Speech Synthesis (`qwen3_tts`)
- Declare neutral TTS contracts in `include/gufo/ext/tts.hpp` (forwarding to `src/ext/tts.hpp`):
  `TtsRunner`, `TtsProvider`, `TtsProbe`, `TtsArtifactResolver`, `TtsRunnerFactory`, `SynthesisRequest`, `SynthesisResult`, `CancellationCheck`.
  Register in `ext::Registry` (`AddTtsProvider`, `FindTtsProvider`, `ProbeTtsProvider`).
- Move `DecodeBase64Wav` to host audio services (`src/cli/serve/audio_buffer.hpp` / `audio_buffer.cpp`).
- Decouple `src/cli/serve/tts_service.hpp`, `tts_service.cpp`, `tts_websocket.cpp`, `audio_tts_api.cpp` from `src/models/qwen3_tts/`.
- Replace `src/models/qwen3_tts/CMakeLists.txt` with `extension.cmake`. Add `extension.cpp`, `provider.cpp`, `probe.hpp`.
- Convert internal includes in `src/models/qwen3_tts/` to extension-relative.

#### 3. Card 17: Image Generation (`qwen_image_21`)
- Declare neutral image contracts in `include/gufo/ext/image.hpp` (forwarding to `src/ext/image.hpp`):
  `ImageRunner`, `ImageProvider`, `ImageProbe`, `ImageArtifactResolver`, `ImageRunnerFactory`, `ImageRequest`, `ImageResult`, `CancellationCheck`.
  Register in `ext::Registry` (`AddImageProvider`, `FindImageProvider`, `ProbeImageProvider`).
- Decouple `src/cli/serve/image_api.hpp`, `image_api.cpp` from `src/models/qwen_image_21/`.
- Replace `src/models/qwen_image_21/CMakeLists.txt` with `extension.cmake`. Add `extension.cpp`, `provider.cpp`, `probe.hpp`.
- Convert internal includes in `src/models/qwen_image_21/` to extension-relative.

#### 4. Card 18: Video Generation (`minimax_h3`)
- Declare neutral video contracts in `include/gufo/ext/video.hpp` (forwarding to `src/ext/video.hpp`):
  `VideoRunner`, `VideoProvider`, `VideoProbe`, `VideoArtifactResolver`, `VideoRunnerFactory`, `VideoRequest`, `VideoResult`, `VideoTelemetry`, `CancellationCheck`.
  Register in `ext::Registry` (`AddVideoProvider`, `FindVideoProvider`, `ProbeVideoProvider`).
- Extract OS subprocess/pipe handling from `src/models/minimax_h3/media_posix.cpp` and `media_windows.cpp` into `src/core/platform/process.hpp`, `process_posix.cpp`, `process_windows.cpp` and `include/gufo/ext/platform/process.hpp`.
- Decouple `src/cli/video/video.cpp`, `src/cli/serve/video_api.cpp`, `src/cli/serve/video_jobs.hpp`, `video_jobs.cpp` from `src/models/minimax_h3/`.
- Replace `src/models/minimax_h3/CMakeLists.txt` with `extension.cmake`. Add `extension.cpp`, `provider.cpp`, `probe.hpp`.
- Convert internal includes in `src/models/minimax_h3/` to extension-relative.

#### 5. Extension API & Shared Infrastructure
- Add `"qwen3_asr"`, `"qwen3_tts"`, `"qwen_image_21"`, `"minimax_h3"` to `extensions.lock`.
- Bump extension API version in `include/gufo/ext/version.hpp` to `0.1.4` (`kApiRevision = 4`, `kApiVersion = "0.1.4"`).
- Record additions in `include/gufo/ext/CHANGES.md`.
- Regenerate `include/gufo/ext/SURFACE` with `python3 tools/ci/check-extension-api.py --write`.
- Update `tools/ci/check-extension-isolation.py` so `--fail-on-extension-boundary` guards all 5 migrated extensions against `src/cli/` includes.
- Run `python3 tools/ci/generate-infrastructure.py --write` to update CMake, Nix, tests, and documentation.

#### 6. Required Verification Gates
Run and document results in `docs/model-extensions/evidence/card15-18/build-and-tests.txt`:
1. `python3 tools/ci/generate-infrastructure.py --check`
2. `python3 -m unittest tests/tools/generate_infrastructure_test.py`
3. `nix develop -c python3 tests/extensions/metadata_test.py --schema`
4. `python3 tools/ci/check-docs.py`
5. `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`
6. `python3 tests/tools/check_extension_isolation_test.py`
7. `python3 tools/ci/check-extension-isolation.py --fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes`
8. `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity' --output-on-failure`
9. `nix develop -c cmake --build --preset cpu-test --target check-pr`
10. `nix build .#checks.x86_64-linux.pr --no-link`

#### 7. Evidence & Handoff
- Write `build-and-tests.txt` and `pending-commands.txt`.
- Write `RESUME-NOTES.md` documenting completed work and an explicit editing-ownership handoff.
