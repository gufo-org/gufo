> Instructions given to the review agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Reviewer brief — Cards 15–18: Migrate Speech, Audio, Image, and Video Models

You are the **read-only reviewer agent** (`reviewer`) for Cards 15, 16, 17, and 18 of the model-extensions RFC in
`/home/mixer/gufo`.

## Absolute constraints

1. **You are strictly read-only.** Do not edit code, do not edit documentation, do not run git/jj write commands.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
3. Evaluate the actual diff (`jj diff`) and files against:
   - `AGENTS.md`
   - `docs/model-extensions/CARDS.md`
   - `docs/model-extensions/cards/README.md`
   - `docs/model-extensions/cards/15-speech-recognition.md`
   - `docs/model-extensions/cards/16-speech-synthesis.md`
   - `docs/model-extensions/cards/17-image-generation.md`
   - `docs/model-extensions/cards/18-video-generation.md`
   - `docs/model-extensions/evidence/card15-18/build-and-tests.txt`
   - `docs/model-extensions/evidence/card15-18/RESUME-NOTES.md`
4. Write your concrete findings directly into `docs/model-extensions/evidence/card15-18/review-findings.md`.
   Every finding must have:
   - Severity: `Critical`, `Major`, `Minor`, or `Nit`
   - Affected paths
   - Concrete problem and violation of requirements
   - Suggested correction
   If there are no findings in a category, explicitly state that.
5. If you do not have shell execution capabilities, emit the exact validation gate commands that the integration agent must run.

## Scope of Review

Review the 4 migrated models and infrastructure:
1. **Card 15 (`qwen3_asr`)**: Neutral contracts in `include/gufo/ext/asr.hpp`, registration in `ext::Registry`, `AudioBuffer` and `ResampleMono16k` in host audio services (`src/cli/serve/audio_buffer.hpp`), `qwen3_asr` extension structure (`extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`), CLI decoupling (`asr_service`, `asr_websocket`, `audio_asr_api`, `transcribe`).
2. **Card 16 (`qwen3_tts`)**: Neutral contracts in `include/gufo/ext/tts.hpp`, `ext::Registry` TTS methods, `DecodeBase64Wav` in host audio services, `qwen3_tts` extension structure, CLI decoupling (`tts_service`, `tts_websocket`, `audio_tts_api`).
3. **Card 17 (`qwen_image_21`)**: Neutral contracts in `include/gufo/ext/image.hpp`, `ext::Registry` image methods, `qwen_image_21` extension structure, CLI decoupling (`image_api`).
4. **Card 18 (`minimax_h3`)**: Neutral contracts in `include/gufo/ext/video.hpp`, `ext::Registry` video methods, `minimax_h3` extension structure, `src/core/platform/process.hpp` platform leaf replacing OS-specific branching, CLI decoupling (`video`, `video_api`, `video_jobs`).
5. **API & Packaging**:
   - `extensions.lock` updated with all 4 extensions.
   - `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake` API version bumped to 0.1.5.
   - `include/gufo/ext/CHANGES.md` updated.
   - `include/gufo/ext/SURFACE` up to date.
   - `tools/ci/check-extension-isolation.py` enforcing boundaries for all 5 migrated extensions.
   - Quality/speed functional tests marked PENDING in `pending-commands.txt`.
