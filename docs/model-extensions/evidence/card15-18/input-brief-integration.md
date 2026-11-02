> Instructions given to the integration agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Integration brief — Cards 15–18: Migrate Speech, Audio, Image, and Video Models

You are the **integration agent** (`codex-exec-writer`) for Cards 15, 16, 17, and 18 of the model-extensions RFC in
`/home/mixer/gufo`. You have **sole editing ownership** of the shared worktree while you run.

## Absolute constraints

1. **Never run `jj commit`, `jj new`, `jj squash`, `git commit`, or anything that
   changes revision state.** The orchestrator commits. Read-only commands (`jj st`,
   `jj diff`, `jj log`) are permitted.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
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

## Tasks to Complete

### 1. Resolve User Finding U1: `nix build` HIP Target Requirement
- **Problem**: In `nix build` (where `ENGINE_ENABLE_HIP=ON`), `gufo_extension_hip_sources` checks `GUFO_EXT_TARGET_${target}_REQUIRES` and requires `"hip"`. But `minimax_h3`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21` declared only `gufo_<name>` with `"requires": []`.
- **Resolution**:
  Align all four extensions with `embedding_gemma2`:
  - Split targets into `gufo_<name>` (unconditional registration target, `requires: []`) and `gufo_<name>_hip` (HIP provider target, `requires: ["hip"]`).
  - In `extension.json`: set provider `"target": "gufo_<name>_hip"` and `"entry": "<name>_provider_register"`, and declare both targets under `"targets"`.
  - In `extension.cmake`: define `gufo_<name>` with `extension.cpp`, probe/config, and link `gufo_ext_runtime`; define `gufo_<name>_hip` with `provider.cpp`, link `gufo_<name>`, and apply `gufo_extension_hip_sources(gufo_<name>_hip SOURCES ...)`.
  - Verify with `nix build`!

### 2. Resolve Reviewer Findings
- **M1**: Remove duplicate `DecodeBase64Wav` from `src/models/qwen3_tts/audio.hpp` and `audio.cpp`.
- **M2**: Add alias `"qwen3-asr"` to `src/cli/serve/asr_service.cpp` (or check for both `"qwen3-asr"` and `"qwen3-asr-1.7b"`).
- **M3**: Consolidate `TakeSpeechSegments`: move tests in `tests/models/qwen3_tts/audio_api_test.cpp` to use `src/cli/serve/text_stream.hpp`, and delete `src/models/qwen3_tts/text_stream.hpp`.
- **N1**: Update `include/gufo/ext/CHANGES.md` to reference `RunCapture` and `RunMediaMux` instead of `RunSubprocess`.
- **N2**: Ensure namespace consistency in `include/gufo/ext/platform/process.hpp`.

### 3. Re-run Verification Gates Once
- Run all affected gates and record output in `docs/model-extensions/evidence/card15-18/integration-reruns.txt`:
  - `python3 tools/ci/generate-infrastructure.py --check`
  - `python3 tools/ci/check-extension-api.py`
  - `python3 tools/ci/check-extension-isolation.py --fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes`
  - `python3 tools/ci/check-docs.py`
  - `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py --fix`
  - `nix develop -c cmake --build --preset cpu-test --target check-pr`
  - `nix build` (default derivation with HIP enabled)
  - `nix build .#checks.x86_64-linux.pr --no-link`

### 4. Update Card Files and Ledger
- Update `docs/model-extensions/cards/15-speech-recognition.md`: mark DoD `[x]`, update Status to `complete`, and fill Execution Record.
- Update `docs/model-extensions/cards/16-speech-synthesis.md`: mark DoD `[x]`, update Status to `complete`, and fill Execution Record.
- Update `docs/model-extensions/cards/17-image-generation.md`: mark DoD `[x]`, update Status to `complete`, and fill Execution Record.
- Update `docs/model-extensions/cards/18-video-generation.md`: mark DoD `[x]`, update Status to `complete`, and fill Execution Record.
- Update `docs/model-extensions/CARDS.md`: update status table for Cards 15–18, and add execution summaries.
- Update `docs/model-extensions/evidence/card15-18/README.md`.
