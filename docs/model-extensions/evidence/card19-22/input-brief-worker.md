> Instructions given to the implementation agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Worker brief (Resumed) — Cards 19–22: Shared Text Contract, Qwen 27B, Flash-Next, and DeepSeek Extensions

You are the replacement **implementation agent** (`worker`) for Cards 19, 20, 21, and 22 of the model-extensions RFC in
`/home/mixer/gufo`. You have **sole editing ownership** of the shared worktree while you run.
The previous worker session was stopped by infrastructure quota (code 429) right as it was running `nix develop -c cmake --preset cpu-test`.

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
   `docs/model-extensions/evidence/card19-22/pending-commands.txt`, never as passes.
4. Scratch goes in `/tmp/card19-22-*` only (logs, caches, temporary build trees) and MUST
   be deleted when the test finishes, failures included.
   No evidence file over 1 MB under `docs/`.
   No raw full build logs, no per-row report trees under `docs/`.
5. Keep tool output small: redirect build and test logs to files in `/tmp/card19-22-logs/`
   and inspect using `tail -n 40` or `grep`. Never output multi-megabyte tool outputs.
6. Host RAM: Before running any GPU-heavy operation, check `free -g` and keep at least
   15 GiB headroom. The `llama-swap.service` must stay running; do not stop services.
7. Preserve unrelated work and user changes.

---

## Current Workspace State & Accomplishments

The previous worker already implemented almost all code refactorings:
1. **Card 19**: Neutral text generation contracts published in `include/gufo/ext/text.hpp`, `src/ext/text.hpp`, `src/ext/text.cpp`, `include/gufo/ext/registry.hpp`, `src/ext/runtime.cpp`. Version bumped to 0.1.6. `CHANGES.md` and `SURFACE` updated.
2. **Card 20**: `src/models/qwen` extension structure created (`extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.cpp`, `probe.hpp`, `runner.cpp`, `runner.hpp`, `drivers.cpp`, `continuation_identity.cpp`).
3. **Card 21**: `src/models/qwen38_flash_next` extension structure created (`extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.cpp`, `probe.hpp`, `runner.cpp`, `runner.hpp`, `drivers.cpp`, `continuation_identity.cpp`, `bench_yarn.cpp`).
4. **Card 22**: `src/models/deepseek_v4_flash` extension structure created (`extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.cpp`, `probe.hpp`, `runner.cpp`, `runner.hpp`, `drivers.cpp`, `continuation_identity.cpp`).
5. `src/cli/serve/inference_backend.cpp`, `bench.cpp`, `prompt.cpp`, `openai_chat.cpp` decoupled from direct model includes.
6. `extensions.lock` updated with all 8 models.
7. `tools/ci/check-extension-isolation.py` updated with all 8 models in `MIGRATED_EXTENSIONS`.

---

## Immediate Tasks to Finish Implementation

### Step 1: Fix CMake Extension Target Definition in `extension.cmake`
In `src/models/deepseek_v4_flash/extension.cmake` and `src/models/qwen38_flash_next/extension.cmake`, `gufo_extension_target(NAME ..._hip)` was placed inside `if (ENGINE_ENABLE_HIP)`.
Because `extension.json` declares the HIP targets with `"requires": ["hip"]`, `GufoExtension.cmake` requires `gufo_extension_target` to be called unconditionally for every declared target. `gufo_extension_target` automatically checks if HIP is enabled and enables/disables the target accordingly.
- Move `gufo_extension_target(NAME gufo_deepseek_v4_flash_hip ...)` and `gufo_extension_target(NAME gufo_qwen38_flash_next_hip ...)` outside `if (ENGINE_ENABLE_HIP)`.
- Guard target-specific link libraries, compile options, and `gufo_extension_hip_sources` with `if(TARGET ..._hip)`.
(Note: `src/models/qwen/extension.cmake` and `src/models/qwen3_asr/extension.cmake` already do this correctly for reference.)

### Step 2: Configure and Build
- Run `nix develop -c cmake --preset cpu-test`.
- Build the affected targets: `nix develop -c cmake --build --preset cpu-test`.
- Fix any compilation or linker errors.

### Step 3: Run All Required Verification Gates
Run and record exact outputs/logs (using redirected log files in `/tmp/card19-22-logs/`):
1. `python3 tools/ci/generate-infrastructure.py --check` (use `--write` if any sync needed)
2. `python3 -m unittest tests/tools/generate_infrastructure_test.py`
3. `nix develop -c python3 tests/extensions/metadata_test.py --schema`
4. `python3 tools/ci/check-docs.py`
5. `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` (use `--fix` if formatting needed)
6. `python3 -m unittest tests/tools/check_extension_isolation_test.py`
7. `python3 tools/ci/check-extension-isolation.py --fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes`
8. `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity' --output-on-failure`
9. `nix develop -c cmake --build --preset cpu-test --target check-pr`
10. `nix build .#checks.x86_64-linux.pr --no-link`
11. `nix build` (verifies production package build and `./result/bin/gufo diagnose`)

### Step 4: Write Evidence and Handoff
1. Write `docs/model-extensions/evidence/card19-22/build-and-tests.txt` documenting commands, exit codes, and summary output for each gate.
2. Write `docs/model-extensions/evidence/card19-22/pending-commands.txt` documenting deferred functional/benchmark commands for the post-Card-14 qualification window.
3. Write `docs/model-extensions/evidence/card19-22/RESUME-NOTES.md` documenting completed work, summary of changes, and an explicit handoff of editing ownership.
