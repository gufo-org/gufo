> Instructions given to the integration agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Integration brief — Cards 19–22: Shared Text Contract, Qwen 27B, Flash-Next, and DeepSeek Extensions

You are the **integration agent** (`codex-exec-writer`) for Cards 19, 20, 21, and 22 of the model-extensions RFC in
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

## Tasks to Complete

### 1. Resolve Reviewer Findings
- **M1 (Major): Register provider metadata constants in `provider.cpp` for text extensions**:
  - In `src/models/qwen/provider.cpp`: `#include <gufo/ext/generated/qwen_constants.hpp>`, call `registry.AddProvider(gufo::ext::generated::extension_qwen::kProvider_qwen27_q4);` in `QwenRegisterProvider`.
  - In `src/models/qwen38_flash_next/provider.cpp`: `#include <gufo/ext/generated/qwen38_flash_next_constants.hpp>`, call `registry.AddProvider(gufo::ext::generated::extension_qwen38_flash_next::kProvider_flash_next_q4);` in `Qwen38FlashNextRegisterProvider`.
  - In `src/models/deepseek_v4_flash/provider.cpp`: `#include <gufo/ext/generated/deepseek_v4_flash_constants.hpp>`, call `registry.AddProvider(gufo::ext::generated::extension_deepseek_v4_flash::kProvider_deepseek4);` in `DeepSeekV4FlashRegisterProvider`.
  - Verify that `gufo diagnose` lists all 8 providers under "Provider Availability" (`embedding_gemma2`, `minimax_h3`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`, `qwen27-q4`, `flash-next-q4`, `deepseek4`).
- **N1 (Nit): Align DeepSeek provider naming**:
  - Clarify in `docs/model-extensions/evidence/card19-22/RESUME-NOTES.md` that `deepseek4` is the canonical provider ID matching the GGUF architecture name.
- **N2 (Nit): Breaking-change notation `!`**:
  - Record in the card execution records that replacing the permissive fallback with `ProbeQwenGguf` is a breaking change requiring `!` in the commit title (`feat(extensions)!: migrate text generation models (Cards 19-22)`).

### 2. Re-run Affected Verification Gates Once
Re-run the verification gates and record exact commands, exit codes, and transcripts in `docs/model-extensions/evidence/card19-22/integration-reruns.txt`:
1. `python3 tools/ci/generate-infrastructure.py --check`
2. `python3 -m unittest tests/tools/generate_infrastructure_test.py`
3. `nix develop -c python3 tests/extensions/metadata_test.py --schema`
4. `python3 tools/ci/check-docs.py`
5. `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py --fix`
6. `python3 -m unittest tests/tools/check_extension_isolation_test.py`
7. `python3 tools/ci/check-extension-isolation.py --fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes`
8. `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity' --output-on-failure`
9. `nix develop -c cmake --build --preset cpu-test --target check-pr`
10. `nix build .#checks.x86_64-linux.pr --no-link`
11. `nix build && ./result/bin/gufo diagnose`

### 3. Update Card Files and Ledgers
- Update `docs/model-extensions/cards/19-shared-text-contract.md`:
  - Change Status to **complete**.
  - Mark all Definition of Done checkboxes `[x]`.
  - Fill in the Execution Record with implementation, review dispositions, and verification.
- Update `docs/model-extensions/cards/20-qwen27b-extension.md`:
  - Change Status to **complete**.
  - Mark all Definition of Done checkboxes `[x]`.
  - Fill in the Execution Record.
- Update `docs/model-extensions/cards/21-flash-next-extension.md`:
  - Change Status to **complete**.
  - Mark all Definition of Done checkboxes `[x]`.
  - Fill in the Execution Record.
- Update `docs/model-extensions/cards/22-deepseek-extension.md`:
  - Change Status to **complete**.
  - Mark all Definition of Done checkboxes `[x]`.
  - Fill in the Execution Record.
- Update `docs/model-extensions/CARDS.md`:
  - Update status table for Cards 19, 20, 21, and 22 to `complete`.
  - Add summary execution records for Cards 19, 20, 21, and 22.
- Update `docs/model-extensions/cards/README.md` if applicable.
- Author `docs/model-extensions/evidence/card19-22/README.md`.
- Update `docs/model-extensions/evidence/card19-22/RESUME-NOTES.md` and explicitly return editing ownership to the orchestrator.
