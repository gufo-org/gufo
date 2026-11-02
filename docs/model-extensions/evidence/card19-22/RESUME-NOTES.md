# Cards 19–22 Resume Notes: Shared Text Contract, Qwen 27B, Flash-Next, and DeepSeek Extensions

## 1. Summary of Accomplished Work

This work bundle successfully implements **Cards 19, 20, 21, and 22** of the model-extensions RFC, completing the extension migration for all text-generation models:

### Card 19: Neutral Text Generation Surface & Extension API
- Published provider-neutral text runtime contracts in `include/gufo/ext/text.hpp`, `src/ext/text.hpp`, `src/ext/text.cpp`, `include/gufo/ext/registry.hpp`, `src/ext/runtime.cpp`, `include/gufo/ext/component_cache_resources.hpp`, and `include/gufo/ext/gguf_identity.hpp`.
- Published `TextModelRunner`, `TextRunnerState`, `TextPreparedPrompt`, `TextDecodeStep`, `TextDecodeSelection`, `TextRunnerDescriptor`, `TextRunnerCapabilities`, `TextExecutionPlan`, `TextRunnerPersistenceDescriptor`, `TextRunnerResourceClaim`, and `TextProvider`.
- Forwarded runtime option descriptors `TextBenchOptions` and `TextPromptOptions` to decouple `src/cli/bench` and `src/cli/prompt` from individual model headers.
- Exported extension surface and bumped extension API version to `0.1.6` with full documentation in `include/gufo/ext/CHANGES.md` and `include/gufo/ext/SURFACE`.

### Card 20: Qwen 27B Model Extension
- Created modular extension structure in `src/models/qwen/`:
  * `extension.cmake`: Defines `gufo_qwen` and `gufo_qwen_hip` extension targets.
  * `extension.cpp`: Calls extension registration.
  * `provider.cpp`: Registers `qwen27-q4` text provider (`ProbeQwen`, `CreateQwenTextRunner`, `RunQwenBenchmark`, `RunQwenPrompt`, `RunQwenChat`).
  * `probe.hpp` / `probe.cpp`: Architecture and quantization probes.
  * `runner.hpp` / `runner.cpp`: HIP-backed `TextModelRunner` implementation for Qwen 27B.
  * `drivers.cpp`: Standalone benchmark and interactive prompt/chat execution logic.
  * `continuation_identity.hpp` / `continuation_identity.cpp`: Model-owned continuation hashing and cache identity.
- Updated `src/models/qwen/extension.json` with packages, targets, and documentation.

### Card 21: Qwen3.8 Flash-Next Model Extension
- Created modular extension structure in `src/models/qwen38_flash_next/`:
  * `extension.cmake`: Defines `gufo_qwen38_flash_next` and `gufo_qwen38_flash_next_hip` targets.
  * `extension.cpp`: Extension registration.
  * `provider.cpp`: Registers `flash-next-q4` text provider.
  * `probe.hpp` / `probe.cpp`: Flash-Next architecture probe (`qwen4exp`).
  * `runner.hpp` / `runner.cpp`: HIP-backed `TextModelRunner` implementation with MTP speculative decoding support.
  * `drivers.cpp`: Standalone benchmark and interactive prompt/chat execution logic.
  * `continuation_identity.hpp` / `continuation_identity.cpp`: Continuation layout and cache identity.
  * `bench_yarn.hpp` / `bench_yarn.cpp`: Flash-Next YaRN RoPE scaling benchmarks.
- Updated `src/models/qwen38_flash_next/extension.json`.

### Card 22: DeepSeek V4 Flash Model Extension
- Created modular extension structure in `src/models/deepseek_v4_flash/`:
  * `extension.cmake`: Defines `gufo_deepseek_v4_flash` and `gufo_deepseek_v4_flash_hip` targets.
  * `extension.cpp`: Extension registration.
  * `provider.cpp`: Registers `deepseek4` text provider (canonical provider ID matching GGUF architecture name).
  * `probe.hpp` / `probe.cpp`: DeepSeek architecture probe (`deepseek4`).
  * `runner.hpp` / `runner.cpp`: HIP-backed `TextModelRunner` implementation with DSpark speculative decoding support.
  * `drivers.cpp`: Standalone benchmark and prompt/chat drivers.
  * `continuation_identity.hpp` / `continuation_identity.cpp`: Continuation identity helpers.
- Updated `src/models/deepseek_v4_flash/extension.json`.

### Infrastructure & Serving Boundary Decoupling
- `src/cli/serve/inference_backend.cpp`, `src/cli/serve/inference_backend.hpp`, `src/cli/serve/openai_chat.cpp`, `src/cli/bench/bench.cpp`, and `src/cli/prompt/prompt.cpp` now operate entirely against provider-neutral interfaces (`ext::TextModelRunner`, `ext::TextProvider`, `ext::ProbeTextProvider`). Direct includes of model-specific headers (`src/models/qwen/...`, `src/models/deepseek_v4_flash/...`, `src/models/qwen38_flash_next/...`) have been removed.
- Updated `extensions.lock` with all 8 discovered extensions.
- Added all 8 extensions to `MIGRATED_EXTENSIONS` in `tools/ci/check-extension-isolation.py`.
- Formatted all 848 repository C++ files to comply with `tools/ci/check-format.py`.

---

## 2. Gate Verification Results Summary

All 11 verification gates were executed and passed with exit code 0:

| Gate | Description | Command | Exit Code | Result |
|------|-------------|---------|-----------|--------|
| **1** | Infrastructure Generation Check | `python3 tools/ci/generate-infrastructure.py --check` | 0 | PASS |
| **2** | Infrastructure Generator Tests | `python3 -m unittest tests/tools/generate_infrastructure_test.py` | 0 | PASS (9/9) |
| **3** | Manifest Schema Validation | `nix develop -c python3 tests/extensions/metadata_test.py --schema` | 0 | PASS (8/8) |
| **4** | Documentation & Link Integrity | `python3 tools/ci/check-docs.py` | 0 | PASS (323 docs, 1609 links) |
| **5** | Code Formatting | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | 0 | PASS (848 files) |
| **6** | Extension Isolation Unit Tests | `python3 -m unittest tests/tools/check_extension_isolation_test.py` | 0 | PASS (27/27) |
| **7** | Static Extension Boundaries | `python3 tools/ci/check-extension-isolation.py --fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes` | 0 | PASS (0 violations) |
| **8** | Continuation Identity Unit Tests | `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity' --output-on-failure` | 0 | PASS (3/3) |
| **9** | Hosted PR Test Suite (`check-pr`) | `nix develop -c cmake --build --preset cpu-test --target check-pr` | 0 | PASS (96/96, 100%) |
| **10** | Hermetic Hosted CPU & Repo Checks | `nix build .#checks.x86_64-linux.pr --no-link` | 0 | PASS (4 derivations) |
| **11** | Production Build & Diagnostics | `nix build && ./result/bin/gufo diagnose` | 0 | PASS (Overall Status: PASS) |

Detailed output logs are documented in `docs/model-extensions/evidence/card19-22/build-and-tests.txt` and `docs/model-extensions/evidence/card19-22/integration-reruns.txt`.

---

## 3. Review Findings & Integration Dispositions

All findings raised in `docs/model-extensions/evidence/card19-22/review-findings.md` have been fully resolved by the integration agent:

1. **`M1` (Major) — Register provider metadata constants in `provider.cpp` for text extensions**:
   - **Resolution**: Updated `src/models/qwen/provider.cpp`, `src/models/qwen38_flash_next/provider.cpp`, and `src/models/deepseek_v4_flash/provider.cpp` to include their generated constants header (`<gufo/ext/generated/..._constants.hpp>`) and invoke `registry.AddProvider(...)` with their respective constants (`kProvider_qwen27_q4`, `kProvider_flash_next_q4`, `kProvider_deepseek4`).
   - **Verification**: Verified via `gufo diagnose` that all 8 providers (`embedding_gemma2`, `minimax_h3`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`, `qwen27-q4`, `flash-next-q4`, `deepseek4`) are discovered and reported as `available`.
2. **`N1` (Nit) — Align DeepSeek provider naming**:
   - **Resolution**: Clarified that `deepseek4` is the canonical registered provider ID in `src/models/deepseek_v4_flash/extension.json` and `provider.cpp`, matching the GGUF architecture name.
3. **`N2` (Nit) — Breaking-change notation `!` in commit title**:
   - **Resolution**: Kept standard Conventional Commit `feat(extensions):` per user confirmation that replacing fallback with strict probing is standard internal model validation rather than a breaking surface change.

---

## 4. Commitments and Invariants Preserved

1. **No Commits**: No `git commit`, `jj commit`, `jj new`, or `jj squash` was executed. All changes remain staged in the Git working tree ready for orchestrator squash/commit.
2. **RFC Integrity**: `docs/model-extensions/RFC.md` remains strictly byte-identical and unmodified.
3. **Scope Discipline**: Functional harness runs (`tests/functional/run.py` rows) and live model benchmarks were deferred to the post-Card-14 qualification window per standing user instruction and recorded in `docs/model-extensions/evidence/card19-22/pending-commands.txt`.
4. **Host RAM**: Preserved >15 GiB headroom throughout execution (110 GiB available).
5. **Conventional Commits**: PR and squash commit title:
   `feat(extensions): migrate text generation models (Cards 19-22)`

---

## 5. Handoff

The implementation and integration of Cards 19, 20, 21, and 22 is complete and verified across all 11 verification gates.
**Sole editing ownership of the shared worktree is hereby formally returned to the orchestrator.**
