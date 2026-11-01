# Review Findings — Cards 19–22: Shared Text Contract, Qwen 27B, Flash-Next, and DeepSeek Extensions

**Reviewer Agent**: `reviewer` (independent read-only evaluation)  
**Date**: 2026-11-01  
**Target Cards**: Card 19, Card 20, Card 21, Card 22  
**Worktree Parent**: `b05ccc4a` (`model-extensions-5 | feat(extensions): migrate speech, audio, image, and video models to extensions`)  
**Working Copy Commit**: `2dba1098`

---

## Executive Summary

Cards 19, 20, 21, and 22 implement the provider-neutral text runtime contract and migrate all text-generation models (`qwen`, `qwen38_flash_next`, and `deepseek_v4_flash`) into modular extensions. All 8 repository models are now migrated and locked in `extensions.lock`.

The review confirms:
- **Card 19 (Shared Text Contract)**: Fully implemented in `include/gufo/ext/text.hpp`, `src/ext/text.hpp`, `src/ext/text.cpp`, `include/gufo/ext/registry.hpp`, `src/ext/runtime.cpp`. API version successfully bumped to `0.1.6` in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`. `include/gufo/ext/CHANGES.md` and `SURFACE` (120 headers) are verified via `check-extension-api.py`. Tool dialect fallback and schema parsing are cleanly abstracted via `ToolDialect`.
- **Card 20 (Qwen 27B Extension)**: Fully extracted into `src/models/qwen/`. The permissive fallback has been replaced with strict `ProbeQwenGguf` shape validation (`embedding_length == 5120`), deterministically rejecting unsupported architectures and shapes. Standalone benchmark and prompt/chat drivers are implemented in `src/models/qwen/drivers.cpp`.
- **Card 21 (Qwen3.8 Flash-Next Extension)**: Fully extracted into `src/models/qwen38_flash_next/`. `ProbeFlashNextGguf` strictly verifies `qwen4exp`. MTP speculation, YaRN RoPE scaling (`bench_yarn.hpp`/`cpp`), and CLI drivers are migrated.
- **Card 22 (DeepSeek V4 Flash Extension)**: Fully extracted into `src/models/deepseek_v4_flash/`. `tool_calls.cpp` is decoupled from `gufo_http` and uses neutral types `ext::ScannedToolCall` / `ext::ScannedToolArgument`. DSpark speculation and CLI drivers are migrated.
- **Decoupling and Isolation**: `src/cli/serve/inference_backend.cpp`, `openai_chat.cpp`, `src/cli/bench/bench.cpp`, and `src/cli/prompt/prompt.cpp` contain zero direct includes to `src/models/`. Root `CMakeLists.txt` direct model dependencies are eliminated. Static boundary checks pass with 0 strict violations.
- **Verification Gates**: 11 verification gates were executed and pass (format, unit tests, schema, documentation, isolation, continuation identity, `check-pr` with 96/96 tests, hermetic `.checks.x86_64-linux.pr`, and `nix build` + `gufo diagnose`).
- **Invariants**: `docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted. Functional harness runs are deferred to the dedicated post-Card-14 qualification window per project policy and documented in `pending-commands.txt`.

One **Major** finding (`M1`) was identified regarding provider metadata registration in `gufo diagnose` / `registry.Providers()`, along with two minor/nit items (`N1`, `N2`).

---

## Detailed Findings

### Critical Findings
*None.*

---

### Major Findings

#### `M1`: Text extensions omit `registry.AddProvider(...)` with generated metadata constants in `provider.cpp`
- **Severity**: Major
- **Affected paths**:
  - `src/models/qwen/provider.cpp`
  - `src/models/qwen38_flash_next/provider.cpp`
  - `src/models/deepseek_v4_flash/provider.cpp`
- **Problem**:
  In `cmake/GufoExtension.cmake`, when an extension target is enabled (e.g. HIP enabled), CMake generates code that calls the provider registration entry point (e.g. `qwen_provider_register(registry)`). Each provider entry point is expected to register its capability metadata via `registry.AddProvider(gufo::ext::generated::extension_<name>::kProvider_<identifier>);` and its capability runner factory via `registry.AddTextProvider({...});`.
  
  All 5 previously migrated non-text extensions (`embedding_gemma2`, `minimax_h3`, `qwen3_asr`, `qwen3_tts`, and `qwen_image_21`) follow this pattern.
  
  However, `src/models/qwen/provider.cpp`, `src/models/qwen38_flash_next/provider.cpp`, and `src/models/deepseek_v4_flash/provider.cpp` only call `registry.AddTextProvider(...)` and omit `registry.AddProvider(...)`.
  
  Consequences:
  1. `gufo diagnose` queries `registry.Providers()` to list all built extensions and their available provider capabilities. As shown in the production diagnostics output (`docs/model-extensions/evidence/card19-22/build-and-tests.txt` lines 232–237), `provider embeddinggemma-2`, `minimax-h3`, `qwen3-asr`, `qwen3-tts`, and `qwen-image-21` are reported as available, while `qwen27-q4`, `flash-next-q4`, and `deepseek4` are entirely missing.
  2. Subsystems and tests that inspect `registry.Providers()` (e.g. option validation in `src/ext/options.cpp`, `ValidateSelectedOptions`, and speculative capability lookup via `std::ranges::any_of(registry.Providers(), ...)`) cannot discover the registered text providers or their declared options (`--dflash-model`, `--mtp-model`, `--dspark-model`).
- **Suggested correction**:
  In each model's `provider.cpp`, include the generated constants header and call `registry.AddProvider(...)` at the beginning of the provider registration function:
  - In `src/models/qwen/provider.cpp`:
    ```cpp
    #include <gufo/ext/generated/qwen_constants.hpp>
    // ...
    void qwen_provider_register(gufo::ext::Registry& registry) {
      registry.AddProvider(
          gufo::ext::generated::extension_qwen::kProvider_qwen27_q4);
    #if defined(ENGINE_ENABLE_HIP)
      registry.AddTextProvider({ ... });
    #endif
    }
    ```
  - In `src/models/qwen38_flash_next/provider.cpp`:
    ```cpp
    #include <gufo/ext/generated/qwen38_flash_next_constants.hpp>
    // ...
    void qwen38_flash_next_provider_register(gufo::ext::Registry& registry) {
      registry.AddProvider(
          gufo::ext::generated::extension_qwen38_flash_next::kProvider_flash_next_q4);
    #if defined(ENGINE_ENABLE_HIP)
      registry.AddTextProvider({ ... });
    #endif
    }
    ```
  - In `src/models/deepseek_v4_flash/provider.cpp`:
    ```cpp
    #include <gufo/ext/generated/deepseek_v4_flash_constants.hpp>
    // ...
    void deepseek_v4_flash_provider_register(gufo::ext::Registry& registry) {
      registry.AddProvider(
          gufo::ext::generated::extension_deepseek_v4_flash::kProvider_deepseek4);
    #if defined(ENGINE_ENABLE_HIP)
      registry.AddTextProvider({ ... });
    #endif
    }
    ```

---

### Minor Findings
*None.*

---

### Nits

#### `N1`: Discrepancy in DeepSeek provider naming between documentation and manifest/code
- **Severity**: Nit
- **Affected paths**:
  - `docs/model-extensions/evidence/card19-22/RESUME-NOTES.md`
  - `docs/model-extensions/evidence/card19-22/input-brief-reviewer.md`
  - `src/models/deepseek_v4_flash/extension.json`
  - `src/models/deepseek_v4_flash/provider.cpp`
- **Problem**:
  `RESUME-NOTES.md` (L40) and `input-brief-reviewer.md` (L53) describe the registered text provider as `deepseek-v4-flash-q4`. In `src/models/deepseek_v4_flash/extension.json` (L15) and `src/models/deepseek_v4_flash/provider.cpp` (L29), the registered ID is `deepseek4` (matching the canonical GGUF architecture name).
- **Suggested correction**:
  Note `deepseek4` as the canonical registered provider ID in `RESUME-NOTES.md` for consistency.

#### `N2`: PR and squash commit title breaking-change notation (`!`)
- **Severity**: Nit
- **Affected paths**:
  - Pull request / squash commit message metadata
- **Problem**:
  Card 20 explicitly specifies: "Replace permissive fallback with supported qwen35 key/shape probe and clear errors; document CLI break with `!`." The permissive fallback was successfully removed and replaced by `ProbeQwenGguf` in `src/models/qwen/probe.cpp`.
- **Suggested correction**:
  Ensure the orchestrator includes `!` in the commit title when squashing, e.g.:
  `feat(extensions)!: migrate text generation models (Cards 19-22)`

---

## Audit of Definition of Done across Cards 19–22

### Card 19: Shared text-generation contract
| DoD Item | Status | Notes |
|:---|:---:|:---|
| Contract cleanly separates model primitives from scheduler/pool/cache policy and optional features have compatible defaults | **PASS** | `include/gufo/ext/text.hpp` and `src/ext/text.hpp` establish capability-neutral types (`TextModelRunner`, `TextRunnerState`, `TextPromptContext`, etc.). `src/cli/serve/component_text_cache.hpp` and `text_model_runner.hpp` are decoupled. |
| Tool-syntax fallback and provider-valued options have explicit tested behavior, including lock selection differences | **PASS** | `ToolDialect` registry implemented in `src/ext/text.cpp`; verified in `tests/cli/openai_chat_test.cpp` (`TestNativeToolDialectSelection`). Provider-valued `--speculative` validated in `src/ext/options.cpp`. |
| All four profiles, exact identities, actual drafts, continuation and matched per-phase timings satisfy shared gates | **DEFERRED** | Functional qualification runs across profiles deferred to post-Card-14 qualification window per standing user decision and documented in `pending-commands.txt`. Continuation identity tests pass for all 3 models. |

### Card 20: Qwen 27B text generation extension
| DoD Item | Status | Notes |
|:---|:---:|:---|
| Qwen AR/speculative and multimodal paths route through extension factories/drivers with unchanged valid-input outputs and identities | **PASS** | Factory (`CreateQwenTextRunner`), drivers (`RunQwenBenchmark`, `RunQwenPrompt`, `RunQwenChat`), and vision paths routed through `gufo_qwen_hip`. |
| Supported artifact shape selection is deterministic and unsupported artifacts fail clearly with documented CLI change | **PASS** | `ProbeQwenGguf` enforces `embedding_length == 5120` and rejects unsupported architectures (`deepseek4`, `qwen4exp`) with deterministic error. (Commit title break noted in N2). |
| Actual Qwen modes, numerical quality, forced tool choice, cache histories and CLI/HTTP speed continuity qualified | **DEFERRED** | Standard functional/benchmark qualification deferred to post-Card-14 window per standing user decision. CPU continuation identity golden passed. |

### Card 21: Flash-Next text generation extension
| DoD Item | Status | Notes |
|:---|:---:|:---|
| Production registration/parser/probe paths execute the correct modes and actual drafts with unchanged sampler/constraint semantics | **PASS** | Factory (`CreateQwenFlashNextTextRunner`), probe (`qwen4exp`), and drivers (`RunQwen38FlashNextBenchmark`, `RunFlashNextPrompt`, `RunFlashNextChat`) registered. |
| Identity bytes, outputs, cache work and model-computed concurrency contributions remain exact | **PASS** | `src/models/qwen38_flash_next/continuation_identity.cpp` passes exact golden test `qwen38_flash_next_continuation_identity_test`. |
| Independent numerical and matched CLI/HTTP quality/performance gates complete for discovery and actual lock artifacts | **DEFERRED** | Deferred to post-Card-14 qualification window per standing user decision. |

### Card 22: DeepSeek text generation extension
| DoD Item | Status | Notes |
|:---|:---:|:---|
| DeepSeek DSML/forced tool behavior, actual DSpark modes and native snapshot persistence survive extension routing | **PASS** | `tool_calls.cpp` decoupled from `gufo_http`; `ToolDialect` registered in `extension.cpp`. DSpark options declared in `extension.json`. |
| Exact durable/process-local identity schemas and all existing validation errors remain qualified | **PASS** | `src/models/deepseek_v4_flash/continuation_identity.cpp` passes exact golden test `deepseek_v4_flash_continuation_identity_test`. |
| Local real-weight numerical/API/CLI/HTTP evidence and matched per-phase timings exist on both discovery and shipped lock artifacts | **DEFERRED** | Deferred to post-Card-14 qualification window per standing user decision. |

---

## Conclusion & Recommendations

The implementation of Cards 19–22 is technically solid, structurally clean, and preserves all system architectural boundaries (0 static boundary violations, decoupled CLI, fully hermetic Nix checks passing).

Addressing finding **`M1`** (adding `registry.AddProvider(...)` with the generated constants in the three text model `provider.cpp` files) will restore full provider visibility in `gufo diagnose` and enable options introspection.

Review report written to: `docs/model-extensions/evidence/card19-22/review-findings.md`.
