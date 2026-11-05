> Instructions given to the review agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Reviewer brief — Cards 19–22: Shared Text Contract, Qwen 27B, Flash-Next, and DeepSeek Extensions

You are the **read-only reviewer agent** (`reviewer`) for Cards 19, 20, 21, and 22 of the model-extensions RFC in
`/home/mixer/gufo`.

## Absolute constraints

1. **You are strictly read-only.** Do not edit code, do not edit documentation, do not run git/jj write commands. (You may only write your findings report to `docs/model-extensions/evidence/card19-22/review-findings.md`).
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
3. Evaluate the actual diff (`jj diff`) and files against:
   - `AGENTS.md`
   - `docs/model-extensions/CARDS.md`
   - `docs/model-extensions/cards/README.md`
   - `docs/model-extensions/cards/19-shared-text-contract.md`
   - `docs/model-extensions/cards/20-qwen27b-extension.md`
   - `docs/model-extensions/cards/21-flash-next-extension.md`
   - `docs/model-extensions/cards/22-deepseek-extension.md`
   - `docs/model-extensions/evidence/card19-22/build-and-tests.txt`
   - `docs/model-extensions/evidence/card19-22/RESUME-NOTES.md`
4. Write your concrete findings directly into `docs/model-extensions/evidence/card19-22/review-findings.md`.
   Every finding must have:
   - Identifier and Title (e.g. `M1`, `N1`)
   - Severity: `Critical`, `Major`, `Minor`, or `Nit`
   - Affected paths
   - Concrete problem and violation of requirements
   - Suggested correction
   If there are no findings in a category, explicitly state that.
5. If you do not have shell execution capabilities, emit the exact validation gate commands that the integration agent must run.

## Scope of Review

Review the 4 cards and all migrated text models:
1. **Card 19 (Shared Text Contract)**:
   - Neutral text contracts in `include/gufo/ext/text.hpp` / `src/ext/text.hpp` (`TextModelRunner`, `TextRunnerState`, `TextPromptContext`, `TextPreparedPrompt`, `TextRunnerToken`, `TextRunnerCapabilities`, `TextProvider`).
   - Registration in `ext::Registry` (`AddTextProvider`, `FindTextProvider`, `ProbeTextProvider`).
   - Extension API version bumped to `0.1.6` in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`.
   - `include/gufo/ext/CHANGES.md` and `SURFACE` updated.
   - Decoupled `src/cli/serve/component_text_cache.hpp` and `src/cli/serve/text_model_runner.hpp`.
2. **Card 20 (Qwen 27B Extension)**:
   - `src/models/qwen` extension structure: `extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`/`probe.cpp`, `runner.hpp`/`runner.cpp`, `drivers.cpp`, `continuation_identity.hpp`/`continuation_identity.cpp`.
   - HIP-free supported qwen35 key/shape probe replacing permissive fallback with explicit clear errors.
   - Provider `qwen27-q4` registration and options.
   - Prompt/bench/chat CLI drivers.
3. **Card 21 (Qwen3.8 Flash-Next Extension)**:
   - `src/models/qwen38_flash_next` extension structure: `extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`/`probe.cpp`, `runner.hpp`/`runner.cpp`, `drivers.cpp`, `continuation_identity.hpp`/`continuation_identity.cpp`, `bench_yarn.hpp`/`bench_yarn.cpp`.
   - Provider `flash-next-q4` registration, MTP options, decode concurrency, YaRN resolution.
   - CLI drivers.
4. **Card 22 (DeepSeek V4 Flash Extension)**:
   - `src/models/deepseek_v4_flash` extension structure: `extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`/`probe.cpp`, `runner.hpp`/`runner.cpp`, `drivers.cpp`, `continuation_identity.hpp`/`continuation_identity.cpp`.
   - Provider `deepseek-v4-flash-q4` registration, DSpark options.
   - `tool_calls.cpp` decoupled from `gufo_http`.
   - CLI drivers.
5. **Decoupling, Isolation, and Verification Gates**:
   - `src/cli/serve/inference_backend.cpp`, `openai_chat.cpp`, `src/cli/bench/bench.cpp`, `src/cli/prompt/prompt.cpp` decoupled from direct model headers.
   - Root `CMakeLists.txt` direct model dependencies removed; all 8 models built via extensions.
   - `extensions.lock` updated with all 8 models.
   - `tools/ci/check-extension-isolation.py` enforcing boundary with 0 strict violations.
   - Continuation identity goldens passing for all models.
   - Gate verification records in `build-and-tests.txt` and pending commands in `pending-commands.txt`.
