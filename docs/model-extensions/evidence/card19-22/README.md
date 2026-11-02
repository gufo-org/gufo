# Cards 19–22 Integration & Verification Evidence

This directory consolidates the execution records, review findings, resolution logs, and verification evidence for **Cards 19, 20, 21, and 22** of the Model Extension Migration roadmap.

## Migrated Text Generation Models & Contracts

| Card | Model ID / Component | Domain | Base Target | HIP Target | Manifest |
|---|---|---|---|---|---|
| **Card 19** | `ext::text` | Neutral Text Runtime Contract | `gufo_ext_runtime` | N/A | `include/gufo/ext/text.hpp` |
| **Card 20** | `qwen` (`qwen27-q4`) | Text Generation (Qwen 27B) | `gufo_qwen` | `gufo_qwen_hip` | `src/models/qwen/extension.json` |
| **Card 21** | `qwen38_flash_next` (`flash-next-q4`) | Text Generation (Flash-Next) | `gufo_qwen38_flash_next` | `gufo_qwen38_flash_next_hip` | `src/models/qwen38_flash_next/extension.json` |
| **Card 22** | `deepseek_v4_flash` (`deepseek4`) | Text Generation (DeepSeek V4 Flash) | `gufo_deepseek_v4_flash` | `gufo_deepseek_v4_flash_hip` | `src/models/deepseek_v4_flash/extension.json` |

---

## Review Findings & Dispositions

All findings identified during independent review have been addressed and verified:

1. **M1 (Major) — Register provider metadata constants in `provider.cpp` for text extensions**:
   - **Resolution**: Added `#include <gufo/ext/generated/<name>_constants.hpp>` and `registry.AddProvider(...)` calls in `src/models/qwen/provider.cpp` (`kProvider_qwen27_q4`), `src/models/qwen38_flash_next/provider.cpp` (`kProvider_flash_next_q4`), and `src/models/deepseek_v4_flash/provider.cpp` (`kProvider_deepseek4`).
   - **Verification**: Verified via `gufo diagnose` that all 8 providers (`embedding_gemma2`, `minimax_h3`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`, `qwen27-q4`, `flash-next-q4`, `deepseek4`) are discovered and reported as `available`.
2. **N1 (Nit) — Align DeepSeek provider naming**:
   - **Resolution**: Documented `deepseek4` as the canonical registered provider ID in `RESUME-NOTES.md` and evidence ledgers, matching the GGUF architecture name.
3. **N2 (Nit) — Breaking-change notation `!`**:
   - **Resolution**: Kept standard Conventional Commit `feat(extensions):` per user confirmation that replacing fallback with strict probing is standard internal model validation rather than a breaking surface change.

---

## Verification Gates Status

| Gate | Description | Command | Exit Code | Result | Details |
|:---|---|---|:---:|:---:|---|
| **1** | Infrastructure Generation Check | `python3 tools/ci/generate-infrastructure.py --check` | 0 | **PASS** | Up to date with model metadata |
| **2** | Infrastructure Generator Tests | `python3 -m unittest tests/tools/generate_infrastructure_test.py` | 0 | **PASS** | 9/9 unit tests passed |
| **3** | Manifest Schema Validation | `nix develop -c python3 tests/extensions/metadata_test.py --schema` | 0 | **PASS** | Schema 1 validation passed |
| **4** | Documentation & Link Integrity | `python3 tools/ci/check-docs.py` | 0 | **PASS** | 327 docs, 1609 links verified |
| **5** | Code Formatting | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py --fix` | 0 | **PASS** | Formatted 848 C++ files |
| **6** | Extension Isolation Unit Tests | `python3 -m unittest tests/tools/check_extension_isolation_test.py` | 0 | **PASS** | 27/27 unit tests passed |
| **7** | Static Boundary Ratchet | `python3 tools/ci/check-extension-isolation.py --fail-on-extension-boundary --fail-on-serving-boundary --fail-on-model-core-includes` | 0 | **PASS** | 0 strict boundary violations across 1401 files |
| **8** | Continuation Identity Unit Tests | `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity' --output-on-failure` | 0 | **PASS** | 3/3 goldens passed (`qwen27b`, `deepseek_v4_flash`, `qwen38_flash_next`) |
| **9** | Hosted PR Test Suite (`check-pr`) | `nix develop -c cmake --build --preset cpu-test --target check-pr` | 0 | **PASS** | 96/96 test suites passed (100%) |
| **10** | Hermetic Hosted Checks | `nix build .#checks.x86_64-linux.pr --no-link` | 0 | **PASS** | 4/4 derivations built and passed |
| **11** | Production Build & Diagnostics | `nix build && ./result/bin/gufo diagnose` | 0 | **PASS** | All 8 extensions active and all 8 providers available |

> [!NOTE]
> End-to-end functional model harness sweeps (`tests/functional/run.py` rows) and live inference benchmarks are deferred to the unified post-Card-14 qualification window per RFC guidelines and documented in [pending-commands.txt](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/pending-commands.txt).

---

## Evidence Artifacts

- [integration-reruns.txt](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/integration-reruns.txt): Complete gate execution transcripts and diagnostic output.
- [review-findings.md](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/review-findings.md): Independent reviewer findings and audit.
- [build-and-tests.txt](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/build-and-tests.txt): Initial worker build logs and diagnostics.
- [pending-commands.txt](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/pending-commands.txt): Deferred functional commands.
- [RESUME-NOTES.md](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/RESUME-NOTES.md): Summary of architectural changes, findings dispositions, and ownership handoff.
- [input-brief-worker.md](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/input-brief-worker.md): Worker execution brief.
- [input-brief-reviewer.md](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/input-brief-reviewer.md): Reviewer execution brief.
- [input-brief-integration.md](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/input-brief-integration.md): Integration agent execution brief.
