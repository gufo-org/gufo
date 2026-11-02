# Card 13 Resume Notes: EmbeddingGemma Pilot Migration

## Summary of Accomplished Work

Card 13 successfully migrated `src/models/embedding_gemma2` from a statically hardwired core dependency to an isolated, discovered Gufo extension conforming to RFC Step 8:

1. **Pilot Coupling Inventory (`docs/model-extensions/evidence/card13/pilot-inventory.md`)**:
   - Documented all 8 includes from `src/cli/serve/` to `src/models/embedding_gemma2/` headers.
   - Identified symbol leaks (`kMaxImageSoftTokens`, `SupportedImageSoftTokens`, `LooksLikeEmbeddingGemma2`).

2. **Public Extension API Contracts & Bump**:
   - Created `include/gufo/ext/embeddings.hpp` and `src/ext/embeddings.hpp`:
     - Declared multimodal embedding interfaces (`EmbeddingBatch`, `EmbeddingInput`, `EmbeddingPart`, `EmbeddingStatus`, `EmbeddingVideoOptions`, `EmbeddingRequestOptions`, `EmbeddingRunner`, `EmbeddingProvider`).
   - Extended `include/gufo/ext/registry.hpp` and `src/ext/runtime.cpp` with `AddEmbeddingProvider`, `RegisterEmbeddingProvider`, `FindEmbeddingProvider`, `ProbeEmbeddingProvider`.
   - Bumped extension API version to `0.1.3` (revision bump 3) in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`.
   - Documented changes in `include/gufo/ext/CHANGES.md`.
   - Updated and passed `tests/extensions/ext_runtime_neutral_contracts_test.cpp`.
   - Updated surface hash with `python3 tools/ci/check-extension-api.py --write` (99 paths).

3. **Decoupled Serving Boundary**:
   - Completely eradicated all model includes in `src/cli/serve/embedding_service.hpp`, `src/cli/serve/embeddings_api.cpp`, and `src/cli/serve/embedding_service.cpp`.
   - Interacted exclusively through `gufo::ext::FindEmbeddingProvider`, `ext::EmbeddingRunner`, and neutral types.

4. **Internal Extension Self-Containment**:
   - Migrated all 24 internal includes in `src/models/embedding_gemma2/` to extension-relative includes (`"config.hpp"`, `"loader.hpp"`, etc.).
   - Maintained FFmpeg/libav and libsndfile/libsoxr audio/video decoding logic inside the extension.

5. **Manifests and Provider Implementation**:
   - Declared `src/models/embedding_gemma2/extension.json` with schema 1:
     - Target `gufo_embedding_gemma2` (`requires: []`) with registration entry `embedding_gemma2_register`.
     - Target `gufo_embedding_gemma2_hip` (`requires: ["hip"]`) with provider entry `embedding_gemma2_provider_register`.
     - Option `--default-prompt-name`.
   - Implemented `src/models/embedding_gemma2/probe.hpp` & `extension.cpp` (pure, HIP-free probe and prompt artifact resolution).
   - Implemented `src/models/embedding_gemma2/provider.cpp` (HIP runner implementation for `EmbeddingRunner`).
   - Created `src/models/embedding_gemma2/extension.cmake` and removed `src/models/embedding_gemma2/CMakeLists.txt`.
   - Added `"embedding_gemma2"` to `extensions.lock`.

6. **Target Ownership & Isolation Ratchet**:
   - Set `"consumers": []` in `src/models/embedding_gemma2/library.json`.
   - Removed `gufo_embedding_gemma2` from `gufo_http` and `gufo` target links in root `CMakeLists.txt`.
   - Added `--fail-on-extension-boundary` / `--fail-on-cli-model-includes` to `tools/ci/check-extension-isolation.py`.
   - Added regression test `test_extension_boundary_blocks_cli_includes_of_migrated_extensions` and repository test `test_repository_satisfies_the_extension_boundary` in `tests/tools/check_extension_isolation_test.py`.

7. **Independent Review & Acceptance Audit**:
   - Independent Reviewer (`6c38a629-ee90-4a8b-be09-cf88cc3f5406`) audited all areas and issued CLEAN PASS with zero findings (F0).
   - Review findings disposition table recorded in Card 13 execution record.
   - All 3 Definition of Done items verified and marked `[x]`.

8. **Integration Verification Re-runs**:
   - All 10 affected verification gates were re-executed once by the integration agent and recorded with exact commands, exit codes, and output excerpts in `docs/model-extensions/evidence/card13/integration-reruns.txt`:
     * Gate 1 (GPU Embedding CTests): 15 tests (9 passed, 6 skipped cleanly code 77, 0 failed; 4.00s).
     * Gate 2 (Runtime Neutral Contracts): 1/1 passed (0.00s).
     * Gate 3 (Extension API Surface): 99 paths, version 0.1.3 (revision 3).
     * Gate 4 (Extension API Policy Unit Tests): 7/7 passed (0.002s).
     * Gate 5 (Extension Isolation & Boundaries): All 6 boundaries clean (`core`, `serving`, `model-core`, `extension`, `target-ownership`, `family-graph`).
     * Gate 6 (Isolation Ratchet Unit Tests): 26/26 passed (22.960s).
     * Gate 7 (Extension Metadata Tests): 12/12 passed (10.415s).
     * Gate 8 (Extension Discovery Tests): 13/13 passed (46.906s).
     * Gate 9 (Continuation Identity Goldens): 3/3 passed on GPU preset (0.01s).
     * Gate 10 (Clang Format): 792 C++ files checked clean.

## Agent Sequence
- **Worker**: Agent `77f3ed36-4902-49c7-b804-b3ab22858d4e`
- **Reviewer**: Agent `6c38a629-ee90-4a8b-be09-cf88cc3f5406` (CLEAN PASS, F0)
- **Integrator**: Agent `b3930d21-4138-46f7-b64f-835f02539530`
- **Orchestrator**: Final closure

## Handoff of Editing Ownership

The integration phase is complete. Card 13 is set to `integrated, ready for orchestrator closure`.
No Git or Jujutsu commits have been made (`jj` / `git` untouched). `docs/model-extensions/RFC.md` remains strictly byte-identical.
Editing ownership is released and handed back to the orchestrator agent (`cddee1bf-02b8-4a7b-a6cc-9074e6be649d`).
