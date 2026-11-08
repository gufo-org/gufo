# Card 11: Isolation Ratchet Verification

## Summary

Card 11 implements the second half of RFC Step 7 by establishing the model core boundary ratchet `--fail-on-model-core-includes` in `tools/ci/check-extension-isolation.py` and verifying that all 8 models have zero production includes of `src/core/**` or `src/cache/**`.

## Include Edge Counts Before and After

The lexical include inventory from `tools/ci/check-extension-isolation.py` records cross-owner include edges. All 152 private core/cache includes across the eight models migrated to public `<gufo/ext/...>` headers (classified under `configuration`):

| Model Directory | Before: Edge to `src/core` | Before: Edge to `src/cache` | After: Edge to `src/core` | After: Edge to `src/cache` | After: Edge to `configuration` (`<gufo/ext/...>`) |
|---|---|---|---|---|---|
| `src/models/embedding_gemma2` | 11 | 0 | **0** | **0** | 11 |
| `src/models/qwen3_asr` | 11 | 0 | **0** | **0** | 11 |
| `src/models/qwen3_tts` | 10 | 0 | **0** | **0** | 10 |
| `src/models/qwen_image_21` | 7 | 0 | **0** | **0** | 7 |
| `src/models/minimax_h3` | 11 | 0 | **0** | **0** | 11 |
| `src/models/qwen` | 48 | 5 | **0** | **0** | 53 |
| `src/models/qwen38_flash_next` | 27 | 2 | **0** | **0** | 29 |
| `src/models/deepseek_v4_flash` | 20 | 0 | **0** | **0** | 20 |
| **Total** | **145** | **7** | **0** | **0** | **152** |

## Ratchet Implementation

1. **CLI Flag & Function**:
   - Flag: `--fail-on-model-core-includes`
   - Validator: `model_core_boundary_violations(report)` in `tools/ci/check-extension-isolation.py`.
   - Rule: For every include edge originating from `src/models/**`, if not within a `tests/` or `test/` subdirectory, the include target must not resolve to `src/core/**` or `src/cache/**`, nor literally start with `src/core/` or `src/cache/`.

2. **Negative Proof & Unit Test Coverage**:
   - Added unit test `test_model_core_boundary_blocks_core_and_cache_includes_from_production_models` in `tests/tools/check_extension_isolation_test.py`.
   - Verified that synthetic production files including `src/core/json.hpp`, `src/cache/types.hpp`, and `src/core/hip/hip_utils.hpp` are flagged as violations.
   - Verified that public `<gufo/ext/...>`, own root-relative `src/models/<self>/...`, family `src/families/qwen/...`, and test-tree sources are allowed.
   - Added test `test_repository_satisfies_the_model_core_boundary` confirming the working tree has 0 violations.

3. **CTest Integration**:
   - Added `extension_model_core_includes_gate` in root `CMakeLists.txt` labeled `cpu;repository`.
   - Verified both in `cpu-test` and `gpu-test` test configurations.
