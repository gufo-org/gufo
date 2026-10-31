# Card 02: Exact continuation identity goldens

Status: **complete**. RFC step and dependencies: **1b; 01**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Writers will move across serving/model boundaries later. Compatibility must be checked against independent pre-migration bytes, including field ordering and settings that do or do not affect identity; a candidate writer compared to itself would conceal drift.

This card is complete. The plan below describes its original intended work; use the execution record as historical evidence rather than rerunning it automatically.

### Original scope

Expose pre-migration identity functions without changing serialization; record independent exact compatibility-byte goldens for Qwen 27B, Flash-Next and DeepSeek persistence descriptors, including draft variants and process-local identity decisions. Run goldens, with DeepSeek locally; retain baseline bytes before moving writers.

## Implementation plan

1. Identify durable and process-local writers for Qwen 27B, Flash-Next and DeepSeek in serving; expose a temporary continuation_identity seam without changing emitted values or ordering.
2. Capture parent writer bodies and literal baseline fixtures before extraction; keep reproduction source, cases, hashes and parent/toolchain provenance under evidence/card02.
3. Add model-owned CPU goldens covering artifacts/drafts/configurations, malformed hashes and identity-neutral options. Test null/process-local spellings independently rather than persist unstable pointer addresses.
4. Compile every inference_backend consumer with the extracted source; keep this seam serving-only until model migrations assign ownership.

## Definition of done

- [x] All 39 durable cases reproduce exact original bytes, and malformed-input errors remain exact.
- [x] Process-local schemas and field ordering are independently pinned while pointer instability is explicit.
- [x] No native snapshot, serialization, fingerprint validation or DeepSeek support-validation behavior changes.

## Validation criteria

- Run the three continuation_identity targets and focused CTests exactly as recorded below in CPU and gpu-test configurations; require 3/3 with zero skips, including local DeepSeek writers.
- Run sha256sum -c docs/model-extensions/evidence/card02/SHA256SUMS; every frozen capture/fixture entry must match.
- Build changed production objects and run required C++ formatting; distinguish writer execution and compilation from model inference or performance qualification.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- Implementation agent: `/root/card02_implement`; independent review agent:
  `/root/card02_review`; integration agent: `/root/card02_integrate`. All three
  stages complete. No commit/revision change by these agents; user-owned
  `RFC.md` unchanged.
- Extracted the original three durable identity writers and two process-local
  writers into the temporary serving-only `continuation_identity.*` seam.
  State ABI selection, fingerprint validation and every emitted field/value/order
  remain unchanged. Every existing executable/test-support compilation of
  `inference_backend.cpp` also compiles the new source. Model migrations will
  move each writer to its owner; this is not public extension API design.
- Captured baseline bytes from literal parent `d1304fd6` writer bodies before
  extracting production writers. Frozen capture source/cases/driver, reproduction
  commands, detailed coverage and SHA-256 inventory are retained in
  [evidence/card02/README.md](../evidence/card02/README.md). The baseline compile
  used `nix develop -c c++ -std=c++20 -I. /tmp/card02_record.cpp -o
  /tmp/card02_record`, then `/tmp/card02_record`, GCC 16.2.0, local Linux x86-64.
  No HIP define, model weights or GPU inference are needed to execute these
  actual writers. DeepSeek executed locally and has no model-skip path.
- Model-owned CPU tests compare 39 independent exact byte fixtures (Qwen 20,
  Flash-Next 11, DeepSeek 8), including numerical-storage/draft/configuration
  distinctions and existing identity-neutral options. Validation tests check
  malformed main/draft hashes and exact errors. DeepSeek's existing writer-only
  support-validation omission stays documented; constructor DSpark validation
  is unchanged. Separate process-local tests account for address/null spelling
  instability rather than persisting pointer bytes.
- `nix develop -c cmake --preset cpu-test` and the matching `gpu-test` configure
  succeeded. CPU target build command:
  `nix develop -c cmake --build --preset cpu-test --target
  qwen27b_continuation_identity_test qwen38_flash_next_continuation_identity_test
  deepseek_v4_flash_continuation_identity_test --parallel 3` passed. Focused
  CTest command: `nix develop -c ctest --test-dir build/cpu-test -R
  '^(qwen27b|qwen38_flash_next|deepseek_v4_flash)_continuation_identity_test$'
  --output-on-failure` passed 3/3, with zero skips.
- Both changed production HIP translation units compiled successfully:
  `nix develop -c cmake --build --preset gpu-test --target
  CMakeFiles/gufo.dir/src/cli/serve/inference_backend.cpp.o
  CMakeFiles/gufo.dir/src/cli/serve/continuation_identity.cpp.o --parallel 2`.
  Existing missing-field-initializer warnings remain in unmodified DeepSeek
  sampler/runner and Flash-Next decode code. This is compile evidence, not a
  full production executable link or GPU inference result; a full GPU target
  dry run showed 136 unrelated pending compilation/link steps.
- Required `nix shell --inputs-from . nixpkgs#clang-tools -c python3
  tools/ci/check-format.py --fix`, followed by the same command without `--fix`,
  passed (708 files). `sha256sum -c
  docs/model-extensions/evidence/card02/SHA256SUMS` passed all 42 entries.
- Numerical execution, scheduling, payload serialization and identity values
  are unchanged; no full functional, numerical or performance matrix was run
  for this serialization-only seam. This card does not establish the later
  runner-migration or extension lock-artifact gates.

- Additional implementation verification: the same three identity targets built
  under `gpu-test`, and `nix develop -c ctest --preset gpu-full -R
  '^(qwen27b|qwen38_flash_next|deepseek_v4_flash)_continuation_identity_test$'
  --output-on-failure` passed 3/3 with zero skips. These standalone tests remain
  HIP/model-load-free even in that configuration. Existing CPU cache uses GCC
  15.3.0; current pinned capture/GPU configuration uses GCC 16.2.0, and both
  reproduce the exact baseline fixture bytes. No performance comparison uses
  mismatched toolchains. Both changed production translation units also compiled
  in the CPU configuration with the analogous object-target command.
- Independent review found one P2 validation gap and no production behavior
  defect: Qwen's null-draft process identity and Flash-Next's disabled-MTP process
  identity had inequality checks without exact serialization checks. Integration
  added literal schema/field/order expectations for both; the Qwen null spelling
  is obtained independently by streaming `static_cast<const void*>(nullptr)`,
  and disabled MTP still pins both draft token limit and concurrency fields.
  Review also verified the durable writers against parent `d1304fd6` and frozen
  capture, and checked all four CMake consumers of `inference_backend.cpp`.
- After integration, all three focused targets rebuilt under both `cpu-test`
  and `gpu-test`; the focused CTest commands above passed 3/3 in each, with zero
  skips. The required formatting fix/check passed all 708 C++ files, and the
  retained SHA-256 inventory passed all 42 entries again. These results cover
  exact identity serialization, not model inference or performance.
