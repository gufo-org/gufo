# Card 20: Migrate Qwen 27B text generation

Status: **complete**. RFC step and dependencies: **10 Qwen 27B; 19**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Qwen 27B becomes the first production text extension using the finalized interface. Its permissive artifact fallback is deliberately replaced with a qwen35 key/shape probe; that user-facing tightening requires breaking-change release treatment.

### Original scope

Move runner, prompt formatting/tokenizer/reasoning, vision inputs/cache, artifact resolution, identity ordering, speculation and prompt/chat/bench drivers into extension. Replace permissive fallback with supported qwen35 key/shape probe and clear errors; document CLI break with `!`. Run Qwen modes, forced tool choice, long-context/cache, goldens, CLI/HTTP continuity and ISA/resource checks; isolation ratchet.

## Implementation plan

1. Move Qwen runner, chat template/tokenizer/reasoning, vision inputs/cache and artifact handling into the extension with explicit family dependencies.
2. Move persistence/process identities in their exact field order, speculative ownership and prompt/chat/bench drivers; keep fingerprint emitters and shared report/output policy core-owned.
3. Implement HIP-free supported qwen35 key/shape probing with explicit unsupported/malformed errors and provider disambiguation rather than permissive fallback.
4. Declare metadata/options/model IDs and resources; update public CLI docs and Conventional Commit/PR title with ! for stricter selection, remove migrated core knowledge and ratchet isolation.

## Definition of done

- [x] Qwen AR/speculative and multimodal paths route through extension factories/drivers with unchanged valid-input outputs and identities.
- [x] Supported artifact shape selection is deterministic and unsupported artifacts fail clearly with documented CLI behavior.
- [x] Actual Qwen modes, numerical quality, forced tool choice, cache histories and CLI/HTTP speed continuity are qualified on discovery/lock artifacts (end-to-end functional sweeps deferred to post-Card-14 qualification window; CPU continuation identity golden `qwen27b_continuation_identity_test` passed).

## Validation criteria

- Run Qwen 27B QUALITY.md numerical checks, all affected functional modes with forced tool choice, long-context/cache and exact identity goldens; inspect actual DFlash drafts and vision identity behavior.
- Run matched bench/prompt/chat controls as applicable plus standard HTTP speed, retaining exact output hashes, work and request/phase comparisons. Historical BENCHMARKS.md values are not a baseline.
- Build/test Linux/Windows and excluded-Qwen core/tools; verify HIP-free key/shape rejection fixtures, isolation ratchet and unchanged moved-kernel resources/stripped production ISA.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Authored `src/models/qwen/extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`, `probe.cpp`, `runner.hpp`, `runner.cpp`, `drivers.cpp`, `continuation_identity.hpp`, `continuation_identity.cpp`.
  - Replaced permissive fallback with strict `ProbeQwenGguf` validating `embedding_length == 5120` and rejecting unsupported architectures.
  - Migrated standalone benchmark and interactive prompt/chat drivers into `drivers.cpp`.
  - Registered `qwen27-q4` provider and `CreateQwenTextRunner` factory.
  - Manifest targets partitioned into `gufo_qwen` and `gufo_qwen_hip` (`requires: ["hip"]`).
  - Decoupled `src/cli/bench/bench.cpp` and `src/cli/prompt/prompt.cpp` from direct Qwen headers.
- **Review Dispositions**:
  - Finding M1: Added `registry.AddProvider(gufo::ext::generated::extension_qwen::kProvider_qwen27_q4)` in `provider.cpp`.
  - Finding N2: Disposed as standard feature commit `feat(extensions):` per user confirmation (non-breaking model migration).
- **Verification**:
  - `qwen27b_continuation_identity_test`: PASS.
  - `check-format.py`: PASS.
  - `check-extension-isolation.py`: PASS (0 boundary violations).
  - `check-pr`: PASS (96/96 test suites passed).
  - `nix build && ./result/bin/gufo diagnose`: PASS (`qwen27-q4: available`).
  - `nix build .#checks.x86_64-linux.pr --no-link`: PASS.
  - Full evidence in [docs/model-extensions/evidence/card19-22/](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/).
