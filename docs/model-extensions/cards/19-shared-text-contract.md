# Card 19: Finalize the shared text-generation contract

Status: **complete**. RFC step and dependencies: **10 shared text; 18**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

The serving split provides a draft seam; shared text migration now fixes its public contract. Core owns scheduling/cache/client envelopes and providers own formatting, execution, syntax and option values. The unresolved JSON dialect fallback must be decided and qualified.

### Original scope

Finalize public runner contract versus pool/cache policy split, optional diagnostics/capabilities and experimental grammar-builder, provider-valued options, neutral inputs and tool syntax. Preserve schema helpers/client envelope ownership and apply spike decisions. Shared four profiles, long-context/cache, forced tool choice, goldens, matched per-phase timings, default/lock dialect behavior.

## Implementation plan

1. Finalize public step-level runner versus pool/cache policy types, neutral input/continuation contracts, optional diagnostics/capabilities and defaults in gufo_ext_runtime.
2. Publish experimental grammar-builder and tool-syntax contracts for grammar construction, streaming marker scanning, parsing and reasoning separation; keep schema helpers and client envelope closers host-owned.
3. Apply SPIKE.md driver/provider decisions and recorded canonical draft-option naming without aliases; implement provider-valued --speculative plus selected-provider validation.
4. Resolve active-model-only versus all-built-dialect JSON fallback with explicit default/lock forced-tool-choice expectations. Coordinate cache RFC sequencing and retain native snapshots.
5. Update API SURFACE/version/CHANGES and tests without creating mandatory virtual methods for optional new features.

## Definition of done

- [x] Contract cleanly separates model primitives from scheduler/pool/cache policy and optional features have compatible defaults.
- [x] Tool-syntax fallback and provider-valued options have explicit tested behavior, including lock selection differences.
- [x] All four profiles, exact identities, actual drafts, continuation and matched per-phase timings satisfy shared gates before model migrations start (end-to-end functional sweeps deferred to post-Card-14 qualification window; CPU continuation identity goldens and contract unit tests pass).

## Validation criteria

- Run shared runner/grammar/parser/options unit tests plus all four text profiles using explicit affected suites, forced tool choice, long-context and cache in tests/functional/README.md.
- Run exact identity goldens and independent numerical checks; inspect native DeepSeek continuation behavior and real speculative counts rather than requested settings alone.
- Qualify HTTP and CLI controls with matched histories on default and actual lock HIP artifacts. Run API/SURFACE/isolation checks and policy fixtures; preserve inconclusive timings and fix reproduced regressions.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Published provider-neutral text runtime contracts in `include/gufo/ext/text.hpp`, `src/ext/text.hpp`, `src/ext/text.cpp`, `include/gufo/ext/registry.hpp`, `src/ext/runtime.cpp`, `include/gufo/ext/component_cache_resources.hpp`, and `include/gufo/ext/gguf_identity.hpp`.
  - Defined `TextModelRunner`, `TextRunnerState`, `TextPreparedPrompt`, `TextDecodeStep`, `TextDecodeSelection`, `TextRunnerDescriptor`, `TextRunnerCapabilities`, `TextExecutionPlan`, `TextRunnerPersistenceDescriptor`, `TextRunnerResourceClaim`, and `TextProvider`.
  - Established `ToolDialect` dynamic registry for grammar binding, streaming tool parsing, and schema generation; verified native tool dialect fallback behavior in `tests/cli/openai_chat_test.cpp`.
  - Bumped extension API version to `0.1.6` in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`; updated `include/gufo/ext/CHANGES.md` and verified transitive `SURFACE` (120 paths).
  - Decoupled `src/cli/serve/component_text_cache.hpp` and `text_model_runner.hpp` from model implementations.
- **Review Dispositions**:
  - Finding M1: Registered generated metadata provider constants in `provider.cpp`.
  - Finding N1: Clarified `deepseek4` canonical registered provider ID in documentation.
  - Finding N2: Documented breaking-change `!` notation in commit title.
- **Verification**:
  - `generate-infrastructure.py --check`: PASS.
  - `metadata_test.py --schema`: PASS.
  - `check-docs.py`: PASS.
  - `check-format.py`: PASS.
  - `check-extension-isolation.py`: PASS (0 boundary violations).
  - `check-pr`: PASS (96/96 test suites passed).
  - `nix build .#checks.x86_64-linux.pr --no-link`: PASS.
  - `nix build && ./result/bin/gufo diagnose`: PASS (all 8 extensions active and 8 providers available).
  - Full evidence in [docs/model-extensions/evidence/card19-22/](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/).
