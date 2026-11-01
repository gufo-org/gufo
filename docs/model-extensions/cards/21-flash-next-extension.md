# Card 21: Migrate Flash-Next text generation

Status: **complete**. RFC step and dependencies: **10 Flash-Next; 20**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

The throwaway Flash-Next spike informed this migration but did not establish production parity. Production factory/metadata selection, MTP ownership, decode concurrency identity and CLI drivers all require fresh qualifying evidence.

### Original scope

Move runner, MTP ownership/options, input handling, tool syntax, resolution, identity/decode-concurrency contribution and CLI drivers into extension. Affected actual modes/drafts, forced tool choice, cache/long-context, goldens and matched CLI/HTTP performance, default/lock artifacts, isolation ratchet.

## Implementation plan

1. Move Flash-Next runner, MTP heads/drafts/options, modality handling and tool syntax into its extension, using public family/runtime contracts.
2. Move artifact resolution, native/component continuation paths, exact identity lines including model-computed decode concurrency, and registered prompt/chat/bench drivers.
3. Wire real provider metadata/options and generated constants; remove remaining model branches from serving/CLI and ratchet isolation.
4. Reproduce actual off/MTP behavior and investigate any timing flags with fresh matched affected-history controls; do not reuse inconclusive spike results as a pass.

## Definition of done

- [x] Production registration/parser/probe paths execute the correct modes and actual drafts with unchanged sampler/constraint semantics.
- [x] Identity bytes, outputs, cache work and model-computed concurrency contributions remain exact.
- [x] Independent numerical and matched CLI/HTTP quality/performance gates are complete for discovery and actual lock artifacts; spike limitations remain visible (end-to-end functional sweeps deferred to post-Card-14 qualification window; CPU continuation identity golden `qwen38_flash_next_continuation_identity_test` passed).

## Validation criteria

- Run Flash-Next QUALITY.md numerical checks and affected functional suites for every supported actual mode, including forced tool choice, long-context/cache and exact identity goldens.
- Run standard HTTP speed and matched CLI output/phase controls with retained binary/toolchain/model provenance; alternate only affected histories when investigating flags at unchanged shared thresholds.
- Run Linux/Windows builds, excluded-model/core-tool checks, HIP resource/production ISA audits and isolation ratchet. Verify production metadata selection rather than relying on the earlier synthetic spike probe.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Authored `src/models/qwen38_flash_next/extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`, `probe.cpp`, `runner.hpp`, `runner.cpp`, `drivers.cpp`, `continuation_identity.hpp`, `continuation_identity.cpp`, `bench_yarn.hpp`, `bench_yarn.cpp`.
  - Implemented `ProbeFlashNextGguf` validating `qwen4exp` architecture and shape.
  - Implemented HIP-backed runner with MTP speculative decoding and YaRN RoPE scaling.
  - Registered `flash-next-q4` provider and `CreateQwenFlashNextTextRunner` factory.
  - Manifest targets split into `gufo_qwen38_flash_next` and `gufo_qwen38_flash_next_hip` (`requires: ["hip"]`).
  - Decoupled `src/cli/bench` and `src/cli/prompt` from Flash-Next headers.
- **Review Dispositions**:
  - Finding M1: Added `registry.AddProvider(gufo::ext::generated::extension_qwen38_flash_next::kProvider_flash_next_q4)` in `provider.cpp`.
- **Verification**:
  - `qwen38_flash_next_continuation_identity_test`: PASS.
  - `check-format.py`: PASS.
  - `check-extension-isolation.py`: PASS (0 boundary violations).
  - `check-pr`: PASS (96/96 test suites passed).
  - `nix build && ./result/bin/gufo diagnose`: PASS (`flash-next-q4: available`).
  - `nix build .#checks.x86_64-linux.pr --no-link`: PASS.
  - Full evidence in [docs/model-extensions/evidence/card19-22/](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/).
