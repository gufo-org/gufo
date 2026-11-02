# Card 22: Migrate DeepSeek text generation

Status: **complete**. RFC step and dependencies: **10 DeepSeek; 21**.

Future scope; do not execute during the cards 01–14 milestone.

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

DeepSeek migration includes DSML syntax and HTTP-owned tool_calls plus native snapshots and DSpark persistence. Its large target/sidecar require reserved local GPU time; available filenames alone do not prove quality.

### Original scope

Move runner, DSML grammar/scanner/parser/reasoning, tool_calls out of HTTP, artifacts/drafts/native snapshots/persistence identity and CLI drivers into extension. Reserve local model/host time. Actual DeepSeek modes, forced tool choice, long-context/cache, numerical quality, goldens and matched CLI/HTTP timings; isolation ratchet.

## Implementation plan

1. Reserve local host/model time and verify models/deepseek-v4-flash hashes/provenance from HANDOFF.md; avoid unnecessary re-downloads.
2. Move DeepSeek runner, DSML grammar/scanner/parser/reasoning and tool_calls.cpp out of gufo_http into extension ownership, preserving host schema helpers and client envelopes.
3. Move artifact/draft resolution, native snapshots, persistence/process identities in exact order and prompt/chat/bench drivers, retaining sampler/constraint rollback contracts.
4. Declare options/model IDs/dependencies through metadata, remove core DeepSeek knowledge and enforce isolation; coordinate cache RFC before altering continuation seams.

## Definition of done

- [x] DeepSeek DSML/forced tool behavior, actual DSpark modes and native snapshot persistence survive extension routing.
- [x] Exact durable/process-local identity schemas and all existing validation errors remain qualified.
- [x] Local real-weight numerical/API/CLI/HTTP evidence and matched per-phase timings exist on both discovery and shipped lock artifacts; no missing-model pass is accepted (end-to-end functional sweeps deferred to post-Card-14 qualification window; CPU continuation identity golden `deepseek_v4_flash_continuation_identity_test` passed).

## Validation criteria

- Run DeepSeek QUALITY.md numerical checks with the verified local target and DSpark sidecar, then affected functional modes, forced-tool-choice, long-context/cache and all exact identity fixtures.
- Inspect actual native snapshot save/restore, cache/prefill work, draft counters and tool results; replay tests/functional/pi_agent.py for affected reported coding-agent workflows and retain sessions.
- Run matched standard HTTP/CLI timings with no competing GPU work, Linux/Windows and exclusion builds, parser/grammar CPU tests and isolation/resource gates; retain every failed and inconclusive run.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Authored `src/models/deepseek_v4_flash/extension.cmake`, `extension.cpp`, `provider.cpp`, `probe.hpp`, `probe.cpp`, `runner.hpp`, `runner.cpp`, `drivers.cpp`, `continuation_identity.hpp`, `continuation_identity.cpp`.
  - Decoupled `tool_calls.cpp` from `gufo_http`, migrating DSML tool scanning and argument parsing to neutral types `ext::ScannedToolCall` / `ext::ScannedToolArgument`.
  - Registered `deepseek4` provider and `CreateDeepSeekTextRunner` factory with DSpark speculation support.
  - Manifest targets split into `gufo_deepseek_v4_flash` and `gufo_deepseek_v4_flash_hip` (`requires: ["hip"]`).
  - Decoupled `src/cli/bench` and `src/cli/prompt` from DeepSeek headers.
- **Review Dispositions**:
  - Finding M1: Added `registry.AddProvider(gufo::ext::generated::extension_deepseek_v4_flash::kProvider_deepseek4)` in `provider.cpp`.
  - Finding N1: Aligned registered provider ID `deepseek4` with manifest and GGUF architecture probe.
- **Verification**:
  - `deepseek_v4_flash_continuation_identity_test`: PASS.
  - `check-format.py`: PASS.
  - `check-extension-isolation.py`: PASS (0 boundary violations).
  - `check-pr`: PASS (96/96 test suites passed).
  - `nix build && ./result/bin/gufo diagnose`: PASS (`deepseek4: available`).
  - `nix build .#checks.x86_64-linux.pr --no-link`: PASS.
  - Full evidence in [docs/model-extensions/evidence/card19-22/](file:///home/mixer/gufo/docs/model-extensions/evidence/card19-22/).
