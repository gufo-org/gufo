# Card 09: Separate serving policy from extension execution

Status: **complete** for its in-card gates; the deferred functional and performance
half stays pending evidence for the post-Card-14 qualification window. RFC step and
dependencies: **6; 08**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Models must use neutral execution contracts while HTTP semantics, scheduler, pool and retention policy stay core-owned. This split coordinates with the cache RFC and preserves both component-cache and native-snapshot continuation behavior.

### Original scope

Extract gufo_serving and extend gufo_ext_runtime with neutral runner/chat/tool/tokenizer/reasoning/continuation types, default HIP probing, host budgets, cache backing/resource contracts and injected logging/budget callbacks. Keep HTTP semantics/scheduler/pool/cache policy in core. All four profiles, long-context/cache, native DeepSeek snapshots, exact goldens and matched timings.

## Implementation plan

1. Extract gufo_serving from currently executable-owned scheduler, runner pool, component text cache and InferenceBackend sources; retain gufo_http as a separate consumer.
2. Extend gufo_ext_runtime with neutral runner/chat/tool/tokenizer/reasoning and continuation state/snapshot bases plus default HIP probing; remove concrete model types from those contracts.
3. Move host snapshot budget queries and neutral backing/resource contracts to runtime/cache owners; place CreateHipComponentCacheResources next to gufo_cache_hip with injected budget/log callbacks.
4. Keep queueing, API envelopes, cache retention/admission and policy in serving/core. Migrate consumers and add focused dependency/behavior tests before publishing the larger surface in Card 10.

## Definition of done

- [x] Models/runtime/cache compile without serving or HTTP implementation headers.
  Production code only: no `src/cli/**` include remains under `src/models`,
  `src/families`, `src/cache`, `src/ext` or `include/gufo/ext`, enforced by
  `extension_serving_boundary_gate`. One exempt in-tree test executable
  (`src/models/qwen3_asr/tests/audio_api_test.cpp`, whose CMake target still
  compiles two serving sources) stays open until Card 12 removes the exemption.
- [ ] Shared runner step/batch, cancellation, publication waits, budgets and snapshot/resource semantics are unchanged.
- [ ] Exact identities and all four profile continuation behaviors are proven, including native DeepSeek snapshots; no cache cutover is inferred.

## Validation criteria

- Build all affected runtime/cache/serving/HTTP targets and run focused unit tests chosen from ctest --preset gpu-full -N; check link dependencies for forbidden serving cycles.
- Run all four text profiles with explicit long-context/cache and affected scheduler/tool suites using tests/functional/README.md. Inspect actual loaded modes/drafts, restore/prefill work and local DeepSeek persistence.
- Run exact identity goldens, affected numerical quality and matched HTTP/CLI timing controls; retain default and actual lock-mode production artifacts separately and fix confirmed regressions.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Three agents, then orchestrator closure. Parent: `uytorytn c03aaa23` (Card 08).
Implementation: **worker** `8e9ab18d-6117-4e80-b033-e5521056fbdc`. Independent
read-only review: **reviewer** `c9a6d209-325e-4f13-9dba-884f29d56fc1` (verdict
BLOCK). Review integration: **codex-exec-writer**
`5cb5551d-5bba-458e-b260-a86101cedec8`, whose sandbox had no Nix daemon, so every
CMake/CTest gate it could not run stayed `PENDING(orchestrator)`. Closure, its own
pinned re-run, the mandated `codex exec review` and its two fixes: the orchestrator.
The orchestrator also found and fixed a defect the integration pass had introduced
(O0 below). Full gate transcript:
[orchestrator-gates.txt](../evidence/card09/orchestrator-gates.txt).

### Implementation and scope audit

Plan item 1 extracted seven translation units into `gufo_serving`: scheduler,
runner, cache facades, disk store, identity and inference backend. The executable
and serving tests link it. `src/cli/serve/continuation_identity.cpp` retains a
second carrier in each of the three model test directories (Qwen 27B, Flash-Next,
DeepSeek), with its own `GUFO_IDENTITY_FIXTURES` definition and HIP-free compile.
There is no conditional compilation in that source today; full de-duplication
is not claimed.

Plan item 2 is **partly delivered**: continuation token/state/snapshot/prefix
identity, `ReasoningTags`, host budgets and the default device probe moved to
`gufo_ext_runtime`. `ChatRequest`, `sampling::JsonConstraint::ToolFormat` and
`TextModelRunner` stay serve-owned; RFC steps 7/10 finish the neutral contract
work. Plan item 3 placed backing/resource contracts with cache and the HIP
factory beside `gufo_cache_hip`, using injected budget/log callbacks. Plan item 4
keeps scheduler, HTTP envelopes and admission/retention policy core-owned;
consumers migrated and focused tests were added. Functional equivalence remains
partially proven, not waived.

### Independent findings and dispositions

The [full independent review](../evidence/card09/review-findings.md) returned
**BLOCK**. This record does not supersede that verdict with an unearned pass.

| Finding | Disposition and retained proof |
| --- | --- |
| F1 Critical | Code restored `include(cmake/Checks.cmake)` under `BUILD_TESTING`, keeping TargetGraph and both tests. The implementer's additions inside Checks.cmake were dead until this include returned. Target resolution and hosted execution **PENDING(orchestrator)**: daemon denied the dry run. |
| F2 Major | Corrected selection to `qwen38_flash_next\.continuation_layout`, confirmed in generated CTestTestfile.cmake. Full corrected GPU/CPU runs **PENDING(orchestrator)**; final log slots explicitly say pending; historical partial tails retained separately. |
| F3 Major | Checker owns forbidden policy, report records HIP, only four named HIP-tier targets may be absent with HIP off, unknown interface items fail configuration. Eleven policy unit tests pass; real CMake rename/revert negative proof and regenerated report gates **PENDING(orchestrator)**. Claims narrowed to fixed/manifest names. |
| F4 Minor | Resolved in flag evidence and TESTING.md: qualification must use `BUILD_TESTING=OFF`; test-preset binaries now assert in serving sources where Card 08 did not. |
| F5 Minor | Resolved: both flag logs regenerated with one labelled exact invocation per matched pair. Seven GPU rows add only `-UNDEBUG`; seven production rows add nothing; no mismatched test/release row remains. |
| F6 Minor | Resolved: removed nonexistent MiniMax edge and `--format json`, corrected logging/fixture claims and generic build path. |
| F7 Minor | Resolved: retained identity carrier documented above and in flag evidence. |
| F8 Minor | Code passes current CMake binary directory; missing supplied report is an assertion failure (negative test exit 1). Hosted integration execution **PENDING(orchestrator)**. |
| F9 Minor | Resolved disclosure: HIP-free fixture immediately returns usable and does not cover HIP pending-at-bound, allocation, timeout or teardown. Covering rows `gufo_gpu_probe`, `ds4.serving`, `inference_backend_gpu_test` remain pending. Device-probe diagnostics now originate in device_probe.cpp. |
| F10 Nit | Resolved: six migrated headers use neighboring `#pragma once` convention. Formatting validation **PENDING(orchestrator)**. |
| F11 Nit | Resolved disclosure: plan item 2 partial; three serve-owned types and RFC completion steps named above. |

### Review findings and dispositions

The [full independent review](../evidence/card09/review-findings.md) returned
**BLOCK** (F1 Critical, F2+F3 Major, F4-F9 Minor, F10-F11 Nit). Every finding was
resolved; the orchestrator executed the proofs the integration sandbox could not.

| Finding | Disposition, decided on orchestrator evidence |
| --- | --- |
| F1 Critical | RESOLVED and proven. `include(cmake/Checks.cmake)` is restored at the end of `CMakeLists.txt` with the Card 09 TargetGraph block kept; `nix develop -c cmake --build build/card09-cpu --target check-pr -- -n` exits 0 and plans the target instead of "unknown target", so `gufo_pr_targets`, `check-pr`, the flake `.pr` check and the Windows job have a target again. |
| F2 Major | RESOLVED and proven. Selection corrected to the registered dotted name; the orchestrator rebuilt both trees and ran the same 43-test selection in `build/card09-gpu` and `build/card09-cpu`: **43/43 passed in both**, complete verdict lists retained ([gpu](../evidence/card09/logs/ctest-gpu-final.txt), [cpu](../evidence/card09/logs/ctest-cpu-final.txt)), so the five serving-semantic suites (`continuation_cache_test`, `component_text_cache_test`, `text_model_runner_test`, `continuation_disk_store_test`, `text_generation_scheduler_test`), all three identity goldens, both layout suites and both new gates are individually visible. |
| F3 Major | RESOLVED and negative-proved. The checker owns `FORBIDDEN`, requires a boolean `hip` in the report, admits `absent` only for four named HIP-tier targets and only with HIP off, and fails on an interface item it cannot classify. Orchestrator proofs: fresh `--preset gpu-test` configure exit 0; gate PASS on both regenerated reports (24 protected targets); **negative proof A** — a protected target silently dropped from the generator fails with "protected target gufo_cache_hip is missing from the link graph" (exit 1); **negative proof B** — an injected `gufo_ext_runtime -> gufo_http` edge fails with "gufo_ext_runtime reaches gufo_http" (exit 1). Both injections reverted and the real builds re-gated clean; overstated prose in `negative-proofs.txt` and `target-ownership.md` corrected to the mechanism actually proven. |
| F4 Minor | RESOLVED: `docs/TESTING.md` and the flag record now state that qualification must run a `BUILD_TESTING=OFF` artifact and that assertion state intentionally differs between test-preset and production serving binaries. |
| F5 Minor | RESOLVED: both flag audits regenerated by the retained `flag-audit.py` with one labelled invocation per matched pair (gpu-test pair adds only `-UNDEBUG`; release pair adds nothing); the mismatched test-baseline row is gone. |
| F6 Minor | RESOLVED: false `gufo_serving -> gufo_minimax_h3` edge, nonexistent `--format json`, overstated logging/fixture prose and the hardcoded `build/card09-gpu` path all corrected. |
| F7 Minor | RESOLVED: `continuation_identity.cpp` is recorded as a retained second carrier in the three model test directories rather than claimed as de-duplicated. |
| F8 Minor | RESOLVED and proven: CMake passes `${CMAKE_BINARY_DIR}` to `check_target_graph_test.py`, a supplied directory with no report fails (exit 1), and inside both real build directories the test passes as part of the 43-test selection instead of skipping. |
| F9 Minor | Disclosed, still uncovered: the neutral device probe's HIP branch is not unit-exercised (the isolated fixture has no `hip::host`, so it compiles the always-usable path). Covering rows `gufo_gpu_probe`, `ds4.serving`, `inference_backend_gpu_test` stay PENDING. Positive control the reviewer verified: the real GPU build compiles `device_probe.cpp` with `ENGINE_ENABLE_HIP=1` and links `hip::host`, so shipped probing is not silently compiled out. |
| F10 Nit | RESOLVED: migrated headers use the neighbouring `#pragma once`; formatting passes. |
| F11 Nit | Resolved as disclosure: plan item 2 is partly delivered; `ChatRequest`, `JsonConstraint::ToolFormat` and `TextModelRunner` stay serve-owned for RFC steps 7/10. |
| O0 (orchestrator) | The integration hardening of the walker aborted configuration of **both** test builds on CMake's own `::@(<id>)` / `::@` item-group markers. Fixed by classifying the markers as delimiters only, with probe evidence that the target names inside them (`gufo_core`, `gufo_logging`, `gufo_kernels`, `gufo_qwen_text`, `gufo_qwen_control_tokens`) still appear in the generated closures, so no edge is hidden. |
| CR1 (codex, P1) | Fixed by the orchestrator: the classifier treated the Windows platform libraries as unclassified items, which would abort any Windows `BUILD_TESTING=ON` configure. Now a classified system-library list including `GUFO_PLATFORM_LIBS`; proven on Linux by simulating the Windows link edge (configure exit 0, gate PASS, no phantom edge in `gufo_platform`'s closure). Windows execution itself remains PENDING. |
| CR2 (codex, P2) | Fixed by the orchestrator: the neutral-contracts budget assertion compared a ratio of two independent live samples and fails on a host or cgroup at or below the 4 GiB headroom, where a zero ceiling is correct. Replaced with the formula relation plus a drift allowance. The low-memory branch could not be reproduced on this host (a 2 GiB `systemd-run` scope exposed no readable `memory.max`), so it stays an unexercised branch, not a pass. |

### Validation and acceptance audit

The orchestrator's own post-edit transcript is
[orchestrator-gates.txt](../evidence/card09/orchestrator-gates.txt); the worker's
and integrator's records ([build-and-tests.txt](../evidence/card09/build-and-tests.txt),
[integration-reruns.txt](../evidence/card09/integration-reruns.txt)) are historical
and are not treated as current proof. After the last edit the orchestrator re-ran:
both reconfigures and full builds (exit 0), the 43-test selection in GPU and CPU
(43/43 each), both link-graph reports (PASS), the four isolation modes (core
boundary, family graph 6 libraries/7 edges, target ownership 17 libraries/179
edges, serving boundary), both python fixtures, `check-docs` (278 files, 1514
links), `check-extension-api` (6 paths, `--write` leaves `SURFACE` unchanged),
formatting (740 C++ files, after formatting the fixed test once), the discovery and
lock production builds and their `gufo diagnose` (exit 0 each; with Timestamp and
availableBytes filtered the discovery report is byte-identical to the parent
artifact), and the hosted `nix build .#checks.x86_64-linux.pr`.

DoD 2 stays unticked on purpose: the shared step/batch, cancellation,
publication-wait, budget and snapshot/resource suites pass in both
configurations and the reviewer verified the moves byte-for-byte, but the probe's
HIP branch is unexercised, and continuation behaviour under real loads belongs to
the deferred qualification window. DoD 3 stays unticked: the three identity
goldens pass byte-exactly in both builds, which proves the identity half only;
the four profiles with `long-context`/`cache`, native DeepSeek snapshot
persistence, matched per-request timings and the shipped lock-mode runtime
qualification are all PENDING.

Shared gates: production flag parity re-audited per source; test assertion state
intentionally differs and is disclosed; no new sanitizer run is claimed; the lock
artifact has no separate compile-command audit beyond the release pair. Nothing
under `include/gufo/ext` changed, so no extension-API bump or `CHANGES.md` entry
is due.

Remaining limitations, with their commands: all functional and performance
qualification (four text profiles with `long-context`/`cache`, DeepSeek local
time and actual weights, matched per-request/phase timings against main, the
standard speed benchmark, `tests/functional/pi_agent.py` replay) is deferred to
the single post-Card-14 qualification window by the 2026-10-31 decision and is
recorded verbatim in [pending-commands.txt](../evidence/card09/pending-commands.txt),
never as a pass. Also still open: Windows execution (this card changed CMake that
Windows consumes; CR1 removed the configure break it introduced, but no Windows
run was performed here); the neutral device probe's HIP branch and the asset-heavy
serving rows (`gufo_gpu_probe`, `ds4.serving`, `inference_backend_gpu_test`);
the low-memory branch of the fixed budget assertion; per-model single-extension
exclusion (no `extension.json` yet); a sanitizer run (no moved HIP source, so
none was repeated); the shipped lock-mode runtime qualification beyond a
`diagnose` smoke; and the exempt ASR test-tree edge until Card 12.
Qualification must use a `BUILD_TESTING=OFF` artifact because assertions now
remain enabled in test-preset serving binaries. `docs/model-extensions/RFC.md`
stayed byte-identical at
`2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448` and is not part
of this commit.
