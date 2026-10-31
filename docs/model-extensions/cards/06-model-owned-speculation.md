# Card 06: Move model-specific speculation out of core

Status: **complete** (functional/performance criteria explicitly deferred —
see validation criteria and execution record; the deferral is not a pass).
RFC step and dependencies: **4; 02, 05**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Core speculation currently knows Qwen implementations. Move those algorithms to their owner while keeping core sampling/constraint contracts neutral. The open cache RFC owns its cutover; native snapshots remain supported until the all-model replacement is proven.

### Original scope

Move Qwen-specific speculative code out of core; neutralize draft tokenizer types and preserve rollback sampler/constraint state. Coordinate cache RFC and keep native snapshots. All four profiles plus long-context/cache (DeepSeek locally), identity goldens and matched timings.

## Implementation plan

1. Inspect core speculative sources, draft tokenizer types and serving callers against the current cache RFC state; record ownership and sequencing before moving interfaces.
2. Move Qwen-specific speculation to model/family-owned code and replace tokenizer dependencies with neutral contracts. Keep core prompt lookup as a host primitive.
3. Preserve sampler, RNG and constraint copies on speculative verification/rollback, cancellation and stop handling; retain native snapshot and persistence paths including DeepSeek.
4. Ratchet the moved boundary, update dependencies and API policy where public contracts change; run exact identities and affected shared text histories before committing.

## Definition of done

- [x] Core speculative code has no Qwen implementation dependency and the neutral tokenizer contract is sufficient for actual modes. Verified by grep (zero model references in `src/core/speculative/`), the enforced `extension_core_model_includes_gate`, compile-graph audit (`hip-build-audit.md`) and independent review.
- [x] Rollback preserves sampler/constraint/RNG semantics and all existing native continuation paths remain reachable. Adapter moved verbatim (review-verified line-by-line); `qwen_dflash_gpu_test` snapshot/persistent-restore/rollback replay PASS on gfx1151; identity goldens exact bytes; weight-free ds4 CPU tests 5/5; DeepSeek persistence path statically audited (analysis only, runtime deferred).
- [ ] All four text profiles, affected long-context/cache histories and matched phase evidence are retained; confirmed regressions are fixed. (Deferred by user decision 2026-10-31 to the post-Card-14 qualification window; DeepSeek rows skipped by user decision. Closure of this card does not claim this item — it remains visibly pending with its exact commands.)

## Validation criteria

User decision (2026-10-31): functional-harness work is deferred from this card
to the single post-Card-14 qualification window, and DeepSeek runtime is dropped
(host capacity). In-card scope is the identity/unit/static gates below;
deferred criteria remain PENDING evidence with exact commands and are never
marked passed. Evidence stays lean: verdict summaries only, no report trees or
raw logs under `docs/`; scratch in `/tmp/cardNN-*` always deleted afterwards.

- Run the existing identity target/CTest selection from Card 02 and affected speculative unit tests discovered with ctest --preset gpu-full -N; require exact bytes and no model skips.
- DEFERRED (post-Card-14 window; PENDING): use tests/functional/run.py and compare.py as documented in tests/functional/README.md with the Qwen27B Q4/Q8 {off,dflash2} and Flash-Next {off,mtp} profiles, explicit affected suites including long-context and cache, actual drafts and fresh `/tmp` output directories pruned afterwards. DeepSeek rows (off/dspark) are SKIPPED BY USER DECISION: retained harness-bug failures stand, re-verify command recorded, no pass claimed.
- DEFERRED (post-Card-14 window; PENDING): existing numerical quality commands and standard speed benchmarks from the affected model QUALITY/BENCHMARKS documents; qualify default and actual lock-mode HIP artifacts separately. Apply shared per-request/phase controls, never pooled averages.
- IN-CARD DeepSeek substitute (user-directed): static correctness analysis of every DSpark-reachable path in the diff plus weight-free DeepSeek CPU tests; explicitly not runtime serving evidence.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Finalized by the orchestrator 2026-10-31; this revision is the closing commit.

Phases and identities (pi subagent run IDs):
- Implementation — pi `worker`: `c95b5e36` (initial; harness 30-min timeout),
  `cf7bb57b`/`d1adffbd`/`8816fad1` (revives; terminated by harness
  timeouts/max-length stops around detached validation queues), aborted launch
  `167d5f5c`, continuation `6bcb8ddb` (completed in-card scope after the user's
  scope changes).
- Independent review — pi `reviewer` `d768952e` (read-only static pass; no
  shell tool, so its listed gates were re-executed by the orchestrator).
- Integration — pi `codex-exec-writer` `7547b33b` (third distinct runtime;
  sandboxed, so pinned-Nix gates it could not reach were re-run by the
  orchestrator).
- Orchestrator — re-ran every pinned gate itself (verdicts below and in
  `integration-gates.txt`), ran the user-mandated final `codex exec review`
  pass, fixed its findings, and made this commit.

| Finding | Disposition |
| --- | --- |
| F1 (P1) | Converted ignored-log references to small tracked-format `.txt` excerpts, preserving every STATUS line and compare verdict. Full identity CTest output now retains all 3 pass lines. Historical build has no retained ninja exit code: explicitly PENDING rather than fabricated. |
| F2 (P2) | Narrowed adapter claim to retained evidence: 22/22 fixed-key methods; 16 identical, 4 address-only differences, 2 destructors + 1 ctor presence-verified; class verbatim at source level. Removed all-27-body claim. |
| F3 (P2) | Attributed 69 changed engine bodies to alias-rename call/jump encodings and moved adapter boundaries, backed by independently confirmed empty normalized source diff; no unretained per-hunk proof claimed. |
| F4 (P2) | Corrected historical Flash-Next compare failure to HARNESS command mismatch, zero measurements, no timing signal. Qualification remains PENDING. |
| F5 (P2) | Removed leading paragraph spaces and repaired ragged ledger and find-command wraps. |

Integration ran inside a sandbox that denied the Nix daemon socket
(`cannot connect to socket at '/nix/var/nix/daemon-socket/socket'`), so its
pinned-environment gate reruns stayed PENDING and it recorded that honestly.
Orchestrator re-run in the pinned dev shell closed them
(`integration-gates.txt` ORCHESTRATOR RE-RUN section): identity trio exit 0
100%/3; affected-unit selection exit 0 100%/8 (`speculative_verification_test`,
`speculative_corpus_test`, `ds4.sampling/template/cli/load`,
`extension_core_model_includes_gate`, `check_extension_isolation_test`);
`check-format.py` exit 0, 727 files; `check-extension-api.py` and `--write`
exit 0, SURFACE unchanged (6 paths); isolation gate exit 0; whole-repo
`check-docs.py` reports exactly the 164 pre-existing `hrx-system/` errors and
zero task-scoped ones; RFC SHA unchanged
`2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448`.
No tests added by integration; evidence/documentation corrections only.

Evidence: `../evidence/card06/README.md`, `integration-gates.txt`,
`unit-identity-goldens.txt`, `configure-gpu-test.txt`, `build-gpu-test.txt`,
`pipeline-status.txt`, `compare-*.txt` and matching JSON, `hip-build-audit.md`,
`isa-audit-output.txt`, `deepseek-static-analysis.md`. Historical evidence is
retained with its limits; new validation does not upgrade deferred runtime work.

User decisions today (2026-10-31): no functional runs during Cards 06–14;
one orchestrator-scheduled post-Card-14 qualification window. DeepSeek runtime
skipped for host capacity, static path analysis plus weight-free tests substitute
in-card only. The user deleted the functional report tree as context pollution;
retain verdict summaries, never recreate report trees under docs.

PENDING commands (not run during this integration):

- ~~Required pinned Nix reruns~~ RESOLVED by the orchestrator re-run above.
  Historical build exit recovery: the original configure/build ran outside the
  harness; captured excerpts stand in `configure-gpu-test.txt` /
  `build-gpu-test.txt`, and the current build directory re-verifies via
  `CMakeCache.txt` audit lines retained there.
- Functional matrix: exact parameterized command recipe and row mapping in
  `../evidence/card06/pending-commands.md`; all Qwen27 Q4/Q8 off/dflash2 and
  Flash-Next off/mtp histories, long-context/cache, CLI output/timings and
  per-request/phase 5% + 10 ms comparisons remain unqualified.
- `prompt_lookup.py`, `compare.py` and model-bench continuity: commands in
  `../evidence/card06/pending-commands.md`; preserve numerical quality gates.
- Qwen27 MTP asset: `GUFO_QWEN27B_MODEL="$Q4" GUFO_QWEN27B_MTP_MODEL="$MTP27" nix develop -c ctest --test-dir build/card06-gpu -R '^qwen_mtp_gpu_test$' --output-on-failure`.
  MTP27 must be a real compatible asset; currently missing.
- DeepSeek GPU trio + serving: `nix develop -c ctest --test-dir build/card06-gpu -R '^(ds4\.target|ds4\.dspark|ds4\.serving)$' --output-on-failure`.
  Runtime exact run.py command in `../evidence/card06/deepseek-static-analysis.md`;
  do not execute without post-Card-14 orchestration and host scheduling.
- Lock-mode HIP gates: `nix develop -c cmake --preset release -DGUFO_EXTENSIONS_FROM_LOCK=ON`;
  `nix develop -c cmake --build --preset release --parallel 4`;
  repeat pending numerical/functional/lookup/bench commands with
  `BIN=build/release/gufo`, retain actual lock cache/compile/binary provenance.
- Hosted closure: `nix build .#checks.x86_64-linux.pr` — RUN by the
  orchestrator after all final edits: exit 0; `check-format`,
  `check-pr-tests` and `check-pr` derivations freshly built (not cached).
  Raw capture: `hosted-pr-check.txt`. Caveat retained from Card 05: the Nix
  source snapshot is taken from tracked paths while the git tree is dirty.

Final review stamp (user-mandated, 2026-10-31): `codex exec review
--dangerously-bypass-approvals-and-sandbox 'review the diff of current jj
revision, you can use jj diff to see changes'` — exit 0, verdict: "The
refactor preserves the speculative adapter behavior and correctly updates
callers and policy propagation. Extension API validation and all six isolation
tests passed"; no critical or major findings, nothing to fix. Its C++ ctest
re-run was unavailable in that environment (no `ctest` on PATH) — covered by
the orchestrator's pinned-gate re-run above. The user directed closing this
card on the codex-review stamp.

Card closure does not claim deferred functional, numerical or shipped-runtime
requirements passed. No forbidden DeepSeek runtime or functional harness run
was performed by integration.
