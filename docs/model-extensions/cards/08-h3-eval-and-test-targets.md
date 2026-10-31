# Card 08: Extract H3, evaluation and model-dependent targets

Status: **complete** (functional/performance, Windows execution and asset-blocked rows
explicitly deferred — see validation criteria and execution record; a deferral is not a
pass). RFC step and dependencies: **5; 07**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

Ownership graph and manifest contracts: [target ownership](../target-ownership.md).

## Context

After Qwen extraction, H3 and utility/test targets still couple core to models. Every such target needs an owner or a build condition so excluding a model does not break unrelated tools. DeepSeek tool_calls remains in HTTP until Card 22.

### Original scope

Extract H3, comparison support and eval targets; move eval dataset to core fixture with Nix filter exception. Give DeepSeek CPU text target without duplicating tool calls (still in HTTP until text migration). Assign or condition every model-dependent core test/tool via manifests. Record sanitizer/assertion state. Linux/Windows, affected CPU/model checks, H3 moved-HIP gate and exclusion-build coverage.

## Implementation plan

1. Extract H3 into its model library and src/testing/compare into separate comparison support; extract src/eval/{dataset,extract,regrade}.cpp into an eval target.
2. Move eval dataset to a core fixture location, update references and the specific Nix source-filter exception without bringing model tests into production sources.
3. Create a CPU-only DeepSeek text target for chat_template.cpp and sampling.cpp, linking rather than duplicating tool_calls.cpp still owned by gufo_http.
4. Inventory every model-dependent core test/tool and assign it to a model/family manifest or condition it on selection. Record old/new sanitizer and assertion state and ratchet the extracted boundaries.

## Definition of done

- [x] H3, eval and comparison targets have explicit ownership and dependency graphs.
- [ ] Every inventoried model-dependent test/tool has manifest registration or a justified condition; no single-extension exclusion leaves unresolved model references.
- [x] DeepSeek CPU text support has one implementation per source and no premature HTTP tool-syntax migration.

## Validation criteria

- Build affected production/test/tool targets on Linux and Windows; run eval/comparison and affected model CPU/unit checks selected from current CTest inventory.
- Run H3 moved-HIP resource and stripped production ISA comparisons with audited source/target flags, plus existing H3 numerical checks appropriate to the moved code.
- Configure/build exclusion cases for each affected extension and the corresponding core tests/tools. Record actual target selections, sanitizer/assertion state and Nix dataset admission evidence.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

Implementation: **worker** `3f4f4c60-7afa-4d25-9f02-14cd8b7070d6`.
Independent read-only review: **reviewer** `e7a49787-2db0-4a73-a81a-83e24f3259ac`.
Review integration: **codex-exec-writer** `313ade6a-5961-4ac8-bcb6-3f1266b7477a`.
Parent `zwllvvnv ec59bed0`; implementation checkpoint `@ uytorytn`; closure and this
commit: the orchestrator.

Full review input:
`/home/mixer/.pi/agent/sessions/--home-mixer-gufo--/subagent-artifacts/e7a49787-2db0-4a73-a81a-83e24f3259ac_reviewer_output.md`.

| Finding | Disposition and evidence |
| --- | --- |
| F1 | RESOLVED (fix by integration, validation by the orchestrator — the integration sandbox has no Nix daemon): record rewritten per configuration and the production `-O3 -ffast-math` tier scoped to `ENGINE_ENABLE_HIP`, so a HIP-off test build keeps the parent's `-O2`/`-UNDEBUG`/no-fast-math state. Fresh `build/card08-sanitizer` (Debug, ASan+UBSan, HIP off): every moved source sanitized, `ds4text` additionally `-UNDEBUG`, 11/11 tests passed with zero sanitizer reports. [Assertion record](../evidence/card08/unit-and-identity-tests.md#sanitizer-assertion-and-code-generation-state-integration-correction), [orchestrator gates](../evidence/card08/orchestrator-gates.txt). |
| F2 | RESOLVED: shared `gufo_compare` edge removed, the four real `CompareLogits` users kept, ownership prose matches. Relinked and re-tested by the orchestrator: `qwen_oracle_test`, `qwen_l1_{norm,ffn,ssm}_module_test` passed inside the 75-test Qwen selection (100%, 172 s). |
| F3 | RESOLVED in code: removed backward DeepSeek consumer and prose; every claim now needs a derived dependency, including compiled_elsewhere claims; negative unrelated-target fixture added. Static verdict in rerun ledger. |
| F4 | RESOLVED: withdrew the pre-commit coverage claim with the dirty-tree warning, then closed it for real after the closing commit — `nix build .#checks.x86_64-linux.pr` exit 0 with freshly built `check-docs`/`check-format`/`check-pr-tests`/`check-pr`, and `nix eval --raw .#packages.x86_64-linux.release.src` proves the H3 tier + core fixture ship, the old model path is gone and exactly one `tests/` file is admitted ([`hosted-post-commit.txt`](../evidence/card08/hosted-post-commit.txt)). |
| F5 | RESOLVED with an orchestrator fix: the duplicate/asymmetry checks were right but made the tool unusable (CMake generates `extensions/gufo/extension_registry.cpp` inside each build directory, so whole-database key sets can never match). `load()` now skips entries whose source lives below its own compile `directory`; doctored-database negatives prove both failures, and the pinned modes were re-run clean (`flags 27/0`, `object 27/0`, `device 8/0`). |
| F6 | RESOLVED: a device image present on one side only now fails while both-sided host wrappers stay allowed; verified by the orchestrator's pinned device re-run (`compared=8 failures=0`, 24 sections byte-identical). |
| F7 | RESOLVED: [provenance README](../evidence/card08/README.md) and [reconstructed resume notes](../evidence/card08/RESUME-NOTES.md); fresh shell identity remains PENDING. |
| F8 | RESOLVED: renamed worker brief to [input-brief-worker.md](../evidence/card08/input-brief-worker.md), explicitly labelled instructions rather than results. |
| F9 | RESOLVED: permanent ownership document linked above; docs gate recorded in rerun ledger. |

No finding was rejected. [Integration rerun ledger](../evidence/card08/integration-reruns.txt)
is authoritative for exact commands, exit codes, verdicts and sandbox denials.
The daemon probe (`nix develop -c python3 --version`) exited 1 with
`cannot connect to socket at '/nix/var/nix/daemon-socket/socket': Operation not permitted`.
No repeated daemon operation or wider-access request was made. Requested pinned
build/parity/format checks remain PENDING; static Python checks use an existing
store interpreter and are labelled separately.

Orchestrator closure (2026-10-31) re-ran the whole pinned gate set itself after the last
edit: formatting 730 files, API write+check with no surface change (6 paths), docs 273
files/1499 links, isolation core boundary + family graph (6 libraries, 7 acyclic edges),
target ownership (17 libraries, 183 edges), 19 gate fixtures, clean reconfigure and builds
of `card08-{gpu,cpu,release,sanitizer}`, identity goldens 3/3, the 70-test affected HIP
selection at 100% with 13 named asset skips, the 75-test Qwen selection at 100% (172 s),
the 36-test HIP-off exclusion selection at 100% with zero skips, moved-source flags 27/27,
host objects 27/27, gfx1151 device sections 8/8, kernel resources `0 worse, 0 better,
2692 same`, and the fresh sanitizer configuration (per-source ASan+UBSan audit plus 11/11
tests, no reports). It also fixed the parity tool defect described under F5. Commands,
exits and verdict lines: `../evidence/card08/orchestrator-gates.txt`.

Mandated review: `codex exec review --dangerously-bypass-approvals-and-sandbox
'review the diff of current jj revision, you can use jj diff to see changes'` exited 0 and
found one actionable issue — recorded as Major, not its stated P2, because the ownership
gate discarded source arguments that still contained a CMake directory variable, leaving
`${PROJECT_SOURCE_DIR}/…` compiled sources invisible to DoD 2. The orchestrator fixed
`expand_cmake()` itself, added the
`test_target_ownership_resolves_directory_variable_sources` regression fixture (20/20
fixtures, all isolation modes re-run clean, edge count unchanged at 183, so nothing was
hidden in this tree), and recorded the exchange in
`../evidence/card08/codex-review.txt`. Nothing was pushed.

Acceptance audit covers all DoD items:

1. Explicit H3/eval/comparison ownership is established by own CMake targets,
   manifests and dependency graph; F3 strengthens the symmetric claim gate.
2. Inventory/registration half is evidenced in [ownership inventory](../evidence/card08/ownership-inventory.md).
   Exclusion half remains unproven: there is no model `extension.json` yet,
   so single-extension exclusion cannot be exercised until later cards.
   HIP-off builds do not substitute for this criterion; box stays unticked.
3. Text sources occur in one production target and tests link that target;
   `tool_calls.cpp` stays in HTTP. The new parity duplicate/source-set checks
   make accidental duplication/drop fatal; fresh pinned validation is pending.

Historical worker gates are in [gates.md](../evidence/card08/gates.md),
[unit and identity tests](../evidence/card08/unit-and-identity-tests.md), and
[moved-source parity](../evidence/card08/moved-source-parity.md): CPU 25/25,
HIP 21 passed + nine asset skips, three identity goldens, production flags
27/27, host objects 27/27, device images 8/8, resources 2692 unchanged.
They predate the integration edits; `orchestrator-gates.txt` supersedes them with the
post-edit runs (CPU 36/36, HIP 70/70 with 13 named skips, Qwen 75/75, parity 27/27/8,
sanitizer 11/11).
The in-card production evidence is `build/card08-release/gufo` (actual cache:
RelWithDebInfo, BUILD_TESTING=OFF, NDEBUG, discovery mode). Earlier filter
inspection verifies the dataset install path; it does not prove a Nix package
build or hosted candidate coverage.

Remaining limitations: functional and
performance qualification deferred to after Card 14; Windows execution and its
pre-existing fileset issue; nine missing-asset H3 oracle rows; unexercisable
single-extension exclusions; pre-existing `kernel_resources_test` budget drift;
shipped lock-mode runtime qualification; production Nix derivation only
path-evaluated, never built (hosted `.pr` checks and the source-snapshot admission
list are established post-commit in `hosted-post-commit.txt`). Exact commands, including
`nix build .#packages.x86_64-linux.release --no-link`, are in
[pending commands](../evidence/card08/pending-commands.md) and the rerun ledger.
No functional runs, timings or speed benchmarks were attempted during integration.
