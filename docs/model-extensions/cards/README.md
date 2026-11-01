# Model extension cards

The [index](../CARDS.md) lists all 29 sequential cards, dependencies, current
statuses and shared acceptance gates. Each numbered file owns its context,
implementation plan, definition of done, validation criteria and execution record.
The [original RFC](../RFC.md) remains the specification; these plans do not amend
its requirements or waive existing quality gates.

Current checkpoint: cards 01–08 are complete. Card 07 extracted the acyclic Qwen family
libraries (`src/families/qwen/{control_tokens,hip_common,config,text,kernels,vision}` with
`library.json` manifests and the enforced `extension_family_graph_gate`), moved Qwen 27B into
`gufo_qwen27b`, took Qwen config extraction out of `GgufReader` into CPU-only
`ExtractQwenConfig`, and relinked the CLI, kernel tools and tests; its unresolved gates are
the deferred functional/performance half, Windows execution, the shipped lock-mode artifact
and the pre-existing `kernel_resources_test` baseline drift, none of which it claims.
Card 08 moved MiniMax H3, the comparison helper, the eval tier and the DeepSeek CPU text
sources out of `gufo_core`/`gufo_deepseek_v4_flash` into owned targets, made the eval
dataset a core fixture with a single Nix source admission, and gated every
model-dependent core test/tool with `extension_target_ownership_gate`; its unresolved
gates are the deferred functional/performance half, Windows execution, the asset-blocked
H3/DeepSeek oracle rows, per-model extension exclusion until `extension.json` exists, the
shipped lock-mode artifact and the still-unbuilt production Nix derivation. See
[target ownership](../target-ownership.md) for the manifest/claim contract later cards
must satisfy. Card 09 is complete for its in-card gates: `gufo_serving` now owns the
scheduler, runner pool, component text cache and `InferenceBackend` sources instead
of the executable and three model test trees re-compiling them; `gufo_ext_runtime`
owns the neutral continuation state/snapshot bases, host snapshot budgets, default
HIP probing and reasoning/tokenizer token types; the backing/resource contracts and
the HIP backing provider moved to `src/cache` and `gufo_cache_hip` behind injected
budget/log callbacks; and `extension_serving_boundary_gate` plus
`extension_link_graph_gate` (each negative-proved) guard the split. Its unresolved
gates are the deferred functional/performance half, Windows execution, the probe's
HIP branch and asset-heavy serving rows, lock-mode runtime qualification, and the
exempt ASR in-tree test edge until Card 12 — none of which it claims. See its
[execution record](09-serving-runtime-boundary.md#execution-record).
Card 10 is complete for its in-card gates: published 44 forwarding headers
(7 platform, 14 experimental, 23 stable); generated and verified the reproducible
transitive `SURFACE` closure (97 paths, 4,235 LoC, 31,365 tokens) with zero serving
leaks and zero model include leaks; bumped the extension API version to `0.1.2` in
`include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`; updated `include/gufo/ext/CHANGES.md`;
extended diff/policy tests in `tests/tools/check_extension_api_test.py` with negative
rejection proof; compiled representative HIP-less and HIP public consumers; and
passed all bounded CPU/GPU CTests including continuation identity goldens. Its
unresolved gates — none of which it claims — are the deferred functional/performance
half, Windows execution, model include migration (deferred to Card 11), and the
exempt ASR in-tree test edge until Card 12. See its
[execution record](10-public-api-surface.md#execution-record).
Card 11 is complete for its in-card gates: migrated all eight model implementations
(152 includes across 95 files) to published `<gufo/ext/...>` headers with zero
non-include edits; added `${CMAKE_SOURCE_DIR}/include` to root `CMakeLists.txt`;
preserved 386 own root-relative and 65 family includes; enforced
`--fail-on-model-core-includes` via `extension_model_core_includes_gate` with negative
proof; and passed bounded CPU/GPU CTests including continuation identity goldens. Its
unresolved gates — none of which it claims — are the deferred functional/performance
half, native Windows binary execution, own/family root-relative coupling (to be migrated
in cards 13–22), and the exempt ASR in-tree test edge until Card 12. See its
[execution record](11-public-header-migrations.md#execution-record).
Card 12 is complete for its in-card gates: added `gufo_test_support` STATIC library
under `BUILD_TESTING` wrapping serving fixtures; migrated all four model API test suites
(Qwen3-ASR, EmbeddingGemma 2, Qwen3-TTS, Qwen-Image-21) to `tests/support/model_api_test_support.hpp`;
removed production serving sources from test targets; tightened the serving boundary
ratchet in `tools/ci/check-extension-isolation.py` (with negative unit test proof) to forbid
`src/cli/` includes across all boundary layers including model test trees; kept test machinery
out of public `SURFACE` (97 paths unchanged) and out of production builds (`BUILD_TESTING=OFF`);
and passed all bounded CPU/GPU CTests, continuation identity goldens, and format checks.
Its unresolved gates — none of which it claims — are the deferred functional/performance half
and remaining CLI text runner coupling in text model integration harnesses. See its
[execution record](12-model-api-test-support.md#execution-record).
Cards 13–29 are pending.
The authorized implementation milestone ends after Card 14 is
verified and committed; cards 15–29 document later scope only. This split is
documentation work and does not itself advance any implementation status.

## Card-by-card workflow

1. Preparation: inspect `jj status`, the card, dependency records and applicable
   `AGENTS.md` files. Identify owned paths, concrete acceptance evidence and
   outstanding jobs. Preserve the user-owned RFC and unrelated work.
2. Implementation: spawn a fresh implementation agent with the complete card,
   relevant decisions, owned paths and required checks. Give it sole editing
   ownership. Require a handoff listing changes, exact checks/results, retained
   evidence, limitations and any live job handles. Do not transfer ownership
   while its editor or mutations remain active.
3. Independent review: use a distinct read-only agent after implementation has
   handed off. Review the actual diff and evidence against the full card and
   RFC, reproduce relevant checks, and report concrete findings with severity,
   paths, practical impact and suggested corrections. Passing implementation
   checks do not replace independent review.
4. Integration: use a third distinct agent to address every finding, add meaningful
   regression coverage when needed, rerun affected checks and update the execution
   record. Even an empty review needs this third agent's acceptance audit and
   verification handoff. Record each finding as resolved, rejected with evidence,
   or still pending; unresolved requirements prevent closure.
5. Closure: the orchestrator inspects the integrated result and proves every
   definition-of-done item with appropriately scoped evidence. Return substantive
   issues to review/integration rather than weakening acceptance. Record all three
   agent names, findings/dispositions, exact checks, artifact/revision evidence and
   limitations. Commit only task-owned paths with `jj` and a Conventional Commit
   message; leave the original user-owned RFC unchanged and uncommitted.
6. Advancement: start the next card only after required verification, integration
   and the closing commit. No competing editors in the shared worktree. Finish or
   explicitly transfer live jobs with their exact handles before ownership changes.

Card 06 is a text-path change, not build infrastructure: its full acceptance
still includes the affected text profiles with `long-context` and `cache` and
matched per-request timings against a main control — but per the user decision
below those functional suites run in the single post-Card-14 qualification
window, not inside the card. Within the card, gate on the identity goldens from
Card 02, the speculative unit tests, static isolation/ISA audits and CPU/hosted
checks, recording every deferred functional command verbatim. Reviewers must read
`docs/model-extensions/evidence/card05/review-and-integration.md` first for the
discovery/selection contracts the core migration now depends on: built extensions
are dependency-first while excluded ones are canonical lexical, the lock is
validated in every mode, and the collision gate is cross-tier only.
Machine/model provenance (weights under `./models`, pinned revisions,
preserved working-copy checkpoint) is recorded in the per-card evidence READMEs
(Card 05: `evidence/card05/README.md`; Card 06: `evidence/card06/README.md`).

## Validation conventions

**User decision (2026-10-31): no functional-harness runs during cards 06–14.**
`tests/functional/run.py` profile rows, `prompt_lookup.py`, standard speed
benchmarks and main-control matched-timing comparisons are deferred to ONE
consolidated qualification window executed after Card 14 is verified, scheduled
solely by the orchestrator. During a card, agents run unit tests (including the
identity goldens), weight-free model tests, CPU/hosted checks, formatting and
static audits only. Evidence stays lean: curated commands, verdict lines,
compact tables and short excerpts; never raw server logs, per-row report trees,
caches or multi-megabyte artifacts under `docs/model-extensions/evidence/`
(the user deleted Card 06's functional report tree as context pollution — keep
small verdict files instead). Deferred requirements are recorded with their
exact commands as pending evidence and are never marked passed before the
qualification window runs them.

Run commands from the repository root in the pinned Nix development environment
unless a card records a different supported environment. Read
[AGENTS.md](../../../AGENTS.md), [testing](../../TESTING.md), the affected model's
`docs/models/<model>/QUALITY.md` and `BENCHMARKS.md`, and the
[functional harness instructions](../../../tests/functional/README.md) before
selecting commands. Resolve actual model paths and drafts under `./models`;
a missing asset or skipped test remains pending evidence.

Discover real target/test names with `ctest --preset gpu-full -N` after configuring
`nix develop -c cmake --preset gpu-test`. Build only the affected target with
`nix develop -c cmake --build --preset gpu-test --target <actual-target>` and run
`nix develop -c ctest --preset gpu-full -R '<anchored-actual-test-regex>'
--output-on-failure`. Angle-bracket examples are placeholders to resolve and
record, not literal commands. New conformance/generator fixtures described in
future cards are to be added; their names are not claims of existing tests.

Functional server caches are scratch, never evidence. A test's server disk
cache must ALWAYS live in a dedicated `/tmp/<test-specific-dir>` (for example
`/tmp/card06-cache/<row-tag>/`) outside the repository, and that directory
MUST ALWAYS be deleted when the test finishes — including failed, aborted and
timed-out runs, which `run.py` does not clean up itself (it removes its own
cache directory only on success). Run each harness invocation with its
`--output` under `/tmp/<test-specific-dir>`; never point `--output` into
`docs/**`. Only small verdict artifacts enter the card's evidence directory
(a per-run summary: status, loaded mode, draft counts, failed cases, phase
deltas); full `report.json` trees and logs stay in `/tmp` during the session
and are deleted with it. Never leave cache extents, model blobs or other
multi-megabyte scratch under `docs/model-extensions/evidence/`. Before closing a card (and after the qualification window), verify
`du -sh docs/model-extensions/evidence/cardNN` and
`find docs/model-extensions/evidence/cardNN -type f -size +5M` show only
explained, durable artifacts.

For hosted CPU scope, configure `cpu-test` and run its relevant tests; use
`nix build .#checks.x86_64-linux.pr` when the card requires bounded hosted checks.
C++ changes need the required formatting check in AGENTS.md. Header/include
changes need `python3 tools/ci/check-extension-api.py --write` followed by a clean
check and the required API policy updates when declarations/contracts change.
No extra GPU matrix is implied for this documentation split.

In the post-Card-14 qualification window, text migrations use explicit
affected suites/modes with `tests/functional/run.py` and `compare.py` per the
harness README, including
`long-context` and `cache` for continuation changes. Record every requested and
actually loaded mode/draft, cache history, work counters, binary/harness/model
hashes and toolchain. Shared text changes need all four profiles; individual
migrations need the affected model/modes. Keep candidate correctness and
independent numerical checks separate from matched main timing controls.
Text timing acceptance is per request/phase at 5% and 10 ms, never averaged.
Non-text service comparisons use their existing harness thresholds (currently
5% and 3 ms for paired service phases), preserving work equivalence and their
own independent quality requirements. Retain failed and inconclusive attempts.
Run the standard speed benchmark as well as the functional suites, and keep
CLI output/timing controls when driver routing changes.

Discovery/default and actual shipped lock-mode HIP artifacts require separate
runtime evidence; a CPU lock build or collision gate proves neither numerical
quality nor performance of the shipped binary. Windows workflow edits are not
Windows execution evidence. Audit actual `CMakeCache.txt`, compile commands,
assertions, sanitizers and stripped production ISA for moved HIP code. Lexical
isolation inventory alone does not prove symbol isolation.

Completed cards retain historical commands/results and their precise limits.
Do not treat a historical `/tmp` artifact as durable new evidence or rerun an
unrelated model sweep. Broaden checks only when scope or failures justify it;
retain current unresolved limitations, including the spike's unproven production
performance parity. Do not graduate API 1.0 without the real elapsed stability
window and all five qualifying API-affecting serving features.
