Orchestrator input brief — instructions and planned commands only; nothing here is a result.

# Card 08 — integration brief (third agent: resolve findings, re-run affected gates once, draft the execution record)

Repo `/home/mixer/gufo`, jj working copy `@` (parent `zwllvvnv ec59bed0`). You are the
third distinct agent on this card (implementation `worker` finished and released
ownership; independent `reviewer` finished read-only). **Do not commit, do not push, do
not `jj new`/`abandon`/`squash`.** Do not touch `docs/model-extensions/RFC.md`
(sha256 `2ce2becbc0bf0f844702b624de6827c2bae6eba921ab8a1ea883659fa2f96448` must hold).

Inputs to read first:
- `docs/model-extensions/cards/08-h3-eval-and-test-targets.md`,
  `docs/model-extensions/CARDS.md` (shared gates + the 2026-10-31 functional deferral),
  `docs/model-extensions/cards/README.md`, `AGENTS.md`.
- Implementation evidence: `docs/model-extensions/evidence/card08/` (`gates.md`,
  `moved-source-parity.md`, `ownership-inventory.md`, `unit-and-identity-tests.md`,
  `pending-commands.md`).
- The full review (F1–F9 with file:line proof, DoD verdicts and its own command list):
  `/home/mixer/.pi/agent/sessions/--home-mixer-gufo--/subagent-artifacts/e7a49787-2db0-4a73-a81a-83e24f3259ac_reviewer_output.md`
  — read it completely. A copy of the diff under review is at
  `/tmp/card08-review/diff.patch`.

Environment rules (unchanged from the run): pinned `nix develop` shell; scratch only
under `/tmp/card08-*` and **always delete it when each step finishes, failures included**;
evidence files under 1 MB; no raw build logs in `docs/`; keep your own tool output small
(head/tail/grep, redirect long output to `/tmp/card08-int/*.log` and quote verdict lines
only). **Never build into `build/card07-*` or `build/card06-*`** — they are the
parent-revision baselines the parity gates read. No functional-harness runs, no
timings, no speed benchmark (deferred by user decision; they stay PENDING in
`pending-commands.md`). If a command is impossible in your sandbox (permission/socket
denial), record the exact command, the denial text and exit code in
`docs/model-extensions/evidence/card08/integration-reruns.txt` and leave the item PENDING
for the orchestrator — never report an unrun command as a pass.

## Findings to close (all of them; each ends RESOLVED, REJECTED-with-evidence, or PENDING)

**F1 Major — the flags/assertion/sanitizer record is wrong and a real CPU-test codegen
change is undocumented** (`unit-and-identity-tests.md:62-73`).
- Rewrite that section per configuration (production release / HIP-test / CPU-test /
  cpu-sanitizer): `gufo_minimax_h3`, `gufo_eval`, `gufo_compare` are `NDEBUG` everywhere
  (they are created before `add_compile_options(-UNDEBUG)`), only
  `gufo_deepseek_v4_flash_text` gets `-UNDEBUG` under `BUILD_TESTING`; the text tier's
  unconditional `-O3 -ffast-math -fno-finite-math-only` is a *new* codegen state for
  `ENGINE_ENABLE_HIP=OFF` test builds (the parent compiled those two files only inside
  test executables at `-O2` without fast-math) and `-UNDEBUG`-armed text objects now
  reach the HIP-test `gufo`. State it plainly, justify or remove it. Judge explicitly
  whether keeping `-O3 -ffast-math` on a CPU-only test build is acceptable for this card
  (the sampler is numerically sensitive); if you keep it, record the reason, and if the
  honest answer is "the production flags should not leak into CPU test builds", scope the
  property so the CPU test state matches the parent (`-O2`, no fast-math) — one
  implementation per source stays a hard requirement either way.
- Then actually build and run a sanitizer configuration in a **new** directory (do not
  reuse `build/cpu-sanitizer`):
  `nix develop -c cmake -S . -B build/card08-sanitizer -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DGUFO_ENABLE_SANITIZERS=ON -DENGINE_ENABLE_HIP=OFF`
  and build only the affected targets (`gufo_deepseek_v4_flash_text gufo_eval gufo_compare
  gufo_minimax_h3 ds4_sampling_test ds4_chat_template_test ds4_eval_test
  eval_regrade_test openai_chat_test minimax_h3_runtime_test minimax_h3_tokenizer_test
  minimax_h3_sampling_test minimax_h3_generation_test`), then
  `nix develop -c ctest --test-dir build/card08-sanitizer -R '^(ds4\.(sampling|template|eval)|eval_regrade_test|openai_chat_test|minimax_h3_(runtime|tokenizer|sampling|generation)_test)$' --output-on-failure`.
  Record the cache line and, per moved source, whether `-fsanitize=address,undefined` is
  present versus the parent's state. Budget ~40 minutes; if it cannot finish, keep the
  partial state out of `docs/`, delete the directory, and write the exact commands as a
  PENDING row.

**F2 Major — `gufo_compare` edge added twice and `target-ownership.md` contradicts the
build** (`tests/models/qwen/CMakeLists.txt:5` vs `:57-60`;
`docs/model-extensions/target-ownership.md:96-101`). Drop `gufo_compare` from the shared
`add_qwen_cpu_test` line and keep only the four targets that call `CompareLogits`; rebuild
`qwen_oracle_test qwen_l1_norm_module_test qwen_l1_ffn_module_test qwen_l1_ssm_module_test`,
re-run `ctest --test-dir build/card08-gpu -R '^(qwen_oracle_test|qwen_l1_(norm|ffn|ssm)_module_test)$' --output-on-failure`
and the ownership gate; fix the doc sentence to match.

**F3 Major — backwards edge published + no over-claim check**
(`src/models/deepseek_v4_flash/library.json:9`, `ownership-inventory.md:21`,
`target-ownership.md:52` vs `text/CMakeLists.txt:39`). Remove the false consumer claim and
its prose, then make claim validation symmetric in
`tools/ci/check-extension-isolation.py` (every declared consumer/check/tool/compiled_elsewhere
entry must correspond to a derived edge for that library) and add one negative fixture
("claims a dependent that does not depend on it"). Re-run
`python3 tests/tools/check_extension_isolation_test.py` and
`python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership`.

**F4 Major — hosted PASS provenance.** Do not argue the mechanism: change
`gates.md:57-72` so the pre-commit hosted row reads "coverage not established (dirty git
tree → Nix snapshot from tracked parent paths)" including the
`warning: Git tree '/home/mixer/gufo' is dirty` line, and make P7 in
`pending-commands.md` the authoritative post-commit re-run with the source-snapshot proof
(`nix build .#checks.x86_64-linux.pr --no-link --print-out-paths`,
`nix eval --raw .#packages.x86_64-linux.release.src`, then list
`$src/src/models/minimax_h3/CMakeLists.txt`, `$src/tests/fixtures/eval/antirez-ds4.json`
and `find $src/tests -type f`). The orchestrator runs those after the closing commit; do
not claim a hosted pass.

**F5 Minor — the parity tool cannot detect a duplicated or dropped source**
(`tools/prof/card08_parity.py:65`, `:291`). Make duplicate compilation of one source across
targets a failure (this turns DoD 3 into a machine check) and make a source present on one
side only a failure (or an explicitly listed, justified exception). Re-run all three modes
against the existing baselines; the verdicts must stay clean on the same inputs.

**F6 Minor — device mode passes when a device image disappears**
(`tools/prof/card08_parity.py:333`): fail when `.hip_fatbin` exists on one side only; keep
the both-sides-wrapper allowance. Re-run device mode (scratch `/tmp/card08-parity`,
deleted afterwards).

**F7 Minor — evidence misses the per-card README and the resume notes.** Add
`docs/model-extensions/evidence/card08/README.md` (revisions + parent, toolchain/compiler
and ROCm identity from the pinned shell and the build caches, the candidate build dirs with
their cache variables, `models/` asset availability with what is missing, untouched
baselines, host/RAM/llama-swap state) and a `RESUME-NOTES.md` in the same directory
reconstructing the actual run phases and remaining commands, labelled as reconstructed at
handoff (the implementation agent deleted its `/tmp` copy).

**F8 Minor — the worker's input brief sits in the evidence directory looking like results**
(`brief-worker.md`, which even contains a `--prefix` flag that does not exist). Rename it to
`input-brief-worker.md` and add a first line: "Orchestrator input brief — planned commands,
none of these are results." Keep it in the tree (it records what the agent was told).

**F9 Minor — `docs/model-extensions/target-ownership.md` is an orphan.** Link it from
`docs/model-extensions/cards/08-h3-eval-and-test-targets.md` (follow the
`INFRASTRUCTURE.md`/`SPIKE.md` precedent) when you write the execution record, and make
sure `python3 tools/ci/check-docs.py` still passes.

Also close the reviewer's validation gap list honestly in the record: the Nix production
derivation was only path-evaluated, never built — record
`nix build .#packages.x86_64-linux.release --no-link` as PENDING with the reason (full
HIP package build; distribution-level gate), and keep the local `build/card08-release`
artifact as the in-card production evidence with its install-rule path verified by the
filter evaluation already recorded.

## Then draft the execution record

Write the `## Execution record` section of
`docs/model-extensions/cards/08-h3-eval-and-test-targets.md` in the style of Card 07's
record: the three agents with their names and run IDs (implementation `worker`
`3f4f4c60-7afa-4d25-9f02-14cd8b7070d6`, independent review `reviewer`
`e7a49787-2db0-4a73-a81a-83e24f3259ac`, integration `codex-exec-writer` — leave the
literal token `<INTEGRATION-RUN-ID>` for the orchestrator to fill), a findings table
(F1–F9: disposition + evidence pointer), the acceptance audit with exact commands/verdicts,
evidence paths, and the remaining limitations (deferred functional/performance half,
Windows execution, asset-blocked H3 oracle rows, per-model extension exclusion not
exercisable until `extension.json` exists, pre-existing `kernel_resources_test` baseline
drift, Nix production derivation unbuilt). Tick only the DoD boxes that evidence actually
proves — DoD 2 stays unticked or explicitly half-proven with the reason. Add the
`target-ownership.md` link (F9) in the card body. Also update `docs/model-extensions/CARDS.md`
(status column for card 08 → `integrating`, not `complete` — the orchestrator flips it) and
the `cards/README.md` checkpoint paragraph only if you can state it accurately; otherwise
leave both to the orchestrator and say so in your handoff.

## Handoff (final message)

List: files changed since the review, each finding's disposition with evidence, every
re-run command with exit code and verdict line, what you REJECTED and why, what stays
PENDING with its exact command, evidence paths, RFC hash re-check, confirmation that no
`/tmp/card08-*` scratch survived and that `build/card07-*`/`card06-*` were untouched, and
the sentence "editing ownership released". Nothing committed.
