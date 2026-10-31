Orchestrator input brief — instructions and planned commands only; nothing here is a result.

# Card 08 — reviewer brief (independent, READ-ONLY, no edits)

You are the independent reviewer for Card 08 of the model-extensions RFC in
`/home/mixer/gufo`. The implementation agent finished and released editing
ownership. **You must not edit anything.** Your profile has no shell: when a claim
needs execution, write the exact command (with cwd, env and expected verdict line) in a
"Commands for the integration agent" section instead of assuming a result. A prepared
diff is provided so you never need `jj`.

## Inputs

- Diff of the current jj revision (working copy vs parent `zwllvvnv ec59bed0`):
  - `/tmp/card08-review/STAT.txt` (stat), `/tmp/card08-review/NAMES.txt` (paths)
  - `/tmp/card08-review/diff.patch` (full, 2609 lines — read in chunks)
  - `/tmp/card08-review/patches/<path-with-underscores>.patch` (one per changed file,
    including full contents of added files)
  - `/tmp/card08-review/newfiles/…` (verbatim copies of the new manifests, the new
    `tools/prof/card08_parity.py` and `docs/model-extensions/target-ownership.md`)
- Contract: `docs/model-extensions/cards/08-h3-eval-and-test-targets.md`
  (implementation plan, definition of done, validation criteria),
  `docs/model-extensions/CARDS.md` (shared acceptance gates, the 2026-10-31 functional
  deferral), `docs/model-extensions/cards/README.md` (validation conventions),
  `AGENTS.md`, and `docs/model-extensions/RFC.md` §Target graph, §Metadata and build
  manifests, §Tests, benchmarks, tools and documentation, plus the step-5 row of the
  implementation-plan table. The worker's brief is
  `docs/model-extensions/evidence/card08/brief-worker.md`.
- Evidence claimed by the worker: `docs/model-extensions/evidence/card08/`
  (`gates.md`, `moved-source-parity.md`, `ownership-inventory.md`,
  `unit-and-identity-tests.md`, `pending-commands.md`). The raw `/tmp/card08-run/`
  logs the worker used were deleted at handoff, so re-derivable claims must be
  re-run, not re-read.
- Parent-state reference for how this repo evidences a card:
  `docs/model-extensions/evidence/card07/orchestrator-gates.txt` and
  `pending-commands.md`.

## What the card had to achieve (verbatim DoD)

1. H3, eval and comparison targets have explicit ownership and dependency graphs.
2. Every inventoried model-dependent test/tool has manifest registration or a
   justified condition; no single-extension exclusion leaves unresolved model
   references.
3. DeepSeek CPU text support has one implementation per source and no premature
   HTTP tool-syntax migration.

## Review questions (answer each with file:line evidence from the diff)

**Build correctness — highest value, be adversarial here.**
- `gufo_minimax_h3`: does it contain exactly the sources `gufo_core` used to compile
  (CPU list incl. the `$<$<PLATFORM_ID:Windows>:…>` media pair, HIP list under
  `ENGINE_ENABLE_HIP`), the `dit.hip` per-source `-O2`, the
  `GUFO_FFMPEG_EXECUTABLE`/`GUFO_FFPROBE_EXECUTABLE` definitions (private vs PUBLIC —
  who else read them?), the `${CMAKE_SOURCE_DIR}` include, and every PUBLIC link edge
  the moved code actually needs (ICU/PNG/JPEG/WEBP/CURL/OpenSSL/OpenMP/HIP/hipblas…)?
  Is the `add_subdirectory()` placed so warnings/sanitizers match what core provided?
  Anything left in `gufo_core` from these directories, or compiled twice anywhere?
- Do all consumers that used to receive H3/compare/eval symbols *through* `gufo_core`
  now link them explicitly — `gufo_http`, `gufo`, `gufo_llm_cli`,
  `gufo-kernel-bench`, `tune_hipblaslt`, `wmma_layout_test`, and each affected test — in
  **every** configuration (HIP ON/OFF, `BUILD_TESTING` ON/OFF, `GUFO_PLATFORM_ONLY`,
  Windows generator expressions)? Static-library link order/INTERFACE propagation is the
  classic silent break: check for a target that only works because it happened to be
  built earlier.
- `gufo_deepseek_v4_flash_text`: exactly `chat_template.cpp` + `runtime/sampling.cpp`;
  `gufo_deepseek_v4_flash` must not compile them again; `tool_calls.cpp` must stay only
  in `gufo_http`; no new tool-syntax movement. Verify the claimed `-O3 -ffast-math`
  preservation and `-UNDEBUG`-only-under-`BUILD_TESTING` against the real files, and
  judge the assertion/sanitizer/warning consequences for *both* the production `gufo`
  binary and the tests in each build mode. Is there exactly one implementation per
  source in every mode (DoD 3)?
- `openai_chat_test` now links `gufo_http`: does that pull HTTP/serving state into a CPU
  test in a way that changes what the test proves (init order, symbol interposition,
  duplicate `openai_chat.cpp` copies), and does it still work on a Windows CPU build
  where much of the tree is excluded?

**Ownership mechanism (DoD 2).**
- Is `--fail-on-target-ownership` a real gate: can it fail (are there negative
  fixtures), is its CMake parsing robust enough for the file it parses
  (`${GUFO_QWEN_LIBRARIES}` indirection, multi-line calls, generator expressions,
  `target_link_libraries` with `PRIVATE/PUBLIC/INTERFACE`, added/renamed tests), and
  would it catch a newly added model-dependent test that nobody registered? Does it
  silently skip files/targets it cannot parse (the dangerous kind of PASS)?
- Does the inventory in `ownership-inventory.md` / `docs/model-extensions/target-ownership.md`
  cover *every* model-dependent test/tool registered outside the model directories
  (compare against root `CMakeLists.txt` yourself), and does each entry name an owner or
  a justified condition? Is `docs/model-extensions/target-ownership.md` the right home
  (it is a permanent repo document: is it indexed/drift-checked, or does it duplicate
  card evidence), and does the docs gate cover it?
- H3 exclusion: the card demands that excluding one model leaves no unresolved
  references. `extension.json` manifests do not exist yet, so `GUFO_EXTENSIONS` cannot
  exclude a model. Is what the worker recorded (HIP-off, platform-only, empty/typo
  `GUFO_EXTENSIONS`) an honest substitute plus an explicitly pending item, or does the
  evidence overclaim exclusion coverage?

**Evidence integrity.**
- Any skipped test (`SKIP_RETURN_CODE 77`, missing weights), unshipped lock-mode
  artifact, Windows build or timing statement presented as a pass rather than PENDING?
  H3 weights are absent under `./models`, so all `minimax_h3_*` HIP/oracle rows are
  pending by definition.
- **Provenance check:** `nix build .#checks.x86_64-linux.pr` and the Nix package/fileset
  read their sources from the **git-tracked tree**, and the working copy was dirty during
  the run (`nix` printed "Git tree … is dirty"). Determine precisely what the recorded
  hosted PASS can and cannot prove about *this* revision, and state the command that must
  be re-run after the closing commit.
- Same for the parity gates: `flags compared=27 failures=0 (21 identical, 6 justified)`
  — read the justification table in `moved-source-parity.md` and reject any justified
  difference that is not actually forced by the move (a dropped warning, include dir,
  definition or optimization would be a real behavior change). Device parity is
  `compared=8 failures=0` while 9 HIP-ish files moved (`…​.hip` × 8 + `runtime_hip.cpp`):
  confirm the count is right and nothing silently fell out of the comparison set.
- `tools/prof/card08_parity.py` vs the existing `tools/prof/card07_parity.py`: is the new
  361-line tool necessary, does it duplicate Card 07 logic that could have been
  parameterized, is it correct (bundler resolution, ELF section parsing, empty-image
  handling, exit code), and is it a durable repo tool or card scratch that should live in
  the evidence directory?

**Hygiene.** One canonical name per behavior (no aliases); model knowledge must not
re-enter core build/CI/Nix files; the eval dataset must be admitted to the production
derivation as exactly one file (`tests/fixtures/eval/antirez-ds4.json`) with no other
model test path admitted and no stale reference to the old path anywhere in tracked
files (`src/cli/eval/eval.cpp`, `tests/models/deepseek_v4_flash/*`, install rule,
`docs/`); `docs/model-extensions/RFC.md` must be untouched; no leftover scratch paths
(`grep` the diff for `/tmp/card08`); nothing under `docs/` over 1 MB.

## Output format

Findings only, each with: ID (`F1`…), severity (Critical/Major/Minor), file:line, what
is wrong, practical impact, and the smallest concrete correction. Then: DoD item-by-item
verdict (proven / partially proven / not proven, with the evidence that decides it),
and a final "Commands for the integration agent" list for every check you could not
execute yourself. Say `No issues found.` if nothing qualifies. Do not soften a finding
because the gates passed, and do not report style preferences.
