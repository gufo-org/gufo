## Review

Read-only pass. No edits made. Evidence base: `/tmp/card08-review/diff.patch` + per-file patches, the current tree, and the card/RFC/AGENTS contract. Line citations are against the working copy.

### Correct (worth keeping, with proof)

- **The extraction itself is right.** `gufo_minimax_h3` carries exactly the 13 CPU entries (incl. both `$<$<PLATFORM_ID:Windows>:media_windows.cpp>` / `media_posix.cpp>`), the 9 HIP entries, the `dit.hip` appended `-O2` and the `GUFO_FFMPEG_EXECUTABLE`/`GUFO_FFPROBE_EXECUTABLE` defines (`src/models/minimax_h3/CMakeLists.txt:16-51`); `gufo_core` lost all of them plus its now-orphan `set_property(SOURCE …dit.hip)` (`CMakeLists.txt` patch hunk `@@ -320,31` / `@@ -373,18`). I verified by grep that no build file outside `src/models/minimax_h3/` references a `src/models/minimax_h3/*` source any more — the remaining hits are the installable data manifest.
- **The warnings/sanitizers trap was handled deliberately**: the new `add_subdirectory()` calls sit after the `GUFO_ENABLE_WARNINGS`/`GUFO_ENABLE_SANITIZERS` block (`CMakeLists.txt:368-379`), so the moved TUs keep the flags they had as `gufo_core` objects, and `-UNDEBUG` (line 542) still only reaches test-scope targets. Only `media_posix.cpp`/`media_windows.cpp` read the FFmpeg defines (grep), and they still get them privately.
- **DoD 3 holds in the tree**: `chat_template.cpp` and `runtime/sampling.cpp` now appear in exactly one target (`src/models/deepseek_v4_flash/text/CMakeLists.txt:17-18`), `gufo_deepseek_v4_flash` links it instead of compiling them (`src/models/deepseek_v4_flash/CMakeLists.txt:57`), `tool_calls.cpp` is compiled only by `gufo_http` (`CMakeLists.txt:420`), and `openai_chat_test` no longer compiles any of the three (`CMakeLists.txt:651-658`). No HTTP tool-syntax moved.
- **The fixture move is complete**: `src/cli/eval/eval.cpp:31`, `CMakeLists.txt:452/510/1120`, `tests/models/deepseek_v4_flash/{CMakeLists.txt:144,dataset_test.py:61,eval_test.cpp:10}` and `.devops/nix/package.nix:57-62` all point at `tests/fixtures/eval/antirez-ds4.json`; grep finds no live reference to the old path (only historical card evidence).
- **The new gate is a real gate**: `--fail-on-target-ownership` derives edges from a paren-balanced CMake scan with variable expansion (`check-extension-isolation.py:557-625`), fails on unclaimed dependents, on compiling a library source without `compiled_elsewhere`, on stale claims, on duplicate names/targets and on a model directory without a manifest; four new fixtures (`tests/tools/check_extension_isolation_test.py:323-435`) prove the failure paths including the repository assertion.
- **Skip handling is honest**: the nine asset-gated H3 rows are recorded as skipped with the reason and the command (`unit-and-identity-tests.md:38-52`, `pending-commands.md` P8), functional/timing rows are PENDING (P1–P7), and the Windows failure is correctly proven pre-existing: `.devops/nix/windows-platform-tests.nix:23-24` admits only two of the three tests that `cmake/PlatformTests.cmake:17` registers, and neither file is in this diff.

### Fixed

None — read-only review.

### Findings

**F1 — Major (P1). The sanitizer/assertion/codegen record is factually wrong for most moved sources and omits a real CPU-build codegen change.**
`docs/model-extensions/evidence/card08/unit-and-identity-tests.md:62-73`.
(a) "test builds compile the moved sources with `-UNDEBUG`" is false for `gufo_minimax_h3`, `gufo_eval`, `gufo_compare`: they are created at `CMakeLists.txt:373-377`, i.e. before `add_compile_options(-UNDEBUG)` at `CMakeLists.txt:542-544`, so they are `NDEBUG` in every configuration — exactly as they were inside `gufo_core`. Only `gufo_deepseek_v4_flash_text` gets `-UNDEBUG` (`src/models/deepseek_v4_flash/text/CMakeLists.txt:35-37`).
(b) The `-O3 -ffast-math -fno-finite-math-only` options (`text/CMakeLists.txt:31-32`) are unconditionally applied to `chat_template.cpp`/`runtime/sampling.cpp`. In an `ENGINE_ENABLE_HIP=OFF` test configuration the parent had **no** such compilation — the only copies were the test-executable ones built at `RelWithDebInfo -O2` without fast-math. So `ds4.sampling`, `ds4.eval` and `openai_chat_test` now exercise a differently optimized sampler, and "reproduces the parent state exactly" (:68) overstates it; the release-vs-release parity run cannot see this. Symmetrically, in `BUILD_TESTING` HIP builds the `gufo` executable now links `-UNDEBUG`-armed text objects where the parent linked the `NDEBUG` copies from `gufo_deepseek_v4_flash`.
(c) No sanitizer configuration was built at all, yet the card's plan requires the old/new sanitizer state; the row infers "unchanged" from three non-sanitizer builds.
Smallest fix: rewrite the section per configuration (production / HIP-test / CPU-test / cpu-sanitizer), stating the `-O3 -ffast-math` gain for the CPU-test compilation and the assertion gain for the test-build `gufo`, and either run `cmake --preset cpu-sanitizer` + `cmake --build --preset cpu-sanitizer` + the `-R '^(ds4\.(sampling|template|eval)|eval_regrade_test|minimax_h3_(runtime|tokenizer|sampling|generation)_test)$'` selection once, or move it to `pending-commands.md` as an explicit PENDING row.

**F2 — Major (P1). `gufo_compare` link edge is added twice and the documentation asserts the opposite of the build.**
`tests/models/qwen/CMakeLists.txt:5` adds `gufo_compare` to *every* `add_qwen_cpu_test` target (10 targets: tokenizer, chat_template, vision_cache, cpu_oracles, forward, generator, ssm, fusion_route, attention_policy, gemm_route), while `:57-60` adds the same edge again to the four targets that actually call `CompareLogits`. `docs/model-extensions/target-ownership.md:96-101` states "other tests include that header without reaching the comparison symbol and therefore do not link the library" — contradicted by line 5. Impact: the ownership graph the card exists to produce records ~10 dependencies that do not exist in intent, the doc is wrong about its own tree, and future readers cannot tell which edge is real. Smallest fix: drop `gufo_compare` from line 5 and keep `:57-60` (the failure recorded in `gates.md:41-44` was exactly those four targets), then rebuild `qwen_oracle_test qwen_l1_norm_module_test qwen_l1_ffn_module_test qwen_l1_ssm_module_test` and re-run `--fail-on-target-ownership`. (If the wider edge is intended, delete the four duplicate calls and rewrite the doc sentence instead — but do not keep both.)

**F3 — Major (P1). The ownership graph publishes a backwards dependency, and the gate has no over-claim check.**
`src/models/deepseek_v4_flash/library.json:9` lists `gufo_deepseek_v4_flash_text` as a *consumer* of `deepseek_v4_flash`; `ownership-inventory.md:21` states it "links the model library", and `target-ownership.md:52` prints it as a production consumer. It does not: `text/CMakeLists.txt:39` links only `gufo_core`, and the real edge runs the other way (already recorded correctly in `text/library.json:7`). The gate cannot catch this: a target declared inside the library directory is skipped outright (`tools/ci/check-extension-isolation.py:588`) and claim validation only tests target existence (`:621`), never that a claim corresponds to a derived edge. Impact: the record that later cards will use to decide "what breaks if I drop this library" contains a false edge, and any future manifest can add unverified edges without failing anything. Smallest fix: delete `gufo_deepseek_v4_flash/library.json:9` (and the corresponding table/inventory text), then make claims symmetric — for each claim require it to appear in `edges` for that library — with one negative fixture ("claims a dependent that does not depend on it").

**F4 — Major (P1). The hosted PASS is recorded against an unreconciled source-provenance claim.**
`gates.md:57-72` asserts the `.pr` check and the `default.src` snapshot saw the *uncommitted* new files, but this repository's own Card 05 provenance record (`docs/model-extensions/evidence/card05/review-and-integration.md:71`) concludes the opposite for the same mechanism: a new file (`tests/extensions/discovery_test.py`) was **absent** from `nix eval --raw .#packages.x86_64-linux.default.src` while a tracked fixture was present. Both cannot be right, and the difference decides whether the hosted gate covers this revision at all. The record also omits the `warning: Git tree '/home/mixer/gufo' is dirty` line that `nix` printed, which is the reader's only clue about which snapshot was built. Smallest fix: after the closing commit, run `nix build .#checks.x86_64-linux.pr` and record the built source hash together with `nix eval --raw .#packages.x86_64-linux.release.src` + `ls $src/src/models/minimax_h3/CMakeLists.txt`, and until then mark the pre-commit hosted row "coverage not established (pre-commit snapshot)" rather than a clean PASS. P7 already schedules the re-run — it must become the authoritative row.

**F5 — Minor (P2). The parity tool can silently shrink its own comparison set, so it cannot prove "one implementation per source".**
`tools/prof/card08_parity.py:65` keys `compile_commands.json` by source path, so if a source were compiled by two targets only the last entry survives, and `:291` requires the source to exist on **both** sides, so a source dropped from the candidate build is simply not compared (no failure, lower `compared=` count). Both are exactly the two regressions a move can introduce, and the recorded `compared=27` matches the expectation only because nothing went wrong this time. Smallest fix: in `load()`, count occurrences per source and fail when any source appears in more than one target's object list (that also turns DoD 3 into a machine check), and fail on sources present in only one side (or list them explicitly).

**F6 — Minor (P2). Device mode passes when a device image disappears.**
`tools/prof/card08_parity.py:333` counts a missing `.hip_fatbin` as `no device image (host wrapper)` and continues, so a moved `.hip` TU that stopped emitting kernels would be reported as skipped, not as a failure — the precise regression this gate exists to catch. Smallest fix: fail when the fatbin exists on one side only (keep the wrapper allowance for a source that has no device image on both sides).

**F7 — Minor (P2). Evidence directory misses the per-card README and the resume notes the run plan required.**
`docs/model-extensions/evidence/card08/` contains six files but no `README.md`; `docs/model-extensions/cards/README.md` states machine/model provenance (weights under `./models`, pinned revisions, working-copy checkpoint) belongs in the per-card evidence README (Card 04/05/06 have one), and the run plan required a `RESUME-NOTES.md` copy at handoff. The provenance that does exist is squeezed into the `gates.md:3-7` header and omits the toolchain identity (compiler/ROCm versions, Nix revision) and the `models/` asset state that the shared gates ask to record. Smallest fix: add a short `README.md` with revision/parent, toolchain, the three candidate build dirs plus their cache variables, asset availability, and a pointer to the RESUME-NOTES content (or restore it).

**F8 — Minor (P2). The worker's instruction sheet is committed as evidence and contains commands that were never run and cannot run.**
`docs/model-extensions/evidence/card08/brief-worker.md` (253 lines) sits next to the results; e.g. it tells the reader to call `tools/prof/card07_parity.py … --prefix src/models/minimax_h3/`, a flag the tool does not implement (the final tool is `card08_parity.py` without `--prefix`). The card's own rule is "a planned command is not a result", and this file is the one place in the evidence directory where that distinction is not visible. Smallest fix: rename/label it (`INPUT-brief-worker.md` or a first line "orchestrator input brief — contains planned commands, none of these are results"), or move it out of the evidence directory.

**F9 — Minor (P2). The new permanent document is unreachable from the card navigation.**
`docs/model-extensions/target-ownership.md` is referenced only from card-08 evidence files; the established convention for permanent docs in this directory is a link from the owning card (`docs/model-extensions/cards/04-…:61` and `05-…:44` → `INFRASTRUCTURE.md`, `03-…:46` → `SPIKE.md`). `check-docs` validates links, not orphans, so nothing catches this. Smallest fix: one link from `docs/model-extensions/cards/08-h3-eval-and-test-targets.md` when the execution record is written.

### Definition-of-done verdicts

1. **"H3, eval and comparison targets have explicit ownership and dependency graphs" — Proven, with a record defect.** Own `CMakeLists.txt` + `library.json` per tier, `gufo_core` stripped, edges derived from the real CMake graph and enforced by `extension_target_ownership_gate`. Blemish: F3 (one false edge in the published graph, no over-claim check).
2. **"Every inventoried model-dependent test/tool has manifest registration or a justified condition; no single-extension exclusion leaves unresolved model references" — Partially proven.** The first half holds: `ownership-inventory.md` covers every root-registered dependent, claims are gated, `tests/models/<model>/` exemption is a documented open decision, and `compiled_elsewhere` names the only two remaining duplicate compilations (`tool_calls.cpp`, `http_client.cpp`). The second half cannot hold yet and is honestly recorded as such (`gates.md:46-54`): no `extension.json` exists, so per-model `GUFO_EXTENSIONS` exclusion is unexercisable until Card 09+; the HIP-off configuration is a substitute, not the criterion.
3. **"DeepSeek CPU text support has one implementation per source and no premature HTTP tool-syntax migration" — Proven.** Grep-verified single compilation of both sources, `tool_calls.cpp` untouched in `gufo_http`, no symbol dependency from `chat_template.cpp` on it. Caveat: the proof is currently inspection-only — the parity tool would not detect a duplicate (F5), and the codegen side effect on CPU test builds is undocumented (F1).

### Validation-criteria gaps (report-only, already partially recorded)

Windows build/run: PENDING, with the blocking cause proven pre-existing (P9). H3 numerical oracles: PENDING (P8). Functional/`prompt_lookup`/timings/speed benchmark: PENDING (P1–P7) per the 2026-10-31 deferral. The **production Nix derivation was only path-evaluated, never built**, so the new install rule + Nix filter combination has no end-to-end proof. Sanitizer state was inferred, never built (F1c).

### Commands for the integration agent

```sh
# F2 — after dropping the redundant gufo_compare edge (or the four duplicates)
nix develop -c cmake --build build/card08-gpu --target qwen_oracle_test qwen_l1_norm_module_test qwen_l1_ffn_module_test qwen_l1_ssm_module_test
nix develop -c ctest --test-dir build/card08-gpu -R '^(qwen_oracle_test|qwen_l1_(norm|ffn|ssm)_module_test)$' --output-on-failure

# F1c — the missing sanitizer configuration
nix develop -c cmake --preset cpu-sanitizer
nix develop -c cmake --build --preset cpu-sanitizer
nix develop -c ctest --preset cpu-sanitizer -R '^(ds4\.(sampling|template|eval)|eval_regrade_test|minimax_h3_(runtime|tokenizer|sampling|generation)_test|openai_chat_test)$' --output-on-failure
# record: grep -m1 'GUFO_ENABLE_SANITIZERS\|CMAKE_BUILD_TYPE' build/cpu-sanitizer/CMakeCache.txt
# and, per moved source, that -fsanitize=address,undefined is present in compile_commands.json exactly where the parent had it

# F3 — after removing the backwards consumer entry
python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership
python3 tools/ci/check-extension-isolation.py --target-ownership-report | grep deepseek
python3 tests/tools/check_extension_isolation_test.py

# F5/F6 — after hardening the parity tool (must still pass on the same inputs)
nix develop -c python3 tools/prof/card08_parity.py --mode flags  --before build/card07-release/compile_commands.json --after build/card08-release/compile_commands.json
nix develop -c python3 tools/prof/card08_parity.py --mode object --before build/card07-release/compile_commands.json --after build/card08-release/compile_commands.json
nix develop -c python3 tools/prof/card08_parity.py --mode device --before build/card07-release/compile_commands.json --after build/card08-release/compile_commands.json --scratch /tmp/card08-parity
# cleanup required: rm -rf /tmp/card08-parity

# F4 — authoritative hosted provenance, run AFTER the closing commit
nix build .#checks.x86_64-linux.pr --no-link --print-out-paths
nix eval --raw .#packages.x86_64-linux.release.src   # then: ls $src/tests/fixtures/eval/antirez-ds4.json $src/src/models/minimax_h3/CMakeLists.txt; find $src/tests -type f

# production derivation end-to-end (the install rule + Nix filter pair has never been built)
nix build .#packages.x86_64-linux.release --no-link   # or record as PENDING with this exact command

# static gates to re-run after the last edit
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
python3 tools/ci/check-extension-api.py --write && python3 tools/ci/check-extension-api.py
python3 tools/ci/check-docs.py
python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-family-graph
python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership
```

**Merge verdict: OK with notes — not merge-ready as-is.** The build/CMake work is sound and its code-identity evidence is strong (byte-identical device sections, unchanged kernel scratch, one implementation per source), but four Major items must be closed first: the inaccurate sanitizer/assertion/codegen record (F1), the duplicated/contradicted `gufo_compare` edge (F2), the backwards ownership record plus missing over-claim check (F3), and the unresolved hosted-check provenance (F4, closed by the post-commit re-run).